"""Run GRPO with local DeepSpeed state mirrored to eduLLM's S3 prefixes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

_TRANSFER = TransferConfig(
    multipart_threshold=64 * 1024**2,
    multipart_chunksize=64 * 1024**2,
    max_concurrency=8,
    use_threads=True,
    # The CRT transfer manager in the boto3/botocore versions resolved by the
    # image injects ContentLength into ExtraArgs, then rejects its own key.
    # Classic transfers still use awscrt for CRC32C checksums without that bug.
    preferred_transfer_client="classic",
)
_UPLOAD_ARGS = {"ChecksumAlgorithm": "CRC32C"}
_DOWNLOAD_ARGS = {"ChecksumMode": "ENABLED"}
_LOCAL_COMPLETE = ".checkpoint_state_complete"


def _validate_tag(tag: str) -> str:
    path = Path(tag)
    if not tag or path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError(f"unsafe checkpoint tag {tag!r}")
    return tag


class S3Tree:
    def __init__(self, uri: str) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"expected s3:// URI, got {uri!r}")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.lstrip("/").rstrip("/") + "/"
        self.client = boto3.client("s3")

    def key(self, relative: str) -> str:
        return self.prefix + relative.lstrip("/")

    def read(self, relative: str) -> bytes | None:
        try:
            return self.client.get_object(
                Bucket=self.bucket,
                Key=self.key(relative),
                ChecksumMode="ENABLED",
            )["Body"].read()
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                return None
            raise

    def put(self, relative: str, body: bytes) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key(relative),
            Body=body,
            ChecksumAlgorithm="CRC32C",
        )

    def upload_tree(self, source: Path, remote_dir: str) -> list[dict]:
        manifest = []
        for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
            relative = path.relative_to(source).as_posix()
            self.client.upload_file(
                str(path),
                self.bucket,
                self.key(f"{remote_dir.rstrip('/')}/{relative}"),
                ExtraArgs=_UPLOAD_ARGS,
                Config=_TRANSFER,
            )
            manifest.append({"path": relative, "size": path.stat().st_size})
        return manifest

    def download_manifest(self, remote_dir: str, destination: Path, manifest: list[dict]) -> None:
        for entry in manifest:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe checkpoint path {entry['path']!r}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(
                self.bucket,
                self.key(f"{remote_dir.rstrip('/')}/{relative.as_posix()}"),
                str(target),
                ExtraArgs=_DOWNLOAD_ARGS,
                Config=_TRANSFER,
            )
            if target.stat().st_size != entry["size"]:
                raise OSError(f"{target} has {target.stat().st_size} bytes; expected {entry['size']}")


def restore_latest(store: S3Tree, checkpoint_dir: Path) -> str | None:
    latest_blob = store.read("latest")
    if latest_blob is None:
        return None
    tag = _validate_tag(latest_blob.decode().strip())
    marker_blob = store.read(f"{tag}/_SUCCESS")
    if marker_blob is None:
        raise RuntimeError(f"remote checkpoint latest={tag!r} has no _SUCCESS marker")
    marker = json.loads(marker_blob)
    target = checkpoint_dir / tag
    if not target.exists():
        temporary = checkpoint_dir / f".{tag}.download"
        temporary.mkdir(parents=True, exist_ok=True)
        store.download_manifest(tag, temporary, marker["files"])
        temporary.replace(target)
    (checkpoint_dir / "latest").write_text(tag + "\n")
    return tag


def upload_latest(store: S3Tree, checkpoint_dir: Path, previous: str | None) -> str | None:
    latest_path = checkpoint_dir / "latest"
    if not latest_path.is_file():
        return previous
    try:
        tag_text = latest_path.read_text().strip()
    except FileNotFoundError:
        return previous
    # DeepSpeed rewrites `latest` with truncate-then-write rather than an
    # atomic rename. A mirror poll can therefore observe an empty file for a
    # moment while a healthy checkpoint is being committed.
    if not tag_text:
        print("checkpoint latest tag is being rewritten; deferring S3 upload", flush=True)
        return previous
    tag = _validate_tag(tag_text)
    source = checkpoint_dir / tag
    if not tag or not source.is_dir() or tag == previous:
        return previous
    if not (source / _LOCAL_COMPLETE).is_file():
        print(f"checkpoint {tag} is still being written; deferring S3 upload", flush=True)
        return previous
    files = store.upload_tree(source, tag)
    if not files:
        raise RuntimeError(f"refusing to publish empty checkpoint {source}")
    marker = json.dumps({"schema_version": 1, "tag": tag, "files": files}, sort_keys=True).encode()
    store.put(f"{tag}/_SUCCESS", marker)
    store.put("latest", (tag + "\n").encode())
    print(f"uploaded complete checkpoint {tag} ({len(files)} files)", flush=True)
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-seconds", type=int, default=60)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("a command is required after --")

    run_id = os.environ["EDULLM_RUN_ID"]
    checkpoint_uri = os.environ["EDULLM_CHECKPOINT_DIR"]
    resume_uri = os.environ.get("EDULLM_RESUME_CHECKPOINT_URI", checkpoint_uri)
    remote_checkpoints = S3Tree(checkpoint_uri)
    resume_checkpoints = S3Tree(resume_uri)
    remote_outputs = S3Tree(os.environ["EDULLM_OUTPUT_PREFIX"])
    experiment = os.environ.get("EXP", "qwen3_30b_a3b_pedagogy_rl")
    run_root = Path(os.environ.get("EDULLM_LOCAL_ROOT", f"/tmp/edullm/{run_id}"))
    checkpoint_root = run_root / "checkpoints"
    checkpoint_dir = checkpoint_root / experiment
    output_dir = run_root / "output"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint destination: {checkpoint_uri}", flush=True)

    restored = restore_latest(resume_checkpoints, checkpoint_dir)
    if restored:
        print(f"restored checkpoint {restored} from {resume_uri}", flush=True)
        if resume_uri != checkpoint_uri:
            upload_latest(remote_checkpoints, checkpoint_dir, previous=None)

    environment = dict(os.environ)
    environment["CKPT_ROOT"] = str(checkpoint_root)
    environment["OUTPUT_DIR"] = str(output_dir)
    process = subprocess.Popen(command, env=environment, start_new_session=True)

    def forward(sig, _frame) -> None:
        if process.poll() is None:
            os.killpg(process.pid, sig)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    stop = threading.Event()
    upload_error: list[BaseException] = []
    latest_state = [restored]

    def mirror() -> None:
        while not stop.wait(args.sync_seconds):
            try:
                latest_state[0] = upload_latest(remote_checkpoints, checkpoint_dir, latest_state[0])
            except BaseException as error:
                upload_error.append(error)
                os.killpg(process.pid, signal.SIGTERM)
                return

    thread = threading.Thread(target=mirror, daemon=True)
    thread.start()
    return_code = process.wait()
    stop.set()
    thread.join()
    if upload_error:
        raise upload_error[0]
    latest_state[0] = upload_latest(remote_checkpoints, checkpoint_dir, latest_state[0])
    if return_code == 0:
        files = remote_outputs.upload_tree(output_dir, "model")
        if not files:
            raise RuntimeError(f"training exited successfully but wrote no model files under {output_dir}")
        remote_outputs.put(
            "model/_SUCCESS",
            json.dumps({"schema_version": 1, "checkpoint": latest_state[0], "files": files}, sort_keys=True).encode(),
        )
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
