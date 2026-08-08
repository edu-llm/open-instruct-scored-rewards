"""Durable S3 staging for local DeepSpeed checkpoint directories."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig

_STEP_DIRECTORY = re.compile(r"^step_(\d+)$")
_MARKER = "COMPLETED"
_SCHEMA_VERSION = 1
_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024**2, multipart_chunksize=64 * 1024**2, max_concurrency=8, use_threads=True
)
_UPLOAD_ARGS = {"ChecksumAlgorithm": "CRC32C", "ChecksumType": "FULL_OBJECT"}


@dataclass(frozen=True)
class S3Prefix:
    bucket: str
    key: str

    @classmethod
    def parse(cls, uri: str) -> S3Prefix:
        location = urlparse(uri)
        if location.scheme != "s3" or not location.netloc:
            raise ValueError(f"expected an s3:// URI, got {uri!r}")
        key = location.path.lstrip("/")
        if key and not key.endswith("/"):
            key += "/"
        return cls(bucket=location.netloc, key=key)

    def child_key(self, relative: str) -> str:
        return self.key + relative.lstrip("/")


class S3CheckpointStore:
    """Upload complete checkpoints and materialize the newest one locally."""

    def __init__(self, uri: str, client: Any | None = None) -> None:
        self.prefix = S3Prefix.parse(uri)
        self.client = client if client is not None else boto3.client("s3")

    def download_latest(self, output_dir: Path) -> Path | None:
        step = self.latest_complete_step()
        if step is None:
            return None

        target = output_dir / f"step_{step}"
        if (target / _MARKER).is_file():
            return target

        marker = self._read_marker(step)
        temporary = output_dir / f".step_{step}.download"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        try:
            for entry in marker["files"]:
                relative = _safe_relative_path(entry["path"])
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(
                    self.prefix.bucket, self.prefix.child_key(f"step_{step}/{relative.as_posix()}"), str(destination)
                )
                if destination.stat().st_size != entry["size"]:
                    raise OSError(
                        f"downloaded {relative} with {destination.stat().st_size} bytes; expected {entry['size']}"
                    )
            (temporary / _MARKER).write_text(f"{_MARKER}\n")
            if target.exists():
                shutil.rmtree(target)
            temporary.replace(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    def upload_step(self, checkpoint_dir: Path) -> None:
        matched = _STEP_DIRECTORY.fullmatch(checkpoint_dir.name)
        if matched is None:
            raise ValueError(f"checkpoint directory must be named step_<number>, got {checkpoint_dir.name!r}")
        if not (checkpoint_dir / _MARKER).is_file():
            raise ValueError(f"refusing to upload incomplete checkpoint {checkpoint_dir}")

        step = int(matched.group(1))
        files = sorted(
            path
            for path in checkpoint_dir.rglob("*")
            if path.is_file() and path.relative_to(checkpoint_dir).as_posix() != _MARKER
        )
        manifest_files = []
        for path in files:
            relative = path.relative_to(checkpoint_dir).as_posix()
            self.client.upload_file(
                str(path),
                self.prefix.bucket,
                self.prefix.child_key(f"step_{step}/{relative}"),
                ExtraArgs=_UPLOAD_ARGS,
                Config=_TRANSFER_CONFIG,
            )
            manifest_files.append({"path": relative, "size": path.stat().st_size})

        marker = {"schema_version": _SCHEMA_VERSION, "step": step, "files": manifest_files}
        self.client.put_object(
            Bucket=self.prefix.bucket,
            Key=self.prefix.child_key(f"step_{step}/{_MARKER}"),
            Body=(json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            ChecksumAlgorithm="CRC32C",
        )
        self._delete_older_steps(step)

    def upload_final_export(self, output_dir: Path) -> None:
        files = sorted(path for path in output_dir.iterdir() if path.is_file())
        manifest_files = []
        for path in files:
            self.client.upload_file(
                str(path),
                self.prefix.bucket,
                self.prefix.child_key(f"final/{path.name}"),
                ExtraArgs=_UPLOAD_ARGS,
                Config=_TRANSFER_CONFIG,
            )
            manifest_files.append({"path": path.name, "size": path.stat().st_size})
        self.client.put_object(
            Bucket=self.prefix.bucket,
            Key=self.prefix.child_key(f"final/{_MARKER}"),
            Body=(
                json.dumps(
                    {"schema_version": _SCHEMA_VERSION, "files": manifest_files}, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode(),
            ChecksumAlgorithm="CRC32C",
        )

    def latest_complete_step(self) -> int | None:
        steps = []
        for entry in self._list_objects():
            relative = str(entry.get("Key", "")).removeprefix(self.prefix.key)
            directory, separator, filename = relative.partition("/")
            matched = _STEP_DIRECTORY.fullmatch(directory)
            if matched is not None and separator and filename == _MARKER:
                steps.append(int(matched.group(1)))
        return max(steps, default=None)

    def _read_marker(self, step: int) -> dict[str, Any]:
        response = self.client.get_object(
            Bucket=self.prefix.bucket, Key=self.prefix.child_key(f"step_{step}/{_MARKER}")
        )
        marker = json.loads(response["Body"].read())
        if marker.get("schema_version") != _SCHEMA_VERSION or marker.get("step") != step:
            raise ValueError(f"invalid remote checkpoint marker for step {step}")
        files = marker.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"remote checkpoint step {step} has no file manifest")
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"remote checkpoint step {step} has an invalid file entry")
            if not isinstance(entry.get("size"), int) or isinstance(entry["size"], bool) or entry["size"] < 0:
                raise ValueError(f"remote checkpoint step {step} has an invalid file size")
            _safe_relative_path(entry["path"])
        return marker

    def _delete_older_steps(self, keep_step: int) -> None:
        keys = []
        for entry in self._list_objects():
            key = str(entry.get("Key", ""))
            relative = key.removeprefix(self.prefix.key)
            directory, separator, _ = relative.partition("/")
            matched = _STEP_DIRECTORY.fullmatch(directory)
            if matched is not None and separator and int(matched.group(1)) < keep_step:
                keys.append({"Key": key})
        for start in range(0, len(keys), 1000):
            self.client.delete_objects(
                Bucket=self.prefix.bucket, Delete={"Objects": keys[start : start + 1000], "Quiet": True}
            )

    def _list_objects(self) -> list[dict[str, Any]]:
        contents = []
        arguments: dict[str, Any] = {"Bucket": self.prefix.bucket, "Prefix": self.prefix.key}
        while True:
            response = self.client.list_objects_v2(**arguments)
            contents.extend(response.get("Contents") or [])
            token = response.get("NextContinuationToken")
            if not response.get("IsTruncated") or not token:
                return contents
            arguments["ContinuationToken"] = token


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe checkpoint path {value!r}")
    return path
