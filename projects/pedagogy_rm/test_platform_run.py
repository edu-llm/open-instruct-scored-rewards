"""Durability checks for the eduLLM S3 checkpoint mirror."""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from boto3.s3.transfer import S3Transfer
from botocore import UNSIGNED
from botocore.config import Config
from botocore.stub import ANY, Stubber

from projects.pedagogy_rm.platform_run import _DOWNLOAD_ARGS, _TRANSFER, _UPLOAD_ARGS, restore_latest, upload_latest


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.events: list[str] = []
        self.fail_upload = False

    def read(self, relative: str) -> bytes | None:
        return self.objects.get(relative)

    def put(self, relative: str, body: bytes) -> None:
        self.events.append(f"put:{relative}")
        self.objects[relative] = body

    def upload_tree(self, source: Path, remote_dir: str) -> list[dict]:
        self.events.append(f"upload:{remote_dir}")
        if self.fail_upload:
            raise OSError("simulated interrupted upload")
        manifest = []
        for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
            relative = path.relative_to(source).as_posix()
            self.objects[f"{remote_dir}/{relative}"] = path.read_bytes()
            manifest.append({"path": relative, "size": path.stat().st_size})
        return manifest

    def download_manifest(self, remote_dir: str, destination: Path, manifest: list[dict]) -> None:
        for entry in manifest:
            target = destination / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.objects[f"{remote_dir}/{entry['path']}"])


def local_checkpoint(root: Path, tag: str = "global_step1") -> None:
    checkpoint = root / tag
    checkpoint.mkdir(parents=True)
    (checkpoint / "model_states.pt").write_bytes(b"model")
    (checkpoint / "zero" / "rank0.pt").parent.mkdir()
    (checkpoint / "zero" / "rank0.pt").write_bytes(b"optimizer")
    (checkpoint / ".checkpoint_state_complete").touch()
    (root / "latest").write_text(tag + "\n")


def test_checksum_arguments_are_supported_by_boto3() -> None:
    assert set(_UPLOAD_ARGS).issubset(S3Transfer.ALLOWED_UPLOAD_ARGS)
    assert set(_DOWNLOAD_ARGS).issubset(S3Transfer.ALLOWED_DOWNLOAD_ARGS)
    assert _TRANSFER.preferred_transfer_client == "classic"


def test_classic_transfer_uploads_crc32c_without_crt_extra_args_bug(tmp_path: Path) -> None:
    pytest.importorskip("awscrt")
    source = tmp_path / "payload"
    source.write_bytes(b"123456789")
    client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
    stub = Stubber(client)
    stub.add_response(
        "put_object",
        {},
        {"Bucket": "bucket", "Key": "key", "Body": ANY, "ChecksumAlgorithm": "CRC32C"},
    )
    stub.activate()

    client.upload_file(
        str(source),
        "bucket",
        "key",
        ExtraArgs=_UPLOAD_ARGS,
        Config=_TRANSFER,
    )

    stub.assert_no_pending_responses()


def test_latest_is_published_only_after_files_and_success_marker(tmp_path: Path) -> None:
    local_checkpoint(tmp_path)
    store = MemoryStore()

    assert upload_latest(store, tmp_path, previous=None) == "global_step1"
    assert store.events == [
        "upload:global_step1",
        "put:global_step1/_SUCCESS",
        "put:latest",
    ]


def test_interrupted_upload_does_not_replace_last_good_checkpoint(tmp_path: Path) -> None:
    local_checkpoint(tmp_path, "global_step2")
    store = MemoryStore()
    store.objects["latest"] = b"global_step1\n"
    store.fail_upload = True

    with pytest.raises(OSError, match="interrupted upload"):
        upload_latest(store, tmp_path, previous="global_step1")

    assert store.objects["latest"] == b"global_step1\n"
    assert "global_step2/_SUCCESS" not in store.objects


def test_half_written_local_checkpoint_is_not_uploaded(tmp_path: Path) -> None:
    local_checkpoint(tmp_path)
    (tmp_path / "global_step1" / ".checkpoint_state_complete").unlink()
    store = MemoryStore()

    assert upload_latest(store, tmp_path, previous=None) is None
    assert store.events == []


def test_blank_latest_during_deepspeed_rewrite_is_deferred(tmp_path: Path) -> None:
    local_checkpoint(tmp_path)
    (tmp_path / "latest").write_text("")
    store = MemoryStore()

    assert upload_latest(store, tmp_path, previous="global_step0") == "global_step0"
    assert store.events == []


def test_complete_checkpoint_round_trips_through_remote_store(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    local_checkpoint(source)
    store = MemoryStore()
    upload_latest(store, source, previous=None)

    destination = tmp_path / "destination"
    destination.mkdir()
    assert restore_latest(store, destination) == "global_step1"
    assert (destination / "latest").read_text() == "global_step1\n"
    assert (destination / "global_step1" / "model_states.pt").read_bytes() == b"model"
    assert (destination / "global_step1" / "zero" / "rank0.pt").read_bytes() == b"optimizer"


def test_restore_refuses_checkpoint_without_success_marker(tmp_path: Path) -> None:
    store = MemoryStore()
    store.objects["latest"] = b"global_step1\n"

    with pytest.raises(RuntimeError, match="no _SUCCESS marker"):
        restore_latest(store, tmp_path)
