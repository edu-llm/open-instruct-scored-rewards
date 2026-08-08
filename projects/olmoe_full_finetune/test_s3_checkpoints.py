import io
import json
from pathlib import Path

import pytest
from projects.olmoe_full_finetune.s3_checkpoints import S3CheckpointStore


class FakeS3Client:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.calls = []
        self.upload_args = []

    def upload_file(self, filename, bucket, key, ExtraArgs=None, Config=None):
        del bucket, Config
        self.objects[key] = Path(filename).read_bytes()
        self.calls.append(("upload", key))
        self.upload_args.append(ExtraArgs)

    def put_object(self, *, Bucket, Key, Body, ChecksumAlgorithm):
        del Bucket
        assert ChecksumAlgorithm == "CRC32C"
        self.objects[Key] = Body
        self.calls.append(("put", Key))

    def get_object(self, *, Bucket, Key):
        del Bucket
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, filename):
        del bucket
        Path(filename).write_bytes(self.objects[key])
        self.calls.append(("download", key))

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        del Bucket, ContinuationToken
        return {
            "Contents": [
                {"Key": key, "Size": len(value)}
                for key, value in sorted(self.objects.items())
                if key.startswith(Prefix)
            ],
            "IsTruncated": False,
        }

    def delete_objects(self, *, Bucket, Delete):
        del Bucket
        for entry in Delete["Objects"]:
            self.objects.pop(entry["Key"], None)
        self.calls.append(("delete", tuple(entry["Key"] for entry in Delete["Objects"])))


def marker(step, files):
    return json.dumps({"schema_version": 1, "step": step, "files": files}).encode()


def test_upload_commits_marker_last_and_prunes_older_step(tmp_path):
    checkpoint = tmp_path / "step_20"
    checkpoint.mkdir()
    (checkpoint / "rank0.pt").write_bytes(b"rank zero")
    (checkpoint / "trainer_state.json").write_text('{"completed_steps": 20}\n')
    (checkpoint / "COMPLETED").write_text("COMPLETED\n")
    client = FakeS3Client(
        {
            "runs/checkpoints/step_10/rank0.pt": b"old",
            "runs/checkpoints/step_10/COMPLETED": marker(10, [{"path": "rank0.pt", "size": 3}]),
            "runs/checkpoints/step_30/rank0.pt": b"newer",
            "runs/checkpoints/step_30/COMPLETED": marker(30, [{"path": "rank0.pt", "size": 5}]),
        }
    )

    store = S3CheckpointStore("s3://bucket/runs/checkpoints/", client)
    store.upload_step(checkpoint)

    marker_key = "runs/checkpoints/step_20/COMPLETED"
    first_delete = next(index for index, call in enumerate(client.calls) if call[0] == "delete")
    marker_put = client.calls.index(("put", marker_key))
    assert all(call[0] == "upload" for call in client.calls[:marker_put])
    assert marker_put < first_delete
    assert not any("step_10/" in key for key in client.objects)
    assert "runs/checkpoints/step_30/COMPLETED" in client.objects
    assert client.upload_args == [
        {"ChecksumAlgorithm": "CRC32C", "ChecksumType": "FULL_OBJECT"},
        {"ChecksumAlgorithm": "CRC32C", "ChecksumType": "FULL_OBJECT"},
    ]


def test_download_uses_newest_complete_step_and_ignores_torn_newer_step(tmp_path):
    files = [{"path": "pytorch_model/rank0.pt", "size": 4}, {"path": "trainer_state.json", "size": 24}]
    client = FakeS3Client(
        {
            "checkpoints/step_20/pytorch_model/rank0.pt": b"zero",
            "checkpoints/step_20/trainer_state.json": b'{"completed_steps": 20}\n',
            "checkpoints/step_20/COMPLETED": marker(20, files),
            "checkpoints/step_40/pytorch_model/rank0.pt": b"torn",
        }
    )

    restored = S3CheckpointStore("s3://bucket/checkpoints", client).download_latest(tmp_path)

    assert restored == tmp_path / "step_20"
    assert (restored / "COMPLETED").read_text() == "COMPLETED\n"
    assert (restored / "pytorch_model/rank0.pt").read_bytes() == b"zero"
    assert not (tmp_path / "step_40").exists()


def test_download_rejects_marker_paths_outside_checkpoint(tmp_path):
    client = FakeS3Client(
        {
            "checkpoints/step_1/COMPLETED": marker(1, [{"path": "../escape", "size": 1}]),
            "checkpoints/step_1/../escape": b"x",
        }
    )

    with pytest.raises(ValueError, match="unsafe checkpoint path"):
        S3CheckpointStore("s3://bucket/checkpoints/", client).download_latest(tmp_path)
