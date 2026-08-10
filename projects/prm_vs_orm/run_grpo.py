"""Stage 7 S3 orchestration: run one GRPO arm (ORM- or PRM-rewarded) around ``grpo_fast.py``.

The two experiment arms are identical except for the reward model: ``--rm-type orm`` vs
``--rm-type prm``. Both drive open-instruct's proven GRPO loop (Ray + vLLM rollouts, group-
normalised advantages, KL to the frozen SFT reference) with the learned RM as the reward, via the
sanctioned plugin hook -- no edit to ``open_instruct``:

  1. download the Stage-2 SFT model (``--sft-uri``): both the GRPO **policy init** and its tokenizer;
  2. download the trained RM (``--rm-uri``) and point the reward bridge at it through the
     ``PRM_VS_ORM_RM_*`` env vars ``rm_verifier`` reads;
  3. download the GRPO prompt corpora (``--prompts-uri``, comma-separated ``s3://`` jsonls) and
     **relabel** each ``{problem, answer, dataset}`` row into the RLVR shape open-instruct expects
     -- ``{"messages": [user], "ground_truth": answer, "dataset": "<rm-name>"}`` -- so
     ``apply_verifiable_reward`` routes every prompt to our RM (``VERIFIER_SOURCE_KEY == "dataset"``);
  4. launch ``grpo_fast.py`` with ``--reward_plugins projects.prm_vs_orm.rm_verifier`` (dotted module
     path, imported under a real name so Ray workers can re-import it; registers ``RMVerifier`` before
     verifiers are built) and ``--chat_template_name tulu`` (the
     template the RM was trained on -- its rendered prompt is byte-identical to ``rm_common``'s);
  5. upload the saved policy (``output_dir/run_name``) to ``$EDULLM_CHECKPOINT_DIR`` for Stage 8 eval.

GPU topology (learners vs vLLM engines) is exposed as flags so it can be tuned at submit without a
code change. ``--selftest`` exercises the pure row-relabel and command builder offline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_common  # noqa: E402 -- overlay-local module (shared vLLM rope-config repair)

REPO_ROOT = Path(__file__).resolve().parents[2]
GRPO_ENTRY = "open_instruct/grpo_fast.py"
# Dotted module path, NOT a file path. ``registry._import_one`` loads a ``*.py`` file under a
# *synthetic* module name (``scored_rewards_plugin_rm_verifier``) that exists only in the driver's
# ``sys.modules``; when grpo_fast pickles the built RMVerifier to the vLLM Ray actors, those workers
# cannot ``import`` that synthetic name and die with ModuleNotFoundError at actor creation. A dotted,
# normally-importable path makes ``RMVerifier.__module__`` a real name every process can re-import
# (given REPO_ROOT on PYTHONPATH, set below) -- the registry's own documented form.
REWARD_PLUGIN = "projects.prm_vs_orm.rm_verifier"


# --------------------------------------------------------------------------------------------
# Pure transforms (offline-testable).
# --------------------------------------------------------------------------------------------
def relabel_row(row: dict, rm_name: str) -> dict:
    """A ``data_prep`` GRPO prompt ``{problem, answer, dataset}`` -> open-instruct RLVR row.

    The ``dataset`` field is the verifier-routing key (``VERIFIER_SOURCE_KEY``); setting it to the
    RM verifier's name makes ``apply_verifiable_reward`` score this prompt with the RM. The single
    user message becomes the prompt; ``add_generation_prompt`` + the tulu template turn it into
    ``<|user|>\\n{problem}\\n<|assistant|>\\n`` at rollout time.
    """
    return {
        "messages": [{"role": "user", "content": row["problem"]}],
        "ground_truth": str(row["answer"]),
        "dataset": rm_name,
    }


def _parse_prompt_blob(raw: bytes, path: str) -> list[dict]:
    """Robustly parse a prompts corpus into a list of dict rows (mirrors ``gen_onpolicy``'s fix).

    ``data_prep`` writes rows via ``json.dumps(..., ensure_ascii=False)``. That escapes ``\\n`` but
    emits Unicode line separators (``\\u2028``/``\\u2029``/``\\x85``/``\\v``/``\\f``) *literally*
    inside string values -- all valid JSON. ``str.splitlines()`` treats every one of those as a line
    boundary, so the old ``read_text().splitlines()`` + strict ``json.loads`` split a row mid-string
    and died with ``Unterminated string ... line 1 column 13`` (the exact crash that killed the ORM
    GRPO arm, and earlier the first Stage-A run; ``gen_onpolicy._parse_prompt_blob`` carries the same
    fix). Parse the whole blob with one streaming ``JSONDecoder(strict=False)``: ``strict=False``
    admits literal control chars inside strings, and ``raw_decode`` finds object boundaries by JSON
    structure rather than physical newlines, so an embedded separator no longer splits a row. A
    leading ``[`` is a single top-level array. Raise (never silently drop) if nothing parses, echoing
    the on-disk head so one re-run diagnoses a genuine truncation via ``edullm logs``.
    """
    text = raw.decode("utf-8", errors="replace")
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"empty prompts corpus at {path} ({len(raw)} bytes)")
    dec = json.JSONDecoder(strict=False)
    if stripped[0] == "[":
        obj = dec.decode(stripped)
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
    rows: list[dict] = []
    idx, n = 0, len(stripped)
    while idx < n:
        while idx < n and stripped[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(stripped, idx)
        if isinstance(obj, dict):
            rows.append(obj)
        idx = end
    if not rows:
        raise ValueError(
            f"could not parse any prompt rows from {path}: {len(raw)} bytes, head={text[:200]!r}"
        )
    return rows


def relabel_file(in_paths: list[str], out_path: str, rm_name: str) -> int:
    """Read each prompt jsonl, relabel every row, write one combined jsonl. Returns row count."""
    n = 0
    with open(out_path, "w") as w:
        for p in in_paths:
            for row in _parse_prompt_blob(Path(p).read_bytes(), p):
                if not (isinstance(row.get("problem"), str) and row["problem"].strip()):
                    continue
                w.write(json.dumps(relabel_row(row, rm_name)) + "\n")
                n += 1
    return n


def build_grpo_cmd(
    *,
    local_prompts: str,
    prompt_count: int,
    sft_dir: str,
    output_dir: str,
    rm_type: str,
    response_length: int,
    max_prompt_len: int,
    pack_length: int,
    num_unique_prompts_rollout: int,
    num_samples_per_prompt_rollout: int,
    total_episodes: int,
    learning_rate: float,
    beta: float,
    temperature: float,
    num_learners: int,
    vllm_num_engines: int,
    vllm_tp: int,
    vllm_gpu_memory_utilization: float,
    with_tracking: bool,
) -> list[str]:
    """Assemble the ``python grpo_fast.py`` argv for one arm (single outer process; Ray fans out)."""
    cmd: list[str] = [
        "python", GRPO_ENTRY,
        "--dataset_mixer_list", local_prompts, str(prompt_count),
        "--dataset_mixer_list_splits", "train",
        "--model_name_or_path", sft_dir,
        "--tokenizer_name_or_path", sft_dir,
        "--chat_template_name", "tulu",
        "--output_dir", output_dir,
        "--exp_name", f"prm-vs-orm-grpo-{rm_type}",
        "--max_prompt_token_length", str(max_prompt_len),
        "--response_length", str(response_length),
        "--pack_length", str(pack_length),
        "--per_device_train_batch_size", "1",
        "--num_unique_prompts_rollout", str(num_unique_prompts_rollout),
        "--num_samples_per_prompt_rollout", str(num_samples_per_prompt_rollout),
        "--apply_verifiable_reward", "true",
        "--ground_truths_key", "ground_truth",
        "--reward_plugins", REWARD_PLUGIN,
        "--temperature", str(temperature),
        "--learning_rate", str(learning_rate),
        "--total_episodes", str(total_episodes),
        "--beta", str(beta),
        "--num_epochs", "1",
        "--num_learners_per_node", str(num_learners),
        "--vllm_num_engines", str(vllm_num_engines),
        "--vllm_tensor_parallel_size", str(vllm_tp),
        "--vllm_gpu_memory_utilization", str(vllm_gpu_memory_utilization),
        "--deepspeed_stage", "2",
        "--gradient_checkpointing",
        "--load_ref_policy", "true",
        "--push_to_hub", "false",
        "--add_bos",
    ]
    if with_tracking:
        cmd.append("--with_tracking")
    return cmd


# --------------------------------------------------------------------------------------------
# S3 plumbing (mirrors run_rm.py).
# --------------------------------------------------------------------------------------------
def _split_uri(uri: str) -> tuple[str, str]:
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def _rel_key(key: str, prefix: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    return key[len(prefix) :] if key.startswith(prefix) else key


def download_prefix(s3: Any, uri: str, local_dir: str) -> int:
    bucket, prefix = _split_uri(uri)
    n = 0
    for obj in s3.list(bucket, prefix):
        key = obj["key"]
        rel = _rel_key(key, prefix)
        if not rel or key.endswith("/"):
            continue
        dst = Path(local_dir) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(s3.get(bucket, key))
        n += 1
    if n == 0:
        raise RuntimeError(f"no objects found under {uri}")
    return n


def download_file(s3: Any, uri: str, local_path: str) -> None:
    bucket, key = _split_uri(uri)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    Path(local_path).write_bytes(s3.get(bucket, key))


def upload_dir(s3: Any, local_dir: str, out_uri: str) -> list[str]:
    bucket, prefix = _split_uri(out_uri)
    prefix = prefix.rstrip("/")
    uploaded: list[str] = []
    for p in sorted(Path(local_dir).rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            s3.put_file(bucket, f"{prefix}/{rel}", str(p))
            uploaded.append(f"s3://{bucket}/{prefix}/{rel}")
    return uploaded


def find_saved_model(output_dir: str) -> str:
    """grpo_fast saves to output_dir/run_name; return the dir that holds the saved HF model."""
    root = Path(output_dir)
    for cfg in sorted(root.rglob("config.json")):
        if (cfg.parent / "tokenizer_config.json").exists() or any(cfg.parent.glob("*.safetensors")):
            return str(cfg.parent)
    raise RuntimeError(f"no saved HF model (config.json + weights) found under {output_dir}")


def _detect_gpus() -> int:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.device_count()
    except (ImportError, RuntimeError):
        return 0


def _selftest() -> None:
    r = relabel_row({"problem": "2+2?", "answer": 4, "dataset": "gsm8k"}, "rm")
    assert r["messages"] == [{"role": "user", "content": "2+2?"}]
    assert r["ground_truth"] == "4" and r["dataset"] == "rm"
    cmd = build_grpo_cmd(
        local_prompts="/tmp/p.jsonl", prompt_count=100, sft_dir="/sft", output_dir="/o",
        rm_type="prm", response_length=512, max_prompt_len=512, pack_length=1024,
        num_unique_prompts_rollout=32, num_samples_per_prompt_rollout=8, total_episodes=1000,
        learning_rate=3e-7, beta=0.05, temperature=0.8, num_learners=4, vllm_num_engines=4,
        vllm_tp=1, vllm_gpu_memory_utilization=0.3, with_tracking=True,
    )
    assert cmd[:2] == ["python", GRPO_ENTRY]
    assert cmd[cmd.index("--reward_plugins") + 1] == REWARD_PLUGIN
    assert cmd[cmd.index("--chat_template_name") + 1] == "tulu"
    assert cmd[cmd.index("--apply_verifiable_reward") + 1] == "true"
    assert cmd[cmd.index("--ground_truths_key") + 1] == "ground_truth"
    assert cmd[cmd.index("--dataset_mixer_list") + 1 : cmd.index("--dataset_mixer_list") + 3] == ["/tmp/p.jsonl", "100"]
    # the OOM guard: engines must be launched with a low KV-cache fraction, never grpo_fast's 0.9 default
    assert cmd[cmd.index("--vllm_gpu_memory_utilization") + 1] == "0.3"
    assert "--with_tracking" in cmd

    # prompt-blob parsing must survive what killed the ORM GRPO arm: a Unicode/control line
    # separator emitted literally inside a value by json.dumps(ensure_ascii=False).
    # str.splitlines() breaks on all of U+2028/U+2029/U+0085/\v/\f; the streaming decoder
    # must not. Cover NDJSON, an embedded LINE SEPARATOR (U+2028) and NEL (U+0085), a
    # top-level JSON array, and a truncated head.
    nd = b'{"problem": "x", "answer": "1"}\n{"problem": "y", "answer": "2"}\n'
    assert [r["problem"] for r in _parse_prompt_blob(nd, "t")] == ["x", "y"]
    sep = (
        '{"problem": "a\u2028b", "answer": "3"}\n'
        '{"problem": "c\u0085d", "answer": "4"}\n'
    ).encode()
    assert [r["problem"] for r in _parse_prompt_blob(sep, "t")] == ["a\u2028b", "c\u0085d"]
    arr = b'[{"problem": "p", "answer": "1"}, {"problem": "q", "answer": "2"}]'
    assert [r["problem"] for r in _parse_prompt_blob(arr, "t")] == ["p", "q"]
    try:
        _parse_prompt_blob(b'{"problem": "', "t")  # truncated head -> must raise, not silently drop
        raise AssertionError("expected truncated blob to raise")
    except ValueError:
        pass
    # end-to-end: relabel_file over a blob with an embedded separator writes clean NDJSON out.
    import tempfile  # noqa: PLC0415 -- test-only
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.jsonl"
        src.write_bytes(sep)
        out = Path(d) / "out.jsonl"
        count = relabel_file([str(src)], str(out), "rm")
        assert count == 2, count
        out_lines = out.read_text().split("\n")
        out_lines = [x for x in out_lines if x]
        assert len(out_lines) == 2 and all(json.loads(x)["dataset"] == "rm" for x in out_lines)
        assert json.loads(out_lines[0])["messages"][0]["content"] == "a\u2028b"
    print("RUN_GRPO SELFTEST OK: row relabel + grpo command builder + robust prompt parse verified")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rm-type", choices=["orm", "prm"])
    ap.add_argument("--sft-uri", help="s3:// prefix of the Stage-2 SFT policy (init + tokenizer)")
    ap.add_argument("--rm-uri", help="s3:// prefix of the trained RM for this arm")
    ap.add_argument("--prompts-uri", help="comma-separated s3:// URIs of GRPO prompt jsonls")
    ap.add_argument("--out", help="s3:// prefix to write the trained policy (EDULLM_CHECKPOINT_DIR)")
    ap.add_argument("--rm-name", default="rm", help="verifier name prompts route to (matches PRM_VS_ORM_RM_NAME)")
    ap.add_argument("--response-length", type=int, default=512)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--pack-length", type=int, default=1024)
    ap.add_argument("--num-unique-prompts-rollout", type=int, default=32)
    ap.add_argument("--num-samples-per-prompt-rollout", type=int, default=8)
    ap.add_argument("--total-episodes", type=int, default=50000)
    ap.add_argument("--learning-rate", type=float, default=3e-7)
    ap.add_argument("--beta", type=float, default=0.05, help="KL penalty to the frozen SFT reference")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--num-learners", type=int, default=4)
    ap.add_argument("--vllm-num-engines", type=int, default=4)
    ap.add_argument("--vllm-tp", type=int, default=1)
    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.3,
                    help="vLLM KV-cache fraction PER ENGINE. grpo_fast's 0.9 default reserves ~72GiB "
                         "on an 80GiB A100 and OOMs the moment Ray co-locates an engine with a learner "
                         "on one card (the crash that killed the PRM arm). A 370M policy at 1024-token "
                         "context needs <10GiB, so 0.3 (~24GiB) fits even co-located, with headroom.")
    ap.add_argument("--max-seq-length", type=int, default=1024, help="RM scoring max length (env for rm_verifier)")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", help="named so the precision guard can see it")
    ap.add_argument("--output-dir", default="/tmp/grpo_out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    for name, val in (("--rm-type", args.rm_type), ("--sft-uri", args.sft_uri), ("--rm-uri", args.rm_uri),
                      ("--prompts-uri", args.prompts_uri), ("--out", args.out)):
        if not val:
            print(f"{name} is required", file=sys.stderr)
            raise SystemExit(2)

    import os  # noqa: PLC0415

    from edullm_data.s3 import Boto3S3  # noqa: PLC0415

    s3 = Boto3S3.default()

    sft_dir, rm_dir = "/tmp/grpo_sft", "/tmp/grpo_rm"
    print(f"downloading SFT policy {args.sft_uri} ...", flush=True)
    download_prefix(s3, args.sft_uri, sft_dir)
    # grpo_fast's vLLM rollout engines load this dir; ensure config.rope_parameters["rope_theta"]
    # exists (vLLM's olmo2 loader requires it on sliding-window layers) using the checkpoint's own base.
    rm_common.ensure_vllm_rope_parameters(sft_dir)
    print(f"downloading RM {args.rm_uri} ...", flush=True)
    download_prefix(s3, args.rm_uri, rm_dir)

    prompt_uris = [u.strip() for u in args.prompts_uri.split(",") if u.strip()]
    local_prompt_files = []
    for i, uri in enumerate(prompt_uris):
        lp = f"/tmp/grpo_prompts_src_{i}.jsonl"
        download_file(s3, uri, lp)
        local_prompt_files.append(lp)
    combined = "/tmp/grpo_prompts.jsonl"
    count = relabel_file(local_prompt_files, combined, args.rm_name)
    print(f"relabelled {count} prompts -> {combined} (dataset={args.rm_name})", flush=True)

    # Put the repo root on PYTHONPATH so ``import projects.prm_vs_orm.rm_verifier`` resolves in every
    # process: the grpo_fast driver (launched as ``python open_instruct/grpo_fast.py`` -> sys.path[0]
    # is ``open_instruct/``, not the root) AND the vLLM Ray actors, which inherit this via grpo_fast's
    # ``ray.init(runtime_env={"env_vars": os.environ ...})`` (PYTHONPATH is not in EXCLUDED_ENV_VARS).
    # This is what lets a worker re-import the dotted reward plugin when it unpickles RMVerifier.
    _pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + _pp if _pp else "")

    # Point the reward bridge (rm_verifier, loaded via --reward_plugins) at the downloaded RM.
    os.environ["PRM_VS_ORM_RM_PATH"] = rm_dir
    os.environ["PRM_VS_ORM_RM_TYPE"] = args.rm_type
    os.environ["PRM_VS_ORM_RM_NAME"] = args.rm_name
    os.environ["PRM_VS_ORM_RM_MAXLEN"] = str(args.max_seq_length)
    os.environ.setdefault("WANDB_PROJECT", f"prm-vs-orm-grpo-{args.rm_type}")

    num_gpus = _detect_gpus()
    print(f"visible GPUs: {num_gpus}", flush=True)

    cmd = build_grpo_cmd(
        local_prompts=combined, prompt_count=count, sft_dir=sft_dir, output_dir=args.output_dir,
        rm_type=args.rm_type, response_length=args.response_length, max_prompt_len=args.max_prompt_len,
        pack_length=args.pack_length, num_unique_prompts_rollout=args.num_unique_prompts_rollout,
        num_samples_per_prompt_rollout=args.num_samples_per_prompt_rollout, total_episodes=args.total_episodes,
        learning_rate=args.learning_rate, beta=args.beta, temperature=args.temperature,
        num_learners=args.num_learners, vllm_num_engines=args.vllm_num_engines, vllm_tp=args.vllm_tp,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
        with_tracking=not args.no_tracking,
    )
    print("launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    model_dir = find_saved_model(args.output_dir)
    print(f"uploading policy {model_dir} -> {args.out} ...", flush=True)
    uploaded = upload_dir(s3, model_dir, args.out)
    print(f"GRPO RUN OK: rm_type={args.rm_type} policy={args.out.rstrip('/')} files={len(uploaded)}", flush=True)


if __name__ == "__main__":
    main()
