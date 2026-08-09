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
  4. launch ``grpo_fast.py`` with ``--reward_plugins projects/prm_vs_orm/rm_verifier.py`` (imports and
     registers ``RMVerifier`` before verifiers are built) and ``--chat_template_name tulu`` (the
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
REWARD_PLUGIN = "projects/prm_vs_orm/rm_verifier.py"


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


def relabel_file(in_paths: list[str], out_path: str, rm_name: str) -> int:
    """Read each prompt jsonl, relabel every row, write one combined jsonl. Returns row count."""
    n = 0
    with open(out_path, "w") as w:
        for p in in_paths:
            for line in Path(p).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                w.write(json.dumps(relabel_row(json.loads(line), rm_name)) + "\n")
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
        vllm_tp=1, with_tracking=True,
    )
    assert cmd[:2] == ["python", GRPO_ENTRY]
    assert cmd[cmd.index("--reward_plugins") + 1] == REWARD_PLUGIN
    assert cmd[cmd.index("--chat_template_name") + 1] == "tulu"
    assert cmd[cmd.index("--apply_verifiable_reward") + 1] == "true"
    assert cmd[cmd.index("--ground_truths_key") + 1] == "ground_truth"
    assert cmd[cmd.index("--dataset_mixer_list") + 1 : cmd.index("--dataset_mixer_list") + 3] == ["/tmp/p.jsonl", "100"]
    assert "--with_tracking" in cmd
    print("RUN_GRPO SELFTEST OK: row relabel + grpo command builder verified")


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
