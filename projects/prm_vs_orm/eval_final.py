"""Stage 8: final evaluation -- the headline PRM-RL vs ORM-RL (vs SFT baseline) numbers.

For each model (SFT, ORM-RL policy, PRM-RL policy) and each eval set (GSM8K primary, MATH
secondary), sample from the policy via vLLM and report:

  * **greedy** accuracy (temperature 0, one sample);
  * **maj@k** accuracy (k samples, majority vote over the extracted final answer).

Correctness comes from ``open_instruct``'s own ``GSM8KVerifier`` / ``MathVerifier`` (reuse, do not
write a grader). Prompts use the same tulu template as every other stage, so the comparison is
apples-to-apples. A JSON report (``eval_report.json``) is uploaded to ``$EDULLM_CHECKPOINT_DIR``.

Models and eval sets are passed as ``name=s3://uri`` lists so this one script produces the whole
comparison table in a single job. ``--selftest`` exercises the pure maj@k aggregation offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_common  # noqa: E402 -- overlay-local (tulu prompt formatting, shared across stages)
from bon_eval import (  # noqa: E402 -- reuse pure helpers (offline-safe imports)
    download_file,
    download_prefix,
    extract_answer,
    majority_correct,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------
# Pure aggregation (offline-testable).
# --------------------------------------------------------------------------------------------
def accuracy(flags: list[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


def parse_named_uris(spec: str) -> list[tuple[str, str]]:
    """``a=s3://x,b=s3://y`` -> ``[("a","s3://x"), ("b","s3://y")]`` (order preserved)."""
    out: list[tuple[str, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, uri = part.partition("=")
        if not sep or not uri.startswith("s3://"):
            raise ValueError(f"expected name=s3://uri, got {part!r}")
        out.append((name.strip(), uri.strip()))
    return out


def _selftest() -> None:
    assert accuracy([True, False, True, True]) == 0.75
    assert accuracy([]) == 0.0
    assert parse_named_uris("sft=s3://a/b,orm=s3://c/d") == [("sft", "s3://a/b"), ("orm", "s3://c/d")]
    # maj@k reuse: '5' twice (correct) beats '6' once (incorrect)
    assert majority_correct(["5", "6", "5"], [True, False, True]) is True
    assert extract_answer("steps...\n# Answer\n\n5") == "5"
    try:
        parse_named_uris("bad_no_uri")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("EVAL_FINAL SELFTEST OK: accuracy, named-uri parse, maj@k reuse verified")


def _grade_fn(dataset: str):
    from open_instruct.ground_truth_utils import GSM8KVerifier, MathVerifier  # noqa: PLC0415

    verifier = GSM8KVerifier() if dataset == "gsm8k" else MathVerifier()

    def is_correct(pred: str, answer: str) -> bool:
        return verifier(tokenized_prediction=[], prediction=pred, label=answer).score >= 0.5

    return is_correct


def _eval_one_model(llm_cls, sampling_cls, model_dir, eval_sets, args) -> dict:
    """Return {dataset: {greedy, maj_k}} for one policy."""
    llm = llm_cls(model=model_dir, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
                  max_model_len=args.max_seq_length, enforce_eager=True)
    stop = ["<|user|>"]
    result: dict[str, dict] = {}
    for dataset, rows in eval_sets.items():
        prompts = [rm_common.tulu_prompt_string(r["problem"]) for r in rows]
        is_correct = _grade_fn(dataset)
        greedy = llm.generate(prompts, sampling_cls(temperature=0.0, max_tokens=args.max_tokens, stop=stop))
        greedy_flags = [is_correct(g.outputs[0].text, r["answer"]) for g, r in zip(greedy, rows)]
        maj_flags: list[bool] = []
        if args.maj_k > 1:
            samp = llm.generate(
                prompts,
                sampling_cls(n=args.maj_k, temperature=args.temperature, top_p=args.top_p,
                             max_tokens=args.max_tokens, stop=stop),
            )
            for out, r in zip(samp, rows):
                cands = [o.text for o in out.outputs]
                correct = [is_correct(c, r["answer"]) for c in cands]
                answers = [extract_answer(c) for c in cands]
                maj_flags.append(majority_correct(answers, correct))
        result[dataset] = {
            "greedy": accuracy(greedy_flags),
            f"maj@{args.maj_k}": accuracy(maj_flags) if maj_flags else None,
            "n": len(rows),
        }
    del llm
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", help="comma-separated name=s3://uri policies (e.g. sft=..,orm_rl=..,prm_rl=..)")
    ap.add_argument("--eval-uris", help="comma-separated dataset=s3://uri (dataset in {gsm8k,math})")
    ap.add_argument("--out", help="s3:// prefix to write eval_report.json (EDULLM_CHECKPOINT_DIR)")
    ap.add_argument("--maj-k", type=int, default=8)
    ap.add_argument("--num-problems", type=int, default=0, help="cap per eval set (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    for name, val in (("--models", args.models), ("--eval-uris", args.eval_uris), ("--out", args.out)):
        if not val:
            print(f"{name} is required", file=sys.stderr)
            raise SystemExit(2)

    models = parse_named_uris(args.models)
    eval_specs = parse_named_uris(args.eval_uris)

    from edullm_data.s3 import Boto3S3  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    s3 = Boto3S3.default()

    eval_sets: dict[str, list[dict]] = {}
    for dataset, uri in eval_specs:
        lp = f"/tmp/eval_{dataset}.jsonl"
        download_file(s3, uri, lp)
        rows = [json.loads(ln) for ln in Path(lp).read_text().splitlines() if ln.strip()]
        if args.num_problems > 0:
            rows = rows[: args.num_problems]
        eval_sets[dataset] = rows
        print(f"loaded {len(rows)} {dataset} eval problems", flush=True)

    report: dict[str, Any] = {"maj_k": args.maj_k, "models": {}}
    for name, uri in models:
        model_dir = f"/tmp/eval_model_{name}"
        print(f"downloading model {name} {uri} ...", flush=True)
        download_prefix(s3, uri, model_dir)
        # vLLM's olmo2 loader needs config.rope_parameters["rope_theta"]; ensure it (SFT and the
        # GRPO-saved policies both derive from the Olmo3 config that can omit it).
        rm_common.ensure_vllm_rope_parameters(model_dir)
        report["models"][name] = _eval_one_model(LLM, SamplingParams, model_dir, eval_sets, args)
        print(f"=== {name} ===  {json.dumps(report['models'][name])}", flush=True)

    bucket, key = uri_split = s3_split(args.out)
    tmp = "/tmp/eval_report.json"
    Path(tmp).write_text(json.dumps(report, indent=2))
    s3.put_file(bucket, key.rstrip("/") + "/eval_report.json", tmp)
    _ = uri_split
    print("=== FINAL EVAL (accuracy) ===", flush=True)
    for name, _ in models:
        for dataset in eval_sets:
            m = report["models"][name][dataset]
            print(f"  {name:10s} {dataset:6s} greedy={m['greedy']:.4f} maj@{args.maj_k}={m[f'maj@{args.maj_k}']}", flush=True)
    print(f"EVAL DONE: report=s3://{bucket}/{key.rstrip('/')}/eval_report.json", flush=True)


def s3_split(uri: str) -> tuple[str, str]:
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


if __name__ == "__main__":
    main()
