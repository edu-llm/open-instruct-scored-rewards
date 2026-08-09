"""Stage 5: best-of-N go/no-go gate for the PRM-vs-ORM experiment.

Sample ``N`` solutions per problem from the Stage-2 SFT policy (vLLM), then rank each problem's
``N`` candidates three ways and measure held-out accuracy:

  * **ORM rank** -- pick the argmax of the last-token ``P(correct)`` (``rm_verifier.orm_reward``);
  * **PRM rank** -- pick the argmax of the product of per-step ``P(correct)`` (``rm_verifier.prm_reward``);
  * **majority** -- pick the most common extracted final answer.

Also report **SFT greedy** (temperature 0) and **pass@N** (oracle upper bound). Correctness comes
from ``open_instruct``'s own ``GSM8KVerifier`` / ``MathVerifier`` (reuse, do not write a grader).

The ORM/PRM scoring here calls the *same* pure helpers the GRPO reward bridge uses
(``rm_verifier`` + ``rm_common``), so the gate measures exactly the signal GRPO would optimise.

**Decision.** If neither RM's best-of-N beats majority vote *and* SFT-greedy, the RMs add no
usable ranking signal -- stop before spending GPU on GRPO. Printed as ``BON GATE: GO`` /
``BON GATE: NO-GO`` with a per-method accuracy table; a JSON report is uploaded to
``$EDULLM_CHECKPOINT_DIR``.

S3 plumbing mirrors ``run_rm.py``. Light enough for a single L40S (a few hundred 370M rollouts).
``--selftest`` exercises the pure aggregation (argmax select, majority vote, pass@N) offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_common  # noqa: E402 -- overlay-local module
import rm_verifier  # noqa: E402 -- overlay-local module (pure ORM/PRM arithmetic)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------
# Pure aggregation (no torch / no vllm) -- the gate arithmetic, unit-tested offline.
# --------------------------------------------------------------------------------------------
def argmax(xs: list[float]) -> int:
    """Index of the max; ties resolve to the earliest (stable) index."""
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def extract_answer(prediction: str) -> str:
    """The final answer string a solution asserts, for majority voting.

    ``data_prep`` templates every solution to end ``# Answer\\n\\n<ans>``; take the text after the
    last ``# Answer`` marker, else fall back to the last non-empty line.
    """
    marker = "# Answer"
    if marker in prediction:
        # answer is the first non-empty line after the last marker
        tail = prediction.rsplit(marker, 1)[1]
        for line in tail.splitlines():
            if line.strip():
                return line.strip()
        return tail.strip()
    # no marker: the answer trails, so take the last non-empty line
    for line in reversed(prediction.splitlines()):
        if line.strip():
            return line.strip()
    return prediction.strip()


def majority_correct(answers: list[str], correct: list[bool]) -> bool:
    """Correctness of the majority-voted answer.

    Vote over extracted answer strings; the winning bucket's correctness is that of any candidate
    in it (all candidates with the same extracted answer share a correctness label). Ties resolve
    to the most common that appears earliest.
    """
    if not answers:
        return False
    counts = Counter(a for a in answers if a)
    if not counts:
        return False
    # tie-break: highest count, then earliest first occurrence
    best: tuple[tuple[int, int], str] | None = None
    for a in counts:
        key = (counts[a], -_first_index(answers, a))
        if best is None or key > best[0]:
            best = (key, a)
    assert best is not None
    idx = _first_index(answers, best[1])
    return correct[idx]


def _first_index(xs: list[str], target: str) -> int:
    for i, x in enumerate(xs):
        if x == target:
            return i
    return 0


def bon_correct(scores: list[float], correct: list[bool]) -> bool:
    """Whether the highest-scored candidate is correct (best-of-N under a ranker)."""
    return correct[argmax(scores)]


def summarize(per_problem: list[dict]) -> dict:
    """Aggregate per-problem method outcomes into mean accuracies."""
    n = len(per_problem)
    if n == 0:
        return {k: 0.0 for k in ("greedy", "majority", "orm_bon", "prm_bon", "pass_at_n", "n")}
    out = {
        "greedy": sum(p["greedy"] for p in per_problem) / n,
        "majority": sum(p["majority"] for p in per_problem) / n,
        "orm_bon": sum(p["orm_bon"] for p in per_problem) / n,
        "prm_bon": sum(p["prm_bon"] for p in per_problem) / n,
        "pass_at_n": sum(p["pass_at_n"] for p in per_problem) / n,
        "n": n,
    }
    out["gate_go"] = max(out["orm_bon"], out["prm_bon"]) > max(out["majority"], out["greedy"])
    return out


# --------------------------------------------------------------------------------------------
# GPU scoring: batched ORM/PRM logits via rm_common, reward via rm_verifier helpers.
# --------------------------------------------------------------------------------------------
def _load_rm(rm_dir: str, rm_type: str, device: str, add_bos_choice: str):
    import olmo3_adapter  # noqa: PLC0415 -- overlay-local
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    olmo3_adapter.register()
    num_labels = 1 if rm_type == "orm" else rm_common.NUM_PRM_CLASSES
    tok = AutoTokenizer.from_pretrained(rm_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if add_bos_choice == "yes":
        add_bos = True
    elif add_bos_choice == "no":
        add_bos = False
    else:
        add_bos = tok.bos_token_id is not None
    model = AutoModelForSequenceClassification.from_pretrained(
        rm_dir, num_labels=num_labels, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tok.pad_token_id
    model.eval().to(device)
    return model, tok, add_bos


def _pooled(model, tok, add_bos, max_len, exchanges, device, batch_size=16):
    """Score a list of (problem, assistant_text) -> list of per-row logit vectors."""
    import torch  # noqa: PLC0415

    rows: list[list[float]] = []
    collator = rm_common.ScoredCollator(pad_token_id=tok.pad_token_id, label_dtype="float")
    for start in range(0, len(exchanges), batch_size):
        chunk = exchanges[start : start + batch_size]
        ids = [
            rm_common.encode_exchange(tok, p, a, add_bos=add_bos, max_length=max_len) for p, a in chunk
        ]
        batch = collator([(row, 0.0) for row in ids])
        with torch.no_grad():
            pooled = rm_common.pooled_scores(
                model, batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
        rows.extend(pooled.float().cpu().tolist())
    return rows


def score_orm(model, tok, add_bos, max_len, problem, solutions, device) -> list[float]:
    rows = _pooled(model, tok, add_bos, max_len, [(problem, s) for s in solutions], device)
    return [rm_verifier.orm_reward(r[0]) for r in rows]


def score_prm(model, tok, add_bos, max_len, problem, solutions, device) -> list[float]:
    # Build every cumulative-prefix exchange across all solutions in one flat batch.
    exchanges: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for sol in solutions:
        steps = rm_verifier.split_steps(sol)
        start = len(exchanges)
        for i in range(len(steps)):
            exchanges.append((problem, rm_common.assistant_text_from_steps(steps[: i + 1])))
        spans.append((start, len(exchanges)))
    rows = _pooled(model, tok, add_bos, max_len, exchanges, device) if exchanges else []
    scores: list[float] = []
    for start, end in spans:
        scores.append(rm_verifier.prm_reward(rows[start:end]) if end > start else 0.0)
    return scores


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


def upload_json(s3: Any, obj: dict, out_uri: str) -> str:
    bucket, key = _split_uri(out_uri)
    tmp = "/tmp/bon_report.json"
    Path(tmp).write_text(json.dumps(obj, indent=2))
    s3.put_file(bucket, key.rstrip("/") + "/bon_report.json", tmp)
    return f"s3://{bucket}/{key.rstrip('/')}/bon_report.json"


# --------------------------------------------------------------------------------------------
def _selftest() -> None:
    assert argmax([0.1, 0.9, 0.3]) == 1
    assert argmax([0.5, 0.5]) == 0  # stable tie
    assert extract_answer("blah\n# Answer\n\n42") == "42"
    assert extract_answer("no marker here\n\n7") == "7"
    # majority: '42' appears twice (correct), '7' once (incorrect) -> majority correct
    assert majority_correct(["42", "7", "42"], [True, False, True]) is True
    # majority wrong: '7' twice (incorrect) beats '42' once (correct)
    assert majority_correct(["7", "42", "7"], [False, True, False]) is False
    # bon picks the highest score
    assert bon_correct([0.2, 0.8, 0.1], [False, True, False]) is True
    assert bon_correct([0.9, 0.1], [False, True]) is False
    per = [
        {"greedy": 0, "majority": 0, "orm_bon": 1, "prm_bon": 1, "pass_at_n": 1},
        {"greedy": 1, "majority": 1, "orm_bon": 1, "prm_bon": 0, "pass_at_n": 1},
    ]
    s = summarize(per)
    assert s["orm_bon"] == 1.0 and s["prm_bon"] == 0.5 and s["pass_at_n"] == 1.0
    assert s["gate_go"] is True  # orm_bon 1.0 > max(majority .5, greedy .5)
    s_nogo = summarize([{"greedy": 1, "majority": 1, "orm_bon": 1, "prm_bon": 1, "pass_at_n": 1}])
    assert s_nogo["gate_go"] is False  # RMs don't beat majority/greedy
    print("BON_EVAL SELFTEST OK: argmax, extract, majority, best-of-N, gate decision verified")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-uri", help="s3:// prefix of the Stage-2 SFT policy")
    ap.add_argument("--orm-uri", help="s3:// prefix of the trained ORM")
    ap.add_argument("--prm-uri", help="s3:// prefix of the trained PRM")
    ap.add_argument("--eval-uri", help="s3:// URI of eval_gsm8k.jsonl")
    ap.add_argument("--out", help="s3:// prefix to write bon_report.json (EDULLM_CHECKPOINT_DIR)")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math"])
    ap.add_argument("--n", type=int, default=16, help="samples per problem")
    ap.add_argument("--num-problems", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--add-bos", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    required = [("--sft-uri", args.sft_uri), ("--orm-uri", args.orm_uri), ("--prm-uri", args.prm_uri),
                ("--eval-uri", args.eval_uri), ("--out", args.out)]
    for name, val in required:
        if not val:
            print(f"{name} is required", file=sys.stderr)
            raise SystemExit(2)

    import torch  # noqa: PLC0415
    from edullm_data.s3 import Boto3S3  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from open_instruct.ground_truth_utils import GSM8KVerifier, MathVerifier  # noqa: PLC0415

    s3 = Boto3S3.default()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sft_dir, orm_dir, prm_dir = "/tmp/bon_sft", "/tmp/bon_orm", "/tmp/bon_prm"
    print(f"downloading SFT {args.sft_uri} ...", flush=True)
    download_prefix(s3, args.sft_uri, sft_dir)
    # vLLM's olmo2 loader reads config.rope_parameters["rope_theta"]; ensure the SFT config carries
    # it (some saved Olmo3 configs omit it) so the engine initialises with the checkpoint's own base.
    rm_common.ensure_vllm_rope_parameters(sft_dir)
    print(f"downloading ORM {args.orm_uri} ...", flush=True)
    download_prefix(s3, args.orm_uri, orm_dir)
    print(f"downloading PRM {args.prm_uri} ...", flush=True)
    download_prefix(s3, args.prm_uri, prm_dir)
    download_file(s3, args.eval_uri, "/tmp/bon_eval.jsonl")

    rows = [json.loads(ln) for ln in Path("/tmp/bon_eval.jsonl").read_text().splitlines() if ln.strip()]
    rows = rows[: args.num_problems]
    prompts = [rm_common.tulu_prompt_string(r["problem"]) for r in rows]
    print(f"loaded {len(rows)} eval problems; sampling N={args.n} each", flush=True)

    llm = LLM(model=sft_dir, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_seq_length, enforce_eager=True)
    stop = ["<|user|>"]
    greedy_out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=stop))
    samp_out = llm.generate(
        prompts,
        SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                       max_tokens=args.max_tokens, stop=stop),
    )

    verifier = GSM8KVerifier() if args.dataset == "gsm8k" else MathVerifier()

    def is_correct(pred: str, answer: str) -> bool:
        return verifier(tokenized_prediction=[], prediction=pred, label=answer).score >= 0.5

    # RMs loaded one at a time to keep VRAM alongside vLLM modest.
    orm_model, orm_tok, orm_bos = _load_rm(orm_dir, "orm", device, args.add_bos)
    prm_model, prm_tok, prm_bos = _load_rm(prm_dir, "prm", device, args.add_bos)

    per_problem: list[dict] = []
    for i, r in enumerate(rows):
        answer = r["answer"]
        cands = [o.text for o in samp_out[i].outputs]
        correct = [is_correct(c, answer) for c in cands]
        greedy_c = is_correct(greedy_out[i].outputs[0].text, answer)
        answers = [extract_answer(c) for c in cands]
        orm_scores = score_orm(orm_model, orm_tok, orm_bos, args.max_seq_length, r["problem"], cands, device)
        prm_scores = score_prm(prm_model, prm_tok, prm_bos, args.max_seq_length, r["problem"], cands, device)
        per_problem.append({
            "greedy": int(greedy_c),
            "majority": int(majority_correct(answers, correct)),
            "orm_bon": int(bon_correct(orm_scores, correct)),
            "prm_bon": int(bon_correct(prm_scores, correct)),
            "pass_at_n": int(any(correct)),
        })

    report = summarize(per_problem)
    report["config"] = {"dataset": args.dataset, "n": args.n, "num_problems": len(rows),
                        "temperature": args.temperature}
    print("=== BEST-OF-N GATE ===", flush=True)
    for k in ("greedy", "majority", "orm_bon", "prm_bon", "pass_at_n"):
        print(f"  {k:10s} {report[k]:.4f}", flush=True)
    uri = upload_json(s3, report, args.out)
    verdict = "GO" if report["gate_go"] else "NO-GO"
    print(f"BON GATE: {verdict}  (report={uri})", flush=True)


if __name__ == "__main__":
    main()
