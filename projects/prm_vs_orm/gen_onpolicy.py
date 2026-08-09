"""On-policy RM data generation for the PRM-vs-ORM experiment (the re-gate recommendation).

The first RM pass trained ORM/PRM on **PRM800K** (GPT-4 solutions). At the Stage-5 best-of-N
gate those RMs failed to out-rank plain majority voting on the 370M SFT policy's *own* GSM8K
samples (NO-GO; see the write-up / memory ``prm-vs-orm-result``). The diagnosis was distribution
shift: an RM trained on a strong model's solutions does not transfer to rank a tiny policy's
on-distribution outputs. This script produces **on-policy** RM training data instead — samples
drawn from *this* SFT policy, labelled by the ground-truth checker — so a re-run of the same
cheap single-GPU gate can decide GO/NO-GO before any A100/GRPO spend.

It emits the **same two corpora the Stage-3/4 trainer already reads** (``reward_modeling_scored.py``
via ``run_rm.py``), so nothing downstream changes:

  * ``orm.jsonl`` -- ``{"messages": [user, assistant], "outcome": 0|1}``. For each prompt we
    sample ``--n-samples`` solutions from the SFT policy and grade each whole solution with
    open_instruct's own ``GSM8KVerifier`` / ``MathVerifier`` (reuse, not a fresh grader). To keep
    the two outcome classes from collapsing on a weak policy we keep every *correct* sample and
    cap *incorrect* samples per problem (``--orm-neg-per-problem``).
  * ``prm.jsonl`` -- ``{"problem", "prefix": [step,...], "step", "rating": -1|0|1}``. Per-step
    process labels via **Math-Shepherd Monte-Carlo estimation**: split a sampled solution into
    steps (the *same* ``STEP_SEP`` split ``rm_verifier`` scores with), and for each step prefix
    roll out ``--mc-rollouts`` continuations from the SFT policy; the step's value is the fraction
    of rollouts that reach the correct final answer. The true final step needs no rollout -- its
    value is the trajectory's own correctness. A value is thresholded to the 3-class rating the PRM
    trains on: ``>= --prm-tau-pos`` -> +1, ``<= --prm-tau-neg`` -> -1, else 0 (neutral). We MC-label
    a few correct + a few incorrect trajectories per problem (``--prm-pos-trajs`` / ``--prm-neg-trajs``)
    so the PRM sees both good and derailed process where the policy can actually solve the problem.

Runs as an eduLLM ``-train`` GPU job (single process, single GPU via vLLM -- no launch waiver),
mirroring ``bon_eval.py``'s S3 + vLLM + rope-fix plumbing::

    python projects/prm_vs_orm/gen_onpolicy.py \\
        --sft-uri     s3://.../runs/<sft_run>/checkpoints/ \\
        --prompts-uri s3://.../runs/<data_run>/checkpoints/prompts_gsm8k.jsonl \\
        --out "$EDULLM_CHECKPOINT_DIR"

The generated ``orm.jsonl`` / ``prm.jsonl`` then feed ``run_rm.py --data-uri`` unchanged, and the
retrained RMs go through the same ``bon_eval.py`` gate.

``--selftest`` exercises the pure selection/labelling logic (ORM balancing, MC value, value->rating
threshold, PRM row assembly, step split) offline -- no torch, no vLLM, no S3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_common  # noqa: E402 -- overlay-local module, added to path above
import rm_verifier  # noqa: E402 -- overlay-local; split_steps is the scoring-time step segmentation


# --------------------------------------------------------------------------------------------
# Pure selection / labelling logic (no torch / no vLLM / no S3) -- unit-tested offline.
# --------------------------------------------------------------------------------------------
def orm_rows_for_problem(
    problem: str, solutions: list[str], correct: list[bool], neg_cap: int
) -> list[dict]:
    """Balanced ORM rows for one problem: every correct solution + up to ``neg_cap`` incorrect.

    A 370M policy is mostly wrong, so keeping all samples would swamp the ORM with ``outcome=0``.
    We keep every positive (they are the scarce, load-bearing class) and cap negatives per problem.
    Order is preserved so the cap is deterministic (earliest incorrect samples kept).
    """
    rows: list[dict] = []
    neg_kept = 0
    for sol, ok in zip(solutions, correct):
        if not ok:
            if neg_kept >= neg_cap:
                continue
            neg_kept += 1
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": sol},
                ],
                "outcome": int(ok),
            }
        )
    return rows


def select_prm_trajectories(
    correct: list[bool], pos_cap: int, neg_cap: int
) -> list[int]:
    """Indices of solutions to MC-label for the PRM: up to ``pos_cap`` correct + ``neg_cap`` incorrect.

    Correct trajectories give the "reachable" process signal; incorrect ones on a solvable problem
    give the derail (where P(correct) drops). Earliest-first for determinism.
    """
    pos = [i for i, ok in enumerate(correct) if ok][:pos_cap]
    neg = [i for i, ok in enumerate(correct) if not ok][:neg_cap]
    return sorted(pos + neg)


def mc_value(rollout_correct: list[bool]) -> float:
    """Math-Shepherd step value = fraction of rollouts from this prefix that reach the answer."""
    if not rollout_correct:
        return 0.0
    return sum(1 for c in rollout_correct if c) / len(rollout_correct)


def value_to_rating(value: float, tau_pos: float, tau_neg: float) -> int:
    """Bin a step's MC value into the PRM's 3-class rating (+1 good / 0 neutral / -1 bad)."""
    if value >= tau_pos:
        return 1
    if value <= tau_neg:
        return -1
    return 0


def prm_rows_from_ratings(problem: str, steps: list[str], ratings: list[int]) -> list[dict]:
    """Assemble ``(problem, prefix, step, rating)`` rows -- one per labelled step.

    ``prefix`` is the steps before ``step``; the trainer joins ``prefix+[step]`` with ``STEP_SEP``
    and pools at the trailing eos, so the label lands on this step's boundary token.
    """
    rows: list[dict] = []
    for i, rating in enumerate(ratings):
        rows.append(
            {"problem": problem, "prefix": list(steps[:i]), "step": steps[i], "rating": int(rating)}
        )
    return rows


# --------------------------------------------------------------------------------------------
# S3 plumbing (mirrors bon_eval.py / run_rm.py).
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


def write_jsonl_s3(s3: Any, out_prefix: str, name: str, rows: list[dict]) -> str:
    body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    uri = out_prefix.rstrip("/") + "/" + name
    bucket, key = _split_uri(uri)
    s3.put(bucket, key, body, content_type="application/x-ndjson")
    return uri


def load_prompts(s3: Any, prompts_uri: str, num_problems: int) -> list[dict]:
    """Load and concatenate one or more ``prompts_*.jsonl`` files (comma-separated URIs).

    Each row is ``{"problem", "answer", "dataset"}`` (data_prep.py). ``num_problems`` caps the
    total across all files (round-robin so a mix of datasets stays balanced when capped).
    """
    per_file: list[list[dict]] = []
    for uri in [u.strip() for u in prompts_uri.split(",") if u.strip()]:
        local = f"/tmp/prompts_{len(per_file)}.jsonl"
        download_file(s3, uri, local)
        rows = [json.loads(ln) for ln in Path(local).read_text().splitlines() if ln.strip()]
        per_file.append(rows)
    # round-robin interleave so a total cap doesn't drop an entire dataset
    merged: list[dict] = []
    idx = 0
    while any(idx < len(f) for f in per_file):
        for f in per_file:
            if idx < len(f):
                merged.append(f[idx])
        idx += 1
    if num_problems > 0:
        merged = merged[:num_problems]
    return merged


# --------------------------------------------------------------------------------------------
def _selftest() -> None:
    # ORM balancing: keep all correct, cap incorrect.
    sols = ["a", "b", "c", "d"]
    corr = [True, False, False, False]
    rows = orm_rows_for_problem("p", sols, corr, neg_cap=2)
    outcomes = [r["outcome"] for r in rows]
    assert outcomes == [1, 0, 0], outcomes  # 1 correct + 2 of 3 incorrect
    assert rows[0]["messages"][0]["content"] == "p" and rows[0]["messages"][1]["content"] == "a"
    # all-correct problem keeps them all regardless of neg_cap
    assert len(orm_rows_for_problem("p", ["x", "y"], [True, True], neg_cap=0)) == 2

    # PRM trajectory selection: up to pos_cap correct + neg_cap incorrect, sorted.
    correct = [False, True, False, True, False]
    assert select_prm_trajectories(correct, pos_cap=1, neg_cap=2) == [0, 1, 2]
    assert select_prm_trajectories(correct, pos_cap=5, neg_cap=0) == [1, 3]
    assert select_prm_trajectories([False, False], pos_cap=2, neg_cap=1) == [0]

    # MC value + rating threshold.
    assert mc_value([]) == 0.0
    assert mc_value([True, True, False, False]) == 0.5
    assert value_to_rating(1.0, tau_pos=0.5, tau_neg=0.0) == 1
    assert value_to_rating(0.3, tau_pos=0.5, tau_neg=0.0) == 0
    assert value_to_rating(0.0, tau_pos=0.5, tau_neg=0.0) == -1

    # PRM row assembly: prefix grows, one row per rating.
    steps = ["s0", "s1", "s2"]
    prm_rows = prm_rows_from_ratings("q", steps, [1, 0, -1])
    assert [r["rating"] for r in prm_rows] == [1, 0, -1]
    assert prm_rows[0]["prefix"] == [] and prm_rows[0]["step"] == "s0"
    assert prm_rows[2]["prefix"] == ["s0", "s1"] and prm_rows[2]["step"] == "s2"

    # step segmentation is exactly the scoring-time split.
    assert rm_verifier.split_steps("a\n\nb\n\nc") == ["a", "b", "c"]

    print(
        "GEN_ONPOLICY SELFTEST OK: ORM balancing, trajectory selection, MC value, "
        "value->rating, PRM row assembly, step split verified"
    )


def _grade_batch(verifiers: dict, dataset: str, predictions: list[str], answer: str) -> list[bool]:
    """Grade a list of predictions against one answer with the dataset's ground-truth verifier."""
    verifier = verifiers[dataset]
    return [
        verifier(tokenized_prediction=[], prediction=pred, label=answer).score >= 0.5
        for pred in predictions
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-uri", help="s3:// prefix of the Stage-2 SFT policy")
    ap.add_argument("--prompts-uri", help="comma-separated s3:// URIs of prompts_*.jsonl")
    ap.add_argument("--out", help="s3:// prefix to write orm.jsonl/prm.jsonl/MANIFEST.json (EDULLM_CHECKPOINT_DIR)")
    ap.add_argument("--num-problems", type=int, default=1200, help="cap total prompts (0 = all)")
    ap.add_argument("--n-samples", type=int, default=16, help="solutions sampled per problem (ORM pool)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    # ORM balancing
    ap.add_argument("--orm-neg-per-problem", type=int, default=4, help="cap incorrect ORM samples/problem")
    # PRM Math-Shepherd MC
    ap.add_argument("--mc-rollouts", type=int, default=4, help="continuations per step prefix (M)")
    ap.add_argument("--prm-pos-trajs", type=int, default=2, help="correct trajectories MC-labelled/problem")
    ap.add_argument("--prm-neg-trajs", type=int, default=2, help="incorrect trajectories MC-labelled/problem")
    ap.add_argument("--prm-max-steps", type=int, default=8, help="cap MC-labelled steps per trajectory")
    ap.add_argument("--prm-tau-pos", type=float, default=0.5, help="MC value >= this -> rating +1")
    ap.add_argument("--prm-tau-neg", type=float, default=0.0, help="MC value <= this -> rating -1")
    ap.add_argument("--mc-temperature", type=float, default=0.8, help="temperature for MC rollouts")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    for name, val in (("--sft-uri", args.sft_uri), ("--prompts-uri", args.prompts_uri), ("--out", args.out)):
        if not val:
            print(f"{name} is required", file=sys.stderr)
            raise SystemExit(2)
    if not args.sft_uri.startswith("s3://") or not args.out.startswith("s3://"):
        print("--sft-uri and --out must be s3:// URIs", file=sys.stderr)
        raise SystemExit(2)

    import torch  # noqa: PLC0415
    from edullm_data.s3 import Boto3S3  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from open_instruct.ground_truth_utils import GSM8KVerifier, MathVerifier  # noqa: PLC0415

    s3 = Boto3S3.default()
    _ = "cuda" if torch.cuda.is_available() else "cpu"  # vLLM manages placement; kept for parity

    sft_dir = "/tmp/onpolicy_sft"
    print(f"downloading SFT {args.sft_uri} ...", flush=True)
    download_prefix(s3, args.sft_uri, sft_dir)
    # vLLM's olmo2 loader needs a hashable, flat rope config (see rm_common.ensure_vllm_rope_parameters).
    rm_common.ensure_vllm_rope_parameters(sft_dir)

    problems = load_prompts(s3, args.prompts_uri, args.num_problems)
    print(f"loaded {len(problems)} prompts; sampling N={args.n_samples} each", flush=True)

    verifiers = {"gsm8k": GSM8KVerifier(), "math": MathVerifier()}

    llm = LLM(
        model=sft_dir, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_seq_length, enforce_eager=True,
    )
    stop = ["<|user|>"]

    # ---- ORM phase: sample n solutions/problem, grade, emit balanced outcome rows -------------
    prompts = [rm_common.tulu_prompt_string(p["problem"]) for p in problems]
    samp = llm.generate(
        prompts,
        SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
                       max_tokens=args.max_tokens, stop=stop),
    )
    orm_rows: list[dict] = []
    orm_pos = 0
    # per-problem: solutions + correctness (reused by the PRM phase), and the chosen trajectories.
    per_problem_solutions: list[list[str]] = []
    per_problem_correct: list[list[bool]] = []
    for p, out in zip(problems, samp):
        sols = [o.text for o in out.outputs]
        correct = _grade_batch(verifiers, p["dataset"], sols, p["answer"])
        per_problem_solutions.append(sols)
        per_problem_correct.append(correct)
        rows = orm_rows_for_problem(p["problem"], sols, correct, args.orm_neg_per_problem)
        orm_rows.extend(rows)
        orm_pos += sum(r["outcome"] for r in rows)
    print(
        f"ORM: {len(orm_rows)} rows ({orm_pos} correct / {len(orm_rows) - orm_pos} incorrect); "
        f"solvable problems (pass@{args.n_samples}>0): "
        f"{sum(1 for c in per_problem_correct if any(c))}/{len(problems)}",
        flush=True,
    )

    # ---- PRM phase: Math-Shepherd MC labels on a few trajectories/problem --------------------
    # Build one flat batch of every non-final step-prefix continuation across all trajectories,
    # so vLLM sees a single large generate() (max throughput). spans map rows back to their step.
    cont_prompts: list[str] = []
    # each job: (problem_idx, traj_solution_idx, step_idx, is_final, prefix_full_text)
    jobs: list[tuple[int, int, int, bool, str]] = []
    traj_registry: list[tuple[int, int, list[str]]] = []  # (problem_idx, sol_idx, steps)
    for pi, (p, sols, correct) in enumerate(zip(problems, per_problem_solutions, per_problem_correct)):
        chosen = select_prm_trajectories(correct, args.prm_pos_trajs, args.prm_neg_trajs)
        for si in chosen:
            steps = rm_verifier.split_steps(sols[si])
            steps = steps[: args.prm_max_steps]
            if not steps:
                continue
            traj_registry.append((pi, si, steps))
            n_steps = len(steps)
            full_steps = rm_verifier.split_steps(sols[si])  # untruncated -> is a step the true final?
            for step_idx in range(n_steps):
                is_final = step_idx == len(full_steps) - 1
                prefix_full = rm_common.assistant_text_from_steps(steps[: step_idx + 1])
                if not is_final:
                    cont_prompts.append(rm_common.tulu_prompt_string(p["problem"]) + prefix_full)
                    jobs.append((pi, si, step_idx, False, prefix_full))
                else:
                    jobs.append((pi, si, step_idx, True, prefix_full))

    n_rollout_jobs = sum(1 for j in jobs if not j[3])
    print(f"PRM: {len(traj_registry)} trajectories, {len(jobs)} steps "
          f"({n_rollout_jobs} need MC rollouts x M={args.mc_rollouts})", flush=True)

    # Rollout only the non-final steps; final steps use the trajectory's own correctness.
    mc_out = (
        llm.generate(
            cont_prompts,
            SamplingParams(n=args.mc_rollouts, temperature=args.mc_temperature, top_p=args.top_p,
                           max_tokens=args.max_tokens, stop=stop),
        )
        if cont_prompts
        else []
    )

    # Grade every rollout and compute each step's MC value -> rating.
    step_value: dict[tuple[int, int, int], float] = {}
    rollout_cursor = 0
    for (pi, si, step_idx, is_final, prefix_full) in jobs:
        p = problems[pi]
        if is_final:
            # final step's value is the trajectory's own correctness (already graded in ORM phase)
            step_value[(pi, si, step_idx)] = 1.0 if per_problem_correct[pi][si] else 0.0
            continue
        rollout_texts = [o.text for o in mc_out[rollout_cursor].outputs]
        rollout_cursor += 1
        # grade the full solution: prefix so far + this rollout continuation
        preds = [prefix_full + rt for rt in rollout_texts]
        flags = _grade_batch(verifiers, p["dataset"], preds, p["answer"])
        step_value[(pi, si, step_idx)] = mc_value(flags)

    prm_rows: list[dict] = []
    rating_hist = {-1: 0, 0: 0, 1: 0}
    for (pi, si, steps) in traj_registry:
        ratings = [
            value_to_rating(step_value[(pi, si, step_idx)], args.prm_tau_pos, args.prm_tau_neg)
            for step_idx in range(len(steps))
        ]
        for r in ratings:
            rating_hist[r] += 1
        prm_rows.extend(prm_rows_from_ratings(problems[pi]["problem"], steps, ratings))
    print(f"PRM: {len(prm_rows)} step rows; rating histogram neg/neu/pos = "
          f"{rating_hist[-1]}/{rating_hist[0]}/{rating_hist[1]}", flush=True)

    # ---- upload corpora + manifest -----------------------------------------------------------
    orm_uri = write_jsonl_s3(s3, args.out, "orm.jsonl", orm_rows)
    prm_uri = write_jsonl_s3(s3, args.out, "prm.jsonl", prm_rows)
    manifest = {
        "kind": "on-policy-rm-data",
        "sft_uri": args.sft_uri.rstrip("/"),
        "prompts_uri": args.prompts_uri,
        "counts": {"orm": len(orm_rows), "orm_correct": orm_pos, "prm": len(prm_rows)},
        "prm_rating_hist": {"neg": rating_hist[-1], "neu": rating_hist[0], "pos": rating_hist[1]},
        "solvable_problems": sum(1 for c in per_problem_correct if any(c)),
        "num_problems": len(problems),
        "config": {
            "n_samples": args.n_samples, "temperature": args.temperature, "top_p": args.top_p,
            "max_tokens": args.max_tokens, "orm_neg_per_problem": args.orm_neg_per_problem,
            "mc_rollouts": args.mc_rollouts, "prm_pos_trajs": args.prm_pos_trajs,
            "prm_neg_trajs": args.prm_neg_trajs, "prm_max_steps": args.prm_max_steps,
            "prm_tau_pos": args.prm_tau_pos, "prm_tau_neg": args.prm_tau_neg,
        },
        "uris": {"orm": orm_uri, "prm": prm_uri},
    }
    write_jsonl_s3(s3, args.out, "MANIFEST.json", [manifest])
    print("GEN_ONPOLICY OK: " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
