"""Fail fast when an AWS tutor-RL image carries mismatched data or heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def item_from(row: dict) -> dict:
    ground_truth = row.get("ground_truth", {})
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    return {**ground_truth, **{key: value for key, value in row.items() if key != "ground_truth"}}


def question_key(item: dict) -> str:
    payload = json.dumps(
        {"question": item.get("question") or item.get("prompt"), "choices": item.get("choices") or []},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_rows(path: Path, prompt_scheme: str) -> tuple[list[dict], set[str]]:
    from projects.tutor_metrics.generate import tutor_messages, tutor_messages_neutral  # noqa: PLC0415

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} has no rows")
    questions = set()
    for index, row in enumerate(rows, start=1):
        item = item_from(row)
        missing = [key for key in ("question", "student_before", "target_level") if not item.get(key)]
        if prompt_scheme == "tutor_metrics" and not item.get("style"):
            missing.append("style")
        if missing:
            raise SystemExit(f"{path}:{index} lacks {sorted(set(missing))}")
        if not isinstance(row.get("messages"), list) or len(row["messages"]) < 2:
            raise SystemExit(f"{path}:{index} needs the exact policy messages")
        problem = {"question": item["question"], "choices": item.get("choices")}
        expected = (
            tutor_messages(problem, item["student_before"], item["style"])
            if prompt_scheme == "tutor_metrics"
            else tutor_messages_neutral(problem, item["student_before"])
        )
        if row["messages"] != expected:
            raise SystemExit(f"{path}:{index} messages do not match the reward head's {prompt_scheme} context")
        questions.add(question_key(item))
    return rows, questions


def main() -> None:
    import numpy as np  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--expected-encoder", default="")
    parser.add_argument("--expected-revision", default="")
    parser.add_argument("--minimum-train-rows", type=int, default=32)
    args = parser.parse_args()

    head = np.load(args.head, allow_pickle=False)
    meta = json.loads(str(head["meta"]))
    if meta.get("schema") != "tutor-metrics/reward-head-v1":
        raise SystemExit(f"{args.head} has unsupported schema {meta.get('schema')!r}")
    if not meta.get("model") or not meta.get("revision"):
        raise SystemExit("reward head must pin model and immutable revision")
    if args.expected_encoder and meta["model"] != args.expected_encoder:
        raise SystemExit(
            f"reward head uses {meta['model']!r}; this launch requires {args.expected_encoder!r} "
            "to fit beside TP=4 vLLM on A100-40GB"
        )
    if args.expected_revision and meta["revision"] != args.expected_revision:
        raise SystemExit(
            f"reward head uses revision {meta['revision']!r}; expected immutable revision "
            f"{args.expected_revision!r}"
        )
    if meta.get("prompt_scheme") not in {"tutor_metrics", "tutor_metrics_neutral"}:
        raise SystemExit(f"unsupported prompt scheme {meta.get('prompt_scheme')!r}")
    dimensions = meta.get("dimensions") or {}
    if "assistance_level" not in dimensions:
        raise SystemExit("reward head lacks assistance_level, the moving-target spine")
    deployable = [key for key, spec in dimensions.items() if spec.get("deployable", True)]
    if "assistance_level" not in deployable:
        raise SystemExit("reward head marks assistance_level non-deployable")
    for key in dimensions:
        for suffix in ("mean", "scale", "coef", "intercept"):
            if f"{key}/{suffix}" not in head:
                raise SystemExit(f"{args.head} lacks {key}/{suffix}")

    train, train_questions = load_rows(args.train, meta["prompt_scheme"])
    _, eval_questions = load_rows(args.eval, meta["prompt_scheme"])
    overlap = train_questions & eval_questions
    if overlap:
        raise SystemExit(f"train/eval leak {len(overlap)} identical questions")
    if len(train) < args.minimum_train_rows:
        raise SystemExit(f"{args.train} has {len(train)} rows; need at least {args.minimum_train_rows}")

    print(
        json.dumps(
            {
                "head": str(args.head),
                "encoder": f"{meta['model']}@{meta['revision']}",
                "prompt_scheme": meta["prompt_scheme"],
                "heads": len(dimensions),
                "deployable_heads": deployable,
                "offline_only_heads": sorted(set(dimensions) - set(deployable)),
                "train_rows": len(train),
                "train_questions": len(train_questions),
                "eval_questions": len(eval_questions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
