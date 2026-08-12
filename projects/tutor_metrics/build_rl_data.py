"""Build question-disjoint GRPO JSONL from state-conditioned probe units."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from projects.tutor_metrics.generate import tutor_messages, tutor_messages_neutral


def split_value(question_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def load_units(path: Path) -> list[dict]:
    text = path.read_text()
    try:
        blob = json.loads(text)
    except json.JSONDecodeError:
        blob = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = blob.get("units", [blob]) if isinstance(blob, dict) else blob
    units = []
    for row in rows:
        if "student_before" in row:
            units.append(row)
            continue
        question_key = json.dumps(
            {"question": row["question"], "choices": row.get("choices") or []}, sort_keys=True, ensure_ascii=False
        )
        if "student_text" not in row:
            # Raw assessment items can enter RL directly without another model
            # pass. Use only states that are truthful without inventing
            # problem-specific work or reasoning.
            question_id = row.get("question_id") or hashlib.sha256(question_key.encode()).hexdigest()[:20]
            lost = split_value(question_id, seed=1) < 0.5
            units.append(
                {
                    "id": row.get("id") or question_id,
                    "question_id": question_id,
                    "question": row["question"],
                    "choices": row.get("choices"),
                    "gold": row.get("answer"),
                    "reference": None,
                    "student_before": (
                        "I don't know how to start this problem."
                        if lost
                        else "Can you just tell me which answer choice is correct?"
                    ),
                    "target_level": "tell_step" if lost else "prompt",
                    "scenario": "lost" if lost else "asks_outright",
                    "style": None,
                    "subject": row.get("subject"),
                    "source_item_id": row.get("id"),
                }
            )
            continue
        choices = row.get("choices") or []
        gold_idx = row.get("gold_idx")
        gold = row.get("answer")
        if gold is None and isinstance(gold_idx, int) and 0 <= gold_idx < len(choices):
            gold = choices[gold_idx]
        units.append(
            {
                "id": f"{row.get('item_id')}:{row['scenario']}",
                "question_id": hashlib.sha256(question_key.encode()).hexdigest()[:20],
                "question": row["question"],
                "choices": row.get("choices"),
                "gold": gold,
                "reference": row.get("solution"),
                "student_before": row["student_text"],
                "target_level": row["target_level"],
                "scenario": row["scenario"],
                "style": None,
                "subject": row.get("subject"),
                "source_item_id": row.get("item_id"),
            }
        )
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument(
        "--train-only-units",
        type=Path,
        action="append",
        default=[],
        help="additional unit files assigned entirely to train (repeatable)",
    )
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--eval-out", type=Path, required=True)
    parser.add_argument("--prompt-scheme", choices=("tutor_metrics", "tutor_metrics_neutral"), required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0 < args.eval_fraction < 1:
        raise SystemExit("--eval-fraction must be between zero and one")
    units = [(unit, False) for unit in load_units(args.units)]
    for path in args.train_only_units:
        units.extend((unit, True) for unit in load_units(path))
    outputs = {"train": [], "eval": []}
    for unit, train_only in units:
        split = (
            "train"
            if train_only
            else "eval"
            if split_value(unit["question_id"], args.seed) < args.eval_fraction
            else "train"
        )
        problem = {"question": unit["question"], "choices": unit.get("choices")}
        messages = (
            tutor_messages(problem, unit["student_before"], unit["style"])
            if args.prompt_scheme == "tutor_metrics"
            else tutor_messages_neutral(problem, unit["student_before"])
        )
        ground_truth = {
            "question": unit["question"],
            "choices": unit.get("choices"),
            "gold": unit.get("gold"),
            "reference": unit.get("reference"),
            "student_before": unit["student_before"],
            "target_level": unit["target_level"],
            "scenario": unit["scenario"],
            "style": unit.get("style"),
            "question_id": unit["question_id"],
            "subject": unit.get("subject"),
            "source_item_id": unit.get("source_item_id"),
        }
        outputs[split].append(
            {
                "messages": messages,
                "ground_truth": json.dumps(ground_truth, sort_keys=True),
                # The completion is a tutor response, not the student's final
                # answer. A math verifier would reward leaking `gold` even when
                # the prescribed assistance level says to withhold it.
                "dataset": "passthrough",
            }
        )

    for split, path in (("train", args.train_out), ("eval", args.eval_out)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in outputs[split]))
    print(f"wrote {len(outputs['train'])} train and {len(outputs['eval'])} eval rows")


if __name__ == "__main__":
    main()
