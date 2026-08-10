"""Fetch and normalize the three licensed pilot datasets.

Raw files are intentionally written under an ignored output directory. The
normalised JSONL keeps provenance and labels but does not silently turn weak
heuristics into gold construct labels.

Example:
    python -m projects.pedagogy_mech_interp.fetch_data \
        --out projects/pedagogy_mech_interp/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from projects.pedagogy_mech_interp.constructs import HelpLevel

URLS = {
    "mrbench": "https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/main/MRBench/MRBench_V2.json",
    "mathdial": "https://huggingface.co/datasets/eth-nlped/mathdial/resolve/main/train.jsonl",
    "bridge": "https://huggingface.co/datasets/rose-e-wang/bridge/resolve/main/train.json",
}

LICENSES = {"mrbench": "CC-BY-SA-4.0", "mathdial": "CC-BY-SA-4.0", "bridge": "CC-BY-NC-4.0"}


def stable_id(*parts: object, size: int = 16) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:size]


def download(url: str, path: Path, force: bool = False) -> str:
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "pedagogy-mech-pilot/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean(text: object) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def is_yes(value: object) -> bool:
    return clean(value).lower() == "yes"


def reveal(value: object) -> bool:
    return clean(value).lower().startswith("yes")


def context_fields(history: str) -> tuple[str, str]:
    text = history.replace("\u00a0", " ")
    question_match = re.search(r"(?:question (?:below )?is:|question:)\s*(.*?)(?:\n\s*\n|Student:)", text, re.I | re.S)
    tutor_turns = re.findall(r"(?:^|\n\s*)Tutor:\s*(.*?)(?=\n\s*(?:Tutor|Student):|\Z)", text, re.I | re.S)
    question = (
        clean(question_match.group(1))
        if question_match
        else clean(tutor_turns[-1] if tutor_turns else text.split("Student:", 1)[0])
    )
    students = re.findall(r"(?:^|\n\s*)Student:\s*(.*?)(?=\n\s*(?:Tutor|Student):|\Z)", text, re.I | re.S)
    student = clean(students[-1]) if students else ""
    return question, student


def closest_contrast(positive: list[dict], negative: list[dict]) -> tuple[dict, dict] | None:
    if not positive or not negative:
        return None
    return min(
        ((p, n) for p in positive for n in negative),
        key=lambda pair: abs(len(pair[0]["response"].split()) - len(pair[1]["response"].split())),
    )


def variant(
    *,
    pair_id: str,
    suffix: str,
    response: dict,
    label: int,
    action: str,
    label_source: str,
    verification_status: str,
    diagnosis_correct: bool | None = None,
    first_error_located: bool | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "variant_id": f"{pair_id}-{suffix}",
        "tutor_turn": clean(response["response"]),
        "model_source": response["model"],
        "label": label,
        "expected_direction": 1 if label else -1,
        "candidate_action": action,
        "annotations": response["annotation"],
        "label_source": label_source,
        "verification_status": verification_status,
    }
    if diagnosis_correct is not None:
        out["diagnosis_correct"] = diagnosis_correct
    if first_error_located is not None:
        out["first_error_located"] = first_error_located
    return out


def adapt_mrbench(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item_index, item in enumerate(raw):
        item_id = f"mrbench-{item['conversation_id']}"
        source_instance = f"{item_id}-{item_index}"
        question, student = context_fields(str(item.get("conversation_history", "")))
        candidates = [
            {"model": model, "response": value.get("response", ""), "annotation": value.get("annotation", {})}
            for model, value in (item.get("anno_llm_responses") or {}).items()
            if clean(value.get("response"))
        ]
        base = {
            "item_id": item_id,
            "template_id": "mrbench-natural",
            "domain": clean(item.get("Data") or "math_tutoring").lower(),
            "source": "MRBench V2",
            "source_split": clean(item.get("Split")),
            "source_license": LICENSES["mrbench"],
            "question": question,
            "student_before": student,
            "reference": clean(item.get("Ground_Truth_Solution")),
            "natural_or_synthetic": "natural",
            "student_state": "reasoned_attempt",
        }

        diagnosed = [
            c
            for c in candidates
            if is_yes(c["annotation"].get("Mistake_Identification"))
            and is_yes(c["annotation"].get("Mistake_Location"))
        ]
        missed = [
            c
            for c in candidates
            if not is_yes(c["annotation"].get("Mistake_Identification"))
            and not is_yes(c["annotation"].get("Mistake_Location"))
        ]
        if pair := closest_contrast(diagnosed, missed):
            pair_id = f"p-{stable_id(source_instance, 'diagnostic_localization')}"
            rows.append(
                {
                    **base,
                    "pair_id": pair_id,
                    "concept": "diagnostic_localization",
                    "prescribed_action": "diagnose_first_error",
                    "variants": [
                        variant(
                            pair_id=pair_id,
                            suffix="positive",
                            response=pair[0],
                            label=1,
                            action="diagnose_first_error",
                            label_source="dataset_direct",
                            verification_status="dataset_direct",
                            diagnosis_correct=True,
                            first_error_located=True,
                        ),
                        variant(
                            pair_id=pair_id,
                            suffix="negative",
                            response=pair[1],
                            label=0,
                            action="generic_feedback",
                            label_source="dataset_direct",
                            verification_status="dataset_direct",
                            diagnosis_correct=False,
                            first_error_located=False,
                        ),
                    ],
                }
            )

        appropriate_help = [
            c
            for c in candidates
            if is_yes(c["annotation"].get("Providing_Guidance"))
            and is_yes(c["annotation"].get("Actionability"))
            and not reveal(c["annotation"].get("Revealing_of_the_Answer"))
        ]
        inappropriate_help = [
            c
            for c in candidates
            if reveal(c["annotation"].get("Revealing_of_the_Answer"))
            or (
                not is_yes(c["annotation"].get("Providing_Guidance"))
                and not is_yes(c["annotation"].get("Actionability"))
            )
        ]
        if pair := closest_contrast(appropriate_help, inappropriate_help):
            pair_id = f"p-{stable_id(source_instance, 'contingent_scaffolding')}"
            negative_action = (
                HelpLevel.DIRECT_ANSWER.value
                if reveal(pair[1]["annotation"].get("Revealing_of_the_Answer"))
                else HelpLevel.WORKED_STEP.value
            )
            rows.append(
                {
                    **base,
                    "pair_id": pair_id,
                    "concept": "contingent_scaffolding",
                    "prescribed_action": HelpLevel.STRATEGY_HINT.value,
                    "variants": [
                        variant(
                            pair_id=pair_id,
                            suffix="positive",
                            response=pair[0],
                            label=1,
                            action=HelpLevel.STRATEGY_HINT.value,
                            label_source="heuristic_proxy",
                            verification_status="needs_construct_specific_human_label",
                        ),
                        variant(
                            pair_id=pair_id,
                            suffix="negative",
                            response=pair[1],
                            label=0,
                            action=negative_action,
                            label_source="heuristic_proxy",
                            verification_status="needs_construct_specific_human_label",
                        ),
                    ],
                }
            )

        generative = [
            c
            for c in candidates
            if is_yes(c["annotation"].get("Actionability"))
            and not reveal(c["annotation"].get("Revealing_of_the_Answer"))
            and "?" in c["response"]
        ]
        non_generative = [
            c
            for c in candidates
            if not is_yes(c["annotation"].get("Actionability"))
            or reveal(c["annotation"].get("Revealing_of_the_Answer"))
        ]
        if pair := closest_contrast(generative, non_generative):
            pair_id = f"p-{stable_id(source_instance, 'generative_elicitation')}"
            rows.append(
                {
                    **base,
                    "pair_id": pair_id,
                    "concept": "generative_elicitation",
                    "prescribed_action": "produce_next_step",
                    "variants": [
                        variant(
                            pair_id=pair_id,
                            suffix="positive",
                            response=pair[0],
                            label=1,
                            action="produce_next_step",
                            label_source="heuristic_proxy",
                            verification_status="needs_construct_specific_human_label",
                        ),
                        variant(
                            pair_id=pair_id,
                            suffix="negative",
                            response=pair[1],
                            label=0,
                            action="tutor_explanation",
                            label_source="heuristic_proxy",
                            verification_status="needs_construct_specific_human_label",
                        ),
                    ],
                }
            )
    return rows


def parse_mathdial_conversation(text: str) -> list[dict[str, str]]:
    turns = []
    for raw_turn in text.split("|EOM|"):
        match = re.match(r"\s*(Teacher|Student|[A-Z][a-z]+):\s*(.*)", raw_turn, re.S)
        if not match:
            continue
        role, body = match.groups()
        move = ""
        if role == "Teacher":
            move_match = re.match(r"\(([^)]+)\)\s*(.*)", body, re.S)
            if move_match:
                move, body = move_match.groups()
        turns.append({"role": "tutor" if role == "Teacher" else "student", "move": move, "text": clean(body)})
    return turns


def adapt_mathdial(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            item = json.loads(line)
            rows.append(
                {
                    "item_id": f"mathdial-{item['qid']}",
                    "source": "MathDial",
                    "source_license": LICENSES["mathdial"],
                    "domain": "math",
                    "question": clean(item.get("question")),
                    "reference": clean(item.get("ground_truth")),
                    "student_before": clean(item.get("student_incorrect_solution")),
                    "student_profile": clean(item.get("student_profile")),
                    "teacher_described_confusion": clean(item.get("teacher_described_confusion")),
                    "trajectory": parse_mathdial_conversation(str(item.get("conversation", ""))),
                    "natural_or_synthetic": "natural",
                    "eligible_for_confirmatory_labels": False,
                    "evidence_limit": "Tutor-move labels are useful for trajectories but are not matched counterfactuals.",
                }
            )
    return rows


def adapt_bridge(path: Path, limit: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    rows = []
    for item in raw[:limit]:
        history = item.get("c_h") or []
        student = next((clean(turn.get("text")) for turn in reversed(history) if turn.get("user") == "student"), "")
        rows.append(
            {
                "item_id": f"bridge-{item['c_id']}",
                "source": "Bridge",
                "source_license": LICENSES["bridge"],
                "domain": clean(item.get("lesson_topic") or "math"),
                "question": clean(next((turn.get("text") for turn in reversed(history) if turn.get("user") == "tutor"), "")),
                "student_before": student,
                "history": [{"role": t.get("user"), "text": clean(t.get("text"))} for t in history],
                "original_response": clean(" ".join(t.get("text", "") for t in item.get("c_r") or [])),
                "expert_revision": clean(" ".join(t.get("text", "") for t in item.get("c_r_") or [])),
                "error_type": clean(item.get("e")),
                "pedagogical_action": clean(item.get("z_what")),
                "pedagogical_intention": clean(item.get("z_why")),
                "natural_or_synthetic": "natural",
                "eligible_for_confirmatory_labels": False,
                "verification_status": "needs_construct_specific_human_label",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("projects/pedagogy_mech_interp/data"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mathdial-limit", type=int, default=500)
    parser.add_argument("--bridge-limit", type=int, default=500)
    args = parser.parse_args()

    raw_dir = args.out / "raw"
    files = {
        "mrbench": raw_dir / "MRBench_V2.json",
        "mathdial": raw_dir / "mathdial_train.jsonl",
        "bridge": raw_dir / "bridge_train.json",
    }
    digests = {name: download(URLS[name], path, args.force) for name, path in files.items()}

    mrbench = adapt_mrbench(json.loads(files["mrbench"].read_text()))
    mathdial = adapt_mathdial(files["mathdial"], args.mathdial_limit)
    bridge = adapt_bridge(files["bridge"], args.bridge_limit)
    write_jsonl(args.out / "sources" / "mrbench_pairs.jsonl", mrbench)
    write_jsonl(args.out / "sources" / "mathdial_trajectories.jsonl", mathdial)
    write_jsonl(args.out / "sources" / "bridge_contrasts.jsonl", bridge)

    metadata = {
        "schema": "pedagogy-mech-download/v1",
        "urls": URLS,
        "licenses": LICENSES,
        "sha256": digests,
        "counts": {"mrbench_pairs": len(mrbench), "mathdial": len(mathdial), "bridge": len(bridge)},
    }
    (args.out / "download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
