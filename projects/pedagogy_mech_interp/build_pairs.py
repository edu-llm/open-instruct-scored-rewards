"""Validate matched contrasts and create leakage-safe grouped splits.

Input is one JSON object per scenario family with exactly two or more
``variants``. The output contains one flattened row per variant. Confirmatory
labels are written to a separate key file so tracing can run while labels stay
sealed.

Example:
    python -m projects.pedagogy_mech_interp.build_pairs \
        --sources projects/pedagogy_mech_interp/data/sources/mrbench_pairs.jsonl \
        --out projects/pedagogy_mech_interp/data/pairs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from projects.pedagogy_mech_interp.constructs import CONSTRUCTS, validate_label

SPLITS = (
    ("discovery_train", 0.50),
    ("discovery_validation", 0.15),
    ("natural_test", 0.15),
    ("confirmatory", 0.20),
)

REQUIRED_FAMILY_FIELDS = {
    "item_id",
    "pair_id",
    "template_id",
    "domain",
    "source",
    "concept",
    "student_state",
    "prescribed_action",
    "natural_or_synthetic",
    "question",
    "student_before",
    "variants",
}


def normalise(text: object) -> str:
    value = str(text or "").lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)
    return value.strip()


def digest(text: object) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()


def assign_split(group_id: str, seed: int, domain: str = "", confirmatory_domain: str = "") -> str:
    if confirmatory_domain and domain.lower() == confirmatory_domain.lower():
        return "confirmatory"
    value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) / 16**16
    available = SPLITS[:3] if confirmatory_domain else SPLITS
    total = sum(share for _, share in available)
    running = 0.0
    for name, share in available:
        running += share / total
        if value < running:
            return name
    return available[-1][0]


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                if line.strip():
                    row = json.loads(line)
                    row["_input"] = f"{path}:{line_no}"
                    rows.append(row)
    return rows


def family_errors(family: dict[str, Any]) -> list[str]:
    errors = []
    missing = sorted(field for field in REQUIRED_FAMILY_FIELDS if family.get(field) in (None, ""))
    if missing:
        errors.append(f"missing fields: {missing}")
    if family.get("concept") not in CONSTRUCTS:
        errors.append(f"unknown concept: {family.get('concept')!r}")
    variants = family.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        errors.append("variants must contain at least two candidates")
        return errors
    labels = [variant.get("label") for variant in variants]
    if 0 not in labels or 1 not in labels:
        errors.append(f"matched family must contain labels 0 and 1, got {labels}")
    texts = [normalise(variant.get("tutor_turn")) for variant in variants]
    if any(not text for text in texts):
        errors.append("empty tutor turn")
    if len(set(texts)) != len(texts):
        errors.append("duplicate variants after text normalization")
    return errors


def flatten(family: dict[str, Any], split: str) -> list[dict[str, Any]]:
    shared = {key: value for key, value in family.items() if key not in {"variants", "_input"}}
    out = []
    for variant in family["variants"]:
        row = {**shared, **variant, "split": split}
        row["text_hash"] = digest(row["tutor_turn"])
        row["context_hash"] = digest(f"{row['question']} {row['student_before']}")
        row["word_count"] = len(str(row["tutor_turn"]).split())
        validate_label(row)
        out.append(row)
    return out


def verifier_findings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    for pair_id, variants in by_pair.items():
        counts = [row["word_count"] for row in variants]
        ratio = min(counts) / max(counts) if max(counts) else 0.0
        reasons = []
        if ratio < 0.65:
            reasons.append(f"length_ratio={ratio:.2f}")
        if len({row["context_hash"] for row in variants}) != 1:
            reasons.append("context_changed_within_pair")
        if any(row.get("verification_status") not in {"dataset_direct", "human_verified"} for row in variants):
            reasons.append("not_independently_verified")
        if reasons:
            findings.append(
                {
                    "pair_id": pair_id,
                    "concept": variants[0]["concept"],
                    "source": variants[0]["source"],
                    "reasons": reasons,
                    "variants": [
                        {
                            "variant_id": row["variant_id"],
                            "label": row["label"],
                            "tutor_turn": row["tutor_turn"],
                        }
                        for row in variants
                    ],
                }
            )
    return findings


def assert_no_cross_split_leakage(rows: list[dict[str, Any]]) -> None:
    seen_items: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    for row in rows:
        split = row["split"]
        item_id = row["item_id"]
        if item_id in seen_items and seen_items[item_id] != split:
            raise ValueError(f"item {item_id} appears in {seen_items[item_id]} and {split}")
        seen_items[item_id] = split
        text_hash = row["text_hash"]
        if text_hash in seen_text and seen_text[text_hash] != split:
            raise ValueError(f"normalized response {text_hash[:12]} appears in {seen_text[text_hash]} and {split}")
        seen_text[text_hash] = split


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_families(families: list[dict[str, Any]], max_families: int, seed: int) -> list[dict[str, Any]]:
    if not max_families or len(families) <= max_families:
        return families
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for family in families:
        buckets[(family["concept"], family["source"])].append(family)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen = []
    keys = sorted(buckets)
    while len(chosen) < max_families and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(chosen) < max_families:
                chosen.append(buckets[key].pop())
    return chosen


def drop_cross_item_text_duplicates(families: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove families that would put a memorisable response in multiple item groups.

    Repeated generic responses such as "Good job" are neither useful evidence
    nor independent examples. Variants reused by different constructs for the
    same item are retained because the entire item is assigned as one group.
    """
    owner: dict[str, str] = {}
    kept = []
    dropped = 0
    for family in families:
        item_id = str(family["item_id"])
        hashes = [digest(variant.get("tutor_turn")) for variant in family["variants"]]
        if any(text_hash in owner and owner[text_hash] != item_id for text_hash in hashes):
            dropped += 1
            continue
        kept.append(family)
        for text_hash in hashes:
            owner[text_hash] = item_id
    return kept, dropped


def deduplicate_pair_ids(families: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept: dict[str, dict[str, Any]] = {}
    dropped = 0
    for family in families:
        pair_id = str(family["pair_id"])
        if pair_id not in kept:
            kept[pair_id] = family
            continue
        first = kept[pair_id]
        first_texts = [normalise(variant["tutor_turn"]) for variant in first["variants"]]
        repeated_texts = [normalise(variant["tutor_turn"]) for variant in family["variants"]]
        if first_texts != repeated_texts:
            raise ValueError(f"pair_id collision with different variants: {pair_id}")
        dropped += 1
    return list(kept.values()), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("projects/pedagogy_mech_interp/data/pairs"))
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-families", type=int, default=500)
    parser.add_argument("--audit-per-stratum", type=int, default=2)
    parser.add_argument(
        "--confirmatory-domain",
        default="bridge",
        help="hold this domain out entirely; empty uses the registered 20%% hash split",
    )
    args = parser.parse_args()

    families = read_jsonl(args.sources)
    invalid = [(family.get("pair_id"), family_errors(family), family.get("_input")) for family in families]
    invalid = [row for row in invalid if row[1]]
    if invalid:
        preview = "\n".join(f"  {pair_id} ({origin}): {errors}" for pair_id, errors, origin in invalid[:20])
        raise SystemExit(f"{len(invalid)} invalid families:\n{preview}")
    families, duplicate_pair_ids_dropped = deduplicate_pair_ids(families)
    families, duplicate_families_dropped = drop_cross_item_text_duplicates(families)
    families = select_families(families, args.max_families, args.seed)

    rows = []
    for family in families:
        split = assign_split(
            str(family["item_id"]), args.seed, str(family.get("domain", "")), args.confirmatory_domain
        )
        rows.extend(flatten(family, split))
    assert_no_cross_split_leakage(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    for split, _ in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        if split == "confirmatory":
            key_fields = {"variant_id", "pair_id", "label", "expected_direction", "concept"}
            write_jsonl(args.out / "confirmatory_key.jsonl", ({k: row[k] for k in key_fields} for row in split_rows))
            public = [{k: v for k, v in row.items() if k not in {"label", "expected_direction"}} for row in split_rows]
            write_jsonl(args.out / "confirmatory.jsonl", public)
        else:
            write_jsonl(args.out / f"{split}.jsonl", split_rows)

    findings = verifier_findings(rows)
    rng = random.Random(args.seed)
    audit: list[dict[str, Any]] = list(findings)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    finding_ids = {finding["pair_id"] for finding in findings}
    for family in families:
        if family["pair_id"] not in finding_ids:
            strata[(family["concept"], family["source"])].append(family)
    for bucket in strata.values():
        rng.shuffle(bucket)
        audit.extend(bucket[: args.audit_per_stratum])
    (args.out / "audit_queue.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    counts = Counter((row["split"], row["concept"]) for row in rows if row["label"] == 1)
    metadata = {
        "schema": "pedagogy-mech-splits/v1",
        "seed": args.seed,
        "confirmatory_domain": args.confirmatory_domain or None,
        "source_files": [str(path) for path in args.sources],
        "families": len(families),
        "variants": len(rows),
        "duplicate_families_dropped": duplicate_families_dropped,
        "duplicate_pair_ids_dropped": duplicate_pair_ids_dropped,
        "split_families_by_concept": {
            f"{split}/{concept}": count for (split, concept), count in sorted(counts.items())
        },
        "audit_findings": len(findings),
        "label_provenance": dict(Counter(row.get("label_source", "unknown") for row in rows)),
        "confirmatory_interpretation": (
            "Only dataset_direct or human_verified constructs are confirmatory; "
            "heuristic_proxy constructs remain exploratory."
        ),
        "confirmatory_key": "confirmatory_key.jsonl (keep sealed until candidate freeze)",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
