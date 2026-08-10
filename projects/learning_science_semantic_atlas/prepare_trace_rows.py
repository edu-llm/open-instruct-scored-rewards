"""Join withheld corpus artifacts into the exact rows consumed by ``trace``.

The corpus builder keeps prompts, generations, and labels apart on purpose.
Tracing needs all three, but joining them by line number would silently attach
the wrong response after a resumed generation.  This module joins by ``unit_id``
and requires the prompt digest in all available inputs to agree.

Only accepted generations are eligible.  Generation files are append-only, so
the last *accepted* row for an item wins; a later refused retry does not erase a
previous usable response.  System prompt, user prompt, and response are copied
without stripping or normalising so the resulting JSONL defines the exact replay
text.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "semantic-atlas/trace-rows-v1"
PROMPT_SCHEMA = "semantic-atlas/generation-prompt-v1"
GENERATION_SCHEMA = "semantic-atlas/generation-v1"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON objects with errors that name the source line."""
    target = Path(path)
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{target}:{number}: expected a JSON object, got {type(row).__name__}")
            rows.append(row)
    return rows


def _unit_id(row: Mapping[str, Any], *, where: str) -> str:
    value = row.get("unit_id", row.get("record_id"))
    if value in (None, ""):
        raise ValueError(f"{where}: row needs unit_id")
    return str(value)


def _index_unique(rows: Iterable[Mapping[str, Any]], *, source: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        unit_id = _unit_id(row, where=f"{source}:{number}")
        if unit_id in indexed:
            raise ValueError(f"{source}: duplicate unit_id {unit_id!r}")
        indexed[unit_id] = row
    return indexed


def generation_is_accepted(row: Mapping[str, Any]) -> bool:
    """Whether a raw generation may be replayed."""
    text = row.get("text", row.get("response"))
    reasons = row.get("refusal_reasons")
    refused = bool(reasons.strip()) if isinstance(reasons, str) else bool(reasons)
    return isinstance(text, str) and bool(text.strip()) and not refused


def newest_accepted_generations(
    rows: Iterable[Mapping[str, Any]], *, source: str = "generations"
) -> dict[str, Mapping[str, Any]]:
    """Return the last accepted row for each unit in append order."""
    accepted: dict[str, Mapping[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        unit_id = _unit_id(row, where=f"{source}:{number}")
        if generation_is_accepted(row):
            accepted[unit_id] = row
    return accepted


def _require_text(row: Mapping[str, Any], field: str, *, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {field} must be non-empty text")
    return value


def _binary(value: Any, *, field: str, unit_id: str) -> int:
    if not isinstance(value, bool):
        raise ValueError(f"{unit_id}: key field {field} must be a JSON boolean, got {value!r}")
    return int(value)


def _matching_prompt_sha(
    unit_id: str,
    prompt: Mapping[str, Any],
    generation: Mapping[str, Any],
    key: Mapping[str, Any],
) -> str:
    values = {
        str(value)
        for value in (prompt.get("prompt_sha"), generation.get("prompt_sha"), key.get("prompt_sha"))
        if value not in (None, "")
    }
    if not values:
        raise ValueError(f"{unit_id}: no prompt_sha in prompt, generation, or key")
    if len(values) != 1:
        raise ValueError(f"{unit_id}: stale join; prompt_sha values disagree: {sorted(values)}")
    return values.pop()


def _provenance(
    generation: Mapping[str, Any],
    key: Mapping[str, Any],
    *,
    model_tag: str,
    prompt_sha: str,
) -> dict[str, Any]:
    names = (
        "source_model",
        "source_policy",
        "temperature",
        "top_p",
        "seed",
        "attempt",
        "block",
        "pool",
    )
    values: dict[str, Any] = {"prompt_sha": prompt_sha, "model_tag": model_tag}
    for name in names:
        if name in generation:
            values[name] = generation[name]
        elif name in key:
            values[name] = key[name]
    return values


def prepare_trace_rows(
    prompts: Sequence[Mapping[str, Any]],
    generations: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    *,
    model_tag: str,
    prompt_source: str = "prompts",
    generation_source: str = "generations",
    key_source: str = "key",
) -> list[dict[str, Any]]:
    """Join prompt, newest accepted generation, and key rows by unit id.

    Key order defines output order.  This lets a round-1 key select its 160 rows
    from prompt and generation files that also contain calibration items.
    """
    if not model_tag.strip():
        raise ValueError("model_tag must be explicit and non-empty")
    prompt_by_id = _index_unique(prompts, source=prompt_source)
    generation_by_id = newest_accepted_generations(generations, source=generation_source)
    _index_unique(keys, source=key_source)

    output: list[dict[str, Any]] = []
    missing_prompts: list[str] = []
    missing_generations: list[str] = []
    for position, key in enumerate(keys, start=1):
        unit_id = _unit_id(key, where=f"{key_source}:{position}")
        prompt = prompt_by_id.get(unit_id)
        generation = generation_by_id.get(unit_id)
        if prompt is None:
            missing_prompts.append(unit_id)
            continue
        if generation is None:
            missing_generations.append(unit_id)
            continue

        prompt_sha = _matching_prompt_sha(unit_id, prompt, generation, key)
        system = _require_text(prompt, "system", where=f"{prompt_source}[{unit_id}]")
        user = _require_text(prompt, "user", where=f"{prompt_source}[{unit_id}]")
        response = generation.get("text", generation.get("response"))
        assert isinstance(response, str)  # established by generation_is_accepted

        item_id = str(key.get("item_id") or "")
        if not item_id:
            raise ValueError(f"{unit_id}: key row needs item_id")
        concept = str(key.get("designed_target_construct", key.get("concept", "")))
        if not concept:
            raise ValueError(f"{unit_id}: key row needs designed_target_construct")
        domain = str(key.get("domain") or "")
        if not domain:
            raise ValueError(f"{unit_id}: key row needs domain")

        label = _binary(key.get("expected_presence"), field="expected_presence", unit_id=unit_id)
        named_label = _binary(key.get("names"), field="names", unit_id=unit_id)
        enacted_label = _binary(key.get("enacts"), field="enacts", unit_id=unit_id)
        provenance = _provenance(generation, key, model_tag=model_tag, prompt_sha=prompt_sha)
        prepared = {
            "schema": SCHEMA,
            "record_id": unit_id,
            "item_id": item_id,
            "sibling_id": str(key.get("sibling_id") or item_id),
            "concept": concept,
            "domain": domain,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "system": system,
            "user": user,
            "response": response,
            "label": label,
            "designed_label": label,
            "named_label": named_label,
            "enacted_label": enacted_label,
            "label_source": str(key.get("label_source") or "designed"),
            "model_tag": model_tag,
            "source": str(provenance.get("source_model") or model_tag),
            "template_id": prompt_sha,
            "provenance": provenance,
        }
        # ``trace`` treats an absent quality as NaN.  JSON null would instead
        # reach ``float(None)`` and fail, so only materialized scores are copied.
        if key.get("quality") is not None:
            prepared["quality"] = key["quality"]
        output.append(prepared)

    problems = []
    if missing_prompts:
        problems.append(f"{len(missing_prompts)} key rows lack prompts (first {missing_prompts[:3]})")
    if missing_generations:
        problems.append(
            f"{len(missing_generations)} key rows lack accepted generations (first {missing_generations[:3]})"
        )
    if problems:
        raise ValueError("; ".join(problems))
    return output


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True, help="withheld/prompts.jsonl")
    parser.add_argument("--generations", type=Path, required=True, help="append-only withheld/generations.jsonl")
    parser.add_argument("--key", type=Path, required=True, help="withheld key selecting the rows to trace")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-tag", required=True, help="explicit filesystem/report tag, such as olmoe-base")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = prepare_trace_rows(
        read_jsonl(args.prompts),
        read_jsonl(args.generations),
        read_jsonl(args.key),
        model_tag=args.model_tag,
        prompt_source=str(args.prompts),
        generation_source=str(args.generations),
        key_source=str(args.key),
    )
    summary = {
        "schema": SCHEMA,
        "records": len(rows),
        "model_tag": args.model_tag,
        "output": str(args.out),
        "dry_run": args.dry_run,
        "label_sources": sorted({str(row["label_source"]) for row in rows}),
        "source_models": sorted({str(row["source"]) for row in rows}),
    }
    if not args.dry_run:
        write_jsonl(rows, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
