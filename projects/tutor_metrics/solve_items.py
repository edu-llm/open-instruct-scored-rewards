"""Write a worked reference solution for each item.

The item bank carries a question, the choices and `gold_idx`, but no working. Two things need
the working and neither can proceed without it.

THE SCENARIOS NEED IT. `slip` asks the student model to produce "the correct method with one
arithmetic error", which it cannot do without knowing the correct method. Left to invent one it
produces a plausible-looking method that is wrong in some other way, and the scenario's prescribed
target - the answer key for the whole reward - is then attached to a student state that does not
exist.

THE RATERS NEED IT. `reference_conflict` is defined as a contradiction with "the reference
solution shown to you", and the guide is explicit that a rater must check against it rather than
re-derive it. A rater re-deriving grade-school arithmetic under time pressure is a rater making
arithmetic errors, and those become label noise attributed to the tutor.

The gold answer is known, so this is generating the WORKING for a known answer rather than solving
anything. Solutions whose stated answer disagrees with `gold_idx` are dropped: the model got lost,
and a wrong reference is worse than none because it silently inverts `reference_conflict`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re

SYSTEM = """You write short worked solutions for school test questions, for use as a reference \
key by human raters. Be correct and complete but brief.

Format, exactly:
- 1 to 4 numbered steps, each one line, showing the actual working.
- Then a final line: "Answer: <the answer>"

No preamble, no restatement of the question, no teaching, no alternative methods."""

TASK = """Question: {question}

Options:
{options}

The correct option is {letter}. {answer}

Write the worked solution that reaches it."""


def options_block(item: dict) -> str:
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(item.get("choices") or []))


def stated_answer(text: str) -> str | None:
    match = re.search(r"Answer:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def agrees(said: str | None, gold: str) -> bool:
    """Is the stated answer the gold one?

    Compared on digits and letters only, because the model writes "275, 234, 163, 147" where the
    option says "275, 234, 163, 147." and a punctuation difference is not a disagreement. Anything
    subtler than that is treated as a disagreement and the item is dropped - a reference solution
    is only useful if it is trustworthy, so the bar for keeping one is high.
    """
    if not said:
        return False
    clean = lambda s: re.sub(r"[^0-9a-z]", "", s.lower())  # noqa: E731
    return clean(said) == clean(gold) or clean(gold) in clean(said)


async def solve(client, model: str, item: dict) -> dict | None:
    choices = item.get("choices") or []
    gold_idx = item.get("gold_idx")
    if not choices or gold_idx is None:
        return None
    gold = choices[gold_idx]

    reply = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": TASK.format(
                    question=item["question"], options=options_block(item), letter=chr(65 + gold_idx), answer=gold
                ),
            },
        ],
        temperature=0.2,
        max_tokens=400,
    )
    text = (reply.choices[0].message.content or "").strip()
    if not agrees(stated_answer(text), gold):
        return None

    return {
        **item,
        "id": item.get("id") or f"{item.get('booklet', 'item')}_{item.get('item_no')}",
        "answer": gold,
        "solution": text,
    }


async def run(args) -> None:
    import openai  # noqa: PLC0415

    client = openai.AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    with open(args.items) as handle:
        items = [json.loads(line) for line in handle if line.strip()][: args.limit]
    print(f"solving {len(items)} items", flush=True)

    gate = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(item: dict):
        nonlocal done
        async with gate:
            try:
                out = await solve(client, args.model, item)
            except Exception as exc:  # noqa: BLE001
                print(f"  failed {item.get('item_no')}: {type(exc).__name__}: {exc}", flush=True)
                return None
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(items)}", flush=True)
            return out

    results = await asyncio.gather(*(one(i) for i in items))
    kept = [r for r in results if r is not None]
    with open(args.out, "w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")

    dropped = len(items) - len(kept)
    print(f"wrote {len(kept)} items with references to {args.out}")
    print(f"dropped {dropped} ({dropped / max(len(items), 1):.0%}) whose solution disagreed with gold")
    if kept and dropped / len(items) > 0.3:
        raise SystemExit("over 30% disagreed with gold; check the prompt or the item bank before continuing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=10**9)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
