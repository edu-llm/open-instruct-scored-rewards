"""Generate scenario-conditioned tutoring moments.

A MOMENT is one problem, one student message written to occupy a prescribed scenario, and one
tutor turn replying to it. The tutor turn is the labelable unit; the scenario carries the target
help level, which is the answer key for contingency.

WHY MOMENTS RATHER THAN FREE DIALOGUES. `pedagogy_rm` ran dialogues and took whatever student
states emerged. They were not evenly spread, nothing recorded which state a turn was answering,
and the interesting cell - a tutor over-helping a student who had already reasoned it out - was
rare precisely because a competent tutor model avoids it. Prescribing the student state fixes all
three: the mix is ours to choose, the target is known before the tutor speaks, and mismatches can
be manufactured by pairing a style with a scenario it suits badly.

WHY THE MISMATCHES ARE NOT SYNTHETIC IN THE WAY THAT BURNED V2. `pedagogy_rm`'s manufactured
negatives were turns corrupted on purpose, and a probe could separate them by style rather than by
pedagogy - the reported -0.37 length coupling was an artefact of the padding and became -0.62 on
real turns. Here a mismatch is a perfectly good turn placed against the wrong student. Nothing
about its surface marks it. A probe has to read the RELATION between turn and student state, which
is the thing we actually want measured.

Usage, against a vLLM server holding Qwen3-30B-A3B:

    python -m projects.tutor_metrics.generate \\
        --items data/items.jsonl --out data/tutor_metrics/moments.jsonl \\
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-30B-A3B-Instruct-2507
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random

from projects.tutor_metrics.metrics import RUNG_ANCHORS, SCENARIOS, Scenario

STUDENT_MAX_TOKENS = 220
TUTOR_MAX_TOKENS = 320
# Long enough that a tell_step turn is never cut off. `pedagogy_rm` capped tutor turns at 160 and
# 11% ended mid-word, which is unlabelable: rating a turn the cap truncated rates the cap.

STUDENT_SYSTEM = """You are simulating a school student working on a problem, for research into \
tutoring. You will be told exactly what state to be in. Occupy that state precisely - this is the \
whole point of the simulation, and a message that drifts out of the assigned state is discarded.

Write as a real student would: 1-3 sentences, plain language, no headings, no meta-commentary. \
Never mention that you are simulating anything. Never apologise. Do not be more articulate than \
the assigned state allows."""

STUDENT_TASK = """Problem:
{question}

The correct solution, for your reference only - the student you are playing does NOT have this:
{solution}

The state you must occupy:
{state}

Write only the student's message."""

TUTOR_SYSTEM = """You are tutoring one student on one problem, by text. Reply with your next turn \
only: no headings, no lists, no sign-off, no meta-commentary. Speak directly to the student."""

# The styles exist to spread `assistance_level` across all five rungs. Without this spread there
# is no contingency variance and the spine of the reward is dead - which is why a style sweep is
# a gate in PLAN.md rather than a nicety. They are written as instructions about HOW MUCH to help,
# not about quality, so that a style is never simply "be a bad tutor".
STYLES: dict[str, str] = {
    "pump": "Say as little as possible. Invite the student to keep going without adding any content.",
    "hint": "Point at where to look or what to consider. Do not name the operation to perform.",
    "prompt": "Name the specific thing the student should do or account for, and leave them to do it.",
    "step": "Work exactly one step of the solution, showing your working, then stop.",
    "tell": "Give the student the answer and a brief justification.",
    "lecture": "Explain the underlying principle thoroughly before saying anything about their attempt.",
    "socratic": "Reply only with questions.",
    "warm": "Be encouraging and supportive first, then help.",
    "terse": "Be extremely brief. One short sentence.",
}


def problem_text(item: dict) -> str:
    """Render the complete problem, including choices needed by multiple-choice stems."""
    choices = item.get("choices") or []
    if not choices:
        return item["question"]
    options = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    return f"{item['question']}\n\nOptions:\n{options}"


def student_messages(item: dict, scenario: Scenario) -> list[dict]:
    solution = item.get("solution") or item.get("explanation") or f"The answer is {item.get('answer', '?')}."
    return [
        {"role": "system", "content": STUDENT_SYSTEM},
        {
            "role": "user",
            "content": STUDENT_TASK.format(
                question=problem_text(item), solution=solution, state=scenario.student_state
            ),
        },
    ]


def tutor_messages(item: dict, student_text: str, style: str) -> list[dict]:
    return [
        {"role": "system", "content": TUTOR_SYSTEM + "\n\n" + STYLES[style]},
        {"role": "user", "content": f"Problem:\n{problem_text(item)}\n\nStudent:\n{student_text}"},
    ]


def tutor_messages_neutral(item: dict, student_text: str) -> list[dict]:
    """Deployment context with no generation-style answer leaked to the head."""
    return [
        {"role": "system", "content": TUTOR_SYSTEM},
        {"role": "user", "content": f"Problem:\n{problem_text(item)}\n\nStudent:\n{student_text}"},
    ]


async def say(client, model: str, messages: list[dict], temperature: float, max_tokens: int) -> tuple[str, bool]:
    """One completion, plus whether the cap cut it off. Truncated turns are flagged, not dropped
    here, so the label-set builder can decide - a rater scoring a truncated turn scores the cap."""
    reply = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    choice = reply.choices[0]
    return (choice.message.content or "").strip(), choice.finish_reason == "length"


async def moment(client, model: str, item: dict, scenario: Scenario, styles: list[str], temperature: float) -> dict:
    """One student message in a prescribed state, and one tutor turn per style answering it.

    The styles share a student message on purpose. It makes the tutor turns directly comparable -
    same problem, same student, same prescribed target - so a rater or a probe sees only the
    tutor's choice varying, and a paired comparison is available for free.
    """
    student_text, _ = await say(client, model, student_messages(item, scenario), 0.9, STUDENT_MAX_TOKENS)

    turns = []
    for style in styles:
        text, truncated = await say(
            client, model, tutor_messages(item, student_text, style), temperature, TUTOR_MAX_TOKENS
        )
        turns.append({"style": style, "text": text, "truncated": truncated})

    return {
        "item_id": item.get("id") or item.get("item_no"),
        "question": item["question"],
        "choices": item.get("choices"),
        "gold_idx": item.get("gold_idx"),
        "solution": item.get("solution") or item.get("explanation"),
        "answer": item.get("answer"),
        "subject": item.get("subject"),
        "scenario": scenario.key,
        # The answer key, written down before any tutor turn existed. Everything downstream
        # depends on this not having been inferred from the text.
        "target_level": scenario.target,
        "scenario_check": scenario.check,
        "student_text": student_text,
        "model": model,
        "tutor_turns": turns,
    }


async def run(args) -> None:
    import openai  # noqa: PLC0415

    client = openai.AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    with open(args.items) as handle:
        items = [json.loads(line) for line in handle if line.strip()][: args.limit]
    scenarios = [s for s in SCENARIOS if not args.scenarios or s.key in args.scenarios.split(",")]
    styles = [] if args.student_only else args.styles.split(",")
    unknown = set(styles) - set(STYLES)
    if unknown:
        raise SystemExit(f"unknown styles {sorted(unknown)}; have {sorted(STYLES)}")

    jobs = [(item, scenario) for item in items for scenario in scenarios]
    random.Random(args.seed).shuffle(jobs)

    # RESUME, BECAUSE THIS RUNS ON A PREEMPTABLE PARTITION. Slurm requeues the script from the
    # top, so a job killed at 90% would otherwise regenerate everything and be killed again at
    # 90%. Reading back what is already on disk makes each attempt strictly additive, and the
    # key is (item, scenario) because that pair is what a moment is.
    already: set[tuple[str, str]] = set()
    if os.path.exists(args.out):
        with open(args.out) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    already.add((str(row["item_id"]), row["scenario"]))
        jobs = [(i, s) for i, s in jobs if (str(i.get("id") or i.get("item_no")), s.key) not in already]
        print(f"resuming: {len(already)} moments already written", flush=True)

    print(f"{len(jobs)} moments x {len(styles)} styles = {len(jobs) * len(styles)} tutor turns", flush=True)

    gate = asyncio.Semaphore(args.concurrency)
    writes = asyncio.Lock()
    done = 0
    # Line-buffered append, flushed per moment: a preemption costs at most the moments in flight
    # rather than the whole run.
    handle = open(args.out, "a", buffering=1)  # noqa: SIM115 - closed in the finally below

    async def one(item: dict, scenario: Scenario):
        nonlocal done
        async with gate:
            try:
                out = await moment(client, args.model, item, scenario, styles, args.temperature)
            except Exception as exc:  # noqa: BLE001 - one bad item must not end a long run
                print(f"  failed {item.get('id')} / {scenario.key}: {type(exc).__name__}: {exc}", flush=True)
                return None
        async with writes:
            handle.write(json.dumps(out) + "\n")
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
        return out

    try:
        results = await asyncio.gather(*(one(item, scenario) for item, scenario in jobs))
    finally:
        handle.close()

    kept = sum(1 for r in results if r is not None)
    print(f"wrote {kept} new moments ({kept * len(styles)} tutor turns) to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True, help="jsonl with question, and ideally solution/answer")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--styles", default=",".join(STYLES))
    parser.add_argument(
        "--student-only",
        action="store_true",
        help="generate prescribed student states without unused tutor turns (for online RL prompts)",
    )
    parser.add_argument("--scenarios", default="", help="comma-separated subset; default all")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=10**9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"rungs: {list(RUNG_ANCHORS)}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
