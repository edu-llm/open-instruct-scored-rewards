"""The gates from PLAN.md phase 3 and 4, run against a generated corpus.

Three questions, in the order that decides whether anything downstream is worth doing.

1. SCENARIO FIDELITY. Did the student model occupy the state it was told to? Everything rests on
   this. The prescribed target level is the answer key for contingency, and it is only the right
   key if the student really was lost when the scenario said lost. A corpus that fails here is not
   partially useful; it is mislabelled at the root.

2. ASSISTANCE SPREAD. Do the styles actually reach all five rungs? If every tutor turn is a hint,
   contingency has no variance, and the spine of the reward is a constant. Nine styles exist to
   prevent this and this is where that is checked rather than assumed.

3. PREVALENCE. Is each metric present often enough to label? A move appearing in 3% of turns gives
   a kappa near zero from prevalence alone and gives a probe nothing to fit. `pedagogy_rm` learned
   this the expensive way, after labelling.

The judge here is a model, which is fine for all three: these are corpus-shaping decisions, not
labels. The one thing a model may never decide is the prescribed target itself - that comes from
the scenario, because tutors agree on the right next action in 18% of cases and a model-inferred
key would bake that coin-flip into everything.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random

from projects.tutor_metrics.metrics import CONTRIBUTIONS, RUNG_ANCHORS, RUNGS, SCENARIO_BY_KEY

FIDELITY_SYSTEM = """You are auditing simulated student messages for a research corpus. Answer \
only "yes" or "no". No explanation."""

RUNG_SYSTEM = """You classify a tutor's turn by how much control it takes, on a fixed ladder. \
Reply with exactly one of: {rungs}. No explanation.

{anchors}"""

PRESENCE_SYSTEM = """You check whether a specific feature is present in a tutor's turn. Answer \
only "yes" or "no". No explanation."""


async def ask(client, model: str, system: str, user: str) -> str:
    reply = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=8,
    )
    return (reply.choices[0].message.content or "").strip().lower()


async def fidelity(client, model: str, moment: dict) -> bool:
    scenario = SCENARIO_BY_KEY[moment["scenario"]]
    answer = await ask(
        client,
        model,
        FIDELITY_SYSTEM,
        f"Problem:\n{moment['question']}\n\nStudent message:\n{moment['student_text']}\n\n{scenario.check}",
    )
    return answer.startswith("y")


async def rung_of(client, model: str, moment: dict, turn: dict) -> str:
    anchors = "\n".join(f"{k}: {v}" for k, v in RUNG_ANCHORS.items())
    answer = await ask(
        client,
        model,
        RUNG_SYSTEM.format(rungs=", ".join(RUNGS), anchors=anchors),
        f"Problem:\n{moment['question']}\n\nStudent:\n{moment['student_text']}\n\nTutor:\n{turn['text']}",
    )
    for rung in RUNGS:
        if rung in answer:
            return rung
    return "unparsed"


async def present(client, model: str, moment: dict, turn: dict, metric) -> bool:
    answer = await ask(
        client,
        model,
        PRESENCE_SYSTEM,
        f"Problem:\n{moment['question']}\n\nStudent:\n{moment['student_text']}\n\n"
        f"Tutor turn:\n{turn['text']}\n\n{metric.question}",
    )
    return answer.startswith("y")


async def run(args) -> None:
    import openai  # noqa: PLC0415

    client = openai.AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    with open(args.moments) as handle:
        moments = [json.loads(line) for line in handle if line.strip()]
    sample = random.Random(0).sample(moments, min(args.sample, len(moments)))
    print(f"auditing {len(sample)} of {len(moments)} moments\n", flush=True)

    gate = asyncio.Semaphore(args.concurrency)

    async def guarded(coro):
        async with gate:
            return await coro

    # --- 1. scenario fidelity -------------------------------------------------
    checks = await asyncio.gather(*(guarded(fidelity(client, args.model, m)) for m in sample))
    by_scenario: dict[str, list[bool]] = collections.defaultdict(list)
    for moment, ok in zip(sample, checks):
        by_scenario[moment["scenario"]].append(ok)

    print("=" * 70)
    print("1. SCENARIO FIDELITY  (gate: >= 80% overall)")
    for key in sorted(by_scenario):
        hits = by_scenario[key]
        print(f"   {key:18s} {sum(hits) / len(hits):6.0%}  ({sum(hits)}/{len(hits)})")
    overall = sum(checks) / max(len(checks), 1)
    print(f"   {'OVERALL':18s} {overall:6.0%}")

    # --- 2. assistance spread -------------------------------------------------
    pairs = [(m, t) for m in sample for t in m["tutor_turns"]][: args.turns]
    rungs = await asyncio.gather(*(guarded(rung_of(client, args.model, m, t)) for m, t in pairs))
    by_style: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for (_, turn), rung in zip(pairs, rungs):
        by_style[turn["style"]][rung] += 1

    print("\n" + "=" * 70)
    print("2. ASSISTANCE SPREAD  (gate: every rung reached by some style)")
    print(f"   {'style':10s} " + " ".join(f"{r:>11s}" for r in RUNGS))
    for style in sorted(by_style):
        counts = by_style[style]
        total = sum(counts.values())
        print(f"   {style:10s} " + " ".join(f"{counts[r] / total:>10.0%} " for r in RUNGS))
    reached = {r for counter in by_style.values() for r, n in counter.items() if n}
    missing = [r for r in RUNGS if r not in reached]
    print(f"   rungs never reached: {missing or 'none'}")

    # --- 3. prevalence --------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. PREVALENCE  (gate: >= 10% each, or cut/oversample before labelling)")
    subset = pairs[: args.prevalence]
    for metric in CONTRIBUTIONS:
        hits = await asyncio.gather(*(guarded(present(client, args.model, m, t, metric)) for m, t in subset))
        rate = sum(hits) / max(len(hits), 1)
        print(f"   {metric.key:28s} {rate:6.0%}  {'OK' if rate >= 0.10 else '<-- too rare'}")

    print("\n" + "=" * 70)
    verdict = []
    if overall < 0.80:
        verdict.append(f"scenario fidelity {overall:.0%} is below the 80% gate")
    if missing:
        verdict.append(f"rungs never reached: {missing}")
    print("FAILED: " + "; ".join(verdict) if verdict else "all gates passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moments", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--sample", type=int, default=300, help="moments to check fidelity on")
    parser.add_argument("--turns", type=int, default=400, help="tutor turns to classify by rung")
    parser.add_argument("--prevalence", type=int, default=200, help="turns per metric for prevalence")
    parser.add_argument("--concurrency", type=int, default=48)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
