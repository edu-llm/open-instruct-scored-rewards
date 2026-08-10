# tutor_metrics

A tutoring reward whose target **moves with the student**, the corpus to fit it on, and a public
app for labelling that corpus.

Read [`PLAN.md`](PLAN.md) for the design and the reasoning. This file is the map and the run order.

## The one idea

Every reward this project has tried was maximised by a short template, because every one of them
scored a turn on its own qualities. `pedagogy_rm`'s withholding-shaped dimensions had an effective
rank of 2.9 and a twelve-word joint optimum, and a trained policy found it. The successor design
that summed "the best K of N good qualities" was worse on inspection: its maximum was a **56-way
tie** whose cheapest member was a fifteen-word template.

So the spine here is not a quality score. It is a **match** between the help the tutor gave and
the help the scenario prescribed. Because we script the student, the right help level is known
before the tutor speaks — and because it changes with the student, no fixed turn can win. The
arithmetic is in [`check_reward.py`](check_reward.py): the best fixed rung scores **0.58 of 1.00**
across the scenario mix, so there is 0.42 per turn that only adapting can earn.

## Files

| | |
|---|---|
| `PLAN.md` | design, evidence, gates, and what could still go wrong |
| `metrics.py` | the metrics, the six scenarios and their prescribed targets, and `reward()` |
| `check_reward.py` | the six tests the previous design failed; run it before changing weights |
| `solve_items.py` | worked reference solutions, needed by both the scenarios and the raters |
| `generate.py` | scenario-conditioned moments: one student state, nine tutor styles |
| `audit.py` | the phase 3 and 4 gates — scenario fidelity, rung spread, prevalence |
| `app/` | the Next.js labelling app ([its own README](app/README.md)) |
| `LABELING_APP_SPEC.md` | the app's technical specification |
| `_redteam_reward_math.py` | the arithmetic that killed the first design, kept as the receipt |

## Run order

```bash
# 1. Model into the shared cache. CPU partition; the only job allowed to reach the network.
sbatch projects/tutor_metrics/scripts/fetch_model.sbatch

# 2. Serve it, solve the items, generate moments, and audit them. One H200, ~6h.
sbatch projects/tutor_metrics/scripts/generate.sbatch

# 3. Check the reward still behaves before wiring it to anything.
uv run python projects/tutor_metrics/check_reward.py
```

Stage 3 of the generation job is a **gate, not a report**. If scenario fidelity is below 80% the
student model did not occupy the state it was told to, the prescribed target is attached to the
wrong student, and nothing downstream measures what it claims to. Fix that before labelling.

## Two rules that are not negotiable

**Never infer a target level from a turn.** Tutors pick the same next action in 18% of cases, and
a tutor agrees with outside raters about their own move at κ 0.34. The target comes from the
scenario, which is why `Scenario.target` is set in `metrics.py` and never computed. With a
prescribed target the same judgement reaches κ 0.55–0.70.

**Never put a `not_eligible_*` metric in the reward.** `ontology.yaml` carries three columns per
construct — decodability, labelability and reward eligibility — and the first draft of this plan
read only the first, which put two ineligible constructs in its reward and a third in a gate.
`answer_withheld` and `icap_class` are collected and reported. They are not optimised.
