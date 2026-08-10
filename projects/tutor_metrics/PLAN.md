# Scenario-prescribed tutoring metrics: a rubric, a large corpus, and a public labelling app

Successor to `projects/pedagogy_rm/`, which trained a 5-dimension probe on 600 turns labelled by
one person. It moved a policy, and its own `FLAWS.md` says why the result is narrow: the
dimensions were near-duplicates, almost all of them scored a turn better simply for being
*shorter*, and the labels were one rater's judgement of one model's output.

**This plan is the second draft.** The first proposed twenty metrics and a "best 1 of A + best 2
of B + best 1 of C" reward. Red-teaming killed that reward, and the arithmetic is preserved in
`_redteam_reward_math.py`. What follows is built around what survived.

---

## 1. Why the obvious design fails

The intuition behind top-K is right: **good tutoring moves are substitutes, not addends.** Chi et
al. 2001 found tutors *suppressed* from explaining produced gains as large as unrestricted tutors.
Summing "explained well" and "elicited well" double-counts a choice the tutor had to make.

The implementation was wrong, in four ways worth recording so they are not reinvented.

**The maximum was a 56-way tie, and its cheapest member was a 15-word template.** Enumerating all
profiles that achieve the maximum score gives 1,767, of which 575 contain no antagonistic pair and
56 are minimal. The cheapest is `asks_next_step + diagnoses_the_error + locates_student_object +
verdict_with_referent` — about fifteen words. V1's degenerate optimum was a twelve-word question,
and this reward would have found a fifteen-word one. When a reward cannot distinguish 56 turns,
what the policy converges on is decided by production cost, which the reward does not model.

**A max over noisy probes rewards noise.** At a probe RMSE of 0.4 on a 1–3 scale, taking the best
1 of 5 adds **+0.47** and the best 2 of 6 adds **+0.76** of pure upward bias when every underlying
value is identical. Simulating a good turn against a bad one, 23% of the true gap disappears into
that bias. Capped top-K therefore scores *worse* than a plain sum at realistic probe accuracy. It
also has a temperature nobody chose: max is a soft-max whose sharpness is set by probe noise, so
improving the probe silently changes the objective.

**Multiplicative gates amplify length coupling instead of confining it.** This was the first
draft's central claim and it was backwards. Each gate is an absence, and absences get harder to
satisfy as a turn grows, so the probability all three pass falls with length. Quality rises 47%
from 15 to 100 words while expected reward **peaks at 30** and then declines. The gates rebuilt the
brevity pressure the design existed to remove.

**Gates also destroy the learning signal.** Three gates at 5% each zero 14.3% of samples. In a
GRPO group of 8, a single zeroed member triples the group standard deviation and halves the mean
advantage among survivors. Only 29% of groups come through clean.

Beyond the reward, the first draft cited `ontology.yaml` for decodability while ignoring the two
other columns it carries per construct. Checking them:

| construct | status in the atlas | first draft used it as |
|---|---|---|
| `answer_withholding_level` | **not_eligible_degenerate** | a gate |
| `icap_engagement_class` | **not_eligible_unreliable**, labelability *low* | warrant for `asks_next_step` |
| `problem_solving_before_instruction` | **measurement_only**, and does not accept `tutor_turn` | warrant for `asks_prediction` |

The first is the sharpest, because the atlas cites *this repository* against us: "A reward that
pays for withholding is maximized by saying nothing, and this repository has the receipts:
dimensions built on withholding had an effective rank of 2.9, their joint optimum was a 12-word
question, and a trained policy found it."

---

## 2. What the atlas says to do instead

One construct is marked as the exception, and the reason generalises:

> `contingent_help_calibration` — "The only construct here whose optimum a fixed policy cannot
> reach by style alone, **because the target moves with the learner.** That makes it the most
> valuable and the hardest to fake."

Contingency is scored as a **match against a prescribed target level**, not as a preference for
less help. A learner with no foothold needs more explicit support than one who has shown the
reasoning; maximal withholding after "I have no idea where to start" is the specific inversion the
construct exists to catch.

There is one hard condition, and it is what the rest of this plan is organised around:

> "Prescribed targets must come from the dataset. A proxy target inferred by a model reintroduces
> the 18%-agreement problem with a machine-made answer key."

Tutors agree on the right next action in **18%** of cases, and a tutor agrees with outside raters
about their own move at κ 0.34. So "what should the tutor have done" cannot be asked of raters or
of a judge model. With a prescribed target it collapses to two codes and an equality check, and
κ rises to **0.55–0.70**.

**We can satisfy that condition, because we build the student.** If the student's state is scripted
rather than emergent — "has produced nothing and says they are lost", "has a correct method with an
arithmetic slip", "has a confident wrong method" — then the appropriate help level is fixed by the
scenario definition, before any tutor turn exists. The answer key is a property of the generator,
not an inference about the output.

That is the whole design. **Scenario-conditioned generation is what turns the most valuable and
least fakeable construct from unusable into the spine of the reward.**

---

## 3. The metrics

### 3.1 The spine: contingency

| field | what it is |
|---|---|
| `target_level` | prescribed by the scenario, never rated: one of `pump`, `hint`, `prompt`, `tell_step`, `tell_answer` |
| `assistance_level` | rated: which single rung the turn actually occupies |
| `contingency` | derived: how far apart they are |

`assistance_level` is **one categorical, not five metrics**. Splitting Graesser's ladder into five
binary heads manufactures five mutually-anticorrelated dimensions from one variable and teaches the
probe the same thing five times — the `actionable`/`elicits` r = 0.96 mistake with the sign flipped.

The first draft's Group A ("five `asks_*` metrics, best 1 counts") was that same mistake. A turn
makes one demand; *which* demand is a categorical, so it becomes:

| field | values |
|---|---|
| `demand_type` | `none`, `self_explanation`, `next_step`, `recall`, `self_check`, `prediction` |
| `demand_is_specific` | does the demand name a particular quantity, step or claim, or is it "why?" / "try again"? |

`demand_is_specific` is the part that carried the measured agreement gain: rewriting "how much
thinking does this ask for" (κ 0.09–0.34) as "does it hand over a *named* next move" reached
0.55–0.66.

### 3.2 Contributions — what the tutor supplies (capped count, not max)

Multi-label presence, each binary, **scored as `min(count, 2)`**. A capped count, not an argmax
over noisy continuous scores, so there is no selection bias to inflate and no soft-max temperature
hiding in the probe's error.

| key | Question | atlas | eligibility | len |
|---|---|---|---|---|
| `diagnoses_the_error` | Does it say *what is wrong*, not merely that something is? | `elaborated_feedback_specificity` | eligible_with_guard | + |
| `applies_principle_here` | Does it name the rule **and** how it applies here? | `elaborated_feedback_specificity` | eligible_with_guard | + |
| `models_one_step` | Does it work one step out in full? | `worked_example_provision` | eligible_with_guard | + |
| `offers_parallel_case` | Does it give a second case **and** say what to compare? | `analogical_case_comparison` | eligible_with_guard | + |
| `names_the_target` | Does it say what the student is trying to establish right now? | `subgoal_labeled_structure` | eligible_with_guard | + |
| `asks_recall_of_established` | Does it ask the student to state from memory something already established? | `retrieval_practice_opportunity` | eligible_with_guard | 0 |

Every one is `eligible_with_guard` in the atlas — none is `not_eligible`. Evidence: elaborated
feedback d = 0.49 against 0.32 for correct-answer and 0.05 for bare right/wrong, largest in
mathematics (Van der Kleij 2015); worked examples g = 0.48 over 55 studies (Barbieri 2023); case
comparison d = 0.50 over 57 experiments (Alfieri 2013); retrieval g = 0.50 (Rowland 2014).

### 3.3 Anchoring and validity

| key | Question | role | len |
|---|---|---|---|
| `locates_student_object` | Does it point at a specific number, step or word the student wrote? | scored | 0 |
| `verdict_with_referent` | Does it say whether a **named** step is right or wrong? | scored | + |
| `no_reference_conflict` | Does anything it asserts conflict with the reference solution? | **penalty** | − |
| `follows_from_dialogue` | Does it contradict something already settled? | **penalty** | − |
| `single_focus` | Is there exactly one thing to attend to? | **penalty** | − |
| `no_empty_praise` | Does it praise without naming anything specific? | **penalty** | − |

Penalties are **subtractive, never multiplicative**. That is the direct fix for both gate problems:
nothing is zeroed, so no GRPO group loses its spread, and the length pressure from absence-shaped
judgements enters linearly with a weight we chose rather than compounding.

### 3.4 Measured, never optimised

| key | why not in the reward |
|---|---|
| `answer_withheld` | atlas `not_eligible_degenerate` — the clearest degenerate optimum there is |
| `icap_class` | atlas `not_eligible_unreliable`, labelability *low*; report four counts per arm, never average |
| `demand_type = prediction` | `problem_solving_before_instruction` is measurement-only and its sign reverses by age and domain |
| `student_state` | a covariate; needed to check contingency was scored against the right scenario |
| `step_size` | non-ordinal (2 is good, 3 is not), like `length_fit` |
| fading, repair completion, impasse | trajectory properties — one turn has no slope |

---

## 4. The reward

```
reward = w_c · contingency(assistance_level, target_level)     # 1.0 exact, 0.5 adjacent, 0 else
       + w_q · min(contributions_present, 2) / 2
       + w_a · locates_student_object
       + w_v · verdict_with_referent
       + w_d · demand_is_specific
       + w_l · length_band                                     # explicit, as in pedagogy_rm
       − p₁ · reference_conflict
       − p₂ · dialogue_conflict
       − p₃ · not_single_focus
       − p₄ · empty_praise
```

**Why this has no fixed degenerate optimum.** `target_level` varies by scenario, so no single
template maximises the spine across the corpus. A turn that maxes contingency for a lost student
(`tell_step`) scores zero on contingency for a student who has already shown the reasoning
(`pump`). This is precisely the property the atlas identifies as unfakeable, and it is absent from
every reward this project has tried before.

**Why it is length-neutral by construction rather than by hope.** The spine is a match to a target
that is *sometimes* long and *sometimes* short. Explaining a step properly takes words; a pump does
not. Averaged over a scenario mix, there is no monotone length gradient to descend, and the length
band remains the only deliberate length term.

**Why nothing is multiplicative.** Every term is additive, so a sample is never zeroed and the
group always retains spread.

The weights are configurable and will be set from measured variance, not guessed, so that no single
term dominates the way the length term did in `pedagogy_rm`.

---

## 5. The corpus

Target **~40,000 scored turns** from ~8,000 dialogues. Generation is cheap and labelling is not, so
the corpus exists to be sampled from well.

- **Model**: `Qwen/Qwen3-30B-A3B-Instruct-2507` — MoE, 30B total and 3B active, so it serves fast.
  Not yet cached; ~60GB. One H200 holds it, and `mit_normal_gpu` has 8-GPU H200 nodes.
- **Items**: the existing standardised bank, plus the physics banks from `student_state_tutor`.

### Scenario conditioning is now mandatory, not a nicety

Each dialogue is generated from a **scenario**: a scripted student state with a prescribed target
help level. The student model is instructed to occupy that state, and the scenario records the
target. A worked first set:

| scenario | student state | prescribed target |
|---|---|---|
| `lost` | no attempt, says they do not know where to start | `tell_step` |
| `slip` | correct method, arithmetic error in one step | `hint` |
| `confident_wrong` | complete method that is wrong in principle | `prompt` |
| `partial` | correct first step, stalled | `hint` |
| `reasoned_attempt` | correct method and reasoning shown, minor gap | `pump` |
| `asks_outright` | requests the answer directly | `prompt` |

The target column is the answer key, and it exists before any tutor turn is generated. **A
generation is discarded if the student model fails to occupy its scenario** — checked by rating
`student_state` and requiring it to match. Otherwise the key is wrong and the whole spine rots.

### Spread has to be engineered

`pedagogy_rm` found an instruct model asked to tutor produces a narrow band of decent turns, which
starves the bottom of every scale.

1. **Tutor styles** — extend `generate.py`'s `STYLES`.
2. **Deliberately mismatched tutors** — prompt a tutor to always lecture, or always withhold. These
   produce *contingency* failures, which is exactly the signal the spine needs, and they are
   natural rather than synthetic: a lecture is a real turn that happens to be wrong for a student
   who has already reasoned.
3. **Student personas** — confident, silent, careless, answer-seeking.
4. **Targeted negatives** — extend `build_negatives.py`.

Source 2 is the important one, and it is structurally better than the manufactured negatives that
misled V2: a mismatch is a real turn placed in the wrong scenario, so a probe cannot separate it by
style. It has to read the relation between the turn and the student.

### Gates before spending labels

- **Scenario fidelity ≥ 80%** — the student model actually occupies the prescribed state. Below
  that the answer key is unreliable and nothing downstream is worth doing.
- **Prevalence ≥ 10% per metric**, measured by LLM judge on 200 turns. `offers_parallel_case` and
  `asks_recall_of_established` are predicted rare; a 3%-prevalence metric gives κ near zero from
  prevalence alone and a probe nothing to fit. Cut or oversample before labelling.
- **`assistance_level` spread** — all five rungs must appear. If tutors only ever hint, contingency
  has no variance and the spine is dead.

---

## 6. The labelling app

Full technical specification in `LABELING_APP_SPEC.md`. The essentials:

Next.js on Vercel with Neon Postgres. The governing number: the same construct names score **κ
0.13–0.30 from cold crowdworkers and 0.65–0.71 from in-house annotators** with anchors and a
calibration pilot. A public link is by default a machine for producing the first regime, so every
feature exists to buy back that 3–5×. 20,000 labels at κ 0.2 are worth less than 2,000 at κ 0.65.

- **One metric at a time**, with the atlas's positive anchor, hard negative and "says-only" negative
  visible.
- **Calibration gate**: ~15 items with known consensus before a rater's labels count.
- **Blinding is a property of the deployment**: ids are salted locally and the key never leaves the
  laptop, so the database cannot leak provenance because it never receives it.
- **Double-labelling as data, not code**: each `(turn, metric)` pair carries `target_labels` of 1 or
  2, so the overlap fraction is an `UPDATE` rather than a redeploy, and κ is measurable in week one.
- **Export** in the exact shape `extract_hidden.py` and `fit_head.py` already consume.

---

## 7. Order of work

| # | Phase | Gate before continuing |
|---|---|---|
| 1 | `metrics.py` and `scenarios.py` in the atlas schema | every metric has a positive anchor, a hard negative, a named surface confound, and an atlas eligibility status that is not `not_eligible` |
| 2 | Fetch Qwen3-30B-A3B, serve it, smoke-test 20 dialogues | turns coherent, styles visibly different |
| 3 | 1,000 pilot dialogues, scenario-conditioned | **scenario fidelity ≥ 80%** |
| 4 | LLM-judge 200 turns on all metrics | prevalence ≥ 10% each; all five assistance rungs present |
| 5 | Re-run `_redteam_reward_math.py` against the revised reward | no tie at the maximum; no length peak below 60 words |
| 6 | Scale to ~8,000 dialogues | — |
| 7 | Build the app, seed calibration | two agents independently reach κ ≥ 0.5 |
| 8 | Label at scale | 2,000+ turns, 20% double-labelled |
| 9 | Fit probes against surface-feature and length-only baselines | probe beats both by more than agreement noise allows |
| 10 | Wire the reward into `plugin.py`, train an arm | — |

---

## 8. What could still go wrong

**The student model may not occupy its scenario reliably.** The entire spine rests on the
prescribed target being right, which rests on the student actually being lost when the scenario
says lost. Phase 3's fidelity gate exists for this, and if it fails the fallback is human-written
student turns for a smaller, higher-quality scenario set.

**Six scenarios may not span the space.** The target distribution is ours by construction, which is
a strength for the answer key and a weakness for external validity: a policy tuned on this mix is
tuned on our beliefs about which states matter.

**Contingency has a κ ceiling of 0.55–0.70 even done well**, and that bounds what any probe can
reach. Report the ceiling alongside every correlation, as `pedagogy_rm` learned to.

**Decodability was measured on OLMo-2, on atlas text.** 19 of 24 constructs being linearly decodable
justifies the attempt; it does not transfer automatically to our corpus and encoder. Re-measure.

**Several headline effects are weaker than usually quoted.** Bloom's 2σ rests on two small
dissertations and has not replicated — human tutoring is nearer d = 0.79, and a PreK-12
meta-analysis puts it at 0.37 SD. Formative assessment's 0.40–0.70 becomes 0.20 in Kingston & Nash
2011, and 0.17 for mathematics. ICAP's strict ordering is contested, which is part of why
`icap_class` is measured and not optimised. Growth mindset (r = 0.10 across 273 studies) is why no
motivational dimension appears at all.

**A twenty-metric rubric is a real labelling cost**, and protocol quality mattered more than
construct choice in every previous experiment here. If the prevalence pass is discouraging, the
fallback is the spine plus the four highest-prevalence contributions, with the rest recorded as
covariates.
