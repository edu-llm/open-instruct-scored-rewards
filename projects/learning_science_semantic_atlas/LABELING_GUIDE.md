# Labeling guide

How to produce the atlas's first label set: **160 unique items, double-rated,
plus 40 blinded retest judgments per rater.**

Read `EVIDENCE_REVIEW.md` §1–2 first. This document assumes you know the four
claim levels and why every construct is written to be answered by pointing at
the artifact.

The single most consequential fact about this protocol: **the same construct
names score κ = 0.13–0.30 from cold crowdworkers and κ = 0.65–0.71 from in-house
annotators given anchors, worked examples, and a calibration pilot.** That is a
3–5× swing from protocol alone, larger than any difference between constructs.
Everything below exists to be the second kind of protocol. Skipping the
calibration session or the retest block does not save time; it changes which
regime you are in.

---

## 1. What you are and are not judging

You see one artifact. You judge whether it **enacts** each of a small set of
named constructs.

You are **not** judging:

- whether the learner understood, engaged, or learned anything;
- whether a different response would have been better;
- whether the artifact is nice, warm, or well written;
- whether *you* would have taught it that way.

The last one is the trap with numbers attached. Three tutors pick the same
action in 18% of cases, so "what should they have done" has no recoverable
answer. If you find yourself deciding by forming an overall impression, **flag
the item instead of guessing.** A high flag rate on a construct is worth more
than a column of guessed numbers, and it is how a broken definition gets found.

---

## 2. Item pool: 160 items

### 2.1 Composition

| Block | Items | Construction |
|---|---|---|
| **Targeted** | 120 | 24 constructs × 5 items |
| **Debunked controls** | 18 | 6 controls × 3 items |
| **Open pool** | 22 | Natural artifacts, no designed target |
| **Total** | **160** | |

The 5 targeted items per construct:

| Role | n | Built to be |
|---|---|---|
| `positive_canonical` | 1 | The construct's `positive_anchor`, unambiguously |
| `positive_atypical` | 1 | A genuine instance in an unusual surface form — no trigger vocabulary, unusual artifact type, or unusual length |
| `positive_partial` | 1 | Meets the definition but violates a `boundary_conditions` entry. Should be **present with low fidelity**, and is the item that tests whether the ordinal carries information |
| `hard_negative` | 1 | The construct's `hard_negative` — the near-miss that shares the surface form |
| `says_only` | 1 | The construct's `says_only_negative` — correct declarative content about the principle, containing none of it |

Every item is drawn verbatim from the corresponding field in `ontology.yaml`
and then instantiated in real content. If an item cannot be written without
also instantiating a second construct, that is fine and expected — this is a
multi-label task and co-occurrence is real. Record it in the key.

### 2.2 Artifact-type mix

Fixed in advance, because four constructs are schedule-level and would otherwise
never receive an applicable item:

| Artifact type | Items |
|---|---|
| `tutor_turn` / `dialogue_episode` | 70 |
| `worked_solution` | 30 |
| `explanation_text` | 25 |
| `practice_item_set` | 20 |
| `study_schedule` / `curriculum_plan` | 15 |

The designed positives for `spaced_practice_schedule`,
`successive_relearning_criterion`, `interleaved_discrimination_practice`, and
`guidance_fading_sequence` live in the bottom 35 rows. Do not put them anywhere
else; a single tutor turn cannot contain a schedule, and an item that pretends
otherwise is testing vocabulary.

**One of the ontology's eight `output_types` is deliberately not sampled.**
Round 1 builds no `assessment_item` rows. Nine constructs accept that type, and
all nine accept at least one type that is in the table above, so no construct
loses its designed items — but it does mean this round says nothing about how
these constructs label on bare assessment items, and the round report must say
so rather than let the omission read as coverage.

### 2.3 Candidate sets, and why the checklist is not 24 items long

Judging 24 constructs on 160 items is 3,840 decisions and would be skimmed.
Instead:

- **The 120 targeted items** are presented with a **candidate set of 6
  constructs**: the designed target, its `sibling_confusable`, and 4 fillers
  drawn from other families. You judge those 6.
- **The 18 debunked-control items** also get 6 constructs, but their designed
  target is a control rather than an atlas construct, so there is no sibling to
  pair with it. Their 6 are chosen instead as the constructs that control is
  most likely to be mistaken for — the ones named in its `expected_label` where
  it names any, and otherwise the warmth- and framing-adjacent constructs a
  drifting instrument would reward. A control is only a control against the
  constructs it is put in front of.
- **The 22 open-pool items** are presented with the **full 24-construct
  checklist**. Their job is precisely to estimate base rates and catch
  over-labeling, which a restricted candidate set cannot do.

On a targeted item the target and its confusable are always both present, so
every one of the 120 is a discrimination test. You cannot tell which of the 6 is
the target — candidate order is randomized per item per rater.

**`sibling_confusable` is directed, not a mutual pairing.** It records what a
construct gets mistaken *for*; the reverse confusion is usually a different and
rarer error. Eleven of the 24 are therefore nobody's sibling and never appear as
a designed distractor. Every construct still gets its own discrimination test on
its own 5 items — that is the test that matters — but the pairing does not
supply these eleven with appearances, and the filler budget has to.

**What this costs, stated plainly:** presence rates within candidate sets are
not population base rates. Base rates come from the 22 open-pool items only, and
are reported with that *n*.

Filler allocation is balanced so that **every construct appears in the candidate
set of at least 45 of the 160 items.** That floor is reachable with room to
spare, and it is worth checking rather than asserting. Each construct is
guaranteed 5 appearances as a target, 5 for each construct that names it as a
sibling, and all 22 open-pool items. The eleven that are nobody's sibling start
at 27 and need 18 fillers each; across all 24 the shortfall totals 319 filler
appearances against the 480 filler slots on the targeted items alone. Any
allocation that satisfies the floor is acceptable; report the realized
appearance count per construct with the round's statistics, because it is the
denominator for everything in §7.

### 2.4 Calibration set: 24 items, disjoint from the 160

One item per construct, built the same way, **not part of the 160 and never
counted in any reliability statistic.** This is deliberate. In a prior round
here, calibration examples were drawn from the labeled pool and the agreement
script had to report "NO CLEAN UNITS: every unit you labelled was used to
calibrate the agents, so there is nothing left to check them against." Keeping
the calibration set disjoint makes that bookkeeping unnecessary.

---

## 3. The judgment schema

Four things are recorded. They are separate on purpose; collapsing any two of
them is how a rubric stops measuring what it names.

### 3.1 Applicability — asked first, for every candidate construct

| Value | When |
|---|---|
| `applicable` | The construct's `required_context` is present and the artifact type is one of its `output_types` |
| `not_applicable_missing_context` | The artifact type fits but a required context field is absent |
| `not_applicable_wrong_artifact` | This construct cannot apply to this kind of artifact at all |

**If applicability is not `applicable`, stop. Do not record presence or
fidelity.** A missing-context item scored 0 is not the same as an item that
genuinely lacks the construct, and pooling them corrupts both the prevalence
estimate and the agreement statistic.

Three constructs are `context_dependent` in the ontology and will hit this
constantly. Their rules are absolute:

- `contingent_help_calibration` — if the scenario does not **prescribe** a
  target help level, mark `not_applicable_missing_context`. Never supply your
  own target. That question agrees at 18%.
- `expertise_reversal_adaptation` — if learner expertise is not **declared as a
  field**, mark `not_applicable_missing_context`. Never infer expertise from
  vocabulary or tone.
- `spaced_practice_schedule` — if there are no wall-clock session boundaries,
  mark `not_applicable_missing_context`. Conversational turns are not occasions.

### 3.2 Presence — binary, requires a quoted span

`0` or `1`: does the artifact instantiate the construct at all?

**Marking `1` requires pasting the span of the artifact that does it.** No span,
no presence. This is the mechanism that keeps the whole task pointing at text,
and it is the single rule most responsible for the difference between the two
reliability regimes above. For constructs whose `positive_anchor` names two
required elements (`belonging_affirmation_framing` needs *common* and
*temporary*; `analogical_case_comparison` needs two cases and an alignment
demand), quote **both**, or mark 0.

### 3.3 Fidelity — ordinal 0–3, only when presence = 1

One scale, shared across all 24 constructs, anchored to the ontology rather than
to your taste.

| Score | Anchor |
|---|---|
| **0 — Nominal** | The surface form is there; the defining requirement is not met. A question with the answer supplied two lines later. A "subgoal label" that reads "Step 3". This is where a hard negative lands if you marked it present. |
| **1 — Partial** | The defining requirement is met, but a required element is missing or a `boundary_conditions` entry is violated. A pretest with no resolving instruction. A completion problem whose blank is a trivial arithmetic step. |
| **2 — Sound** | Meets the definition with all required elements present. **This is the modal score for a real instance and is not a criticism.** |
| **3 — Exemplary** | Meets the definition *and* satisfies the boundary conditions that apply to this artifact. Rare. If you are scoring 3 more than about one time in six, re-read the boundary conditions. |

Three points plus a nominal floor, not five. A longer scale invites you to
express confidence in the score, and confidence is variance rather than signal.

### 3.4 General quality — once per item, 1–3

| Score | Anchor |
|---|---|
| 1 | You would not want a learner to receive this |
| 2 | Adequate |
| 3 | You would be pleased if a tutor produced this |

**This dimension is known to be bad and is collected anyway.** Two careful
raters agreed on holistic quality 39% of the time (*r* = 0.15). It is here for
exactly two jobs:

1. To test whether the atlas constructs explain any of it. If 24 evidence-graded
   constructs predict nothing about what a person thinks is good teaching, that
   is a finding about the atlas.
2. To detect a rater who is scoring overall impression instead of the construct.
   If a rater's fidelity scores correlate with their own general-quality score
   above ~0.7 within a construct, they have collapsed the two.

It is **never** a reward term, never averaged with construct scores, and never
reported without its reliability beside it.

### 3.5 Flags

| Flag | Set when |
|---|---|
| `rubric_misfit` | The definition did not fit the artifact and you would have had to guess |
| `factual_error` | The artifact contradicts the supplied reference solution. Check *against* the reference; do not re-derive it. The re-derivation form of this question failed at κ = 0.18 here |
| `says_only` | The artifact discusses a pedagogical principle without enacting it — record which construct keys |
| `debunked_claim` | The artifact asserts something from `debunked_controls` (learning styles, pyramid percentages) |

A construct flagged `rubric_misfit` on more than **20% of items where it appears
in the candidate set** is rewritten or dropped regardless of its κ. A high flag
rate is the same signal as low agreement, arriving more politely.

---

## 4. Retest: 40 blinded judgments

Forty of the 160 are re-presented to the same rater. **They will not know
which.**

| Stratum | n |
|---|---|
| Targeted items, one per construct, rotating role across constructs | 24 |
| Debunked controls | 8 |
| Open pool | 8 |

Rules:

- **≥ 72 hours** after the original judgment, and never in the same session.
- Re-salted item id, so it cannot be matched to the original by lookup.
- Different position in the session, different candidate-set order.
- Same rater. This measures a rater against *themselves*.

Total judgments per rater: **200 items** (160 + 40).

Intra-rater retest is not a formality — it is the ceiling on everything else. A
rater who disagrees with themselves at κ = 0.5 cannot agree with a colleague
above that, and a construct where intra-rater agreement is low is unstable in a
way no amount of adjudication will fix. In a prior round here **no turns were
re-presented at all**, so a single human's signal could not be separated from
her own noise, and every downstream comparison inherited that gap.

---

## 5. Sessions, order, and blinding

**Sessions.** Maximum 90 minutes and 40 items. Expect roughly:

| Block | Rate | Time |
|---|---|---|
| 138 items × 6 candidates | ~3.5 min/item | ~8.1 h |
| 22 open-pool items × 24 candidates | ~7 min/item | ~2.6 h |
| 40 retest | ~3.5 min/item | ~2.3 h |
| **Total per rater** | | **~13 h over ~9 sessions** |

**Order.** Item order randomized per rater. Never label all items of one
construct consecutively — a run of retrieval-practice items trains you to see
retrieval practice. Candidate-set order randomized per item per rater, so
position cannot become a cue for "this is the target".

**Withheld from raters** — the fields that would tell you the answer:

```
designed_target_construct, item_role, source_model, source_policy,
temperature, sample_index, prior_scores, agent_labels, other_rater_labels
```

Item ids are salted hashes. The key file carries the mapping and **is not opened
until every label is in.** Blinding is not a formality here: a pool that labels
its own answers measures the labeler's belief about the answer.

---

## 6. Agent calibration

Model raters label the same 160 items, for throughput and as a candidate label
source. They are calibrated and then checked, in that order.

### 6.1 Protocol

1. **Agents run first**, before any human labels exist. Anything the human
   labels afterward is then untainted by agent output.
2. Agents receive this guide verbatim, plus the 24-item calibration set as
   few-shot examples **with gold labels and one-sentence rationales**. Record
   the shot ids per construct in the run manifest.
3. Every construct an agent is asked to score **must be defined in the prompt.**
   This is not optional. In a prior round two dimensions were requested without
   being defined; the agents guessed from the names and the few-shot examples,
   and one of them — whose entire point was that 1 and 3 are both bad — came back
   `2` on all 25 holdout turns, correlating **−0.36** with the human. An
   undefined dimension does not fail loudly. It comes back plausible and empty.
4. Agents see the same withheld-field list as humans and the same randomized
   candidate order.
5. Human labels are collected blind to agent output.

### 6.2 What to report, per construct

Three numbers, and they answer different questions:

| Statistic | Question |
|---|---|
| **Inter-agent κ** | Do the models agree with each other? |
| **Agent–human κ** | Does each model agree with the reference? |
| **Consensus–human κ** | Does the rounded mean of the agents agree with the reference? |

The consensus row is the one to read, because the mean of the agents is what any
downstream model would actually be trained on. Round the consensus to the
rubric's scale so it can be compared with a single rater on equal terms.

### 6.3 The failure mode to watch for

**A construct where the agents agree with each other but not with the human is
the dangerous case.** It looks reliable and is measuring something else. This is
not hypothetical: in a prior round here, six model raters from six labs agreed
with the human at **κ = 0.18** on one dimension while agreeing with **each other
at 0.41**. Six models can agree closely because they share training data and a
house style. That is consistency, not accuracy.

### 6.4 Agent usability gate

Agent labels may substitute for human labels on a construct only when **both**
hold:

- agent–human κ_w ≥ **0.60**; and
- agent–human κ_w ≥ **0.80 × human–human κ_w** for that construct.

Both, because the two catch different failures. Agreeing with a noisy human at
0.5 when two humans agree at 0.5 is as good as it is possible to be; agreeing at
0.6 when two humans agree at 0.9 is a real deficit that the absolute threshold
alone would wave through.

### 6.5 Mandatory agent bias probes

- **Length.** Spearman correlation between agent fidelity and token count, per
  construct, reported next to the human's. Precedent: agent leak scores tracked
  length at **+0.59** against the human's **+0.43**, and that bias got a policy
  comparison backwards — an arm was ranked below another that the human
  preferred 29–11.
- **Says-only.** Agent says-only false-positive rate, reported separately.
  Expect agents to be worse than humans here, because reading topic is exactly
  what they are good at. This number is the atlas's core question asked of the
  instrument itself.
- **Position.** Presence rate by candidate-set position. Should be flat.

---

## 7. Per-concept reliability gates

Computed **per construct**. All 160 items are rated by both human raters; there
is no partial overlap. A prior round shipped slices with zero overlap and
agreement was simply unmeasurable.

The denominator changes from statistic to statistic and must not be allowed to
drift between them. Three bases are used:

- **appearances** — every double-rated item where the construct was in the
  candidate set, whatever either rater then said about applicability;
- **both-applicable** — appearances where both raters marked `applicable`;
- **both-present** — both-applicable appearances where both also marked
  presence = 1.

`applicability_kappa` and `flag_rate` are computed over **appearances** and
nothing narrower. Restricting either to both-applicable items would condition on
the thing being measured: an applicability κ over items both raters called
applicable is 1.0 by construction and reports nothing.

Two statistics are within-rater rather than between-rater and take the same
bases one head at a time: S8 over a rater's retest appearances that they marked
applicable in both passes, S9 over the items that rater marked present.

### 7.1 Statistics

| # | Statistic | Base | Definition |
|---|---|---|---|
| S1 | `n_judged` | appearances | Double-rated items where this construct was in the candidate set |
| S2 | `n_present` | both-applicable | Items where ≥1 rater marked presence = 1 |
| S3 | `applicability_kappa` | appearances | Cohen's κ on the boolean `applicable`, reported with a breakdown of the two not-applicable reasons |
| S4 | `presence_kappa` | both-applicable | Cohen's κ on the binary, **reported with Gwet's AC1** |
| S5 | `fidelity_kappa_w` | both-present | **Linearly** weighted κ on 0–3 |
| S6 | `says_only_fp` | both-applicable | Fraction of judgments on that construct's says-only item marking presence = 1 |
| S7 | `hard_negative_fp` | both-applicable | Same on its hard negative, at presence and separately at fidelity ≥ 2 |
| S8 | `retest_presence_kappa` | retest, within rater | Intra-rater κ on the binary, over the retest appearances of this construct |
| S9 | `length_rho` | present, within rater | Spearman(fidelity, token count), reported for each rater |
| S10 | `flag_rate` | appearances | Fraction flagged `rubric_misfit` |

Three choices worth defending:

**Linear, not quadratic, weights (S5).** Quadratic is conventional and forgives
one-point disagreements so heavily that a scale nobody agrees on can still score
well — the opposite of what this is for.

**Both κ and AC1 for presence (S4).** Presence is rare for most constructs, and
Cohen's κ is unstable under skewed marginals: two raters agreeing on 95% of
items can post a κ near zero purely because almost everything is a 0. Reporting
κ alone would drop constructs for being rare rather than for being unreliable.
Reporting AC1 alone would flatter them. Report both and the prevalence, and
resolve disagreements between the two by inspecting the confusion table.

**S6 and S7 rest on one item each, and the round report must say so.** The pool
gives every construct exactly one says-only item and one hard negative (§2.1),
double-rated, so each of these "rates" has a denominator of two judgments and
can only ever come back 0, 0.5, or 1. They are counts wearing the costume of
rates. Report them as counts — *k* of 2 — alongside the fraction, and do not
compare them across constructs as though a difference of 0.5 were an
effect. Round 2 raises both to five items per construct; until then G4 and G5
below are screens, not estimates.

### 7.2 Gates

A construct earns **confirmatory** status only if all of G1–G8 pass.

| Gate | Threshold | On failure |
|---|---|---|
| **G1 Power** | `n_judged` ≥ 20 **and** `n_present` ≥ 8 | `insufficient_evidence` — *not* a pass and *not* a fail. Oversample in round 2 |
| **G2 Presence** | `presence_kappa` ≥ 0.60 | 0.40–0.60 → **exploratory**: usable, but report results against the reliability ceiling rather than against 1.0. < 0.40 → **drop or rewrite** |
| **G3 Fidelity** | `fidelity_kappa_w` ≥ 0.40 with ≥ 8 both-present items | Keep presence, **discard the ordinal**. Report as binary only |
| **G4 Says-only** | Round 1: **0 of 2** says-only judgments marked present. Round 2, at 5 items: `says_only_fp` ≤ 0.15 | **Blocked from reward.** The construct is being read as topic, not structure |
| **G5 Hard negative** | Round 1: **0 of 2** at presence and 0 of 2 at fidelity ≥ 2. Round 2, at 5 items: ≤ 0.30 and ≤ 0.10 | Rewrite the discriminator in the ontology's `sibling_confusable` field, re-label the slice |
| **G6 Applicability** | `applicability_kappa` ≥ 0.60 for `context_dependent` constructs | Nothing downstream is interpretable. Fix the required-context fields before re-labeling |
| **G7 Retest** | `retest_presence_kappa` ≥ 0.60 | The definition is unstable within one head. Rewrite; adjudication will not fix it |
| **G8 Confound** | \|`length_rho`\| ≤ 0.50 | Report as length-confounded and **block from reward**. Keep as a descriptor |

Plus the standing rule from §3.5: `flag_rate` > 0.20 forces a rewrite whatever
the κ says.

**G4 and G5 are the strictest gates in the table and the least powered, and that
combination is deliberate.** At two judgments apiece the round-1 form is
all-or-nothing: one says-only item read as present blocks the construct from
reward. A single judgment is thin evidence for blocking, and blocking is the
cheap direction to be wrong in — a construct wrongly held out of a reward is
still labeled, still reported, and still available next round, whereas a
construct that reads vocabulary as structure and reaches a reward takes the
policy with it.

**G7 carries no power floor and needs one read into it.** The retest block puts
each construct in front of a rater about 16 times on average (§4), so a construct's
retest κ has a wide interval and should be read as a screen for instability
rather than as an estimate. Report the retest *n* beside the κ, and treat
`n < 10` as `insufficient_evidence` on G7 rather than as a pass.

### 7.3 Round-2 diagnostic pool

Round 2 is triggered per construct by a round-1 G4 or G5 failure, or when a
construct is otherwise being reconsidered for reward use. For each triggered
construct, add **four new says-only items and four new hard negatives**. Combined
with round 1, this yields five items of each role (ten artifacts, twenty
double-rated judgments) for that construct. New items must:

- use lineages absent from calibration and round 1;
- span at least three domains and two accepted artifact types where the ontology
  permits that coverage;
- preserve the same six-candidate blinding and sibling inclusion rules;
- be double-rated independently, with no adjudication before G4/G5 are computed.

The round-2 thresholds in G4 and G5 apply to the **combined five-item-per-role
pool**, not to the four new items alone. Constructs not triggered for round 2
receive no added items; this pool is a targeted diagnostic, not a new prevalence
sample.

### 7.4 Probe ceiling

For any construct that will be predicted by a model, report

```
ceiling = sqrt(max(r, 0))
```

where `r` is the inter-rater correlation on fidelity. A predictor cannot
correlate with a noisy label above the square root of that label's reliability.
Compare a probe against this column, **not against 1.0**. Approaching it means
the probe is done and the labels are the limit. A prior project spent a long
time treating a 0.581 correlation as a representation failure when it was a
label-noise ceiling.

### 7.5 Expected outcomes

From the ontology's `labelability` bands, stated in advance so that hitting them
is not a surprise and missing them is informative:

| Band | Constructs | Expectation |
|---|---|---|
| `high` (10) | retrieval, successive relearning, pretest, worked example, completion, subgoal, teach-back, answer withholding, calibration elicitation, utility value | Pass G2 confirmatory |
| `moderate` (9) | interleaving, fading, analogical, problem-first, self-explanation, elaborated feedback, metacognitive, autonomy, belonging | Land in the 0.40–0.60 exploratory band |
| `context_dependent` (3) | spacing, contingency, expertise reversal | Pass **only** where required context is supplied; G6 is the real gate |
| `low` (2) | ICAP class, attributional framing | Expected to fail G2. Both are already `not_eligible` for reward. If they fail, that is a confirmation, not a loss |

If a `high`-band construct fails G2, do not label more of it. Rewrite the
definition first. A probe cannot beat the noise in its own labels, and no volume
of labeling, no bigger model, and no better architecture fixes it — the ceiling
is set by the question, not by the method.

---

## 8. Adjudication

Disagreements are resolved in batch, at session boundaries, never mid-session.

1. **Mandatory** for `low`-band constructs on every disagreement; **optional
   elsewhere**, and only after the round's statistics are computed. Adjudicating
   before computing agreement destroys the measurement you were trying to take.
2. Adjudication produces two things: a third label, **and** a rule added to this
   guide.
3. Rules are versioned. A rule added mid-round retroactively changes the
   question for items already labeled, so any construct receiving a new rule has
   its slice re-labeled or is reported at the older version. Both files carry
   `construct_version`; keep them in step.
4. **Do not adjudicate away a low κ.** The purpose of the gate is to find
   questions that two careful people cannot answer the same way. A construct
   forced to agreement by a third rater's tiebreak has not improved; it has
   hidden the evidence that it needed rewriting.

---

## 9. Output format

One JSON file per rater. Consumed by the agreement analysis alongside the key.

The three-valued applicability judgment of §3.1 is serialized as a **boolean
`applicable`** plus a **diagnostic `applicability` reason**, because the gate
computation treats applicability as a binary code and only G6 and the round
report need the reason. Recording both keeps §3.1 intact without giving the
agreement statistic a three-way code it would have to collapse anyway.

```json
{
  "rater": "sophia",
  "construct_version": "atlas-v1.0.0",
  "guide_version": "1.0.0",
  "session_log": [
    {"session": 1, "started": "2026-08-11T09:02:00Z", "ended": "2026-08-11T10:28:00Z", "item_ids": ["a1f3c2d8e9b0a1", "..."]}
  ],
  "labels": [
    {
      "id": "a1f3c2d8e9b0a1",
      "is_retest_of": null,
      "quality": 2,
      "flags": ["says_only"],
      "flag_note": "describes retrieval practice, does not create one",
      "constructs": {
        "retrieval_practice_opportunity": {
          "applicable": true,
          "applicability": "applicable",
          "presence": false,
          "fidelity": null,
          "span": null
        },
        "self_explanation_prompt": {
          "applicable": true,
          "applicability": "applicable",
          "presence": true,
          "fidelity": 2,
          "span": "Why is that step allowed here, and what would we check first?"
        },
        "contingent_help_calibration": {
          "applicable": false,
          "applicability": "not_applicable_missing_context",
          "presence": null,
          "fidelity": null,
          "span": null
        }
      }
    }
  ]
}
```

Validation rules, enforced at load:

- `presence = true` requires a non-empty `span`.
- `fidelity` is non-null **iff** `presence = true`.
- `presence` and `fidelity` are null **iff** `applicable = false`.
- `applicable == (applicability == "applicable")`; the two may not disagree.
- `fidelity ∈ {0,1,2,3}`; `presence` is boolean; `quality ∈ {1,2,3}`.
- Every construct key resolves against `ontology.yaml`.
- `construct_version` matches the ontology's.

A record failing any of these is rejected rather than coerced. Nulls carry
meaning in this schema and must never be imputed to zero: a construct the rater
called inapplicable has no presence to average, and one with `presence = false`
has no fidelity to score. Filling either with a 0 silently converts "the
question did not apply" into "the artifact did it badly."

---

## 10. Order of operations

Nothing below may be reordered. Each step exists because doing it later costs
something that cannot be recovered.

1. Freeze `ontology.yaml`. Definitions are cheap to change now and expensive to
   change once 160 labeled items depend on them.
2. Build the 24-item calibration set and the 160-item pool. Blind and salt.
3. Run agents. Store output sealed.
4. Calibration session: both human raters label the 24 calibration items
   together, out loud, resolving every disagreement into a written rule. This
   session is the 3–5× protocol effect. Do not skip it.
5. Human raters label all 160 independently, blind.
6. Wait ≥ 72 h. Run the 40-item retest block.
7. **Compute §7 before opening the key.** Agreement is a property of the labels
   and does not need the answers.
8. Open the key. Compute says-only and hard-negative false-positive rates,
   agent–human agreement, and base rates from the open pool.
9. Publish the per-construct gate table — every construct, including the failures
   and the `insufficient_evidence` rows. A gate table that reports only the
   survivors is a selection effect with a decimal point.

---

## 11. Before you start

A short list of the mistakes this protocol is built to prevent, each of which
has already happened once in this repository:

- Labeling before agreement is measurable, because slices had no overlap.
- Calibrating agents on the units later used to check them, leaving no clean
  units to check them against.
- Asking for a dimension without defining it, and receiving a plausible,
  empty, constant column.
- Treating a non-ordinal scale as ordinal and averaging it into meaninglessness.
- Reading agent-versus-agent consistency as evidence of accuracy.
- Optimizing a rubric made entirely of withholding measures, and discovering its
  optimum is a 12-word question.
- Never re-presenting an item to the same rater, and so never being able to
  separate one person's signal from her own noise.
