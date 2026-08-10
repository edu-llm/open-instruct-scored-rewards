# Evidence review

What the learning-science literature actually supports, what it does not, and
which of the supported things can be read off a single artifact by two people
who agree with each other.

This document is the justification layer for `ontology.yaml`. Every construct
there points back to a section here; every grade here is defined before it is
used. `LABELING_GUIDE.md` turns both into a protocol.

---

## 1. The distinction the whole project rests on

Four different things get called "the model is good at teaching". They are
evidence for four different claims, and conflating them is the most common
error in evaluating tutoring systems.

| Level | Name | What it is | What it licenses |
|---|---|---|---|
| 1 | **Declarative knowledge** | The artifact *states* a principle | A claim about what the system can say |
| 2 | **Enacted output** | The artifact's *structure* instantiates the principle | A claim that this artifact affords the activity |
| 3 | **Learner enactment** | A learner *did* the targeted cognitive work | A claim that the offer was taken up |
| 4 | **Learning outcome** | Durable, independently measured change | A learning claim, bounded by delay and distance |

The gap between levels 1 and 2 is why this atlas exists. A tutoring model can
produce a fluent paragraph on why retrieval practice beats rereading, and then
hand the student the answer. That artifact is level-1 positive and level-2
negative, and any evaluation that scores it by topic or vocabulary will call it
excellent. Every construct in the ontology therefore carries a **says-only
negative**: a text that is correct *about* the principle and contains none *of*
it. Those items are in the labeling pool on purpose, and the rate at which a
rater or an agent marks them present is a direct measure of whether the
instrument is reading structure or reading vocabulary.

The gap between 2 and 3 is where most tutoring papers overclaim. An artifact
that asks a good question has made an offer. Whether the learner retrieved,
explained, or thought at all is a fact about the *next* turn, and it is not
knowable from the artifact under any amount of careful reading.

The gap between 3 and 4 is where the field's own literature is strictest, and
this project inherits that strictness. Uptake is not retention. Immediate
correctness is not learning — under some of the best-evidenced techniques here
it is *anti*-correlated with learning, because spacing, interleaving, and
retrieval all depress immediate performance while improving delayed retention
(Soderstrom & Bjork, 2015). Students in active-learning conditions measurably
learned more while reporting that they had learned less (Deslauriers et al.,
2019). Any metric built on immediate ease is pointed the wrong way.

**Scope limit.** Every construct in this atlas is scored at level 2. That is
not modesty, it is the ceiling of the data: a single artifact cannot reach
level 3 or 4. Constructs whose interesting content lives higher up carry an
`escalation` note saying exactly what data would be needed.

---

## 2. Two independent gradings

A construct has two properties that get confused with each other:

- **Evidence grade (A–D)** — how good the evidence is for the *instructional
  principle*.
- **Labelability** — whether two humans can agree on whether an artifact
  instantiates it.

These are independent, and the ontology grades them separately. Spaced practice
is grade A and nearly unlabelable from a tutor turn, because a turn cannot
contain a schedule. Heading structure is trivially labelable and rests on
nothing. Both failure modes produce useless labels, for opposite reasons.

### Evidence grades

| Grade | Meaning | Bar |
|---|---|---|
| **A** | Robust | ≥2 independent meta-analyses, or one plus a large preregistered multi-site trial. Direction stable across labs, materials, populations. Known moderators do not reverse the sign in common conditions. |
| **B** | Well supported, substantial moderators | ≥1 meta-analysis or several independent controlled experiments. Direction generally stable, size heavily moderator-dependent, sign can reverse in *named* conditions. |
| **C** | Promising but contested or narrow | Concentrated in few labs, or a meta-analysis with serious heterogeneity, publication-bias, or active-control problems. Credible specialists disagree about the size. |
| **D** | Contested; best evidence near zero | Credible meta-analyses reach opposing conclusions, or the highest-quality subset shows ~zero. Retained to be **measured** and to serve as a negative control — never assumed, never rewarded. |

Grade distribution across the 24 constructs: **A** ×4, **B** ×16, **C** ×3,
**D** ×1. The skew toward B is honest rather than hedging. Most of this
literature is "real effect, large moderators", and a review that graded most of
it A would be describing a different literature.

### Labelability bands

Grounded in measurements, not intuition. This repository has labeled pedagogical
constructs before and the numbers are unforgiving:

| Measured in prior work here | Result |
|---|---|
| Holistic "goodness", two careful raters | **39% exact agreement**; two annotators at **r = 0.15** |
| Concrete `leak` (answer withholding), same raters | **69% exact agreement** |
| `correct` ("could a student take away anything wrong?") | **κ = 0.18** vs human, while agents agreed with *each other* at 0.41 |
| Cognitive demand ("how much thinking does this ask for") | **κ = 0.09–0.34** across three independent studies |
| The same construct rephrased behaviorally ("does it hand over the next move, and name it") | **κ = 0.55–0.66** |
| Uptake / revoicing, the field's flagship discourse measure | **Fleiss κ = 0.286**, and learner-token overlap alone predicts it at **r = 0.52** |
| Learner internal state (goals, motivation, interest) | **Krippendorff α = 0.03, 0.02, 0.07** — below chance, replicated |
| Tone, encouragement, warmth | **α = 0.24–0.30** |
| Which strategy the tutor *should* have used | three tutors agree **18%** of the time; a tutor agrees with outside raters about their own move at **κ = 0.34** |

Two conclusions follow, and they shape every definition in the ontology.

**First: phrasing is the intervention.** Cognitive demand went from κ ≈ 0.2 to
κ ≈ 0.6 by being reworded from a mental-state question into a behavioral one.
Nothing about the underlying construct changed. Every ontology entry is
therefore written to be answered by *pointing at the artifact*, and where a
construct cannot be phrased that way it is excluded rather than softened.

**Second: protocol is worth more than construct choice.** The same construct
names score κ = 0.13–0.30 from cold crowdworkers and κ = 0.65–0.71 from
in-house annotators given anchors, worked examples, and a calibration pilot
(Daheim et al., 2024, against Maurya et al., 2025). That is a 3–5× swing from
protocol alone, larger than any difference between constructs. `LABELING_GUIDE.md`
is built to be the second kind of protocol.

Those numbers set the four bands the ontology assigns, and the bands are
predictions registered before the labeling rather than descriptions after it.

| Band | *n* | Expected κ_w | What it means |
|---|---|---|---|
| **high** | 10 | ≥ 0.60 | Answerable by pointing at a structural feature. Eligible for confirmatory use if it clears its gate |
| **moderate** | 9 | 0.40–0.60 | Answerable by pointing, but with a judgment call raters can legitimately split on. Usable, reported against the reliability ceiling rather than against 1.0 |
| **low** | 2 | < 0.40 | Historically unreliable here or in the published annotation literature. Mandatory adjudication, exploratory only |
| **context_dependent** | 3 | varies | Reliability depends entirely on whether the required context was supplied. Unusable without it, and the applicability judgment is the real gate |

`LABELING_GUIDE.md` §7.4 states the per-construct expectations, so that hitting
them is unremarkable and missing them is informative. A band is not a pass: two
of the ten `high` constructs failing their gate would be a fact about this
review's judgment, and it is recorded this way so that it can become one.

---

## 3. Family 1 — Practice scheduling and generation

### Retrieval practice (grade A)

The most robust finding in the set. Three independent meta-analyses converge:
testing beats restudy, the advantage grows with retention interval, and feedback
amplifies it (Rowland, 2014; Adesope et al., 2017; Yang et al., 2021 for the
classroom case). The primary demonstrations are unambiguous: the restudy
advantage *reverses* as delay lengthens (Roediger & Karpicke, 2006), and it is
retrieval rather than repeated exposure that does the work (Karpicke & Roediger,
2008).

The measurement problem is that "asks a question" is not the construct. The
construct is *the answer is withheld and a production is required*. Rhetorical
questions, comprehension checks, and questions whose answers appear two
sentences later all have the surface form and none of the content. The
discriminating test used in the ontology is mechanical: could the learner
produce the answer by copying a span of the artifact?

### Spacing (grade A) and successive relearning (grade B)

Distributed practice is one of the oldest replicated effects in psychology, with
a quantitative synthesis of how optimal lag scales with retention interval
(Cepeda et al., 2006) and a recent meta-analysis specific to spacing retrieval
episodes (Latimier et al., 2021). Combining spacing with retrieval is the
highest-utility pairing in the Dunlosky et al. (2013) review, and successive
relearning — retrieval to a stated criterion, repeated across sessions — is that
combination made explicit (Rawson & Dunlosky, 2011).

Both are graded A/B on the principle and are **not rewardable at turn level**,
because a turn cannot contain a schedule. This is the cleanest case in the atlas
of a construct where a turn-level reward would necessarily be rewarding the
*word* "spaced". Sessions must be real occasions separated in wall-clock time;
turns in one conversation are not spacing.

Successive relearning has one specific counterfeit worth naming: a stopping rule
based on learner-judged confidence. Learners are systematically overconfident
and terminate study early (Dunlosky & Rawson, 2012), so "review until you feel
confident" replaces the criterion with the exact signal the criterion exists to
override.

### Interleaving (grade B — and the sign reverses)

Brunmair & Richter (2019), 59 studies and 238 effect sizes, is the reason this
is B rather than A. Overall *g* = 0.42, but decomposed:

| Material | *g* |
|---|---|
| Paintings / visual category learning | 0.67 |
| Mathematics | 0.34 |
| Expository text, tastes | ns |
| **Words** | **−0.39 (blocking wins)** |

The meta-regression is the useful part: interleaving works better when
*between*-category similarity is high and *within*-category similarity is low.
That is a precise statement that the effect is about **discrimination**, and it
justifies the ontology's hard negative — a set alternating French vocabulary and
calculus is variety, not interleaving, because no learner could confuse them.
Nothing in this literature supports mixing unrelated topics.

Interleaving also depresses practice-phase accuracy while improving delayed
performance (Taylor & Rohrer, 2010), so practice accuracy is the wrong success
metric for it, in the same way it is wrong for spacing.

### Pretesting (grade B, with a documented cost)

Failed retrieval attempts before instruction improve later learning of the
attempted material (Richland et al., 2009), including for errorful generation
where the guess is related to the target (Kornell et al., 2009), and in applied
video-lesson settings (Carpenter & Toftness, 2017).

The reason this is B and carries a boundary condition rather than a
recommendation: prequestions can *impair* memory for the non-pretested content
in the same lesson (Pan & Sana, 2021). Attention is redistributed, not created.
A design that pretests everything is not a design.

Position, not wording, is definitional. A question about studied material is
retrieval practice; the identical sentence about unstudied material is a
pretest. And the resolving instruction must actually follow — an unresolved
failed attempt is not the effect.

---

## 4. Family 2 — Example-based instruction and problem-first design

### Worked examples (grade A)

Four decades of controlled experiments (Sweller & Cooper, 1985; Atkinson et al.,
2000; Renkl, 2014) and now a meta-analysis: 55 studies, 181 effect sizes,
*g* = 0.48 on mathematics performance (Barbieri et al., 2023).

Two moderators from that meta-analysis matter more than the headline:

1. **Correct examples outperform erroneous examples** and correct-plus-erroneous
   mixtures, for overall learning.
2. **Pairing worked examples with self-explanation prompts was a *negative*
   moderator.** The two most-recommended techniques in this family did not
   stack; adding the prompt reduced the examples' benefit.

The second finding is why `self_explanation_prompt` is graded B rather than A
despite having its own supporting meta-analysis, and it is a good example of the
conservatism this review is trying to practice. The common assumption — that
evidence-based techniques compose additively — is not supported here, and an
ontology that silently assumed it would be encoding a claim the literature
contradicts.

### Completion problems and fading (grade B)

Completion problems (van Merriënboer & Krammer, 1987; Paas, 1992) and faded
worked examples (Renkl et al., 2002; Atkinson et al., 2003) are the designed
bridge from studying to solving. Backward fading — removing the last steps first
— has the most direct support, so the *direction* of removal is part of the
manipulation, not a detail.

Fading is a **trajectory** property and can never be labeled from one item. Its
measurement trap is that decreasing verbosity looks identical to decreasing
scaffolding. The ontology requires counting supplied *reference-solution steps*
rather than words, because a turn-length proxy would score a policy that simply
got terser as one that scaffolded well. This repository has already produced a
policy that got terser.

### Subgoal labeling (grade B)

Segmenting a solution under labels that name what each block *accomplishes*
improves solving of novel problems requiring adaptation (Catrambone, 1998;
Margulieux & Catrambone, 2016). The effect is on transfer more than on
reproduction.

Functional versus positional labeling is the whole construct. "Isolate the
variable" is a subgoal label; "Step 3" and "Subtract 4 from both sides" are not.
This is one of the few **pure contribution measures** in the atlas — it adds
structure without removing learner work — which makes it a valid pairing partner
for the withholding measures in Family 4. See §10.

### Analogical case comparison (grade B)

Comparing two cases side by side supports transfer better than studying them
separately (Gentner et al., 2003), with meta-analytic support for case
comparison generally (Alfieri et al., 2013). Cases must share deep structure
while differing on surface features, and must be simultaneously available.

A single illustrative analogy ("electricity is like water in pipes") is not this
construct: there is no second case to align and no alignment demand.

### Problem solving before instruction (grade C — genuinely contested)

This is the one place where the review declines to take a side, and the ontology
records both.

**For:** Sinha & Kapur (2021), 53 studies and 166 comparisons, *g* = 0.36
[0.20, 0.51] favoring problem-solving-first for conceptual knowledge and
transfer, rising to 0.37–0.58 when implemented with high fidelity to productive
failure design principles. Mechanisms are specified (Loibl et al., 2017).

**Against:** cognitive-load researchers report the *opposite* ordering when
element interactivity is high and learners lack prerequisites (Ashman et al.,
2020), against a background argument that minimal guidance underperforms
generally (Kirschner et al., 2006).

**And from inside the supporting meta-analysis:** effect sizes favored
instruction-first for grades 2–5 and for domain-general skills. The sign
reverses by population.

So: a real effect in a defined region, with a boundary that a single artifact
cannot observe, disputed by serious people. That is what grade C means, and it
is why the construct is `measurement_only` — collected and reported, never
optimized. Rewarding a contested ordering would bake one side of a live
disagreement into a policy.

The fidelity requirement also matters for labeling. "Try it yourself first"
followed by canonical instruction that ignores what the learner produced is the
low-fidelity version, and its effects are markedly smaller. Without the
consolidating instruction it is unguided discovery, which is separately not
supported (Alfieri et al., 2011).

---

## 5. Family 3 — Generative explanation

### Self-explanation (grade B)

Spontaneous self-explanation distinguishes successful from unsuccessful example
study (Chi et al., 1989), and prompting it beats unprompted study
meta-analytically (Bisra et al., 2018). Downgraded from A for two reasons: the
negative interaction with worked examples in mathematics (§4), and constraint
analyses showing the advantage narrows against equally active comparison
conditions (Rittle-Johnson & Loehr, 2017).

The labeling trap is the generic prompt. "Can you explain your answer?" names no
object, and this repository's measurements identify exactly this class of vague
handover as the case raters cannot score consistently — which is why the working
version of the dimension distinguishes *named* from *unnamed* handover.

### Teach-back (grade B)

Two meta-analyses. Kobayashi (2019): *g* = 0.35 for preparing-to-teach and
*g* = 0.56 for teaching after preparation, relative to studying without teaching
expectancy, with interactive teaching outperforming non-interactive. Fiorella &
Mayer (2013) separate expectancy from actual explaining and find the durable
benefit comes from actually explaining.

Two constraints the ontology encodes. **Prior expectancy is the moderator that
decides whether the effect exists** — teaching after studying *without* prior
teaching expectancy has been estimated at essentially zero. And the benefit
accrues to the *teacher*; this construct says nothing about the audience.
Teaching also does not automatically produce knowledge-building rather than
knowledge-telling (Roscoe & Chi, 2007), which is a level-3 property.

### ICAP (grade B on the principle, low on labelability)

The ordinal prediction — interactive > constructive > active > passive — is
supported across classroom translations (Chi & Wylie, 2014; Chi et al., 2018).
ICAP is a *classification framework*, not an intervention, and its own authors
document how hard teachers find the active/constructive boundary.

This is the atlas's clearest case of a well-supported principle that fails on
labelability. Expected κ is 0.30–0.50, in the band where cognitive-demand codes
have historically landed. It is included because the taxonomy is genuinely
useful for describing an item pool, and it is marked `not_eligible_unreliable`
for reward, with a second reason: it is an ordinal *category* set, and any
reward would silently treat it as an interval scale. Report four counts; never
average.

---

## 6. Family 4 — Feedback and assistance

### Elaborated feedback (grade A, including the harm)

The clearest quantitative ordering in the review, from 70 effect sizes in
computer-based environments (Van der Kleij et al., 2015):

| Feedback type | *d* |
|---|---|
| Elaborated feedback | 0.49 |
| Knowledge of correct response | 0.32 |
| Knowledge of results (bare right/wrong) | 0.05 |

Corroborated by Wisniewski et al. (2020) for information-rich feedback, and
framed by Hattie & Timperley (2007), whose four levels place feedback directed
at the *self* as least effective.

The grade-A designation includes the negative result, which is as well
established as the positive one: in Kluger & DeNisi (1996), **over a third of
feedback interventions decreased performance**, with attention directed to the
self as the identified mechanism. Feedback is not a monotone good.

This has a direct consequence for reward design. A reward built only from
*withholding* measures has its optimum at maximal withholding — which is the
*d* = 0.05 end of that table. This repository built one and the policy found it.
Elaborated feedback is therefore the atlas's primary **contribution** measure
and the required pairing partner for anything that pays for restraint.

One guard applies to the construct itself: token overlap with the learner's
message predicts human uptake labels at *r* = 0.52, so a specificity measure that
can be satisfied by quoting the learner is a parroting reward. And a weak
correctness signal in a reward plausibly teaches confident-sounding rather than
correct, so the accuracy check must be against a supplied reference solution
rather than a re-derivation.

### Answer withholding (grade B — and explicitly not monotone)

Assistance level is a real design variable with a documented trade-off rather
than an optimum: the **assistance dilemma** states plainly that both giving and
withholding information have costs and that the optimum is interior and
learner-dependent (Koedinger & Aleven, 2007). Help availability changes learner
strategy rather than simply helping (Aleven et al., 2003; Aleven et al., 2016),
and learners game hint systems to extract answers (Baker et al., 2004).

Two measurement notes, both learned expensively here.

**Length is the dominant confound.** Agent raters tied leakage to length at
*r* = 0.59 against a human's 0.43, and got a policy comparison backwards as a
result — ranking an arm below another that the human preferred 29–11.

**A confidently wrong answer is the worst case, not the middle one.** An earlier
version of this scale anchored 1 at "never points at one option" and 3 at
"states the *correct* option", so a turn stating a *wrong* option satisfied
neither and fell to 2 by default. Pedagogically that case both removes the work
and misleads. The scale must ask how much of the learner's remaining work is
taken away, independent of whether the answer given is right.

This construct is `not_eligible_degenerate` for reward. The evidence for the
degeneracy is internal and specific: five withholding-flavored dimensions had an
effective rank of 2.9, the nine turns that maxed four of them at once had a
median length of **12 words** against a corpus median of 30, and a trained
policy converged there.

### Contingent scaffolding (grade B)

Contingency — matching support to what the learner has demonstrated — is the
defining property of scaffolding (Wood et al., 1976; van de Pol et al., 2010),
with classroom evidence separating it from independent working time (van de Pol
et al., 2015). VanLehn (2011) is the useful corrective on magnitude: differences
between tutoring conditions are smaller than commonly assumed.

This is the most valuable construct in the atlas for one structural reason: **it
is the only one whose optimum a fixed policy cannot reach by style alone**,
because the target moves with the learner. It is also the one most easily
ruined, because scoring it requires a target and there is no reliable way to
recover the target from intuition — three tutors pick the same action 18% of the
time. The target must be *prescribed by the scenario*. Where a scenario does not
prescribe one, the item is not applicable, not a guess.

Conveniently, contingency needs no new tutor-side code: the help-level scale is
the same one `answer_withholding_level` already uses. It needs the *learner*-side
code, and is then a lookup over (learner state, help level).

### Expertise reversal (grade B)

Techniques that help novices can harm more knowledgeable learners; the sign
genuinely flips (Kalyuga et al., 2003; Kalyuga, 2007; Sweller et al., 2019).
This is what makes "more explanation is better" false, and it is the natural
counterweight to a contribution-only reward.

It requires a *declared* expertise level. Inferring expertise from a learner's
vocabulary or tone is precisely the latent inference this project excludes
(§8), so a missing declaration makes the item not applicable.

---

## 7. Family 5 — Metacognitive and motivational framing

This family has the weakest evidence and the worst labelability in the atlas,
and it is included largely so that both facts are measured rather than assumed.

### Metacognitive regulation (grade B)

Strategy and self-regulation instruction show positive effects (Dignath &
Büttner, 2008; Donker et al., 2014), with the important qualification that
long-term follow-up effects are **considerably smaller** than immediate ones
(de Boer et al., 2018). Effects are larger when embedded in domain content than
taught as a generic study skill — which is also the labeling rule: a prompt must
name its object. "Make sure you really understand this" has no object, no
criterion, and no producible response.

### Calibration (grade B, diagnostic before therapeutic)

Overconfidence is well documented and predicts premature termination of study
(Dunlosky & Rawson, 2012; Bjork et al., 2013), and delaying judgments of
learning substantially improves their accuracy (Rhodes & Tauber, 2011). What is
*weak* is the therapeutic claim: reactivity effects of merely making judgments
are small (Double et al., 2018).

So calibration elicitation is worth collecting and is not worth rewarding.
A single confidence statement is not calibration; calibration is a relationship
between judgments and outcomes across many items, which is a level-3 statistic.

### Attribution (grade D)

The theory is well developed (Weiner, 1985) and one half of the claim is
meta-analytically solid: feedback directed at the self is the case associated
with *reduced* performance (Kluger & DeNisi, 1996; Hattie & Timperley, 2007).

The other half — that re-framing attributions in a tutoring turn improves
learning — rests on evidence that is small, old, heterogeneous, and adjacent to
the mindset literature whose highest-quality subset shows ~zero (§9). Expected
labeling agreement is 0.30–0.50, near the neighborhood where tone codes live.

It is retained for one specific job: **drift monitoring**. A reward trained on
other constructs may converge toward trait praise, because praise is cheap and
correlates with everything warm. A rising trait-praise rate is an alarm. It is
never a target.

### Utility value (grade C)

Real randomized field effects, but concentrated: learner-generated relevance
writing improved outcomes for students with low success expectations (Hulleman &
Harackiewicz, 2009) and narrowed gaps for first-generation underrepresented
students (Harackiewicz et al., 2016), with mixed results in follow-up work
(Rosenzweig et al., 2020).

Two constraints. **Learner generation is the manipulation** — telling learners
why material is useful performed worse and can backfire for low-confidence
learners. And the intervention was administered a small number of times; a
policy that appends a relevance prompt to every turn is not implementing it.

### Autonomy support (grade B)

The behavioral cluster — rationale, bounded choice, informational language,
perspective-taking — is learnable and produces motivational gains (Su & Reeve,
2011; Reeve & Cheon, 2021), with choice effects meta-analyzed separately
(Patall et al., 2008). Effects on *achievement* are smaller and more indirect
than effects on motivation.

Friendly tone alone is not autonomy support, and the two are separable. Coded as
a global impression the construct converges with tone (α = 0.24–0.30); coded as
component behaviors it stays pointable. It is nonetheless
`not_eligible_degenerate` for reward, because at the surface it is nearly
collinear with politeness, and rewarding politeness means rewarding praise.

### Belonging (grade C)

Walton & Cohen (2011) is a genuine result with three-year follow-up. The large
multi-site test (Walton et al., 2023) is the reason the grade is C rather than
B: its central finding is that effects are **concentrated in specific
institutional contexts and groups** rather than general. A null in an
unsupportive context is the expected result, not a failed replication.

The delivery vehicle also matters. The original was a single early
administration with a saying-is-believing component. A tutoring turn is not that
vehicle and inherits little of the evidence. What is labeled here is only the
*framing*; belonging itself is a learner state and is excluded (§8).

---

## 8. What is excluded, and why

Eleven constructs are excluded from labeling despite mattering instructionally.
Each is excluded for a recorded reason, so nobody re-proposes it on the theory
that the wording just needed work. The reasons are of two kinds, and the
difference matters.

**Six were measured and failed** — motivation, correct-strategy, tone, holistic
quality, uptake, and could-be-misled. A better question could in principle
rescue these, and the number says how far it would have to come.

**Five are structurally impossible** — cognitive load, understanding or mastery,
desirable difficulty, retention and transfer, and realized engagement. Nothing
was measured on an artifact because the thing named is not a property of an
artifact. That is the stronger reason rather than the weaker one: no rewording
reaches it, and each entry names the different data that would be required
instead.

Full list in `ontology.yaml: excluded_latent_constructs`; the load-bearing cases:

- **Holistic teaching quality** — 39% exact agreement, *r* = 0.15. Collected in
  the protocol as a diagnostic covariate only. A probe once reached 0.581
  predicting it, which looked like a representation failure until the label
  noise was measured; the ceiling was set by the question, not the method.
- **Learner internal states** (motivation, interest, goals) — α = 0.03, 0.02,
  0.07. Below chance, replicated across two model conditions.
- **"Which strategy should the tutor have used"** — 18% three-way agreement.
  Included only in the prescribed-target form, as contingency.
- **Uptake / revoicing** — Fleiss κ = 0.286, and a single overlap feature
  predicts it at *r* = 0.52, so as a reward target it is gamed by parroting.
- **"Could a student take away anything wrong"** — κ = 0.18 against a human
  while agents agreed with each other at 0.41. Rewriting moved their behavior
  a long way and their agreement with the human hardly at all, which is the
  signature of a question that cannot be transferred rather than one that was
  badly worded. Replaced by the reference-checkable form.
- **Tone / warmth** — α = 0.24–0.30, and rewarding it means rewarding praise.
- **Learner cognitive load** — a property of the learner. Observable *design
  risks* (redundancy, split attention, missing signaling) are codable; the load
  is not.
- **Desirable difficulty** — definitionally requires an immediate cost paired
  with a delayed gain, so it cannot exist in an artifact. Frustration,
  obscurity, and missing prerequisites are not desirable difficulties.
- **Retention and transfer** — level 4. Nothing in an artifact bears on them.

The pattern is consistent: what fails is anything requiring a model of the
learner's mind, and anything requiring a counterfactual. What survives is what
can be checked against the text or against a supplied reference.

---

## 9. Debunked and contested controls

Six items are in the pool as **negative controls**. They should attract zero
presence labels on every construct. If a human or an agent rates them as
pedagogically strong, that is a measurable defect in the instrument, which is
the point of including them.

**Learning-style meshing.** The hypothesis requires a crossover interaction
between assessed style and instructional modality. Studies using that design do
not find it (Pashler et al., 2008; Rogowsky et al., 2015), and the belief
persists anyway (Kirschner, 2017) — which is why it must be tested rather than
assumed absent.

**Generic growth-mindset language.** The most instructive contested case in the
review, and the one where taking a side would be easiest and wrong. Two
meta-analyses of the same literature reached opposing conclusions in the same
issue of the same journal:

- Macnamara & Burgoyne (2023): 63 studies, *N* = 97,672. Overall *d* = 0.05
  [0.02, 0.09], **nonsignificant after publication-bias correction**. Highest-
  quality subset (6 studies, *N* = 13,571): *d* = 0.02 [−0.06, 0.10]. Authors
  with a financial incentive published significantly larger effects.
- Burnette et al. (2023): 53 independent samples, heterogeneity-attuned
  multilevel meta-regression. For targeted subsamples with high implementation
  fidelity, academic achievement *d* = 0.14 [0.06, 0.22] — **with a 95%
  prediction interval of −0.08 to 0.35.**

The dispute is methodological and unresolved (see the commentary exchange).
Note what the two agree on: even the favorable synthesis predicts that a new
intervention may do nothing. And both concern *interventions*; detached "your
brain is a muscle" phrasing attached to no task carries none of whatever effect
exists. Yeager et al. (2019) is the best single trial and finds small effects
concentrated in lower-achieving students in supportive school norms.

**Learning-pyramid retention percentages.** "People remember 10% of what they
read, 90% of what they teach" traces to no primary study; the numbers are not
empirical (Letrud & Hernes, 2018). An artifact asserting them has made a factual
error independent of its pedagogy.

**Unguided discovery.** Unassisted discovery does not outperform explicit
instruction; *enhanced* discovery with guidance and feedback does (Alfieri
et al., 2011; Kirschner et al., 2006). This control exists specifically to
separate problem-solving-before-instruction, which requires consolidating
instruction, from the unguided version sharing its surface form.

**Praise and warmth as stand-alone quality.** Over a third of feedback
interventions reduced performance, with self-directed attention as the mechanism
(Kluger & DeNisi, 1996). Used here as a drift alarm.

**Felt learning and fluency.** Inverted indicators. Active-learning students
learned more and felt they learned less (Deslauriers et al., 2019); learning and
performance dissociate systematically (Soderstrom & Bjork, 2015).

---

## 10. Reward eligibility policy

Twelve of 24 constructs are eligible for a reward, all with named guards. The
rest are blocked, each for one of four recorded reasons:

| Status | Count | Reason |
|---|---|---|
| `eligible_with_guard` | 12 | Enacted, structurally checkable, adequate evidence — with a specific named guard |
| `not_eligible_wrong_level` | 4 | Content lives above level 2; a turn cannot reach it |
| `measurement_only` | 4 | Covariate, negative control, or descriptor |
| `not_eligible_degenerate` | 2 | Has a reachable degenerate optimum an optimizer will find |
| `not_eligible_unreliable` | 2 | Expected agreement below the 0.40 floor |

Eligibility here means "may be considered after it passes its reliability gate",
not "approved". Several of the twelve are expected to fail the gate in
`LABELING_GUIDE.md`, and the gate outranks this table.

Three policies govern the eligible twelve.

**1. No withholding measure may be rewarded without a paired contribution
measure.** This is the central lesson from this repository's prior work and it
is not a style preference. Withholding-only rubrics are maximized by silence;
against 70 effect sizes, bare right/wrong feedback is *d* = 0.05 and elaborated
feedback is *d* = 0.49. A reward made of restraint measures optimizes toward the
0.05 end. The pairing partners available here are
`elaborated_feedback_specificity`, `subgoal_labeled_structure`, and
`completion_problem_scaffold` — the last being unusually robust because it is
simultaneously a contribution and a withholding measure.

**2. Length must be a measured covariate for every eligible construct.** It is
the top-listed surface confound for the majority of them, it is what agent
raters latch onto (*r* = 0.59 vs a human's 0.43), and it is separately known to
predict human preference better than any rated dimension — which means a rubric
that ignores it is being partly scored by it anyway.

**3. Correctness is checked against a supplied reference, never re-derived.**
The re-derivation form of this question failed its agreement gate at κ = 0.18.
The reference-checking form passed.

---

## 11. What this project may and may not claim

**May claim,** on completing the labeling protocol: that specified constructs
can be labeled in artifacts at a measured reliability; that a given corpus or
model enacts them at a measured rate; that enactment dissociates from
description, quantified by the says-only false-positive rate.

**May not claim,** on any evidence this project can produce: that any artifact
improved learning; that a higher atlas score means better teaching; that
constructs compose additively — the worked-example / self-explanation
interaction is a direct counterexample; that an agent panel's agreement with
itself licenses using its labels. Six models can agree closely because they
share training data and a house style. That is consistency, not accuracy, and
the case where agents agree with each other but not with the human is the
dangerous one: it looks reliable and is measuring something else.

---

## Bibliography

All entries are primary sources or meta-analyses. DOIs resolve; the ontology
carries the same list machine-readably with per-claim attribution.

**Practice scheduling and generation**

- Adesope, O. O., Trevisan, D. A., & Sundararajan, N. (2017). Rethinking the use of tests. *Review of Educational Research*. https://doi.org/10.3102/0034654316689306
- Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of interleaved learning and its moderators. *Psychological Bulletin, 145*(11), 1029–1052. https://doi.org/10.1037/bul0000209
- Carpenter, S. K., Pan, S. C., & Butler, A. C. (2022). The science of effective learning with spacing and retrieval practice. *Nature Reviews Psychology, 1*, 496–511. https://doi.org/10.1038/s44159-022-00089-1
- Carpenter, S. K., & Toftness, A. R. (2017). The effect of prequestions on learning from video presentations. *Journal of Applied Research in Memory and Cognition, 6*(1), 104–109. https://doi.org/10.1016/j.jarmac.2016.07.014
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks. *Psychological Bulletin, 132*(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354
- Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques. *Psychological Science in the Public Interest, 14*(1), 4–58. https://doi.org/10.1177/1529100612453266
- Karpicke, J. D., & Roediger, H. L. (2008). The critical importance of retrieval for learning. *Science, 319*(5865), 966–968. https://doi.org/10.1126/science.1152408
- Kornell, N., Hays, M. J., & Bjork, R. A. (2009). Unsuccessful retrieval attempts enhance subsequent learning. *JEP: LMC, 35*(4), 989–998. https://doi.org/10.1037/a0015729
- Latimier, A., Peyre, H., & Ramus, F. (2021). A meta-analytic review of the benefit of spacing out retrieval practice episodes. *Educational Psychology Review, 33*, 959–987. https://doi.org/10.1007/s10648-020-09572-8
- Pan, S. C., & Sana, F. (2021). Pretesting versus posttesting. *JEP: Applied, 27*(2), 237–257. https://doi.org/10.1037/xap0000345
- Rawson, K. A., & Dunlosky, J. (2011). Optimizing schedules of retrieval practice. *JEP: General, 140*(3), 283–302. https://doi.org/10.1037/a0023956
- Richland, L. E., Kornell, N., & Kao, L. S. (2009). The pretesting effect. *JEP: Applied, 15*(3), 243–257. https://doi.org/10.1037/a0016496
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning. *Psychological Science, 17*(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x
- Rohrer, D., Dedrick, R. F., & Stershic, S. (2015). Interleaved practice improves mathematics learning. *Journal of Educational Psychology, 107*(3), 900–908. https://doi.org/10.1037/edu0000001
- Rowland, C. A. (2014). The effect of testing versus restudy on retention. *Psychological Bulletin, 140*(6), 1432–1463. https://doi.org/10.1037/a0037559
- Taylor, K., & Rohrer, D. (2010). The effects of interleaved practice. *Applied Cognitive Psychology, 24*(6), 837–848. https://doi.org/10.1002/acp.1598
- Yang, C., Luo, L., Vadillo, M. A., Yu, R., & Shanks, D. R. (2021). Testing (quizzing) boosts classroom learning. *Psychological Bulletin, 147*(4), 399–435. https://doi.org/10.1037/bul0000309

**Example-based instruction and problem-first design**

- Alfieri, L., Brooks, P. J., Aldrich, N. J., & Tenenbaum, H. R. (2011). Does discovery-based instruction enhance learning? *Journal of Educational Psychology, 103*(1), 1–18. https://doi.org/10.1037/a0021017
- Alfieri, L., Nokes-Malach, T. J., & Schunn, C. D. (2013). Learning through case comparisons: A meta-analytic review. *Educational Psychologist, 48*(2), 87–113. https://doi.org/10.1080/00461520.2013.775712
- Ashman, G., Kalyuga, S., & Sweller, J. (2020). Problem-solving or explicit instruction: Which should go first when element interactivity is high? *Educational Psychology Review, 32*, 229–247. https://doi.org/10.1007/s10648-019-09500-5
- Atkinson, R. K., Derry, S. J., Renkl, A., & Wortham, D. (2000). Learning from examples. *Review of Educational Research, 70*(2), 181–214. https://doi.org/10.3102/00346543070002181
- Atkinson, R. K., Renkl, A., & Merrill, M. M. (2003). Transitioning from studying examples to solving problems. *Journal of Educational Psychology, 95*(4), 774–783. https://doi.org/10.1037/0022-0663.95.4.774
- Barbieri, C. A., Miller-Cotto, D., Clerjuste, S. N., & Chawla, K. (2023). A meta-analysis of the worked examples effect on mathematics performance. *Educational Psychology Review, 35*, 11. https://doi.org/10.1007/s10648-023-09745-1
- Catrambone, R. (1998). The subgoal learning model. *JEP: General, 127*(4), 355–376. https://doi.org/10.1037/0096-3445.127.4.355
- Gentner, D., Loewenstein, J., & Thompson, L. (2003). Learning and transfer: A general role for analogical encoding. *Journal of Educational Psychology, 95*(2), 393–408. https://doi.org/10.1037/0022-0663.95.2.393
- Kapur, M. (2008). Productive failure. *Cognition and Instruction, 26*(3), 379–424. https://doi.org/10.1080/07370000802212669
- Kirschner, P. A., Sweller, J., & Clark, R. E. (2006). Why minimal guidance during instruction does not work. *Educational Psychologist, 41*(2), 75–86. https://doi.org/10.1207/s15326985ep4102_1
- Loibl, K., Roll, I., & Rummel, N. (2017). Towards a theory of when and how problem solving followed by instruction supports learning. *Educational Psychology Review, 29*, 693–715. https://doi.org/10.1007/s10648-016-9379-x
- Margulieux, L. E., & Catrambone, R. (2016). Improving problem solving with subgoal labels. *Learning and Instruction, 42*, 58–71. https://doi.org/10.1016/j.learninstruc.2015.12.002
- Paas, F. (1992). Training strategies for attaining transfer of problem-solving skill in statistics. *Journal of Educational Psychology, 84*(4), 429–434. https://doi.org/10.1037/0022-0663.84.4.429
- Renkl, A. (2014). Toward an instructionally oriented theory of example-based learning. *Cognitive Science, 38*(1), 1–37. https://doi.org/10.1111/cogs.12086
- Renkl, A., Atkinson, R. K., Maier, U. H., & Staley, R. (2002). From example study to problem solving. *Journal of Experimental Education, 70*(4), 293–315. https://doi.org/10.1080/00220970209599510
- Sinha, T., & Kapur, M. (2021). When problem solving followed by instruction works. *Review of Educational Research, 91*(5), 761–798. https://doi.org/10.3102/00346543211019105
- Sweller, J., & Cooper, G. A. (1985). The use of worked examples as a substitute for problem solving. *Cognition and Instruction, 2*(1), 59–89. https://doi.org/10.1207/s1532690xci0201_3
- van Merriënboer, J. J. G., & Krammer, H. P. M. (1987). Instructional strategies and tactics for the design of introductory computer programming courses. *Instructional Science, 16*, 251–285. https://doi.org/10.1007/BF00120253
- van Merriënboer, J. J. G., Kirschner, P. A., & Kester, L. (2003). Taking the load off a learner's mind. *Educational Psychologist, 38*(1), 5–13. https://doi.org/10.1207/S15326985EP3801_2

**Generative explanation**

- Bisra, K., Liu, Q., Nesbit, J. C., Salimi, F., & Winne, P. H. (2018). Inducing self-explanation: A meta-analysis. *Educational Psychology Review, 30*, 703–725. https://doi.org/10.1007/s10648-018-9434-x
- Chase, C. C., Chin, D. B., Oppezzo, M. A., & Schwartz, D. L. (2009). Teachable agents and the protégé effect. *Journal of Science Education and Technology, 18*, 334–352. https://doi.org/10.1007/s10956-009-9180-4
- Chi, M. T. H., Bassok, M., Lewis, M. W., Reimann, P., & Glaser, R. (1989). Self-explanations. *Cognitive Science, 13*(2), 145–182. https://doi.org/10.1207/s15516709cog1302_1
- Chi, M. T. H., Adams, J., Bogusch, E. B., Bruchok, C., Kang, S., et al. (2018). Translating the ICAP theory of cognitive engagement into practice. *Cognitive Science, 42*(6), 1777–1832. https://doi.org/10.1111/cogs.12626
- Chi, M. T. H., & Wylie, R. (2014). The ICAP framework. *Educational Psychologist, 49*(4), 219–243. https://doi.org/10.1080/00461520.2014.965823
- Fiorella, L., & Mayer, R. E. (2013). The relative benefits of learning by teaching and teaching expectancy. *Contemporary Educational Psychology, 38*(4), 281–288. https://doi.org/10.1016/j.cedpsych.2013.06.001
- Kobayashi, K. (2019). Learning by preparing-to-teach and teaching: A meta-analysis. *Japanese Psychological Research, 61*(3), 192–203. https://doi.org/10.1111/jpr.12221
- Rittle-Johnson, B., & Loehr, A. M. (2017). Eliciting explanations: Constraints on when self-explanation aids learning. *Psychonomic Bulletin & Review, 24*, 1501–1510. https://doi.org/10.3758/s13423-016-1079-5
- Roscoe, R. D., & Chi, M. T. H. (2007). Understanding tutor learning. *Review of Educational Research, 77*(4), 534–574. https://doi.org/10.3102/0034654307309920

**Feedback and assistance**

- Aleven, V., Stahl, E., Schworm, S., Fischer, F., & Wallace, R. (2003). Help seeking and help design in interactive learning environments. *Review of Educational Research, 73*(3), 277–320. https://doi.org/10.3102/00346543073003277
- Aleven, V., Roll, I., McLaren, B. M., & Koedinger, K. R. (2016). Help helps, but only so much. *IJAIED, 26*, 205–223. https://doi.org/10.1007/s40593-015-0089-1
- Baker, R. S., Corbett, A. T., Koedinger, K. R., & Wagner, A. Z. (2004). Off-task behavior in the Cognitive Tutor classroom. *CHI '04*. https://doi.org/10.1145/985692.985741
- Hattie, J., & Timperley, H. (2007). The power of feedback. *Review of Educational Research, 77*(1), 81–112. https://doi.org/10.3102/003465430298487
- Kalyuga, S. (2007). Expertise reversal effect and its implications for learner-tailored instruction. *Educational Psychology Review, 19*, 509–539. https://doi.org/10.1007/s10648-007-9054-3
- Kalyuga, S., Ayres, P., Chandler, P., & Sweller, J. (2003). The expertise reversal effect. *Educational Psychologist, 38*(1), 23–31. https://doi.org/10.1207/S15326985EP3801_4
- Kluger, A. N., & DeNisi, A. (1996). The effects of feedback interventions on performance. *Psychological Bulletin, 119*(2), 254–284. https://doi.org/10.1037/0033-2909.119.2.254
- Koedinger, K. R., & Aleven, V. (2007). Exploring the assistance dilemma in experiments with Cognitive Tutors. *Educational Psychology Review, 19*, 239–264. https://doi.org/10.1007/s10648-007-9049-0
- Shute, V. J. (2008). Focus on formative feedback. *Review of Educational Research, 78*(1), 153–189. https://doi.org/10.3102/0034654307313795
- Sweller, J., van Merriënboer, J. J. G., & Paas, F. (2019). Cognitive architecture and instructional design: 20 years later. *Educational Psychology Review, 31*, 261–292. https://doi.org/10.1007/s10648-019-09465-5
- Van der Kleij, F. M., Feskens, R. C. W., & Eggen, T. J. H. M. (2015). Effects of feedback in a computer-based learning environment on students' learning outcomes. *Review of Educational Research, 85*(4), 475–511. https://doi.org/10.3102/0034654314564881
- van de Pol, J., Volman, M., & Beishuizen, J. (2010). Scaffolding in teacher–student interaction. *Educational Psychology Review, 22*, 271–296. https://doi.org/10.1007/s10648-010-9127-6
- van de Pol, J., Volman, M., Oort, F., & Beishuizen, J. (2015). The effects of scaffolding in the classroom. *Instructional Science, 43*, 615–641. https://doi.org/10.1007/s11251-015-9351-z
- VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist, 46*(4), 197–221. https://doi.org/10.1080/00461520.2011.611369
- Wisniewski, B., Zierer, K., & Hattie, J. (2020). The power of feedback revisited. *Frontiers in Psychology, 10*, 3087. https://doi.org/10.3389/fpsyg.2019.03087
- Wood, D., Bruner, J. S., & Ross, G. (1976). The role of tutoring in problem solving. *Journal of Child Psychology and Psychiatry, 17*(2), 89–100. https://doi.org/10.1111/j.1469-7610.1976.tb00381.x

**Metacognitive and motivational**

- Bjork, R. A., Dunlosky, J., & Kornell, N. (2013). Self-regulated learning: Beliefs, techniques, and illusions. *Annual Review of Psychology, 64*, 417–444. https://doi.org/10.1146/annurev-psych-113011-143823
- de Boer, H., Donker, A. S., Kostons, D. D. N. M., & van der Werf, G. P. C. (2018). Long-term effects of metacognitive strategy instruction. *Educational Research Review, 24*, 98–115. https://doi.org/10.1016/j.edurev.2018.03.002
- Dignath, C., & Büttner, G. (2008). Components of fostering self-regulated learning among students. *Metacognition and Learning, 3*, 231–264. https://doi.org/10.1007/s11409-008-9029-x
- Donker, A. S., de Boer, H., Kostons, D., Dignath van Ewijk, C. C., & van der Werf, M. P. C. (2014). Effectiveness of learning strategy instruction. *Educational Research Review, 11*, 1–26. https://doi.org/10.1016/j.edurev.2013.11.002
- Double, K. S., Birney, D. P., & Walker, S. A. (2018). A meta-analysis and systematic review of reactivity to judgements of learning. *Memory, 26*(6), 741–750. https://doi.org/10.1080/09658211.2017.1404111
- Dunlosky, J., & Rawson, K. A. (2012). Overconfidence produces underachievement. *Learning and Instruction, 22*(4), 271–280. https://doi.org/10.1016/j.learninstruc.2011.08.003
- Harackiewicz, J. M., Canning, E. A., Tibbetts, Y., Priniski, S. J., & Hyde, J. S. (2016). Closing achievement gaps with a utility-value intervention. *JPSP, 111*(5), 745–765. https://doi.org/10.1037/pspp0000075
- Hulleman, C. S., & Harackiewicz, J. M. (2009). Promoting interest and performance in high school science classes. *Science, 326*(5958), 1410–1412. https://doi.org/10.1126/science.1177067
- Patall, E. A., Cooper, H., & Robinson, J. C. (2008). The effects of choice on intrinsic motivation. *Psychological Bulletin, 134*(2), 270–300. https://doi.org/10.1037/0033-2909.134.2.270
- Reeve, J., & Cheon, S. H. (2021). Autonomy-supportive teaching. *Educational Psychologist, 56*(1), 54–77. https://doi.org/10.1080/00461520.2020.1862657
- Rhodes, M. G., & Tauber, S. K. (2011). The influence of delaying judgments of learning on metacognitive accuracy. *Psychological Bulletin, 137*(1), 131–148. https://doi.org/10.1037/a0021705
- Rosenzweig, E. Q., Wigfield, A., & Hulleman, C. S. (2020). More useful or not so bad? *Journal of Educational Psychology, 112*(1), 166–182. https://doi.org/10.1037/edu0000370
- Su, Y.-L., & Reeve, J. (2011). A meta-analysis of the effectiveness of intervention programs designed to support autonomy. *Educational Psychology Review, 23*, 159–188. https://doi.org/10.1007/s10648-010-9142-7
- Walton, G. M., & Cohen, G. L. (2011). A brief social-belonging intervention improves academic and health outcomes of minority students. *Science, 331*(6023), 1447–1451. https://doi.org/10.1126/science.1198364
- Walton, G. M., Murphy, M. C., Logel, C., Yeager, D. S., et al. (2023). Where and with whom does a brief social-belonging intervention promote progress in college? *Science, 380*(6644), 499–505. https://doi.org/10.1126/science.ade4420
- Weiner, B. (1985). An attributional theory of achievement motivation and emotion. *Psychological Review, 92*(4), 548–573. https://doi.org/10.1037/0033-295X.92.4.548

**Contested and debunked**

- Burnette, J. L., Billingsley, J., Banks, G. C., Knouse, L. E., Hoyt, C. L., Pollack, J. M., & Simon, S. (2023). A systematic review and meta-analysis of growth mindset interventions. *Psychological Bulletin, 149*(3–4), 174–205. https://doi.org/10.1037/bul0000368
- Deslauriers, L., McCarty, L. S., Miller, K., Callaghan, K., & Kestin, G. (2019). Measuring actual learning versus feeling of learning. *PNAS, 116*(39), 19251–19257. https://doi.org/10.1073/pnas.1821936116
- Kirschner, P. A. (2017). Stop propagating the learning styles myth. *Computers & Education, 106*, 166–171. https://doi.org/10.1016/j.compedu.2016.12.006
- Letrud, K., & Hernes, S. (2018). Excavating the origins of the learning pyramid myths. *Cogent Education, 5*(1). https://doi.org/10.1080/2331186X.2018.1518638
- Macnamara, B. N., & Burgoyne, A. P. (2023). Do growth mindset interventions impact students' academic achievement? *Psychological Bulletin, 149*(3–4), 133–173. https://doi.org/10.1037/bul0000352
- Pashler, H., McDaniel, M., Rohrer, D., & Bjork, R. (2008). Learning styles: Concepts and evidence. *Psychological Science in the Public Interest, 9*(3), 105–119. https://doi.org/10.1111/j.1539-6053.2009.01038.x
- Rogowsky, B. A., Calhoun, B. M., & Tallal, P. (2015). Matching learning style to instructional method. *Journal of Educational Psychology, 107*(1), 64–78. https://doi.org/10.1037/a0037478
- Sisk, V. F., Burgoyne, A. P., Sun, J., Butler, J. L., & Macnamara, B. N. (2018). To what extent and under which circumstances are growth mind-sets important to academic achievement? *Psychological Science, 29*(4), 549–571. https://doi.org/10.1177/0956797617739704
- Soderstrom, N. C., & Bjork, R. A. (2015). Learning versus performance: An integrative review. *Perspectives on Psychological Science, 10*(2), 176–199. https://doi.org/10.1177/1745691615569000
- Yeager, D. S., Hanselman, P., Walton, G. M., Murray, J. S., et al. (2019). A national experiment reveals where a growth mindset improves achievement. *Nature, 573*, 364–369. https://doi.org/10.1038/s41586-019-1466-y

**Annotation reliability**

Reliability figures cited in §2 without a DOI are internal measurements from
`projects/pedagogy_rm` (recorded in its `rubric.py`, `agreement.py`, and
`FLAWS.md`) and from `projects/pedagogy_mech_interp/METRIC_TAXONOMY.md`. The
crowdworker-versus-in-house comparison (κ 0.13–0.30 against 0.65–0.71) is the
contrast between Daheim et al. (2024) and Maurya et al. (2025) as recorded
there.
