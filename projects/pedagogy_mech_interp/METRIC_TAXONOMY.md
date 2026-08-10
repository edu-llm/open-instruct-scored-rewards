# Learning-science metric taxonomy

This project tests whether language models represent and causally use concrete
pedagogical decisions. It does not treat a response that *looks educational* as
evidence that a student learned.

Every measure belongs to one of three levels:

1. **Tutor opportunity**: an observable action in a tutor turn.
2. **Learner enactment**: what the learner does in response.
3. **Learning outcome**: durable, independent change measured later.

A single tutor response can usually measure only the first level. It cannot
establish engagement, cognitive load, calibration, mastery, retention, or
transfer.

## Pilot constructs

### Diagnostic localization

- **Claim measured:** the tutor identifies the first causal error, faulty rule,
  or misconception in the learner's work rather than responding generically.
- **Unit:** learner attempt plus tutor response.
- **Primary score:** `diagnostic_localization` in `{0, 1}`.
- **Positive anchor:** names the specific erroneous step or proposition and
  describes its conflict with the reference reasoning.
- **Negative anchor:** generic feedback, diagnosis from the final answer only,
  an invented learner intention, or correction of a downstream symptom.
- **Required data:** step-level reference solution, acceptable alternative
  strategies, and an annotated first causal error.
- **Related V2 measures:** `locates`, `verdict`, and `correct`.
- **Evidence:** feedback is most useful when it closes a specific gap between a
  goal and the learner's current state (Hattie & Timperley, 2007; Shute, 2008).
  Model tracing work should not infer the learner's misconception from answer
  correctness alone.

### Contingent scaffolding

- **Claim measured:** the amount and form of help match what the learner has
  demonstrated.
- **Unit:** at least a learner turn and tutor response; fading requires a
  trajectory.
- **Help levels:** `question`, `strategy_hint`, `worked_step`, `direct_answer`.
- **Student states:** `no_attempt`, `attempt`, `reasoned_attempt`.
- **Primary score:** whether the prescribed and selected help levels match.
  The target is scenario-specific; lower leakage is not always better.
- **Response-level components:** useful guidance, a named next move, factual
  correctness, and work left for the learner.
- **Trajectory components:** escalation after failure, fading after success,
  and increasing learner-owned steps.
- **Required data:** learner state, prescribed action, response action, and
  subsequent attempt when enactment is evaluated.
- **Invalid shortcut:** maximizing `hands_over - leak` for every learner. A
  novice with no foothold can need more explicit support than a learner who has
  already shown the relevant reasoning.
- **Evidence:** van de Pol et al. (2010); Kalyuga (2007).

### Generative elicitation

- **Claim measured:** the tutor creates an opportunity for the learner to
  retrieve, justify, self-explain, compare, or produce the next step rather
  than performing that work for them.
- **Unit:** tutor turn for opportunity; tutor/learner pair for enactment.
- **Primary opportunity score:** `generative_elicitation` in `{0, 1}`.
- **Positive anchor:** the requested response requires information beyond what
  the tutor has supplied and names the object of the learner's work.
- **Negative anchor:** rhetorical questions, yes/no checks, copying, generic
  "explain your answer", or the tutor supplying the explanation before asking.
- **Enactment score:** number and correctness of learner-generated conceptual
  links or independently retrieved target facts.
- **Required data:** target knowledge component and a rubric for acceptable
  inferences.
- **Evidence:** self-explanation meta-analysis (Bisra et al., 2018), retrieval
  practice meta-analyses (Rowland, 2014; Adesope et al., 2017), and ICAP
  (Chi & Wylie, 2014).

## Tier 1: response-level tutor opportunities

These can be scored from a response when the learner context and reference are
available. They remain behavior measures, not learning outcomes.

### Feedback gap closure

Score separately whether a response identifies the goal, accurately describes
the current state, explains the gap, and gives an executable next step. Do not
reward unsupported praise or an answer without diagnosis.

### Misconception refutation

Record whether the response explicitly distinguishes the misconception from a
correct alternative, gives a causal explanation or discriminating example, and
asks the learner to re-predict or reapply. Exposure to a refutation is not
evidence of repair.

### Retrieval-practice opportunity

Count an opportunity only when the answer remains hidden and the learner must
produce the target from memory. Recognition, copying, and rhetorical questions
do not count. Corrective feedback and delayed retention are separate measures.

### Constructive activity opportunity

Classify the requested action as passive, active, constructive, or interactive.
Report categories rather than averaging them as an interval scale. An
invitation is not realized engagement.

### Autonomy-supportive behavior

Code perspective-taking, meaningful rationale, bounded choice, informational
language, and responsiveness separately from commands, coercion, guilt, or
answer takeover. Friendly tone alone is not autonomy support.

### Cognitive-load design risks

Observable risks include irrelevant detail, redundancy, unintegrated
references, too many simultaneously introduced elements, missing signaling,
and missing prerequisite support. Token count and readability are not measures
of the learner's cognitive load.

## Tier 2: trajectory/process measures

These require multiple turns or attempts.

### Complete formative-assessment cycles

Measure complete `Elicit -> Student response -> Recognize -> Use` cycles and
whether the tutor's use is contingent on the evidence. Asking a question
without adapting is not formative assessment.

### Feedback uptake and repair closure

Measure addressed errors that the learner correctly revises, followed by an
unprompted isomorphic item. Separate parroting, same-item correction, and
independent repair.

### Scaffolding escalation and fading

Track help level through a trajectory. Appropriate behavior escalates after
failed attempts and fades after demonstrated success; fewer hints are not
inherently better.

### Spaced successive relearning

For each target, count successful effortful recalls across distinct sessions
and report lag relative to the desired retention interval. Turns in one
conversation are not spacing.

### Interleaved discriminative practice

Measure practice transitions among confusable categories plus opportunities to
choose and justify the applicable procedure. Randomly mixing unrelated topics
does not count.

### Metacognitive calibration and regulation

Collect confidence before feedback, then report calibration error or Brier
score and whether later study/help choices target weak items. A single
confidence statement cannot measure calibration.

### Knowledge-state prediction and adaptation

Evaluate mastery predictions with held-out log loss/Brier score and calibration
by skill. Separately measure whether task and hint selection actually respond
to the estimate. Predictive accuracy does not prove instructional benefit.

## Tier 3: learning outcomes

These require independent assessments and are outside the frozen-model pilot.

### Retention

Use delayed, independently scored retrieval with no tutor answer visible.
Immediate fluency and same-session correction are not retention.

### Transfer

Predeclare content and context distance, cue similarity, response format,
delay, and assistance. A paraphrased practice item is near transfer, not far
transfer.

### Preparation for future learning

Measure whether learners can use a new resource to solve a novel problem after
the intervention. This is distinct from solving a practiced problem faster.

### Calibrated desirable difficulty

Require an immediate-performance cost paired with a delayed retention or
transfer gain, conditioned on prior knowledge. Frustration, obscurity, and
missing prerequisites are not desirable difficulties.

## Excluded primary metrics

- Learning-style matching: the meshing hypothesis lacks credible support.
- Generic growth-mindset or belonging language without context and outcomes.
- Praise, warmth, verbosity, or question counts as stand-alone quality scores.
- Immediate correctness, completion speed, or fluency as learning.
- Generic "Socratic" style without a defined learner action.
- Routing frequency, neuron activation magnitude, or probe accuracy as proof
  of causal representation.
- One holistic "learning science" scalar before independent validation against
  delayed retention and transfer.

## Required reporting fields

Every metric artifact records:

- `level`: `tutor_opportunity`, `learner_enactment`, or `learning_outcome`;
- `unit_of_analysis`;
- `required_context`;
- `label_source` and inter-rater reliability;
- `construct_version`;
- `natural_or_synthetic`;
- `known_surface_confounds`;
- `eligible_for_reward`;
- `evidence_limit`.

## Core references

- Adesope, Trevisan, and Sundararajan (2017),
  https://doi.org/10.3102/0034654316689306
- Bisra et al. (2018), https://doi.org/10.1007/s10648-018-9434-x
- Chi and Wylie (2014), https://doi.org/10.1080/00461520.2014.965823
- Hattie and Timperley (2007), https://doi.org/10.3102/003465430298487
- Kalyuga (2007), https://doi.org/10.1007/s10648-007-9054-3
- Rowland (2014), https://doi.org/10.1037/a0037559
- Shute (2008), https://doi.org/10.3102/0034654307313795
- van de Pol, Volman, and Beishuizen (2010),
  https://doi.org/10.1007/s10648-010-9127-6
