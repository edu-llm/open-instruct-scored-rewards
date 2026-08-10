# Preregistered pilot design

Status: implementation protocol. Confirmatory labels and outputs must remain
sealed until the discovery analysis and candidate sites are frozen.

## Scope

Primary model: `allenai/OLMoE-1B-7B-0924-Instruct`.

Dense replication: `allenai/OLMo-2-1124-7B-Instruct`.

The models differ in training lineage, total parameters, active parameters, and
architecture. All estimands are therefore within-model. Cross-model summaries
test replication, not MoE-versus-dense superiority.

The pilot tests three constructs:

1. diagnostic localization;
2. contingent scaffolding;
3. generative elicitation.

It tests behavioral selection, domain-general decodability, and causal use.
Reward-model training and claims about student learning are explicitly outside
the confirmatory pilot.

## Data split

The immutable grouping unit is the source problem or natural tutoring episode.
All paraphrases, response variants, templates, and synthetic edits derived from
one source stay in the same split.

- 50% discovery train;
- 15% discovery validation;
- 15% natural in-domain test;
- 20% sealed cross-domain confirmatory test.

`build_pairs.py` hashes the group identifier with the registered seed. It
creates the split before generating or accepting variants, preventing sibling
counterfactuals from crossing splits. Exact and normalized-text hashes are
checked across splits.

The confirmatory split may be traced before analysis only when artifact IDs are
blinded. Labels, candidate-site scores, and intervention results stay sealed
until `probe.py freeze` writes the analysis manifest.

## Scenario schema

Every record contains:

- `item_id`: source item;
- `pair_id`: matched contrast;
- `variant_id`: individual prompt/response;
- `template_id`: wording family;
- `domain` and `source`;
- `concept`;
- `student_state`;
- `prescribed_action`;
- `candidate_action`;
- `label` in `{0, 1}`;
- `expected_direction` in `{-1, 1}`;
- `natural_or_synthetic`;
- source license and provenance;
- normalised text and hashes;
- verification status and notes.

Synthetic pairs must attempt to change one construct while preserving content,
correctness, approximate token length, syntax, and tone. Natural and synthetic
results are never pooled as if they were one sampling process.

MRBench directly annotates mistake identification and location, so the
diagnostic-localization contrast is `dataset_direct`. It does not directly
annotate contingent scaffolding or generative elicitation. Contrasts derived
from guidance, actionability, answer reveal, or punctuation are
`heuristic_proxy` candidates and remain exploratory until construct-specific
human labels replace those proxies. A held-out domain does not make a heuristic
label confirmatory.

## Behavioral hypotheses

### H-A1: pedagogical action selection

For each construct, the model's log probability of the prescribed action
exceeds the matched alternative on the sealed cross-domain split.

Primary endpoint:

`action_logit_difference = logit(prescribed) - logit(alternative)`.

Open-generation rubric scores are secondary and are evaluated blind to model,
condition, and intervention.

Advance a construct to representation analysis only if:

- the paired 95% confidence interval for the mean logit difference excludes
  zero in the expected direction; and
- balanced forced-choice accuracy is at least 0.60 on the confirmatory split.

The threshold is a feasibility gate, not a universal effect-size claim.

## Representation hypotheses

### H-B1: cross-domain linear decodability

At a site selected on discovery data, a regularized linear probe predicts the
construct label on held-out domains better than lexical and permuted-label
controls.

Primary endpoint:

`selectivity = balanced_accuracy(linear_probe) -
balanced_accuracy(best_lexical_control)`.

Required controls:

- grouped nested cross-validation;
- bag-of-words/TF-IDF or hashed lexical features;
- token count, question marks, digits, and extraction position;
- labels permuted within source and template strata;
- identical probe capacity and regularization search;
- unseen template and domain evaluation;
- no selection and significance testing on the same examples.

Advance a construct to causal testing only if confirmatory selectivity is at
least 0.05 and its cluster-bootstrap 95% confidence interval excludes zero.
Probe performance licenses a decodability claim only.

### H-B2: MoE routing association

Router probabilities or selected-expert identities differ across matched
construct variants after accounting for expert base rate.

This is descriptive. Routing skew, utilization, and activation magnitude never
license a specialization or causal-use claim.

## Causal hypotheses

Candidate layers, positions, directions, and experts are frozen from discovery
data before the confirmatory split is opened.

### H-C1: necessity

Removing a candidate direction or ablating a candidate expert output reduces
the target action logit difference in scenarios where the construct is
appropriate.

### H-C2: sufficiency

Adding or patching the candidate representation increases the target action
logit difference in the matched counterfactual.

### H-C3: rescue

Restoring the candidate representation after ablation recovers a positive
fraction of the lost target effect.

Primary causal endpoint:

`repair_fraction = (patched - corrupted) / (clean - corrupted)`,

reported with denominator-near-zero cases excluded by a preregistered absolute
logit-gap floor of `1e-4`.

Required controls:

- both intervention directions;
- at least three signed doses including zero;
- norm-matched random and shuffled directions;
- random active experts and equal-routing-rank experts;
- irrelevant token positions and contraindicated scenarios;
- router intervention and expert-output intervention analyzed separately;
- single experts and small active-expert coalitions;
- factual correctness, response fluency, held-out language-model loss, and an
  unrelated educational decision as off-target measures.

A candidate is called causally relevant only if:

1. the target effect has the expected sign and a cluster-bootstrap 95%
   confidence interval excluding zero;
2. the effect exceeds its matched random control;
3. ablation and rescue both succeed;
4. the signed-dose slope is monotonic in the expected direction; and
5. no off-target measure crosses its registered non-inferiority margin.

Steering without necessity and rescue is reported as controllability, not
natural causal use.

The probe direction is learned from `content_last`, after the model has read a
candidate response. Applying that direction at the pre-answer forced-choice
decision position is therefore a cross-position controllability test. It cannot
explain how the earlier response tokens were naturally generated. A
generation-mechanism claim requires a direction selected at the decision
position from learner-state counterfactuals whose prescribed actions differ.

## MoE intervention definitions

### Router intervention

Modify router logits before top-k selection, then recompute top-k weights and
preserve the checkpoint's native `norm_topk_prob` setting. OLMoE-1B-7B uses
`false`, so retained top-k mass generally does not sum to one. A separately
named mass-preserving renormalized intervention is an off-target norm control,
not the primary intervention. Forcing, suppressing, and swapping routing are
separate conditions. Router interventions change selection and must not be
interpreted as expert-output effects.

### Expert-output intervention

Hold router selections and native top-k weights fixed. Zero, resample, patch, or
interchange the selected expert's weighted contribution before summation.
This isolates transformation from routing.

### Coalition intervention

A coalition is defined only among experts active at the same layer and token.
Compare candidate coalitions with random active coalitions of equal size and
similar routing mass.

## Statistical analysis

The sampling unit is the source `item_id`, not a token, generation, expert
decision, or derived pair. All confidence intervals and permutation tests
resample or permute at item level so multiple constructs or moments from one
conversation do not count as independent evidence.

- Discovery: Benjamini-Hochberg FDR within construct and analysis family.
- Confirmatory: Holm correction over the preregistered primary contrasts.
- Report paired mean differences, standardized paired effects, confidence
  intervals, and sample counts.
- Treat layers and tokens as repeated measurements, not independent examples.
- Show natural and synthetic estimates separately.
- Save random seeds, package versions, model revisions, split hashes, prompt
  hashes, and candidate-freeze manifests.

## Exclusions

Exclude before opening labels:

- malformed records or missing required fields;
- duplicate normalized texts across splits;
- tokenizer truncation that removes the manipulated span;
- pairs whose verified response differs on multiple target constructs;
- interventions with failed no-op logit parity;
- repair fractions with a clean/corrupted denominator below `1e-4`.

Do not exclude examples because the model answered incorrectly or because the
intervention moved in the wrong direction.

## Stop/go rules

1. Stop a construct after behavior if H-A1 fails.
2. Stop causal localization if H-B1 fails.
3. Stop expert claims if only routing association succeeds.
4. Report a steering result as controllability if necessity/rescue fails.
5. Do not train a reward model unless the construct passes independent ratings,
   held-out-item decodability, and at least one selective causal test.
   Cross-domain decodability requires a future independently labeled domain;
   Bridge currently supplies held-out intervention prompts, not probe labels.
6. Do not claim learning benefit without a separate randomized learner study
   with delayed transfer outcomes.

## Reproducibility artifacts

Each run writes:

- resolved configuration and model revision;
- input and split hashes;
- exact package versions;
- per-example predictions and interventions;
- aggregate statistics with bootstrap seeds;
- discovery candidate manifest;
- confirmatory gate decisions;
- a machine-readable claim status: `failed_behavior`, `decodable`,
  `associated_only`, `controllable`, or `causally_relevant`.
