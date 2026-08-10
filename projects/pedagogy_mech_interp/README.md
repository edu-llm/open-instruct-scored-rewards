# Pedagogy mechanistic-interpretability pilot

This is a new, self-contained project that asks whether open language models
encode and causally use three concrete pedagogical decisions:

- diagnostic localization;
- contingent scaffolding; and
- generative elicitation.

OLMoE-1B-7B is the primary subject. OLMo-2-7B is a dense replication case, not
an architecture-controlled comparator.

The claim ladder is intentionally strict:

1. the model behaviorally distinguishes the preferred response;
2. a representation is decodable across held-out items and domains;
3. interventions selectively change the decision;
4. only necessity, rescue, specificity, and off-target checks license a causal
   relevance claim.

High router utilization, large neuron activations, and accurate probes are
association evidence only.

## Project layout

- `METRIC_TAXONOMY.md`: evidence tiers and invalid shortcuts.
- `RESEARCH_DESIGN.md`: frozen hypotheses, splits, gates, and claim language.
- `DATA_MANIFEST.json`: source, license, and contamination notes.
- `fetch_data.py`: downloads and normalizes MRBench, MathDial, and Bridge.
- `build_pairs.py`: validates contrasts and creates grouped, sealed splits.
- `trace.py`: token-aligned dense and MoE activation tracing.
- `probe.py`: behavior gates, lexical controls, probes, routing atlas, and
  candidate freeze.
- `intervene.py`: activation, router, and selected-expert interventions.
- `analyze.py`: item-clustered inference and final reporting.
- `tests/`: model-free tests for schemas, splits, hooks, routing, and statistics.
- `scripts/`: resumable Slurm entrypoints.

The project imports no reward-model implementation from `pedagogy_rm`. It
copies only its useful design constraints: exact chat reconstruction, concrete
rubrics, grouped splits, and adversarially matched examples.

## Build the data

From the repository root:

```bash
python -m projects.pedagogy_mech_interp.fetch_data \
  --out projects/pedagogy_mech_interp/data

python -m projects.pedagogy_mech_interp.build_pairs \
  --sources projects/pedagogy_mech_interp/data/sources/mrbench_pairs.jsonl \
  --out projects/pedagogy_mech_interp/data/pairs
```

The registered split holds the Bridge domain out from site selection. The
held-out rows omit labels; `confirmatory_key.jsonl` is opened only after
`probe.py` writes `candidate_freeze.json`. A held-out domain does not upgrade a
heuristic label to confirmatory evidence.

Current normalized core:

- 530 MRBench matched families before exact-text deduplication;
- 500 MathDial trajectories;
- 419 Bridge novice/expert contrasts;
- 470 retained matched families (940 response variants);
- 133 Bridge-domain confirmatory families.

`audit_queue.json` contains every automatic verifier disagreement plus a
stratified sample. Diagnostic localization is derived directly from MRBench
mistake-identification and mistake-location annotations. Contingent scaffolding
and generative elicitation are explicitly marked `heuristic_proxy`; they remain
exploratory until construct-specific human labels replace those proxies.

## Local verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest projects/pedagogy_mech_interp/tests -q
ruff check projects/pedagogy_mech_interp
```

The tests do not download or instantiate a 7B model. Toy dense, Transformers
4-style MoE, and Transformers 5-style router modules test:

- no-op hook logit parity;
- selected-position isolation;
- native normalized and unnormalized top-k routing;
- forced routing;
- selected expert-output subtraction;
- expert decomposition reconstruction;
- grouped split integrity; and
- deterministic pair-clustered statistics.

## One-GPU smoke trace

```bash
python -m projects.pedagogy_mech_interp.trace \
  --input projects/pedagogy_mech_interp/data/pairs/discovery_validation.jsonl \
  --output projects/pedagogy_mech_interp/results/olmoe/smoke.npz \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --limit 4 \
  --verify-decomposition
```

The trace stores the last tutor content token at five spread layers:

- residual and MLP inputs/outputs;
- true pre-softmax OLMoE router logits and probabilities;
- selected expert identities and native top-k weights; and
- each selected expert's weighted output before summation.

The expert decomposition check must pass before a full trace is accepted.

## Slurm pipeline

On ORCD:

```bash
MODEL=allenai/OLMoE-1B-7B-0924-Instruct \
MODEL_TAG=olmoe \
bash projects/pedagogy_mech_interp/scripts/submit_pipeline.sh
```

Dense replication:

```bash
MODEL=allenai/OLMo-2-1124-7B-Instruct \
MODEL_TAG=olmo2_7b \
bash projects/pedagogy_mech_interp/scripts/submit_pipeline.sh
```

For plumbing only, set `LIMIT=8 LIMIT_PAIRS=2`. A full run uses one frozen
checkpoint and no training.

The submission chain is:

```text
three split traces -> probe/candidate freeze -> confirmatory interventions -> report
```

## Completed pilot outcome

Hardened frozen-model reruns completed for both registered checkpoints.
Artifacts are under `results/olmoe_validity/` and
`results/olmo2_7b_validity/`.

- The one confirmatory construct, directly annotated diagnostic localization,
  passed forced-choice behavior in both models: 77.4% accuracy for OLMoE and
  87.1% for OLMo-2. It did not pass held-out-item decodability or necessity.
- OLMoE generative elicitation had exploratory activation selectivity above
  lexical control and an exploratory routing association. Its construct label
  is a heuristic proxy, and interventions failed necessity, rescue, and
  specificity.
- Contingent scaffolding and the remaining generative results are exploratory
  because MRBench does not directly annotate those constructs.
- No construct/model combination passed the full causal ladder. Apparent
  isolated rescue or specificity results do not license a mechanism claim
  without the preceding decodability and necessity gates.
- The registered decision is **no reward-model follow-up yet**. The next useful
  study is a learner-state counterfactual bank that changes the prescribed
  action while holding candidate wording fixed, followed by decision-point
  patching. More probe fitting or expert-frequency mining would not repair the
  missing causal evidence.

## Interpretation limits

- MRBench supplies natural same-context response contrasts, but it is public
  and potentially present in post-training data.
- MathDial uses simulated students and GSM8K-derived problems.
- Bridge supplies real tutoring excerpts but not delayed learning outcomes.
- A response-level construct measures an instructional opportunity, not learner
  enactment or learning.
- The current forced-choice intervention endpoint tests model evaluation of
  pedagogical responses. A later interactive generation study is needed before
  claiming that the same mechanism controls unconstrained tutoring.
- Probe directions are learned after the model reads a candidate response and
  applied at a pre-answer decision point. That cross-position test measures
  controllability, not the natural mechanism that generated the response.
- Transformers 5.4 returns router probabilities where later versions return
  logits. The tracer now detects both contracts, recomputes true logits from
  the router input, and preserves OLMoE's native `norm_topk_prob=false` mass.
- The identity expert-output rescue is an implementation control. Independent
  semantic rescue requires a future learner-state counterfactual bank with
  aligned internal variables.
