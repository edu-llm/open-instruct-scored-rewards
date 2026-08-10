# Learning-science semantic atlas

This project asks a narrower and more testable question than “is this model a
good tutor?”:

> Do dense and mixture-of-experts language models represent distinct
> learning-science concepts in what they know and generate, beyond terminology,
> formatting, generic instructional quality, and prompt compliance?

The atlas covers tutor turns, explanations, study plans, worked solutions,
practice sets, assessments, and curriculum plans. OLMoE-1B-7B is the MoE case;
OLMo-2-7B is a dense replication. They are different model families, so
cross-model results are replication or non-replication—not a controlled claim
that one architecture is more interpretable.

## Completed exploratory run

The August 2026 run is complete. See [`FINAL_REPORT.md`](FINAL_REPORT.md) for
the evidence-limited conclusions and exact result paths.

In brief, both model families showed clear declarative knowledge but no reliable
preference for enacting the target strategy over a matched sibling strategy.
Representations carried modest control-adjusted signal, while ontology RSA and
MoE expert-path specialization were null. Neither pedagogy-RL adapter improved
the enacted-strategy endpoint, so the causal-intervention gate did not open.

This run used the generated corpus's designed key without independent
round-1 human verification. It is exploratory and design-grounded. The
named-versus-enacted endpoint is excluded because only 4 of 24 intended
says-only artifacts actually used construct vocabulary.

## Claim ladder

The project keeps four claims separate:

1. **Declarative knowledge:** the model can define or discuss a principle.
2. **Enacted output:** the generated artifact instantiates the principle.
3. **Learner enactment:** a learner actually performs the afforded activity.
4. **Learning outcome:** delayed, independent retention or transfer improves.

The corpus and tracing pipeline are designed to test the first two. This run
supported the first and did not establish the second. The last two require
learner-response and delayed-assessment data.

Representation claims are gated separately:

1. concept labels are reliable;
2. quality-matched sibling concepts are discriminable;
3. activation features beat lexical, register, token, and global-quality
   controls;
4. the signal survives named-versus-enacted mismatch and held-out formats;
5. routing association is reported as association, not expert specialization;
6. causal relevance requires necessity, rescue, specificity, and dose response.

## Project map

- `EVIDENCE_REVIEW.md`: cited review and construct-selection rationale.
- `FINAL_REPORT.md`: completed exploratory findings and causal-gate decision.
- `ontology.yaml`: 24-construct faceted ontology, exclusions, and controls.
- `LABELING_GUIDE.md`: 160-item human calibration plus 40 blinded retests.
- `schema.py`: validated ontology, artifact, label, and split schemas.
- `build_corpus.py`: deterministic prompt/contrast manifest builder, generator,
  and the audit of every count the labeling guide fixes.
- `labeling.py`: blinded per-construct judge packets and strict ingestion.
- `agreement.py`: weighted kappa, Gwet AC1, consensus, and reliability gates.
- `ppi.py`: prediction-powered correction against a random human gold slice.
- `behavior.py`: declarative and enacted semantic batteries.
- `trace.py`: multi-position dense and OLMoE teacher-forced replay tracing.
- `analyze_representations.py`: sibling, mismatch, layer, and routing analyses.
- `semantic_geometry.py`: held-out-concept decoding and concept geometry.
- `checkpoint_delta.py`: fixed-text base-versus-LoRA representation analysis.
- `export_checkpoint.py`: ZeRO-state inspection and consolidation into the
  adapter directory the tracer will load.
- `scripts/`: dry-run-friendly ORCD jobs.
- `tests/`: CPU tests; no 7B model download.

## Why matched siblings matter

A probe that separates “good teaching” from “bad teaching” can learn one global
quality direction. The primary analysis instead distinguishes two valid,
quality-matched concepts—for example, spacing from interleaving or a worked
example from a completion problem. Each construct also receives four cells:

- named and enacted;
- named but not enacted;
- enacted without being named;
- neither named nor enacted.

The says-only and enacted-without-naming cells prevent learning-science
vocabulary from masquerading as semantic understanding.

## Human and agent label protocol

The planned confirmatory protocol makes the human calibration set a probability
sample, not a hand-picked error analysis. Agent judges label the larger corpus,
and confirmatory estimates are rectified against the random human slice.
Few-shot examples and every sibling sharing their source item are quarantined
from evaluation.

Presence, enactment fidelity, applicability, and generic quality remain
separate labels. Constructs below the reliability gate are rewritten or
retired before tracing.

The completed exploratory run did not execute this confirmatory round-1
protocol; it used the designed key after agent calibration was accepted. Reports
therefore set `confirmatory: false`, and no human-reliability estimate is
claimed.

## Pedagogy-RL checkpoint comparison

Two comparisons are supported:

- `ckpt/pedagogy-tutor-armF.zip` is a dense OLMo-2 LoRA. It has no experts.
- `/orcd/scratch/orcd/013/zsophia/ckpt/pedagogy_olmoe/global_step{180,190,200}`
  contains late DeepSpeed states from the OLMoE pedagogy-RL run.

The OLMoE router parameters were frozen during RL, but routing need not remain
fixed: attention and expert adapters change the hidden state entering later
routers. The delta analysis therefore separates router-input shift, direct
expert adaptation under fixed routing, selection shift, and downstream path
composition.

Raw DeepSpeed states are deliberately rejected by the tracer until they are
consolidated and exported with their adapter configuration. The pipeline never
guesses how to apply a 27 GB training-state shard.

`export_checkpoint.py` is the step that removes that rejection.
`scripts/export_rl_checkpoints.sbatch` runs it over all three steps and the
dense zip, on CPU:

```bash
# on ORCD, from /orcd/scratch/orcd/013/zsophia/open-instruct
STAGE=inspect bash projects/learning_science_semantic_atlas/scripts/export_rl_checkpoints.sbatch
DRY_RUN=1 bash projects/learning_science_semantic_atlas/scripts/export_rl_checkpoints.sbatch
sbatch projects/learning_science_semantic_atlas/scripts/export_rl_checkpoints.sbatch
SPLIT=<split> sbatch projects/learning_science_semantic_atlas/scripts/compare_rl_checkpoints.sbatch
```

`STAGE=inspect` reads directory listings only—no torch, no deepspeed, no
GPU—so it runs on a login node and reports the shard sizes that set the export
job's memory request. `DRY_RUN=1` additionally resolves the adapter
configuration and checks it against the tensors in the shard without writing
anything.

The rank, alpha, and target lists are training-time facts that a ZeRO state does
not record, so the export refuses to invent an `adapter_config.json`: it uses
the one PEFT wrote beside this run's HF-format saves, and two candidates whose
bytes differ stop the job. Per-expert LoRA names are positional—two parameters
on one module differ only by `base_layer` nesting depth—so tensor names are
copied through unrenamed and travel with the configuration that reproduces the
nesting. Nothing is deleted or overwritten: each export is built in a sibling
`.partial-<pid>` directory and renamed into place, a matching export already
present is left alone, and one that differs stops the job.

Every checkpoint comparison has two arms:

- **fixed-text replay**, where base and tuned models read byte-identical prompts
  and outputs, isolates representation change;
- **native generation**, reported separately, measures behavior but is
  confounded by output-content differences.

## Local verification

From the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest projects/learning_science_semantic_atlas/tests -q
ruff check projects/learning_science_semantic_atlas
```

Build a prompt manifest without calling a model:

```bash
python -m projects.learning_science_semantic_atlas.build_corpus \
  --ontology projects/learning_science_semantic_atlas/ontology.yaml \
  --out projects/learning_science_semantic_atlas/data/corpus
```

That writes the 24-item calibration manifest, the 160-item round-1 manifest, one
generation brief per item, and a manifest whose audit checks every count
`LABELING_GUIDE.md` fixes: the 120/18/22 composition, the 70/30/25/20/15
artifact-type budget, one says-only item and one hard negative per construct, all
four cells of the names-versus-enacts 2x2, the 24 quality-matched sibling
contrasts, and the 45-appearance candidate-set floor. Adding `--mode all
--no-model` also assembles a corpus whose artifact texts are marked scaffolds, so
`labeling` and `agreement` can be run before a GPU is asked for.

The briefs state each item's target construct and role, so they are withheld:
everything under `<out>/withheld/` is opened after labelling, and only
`corpus.jsonl`, `items.jsonl`, `calibration_corpus.jsonl`, and `manifest.json`
are blinded.

Trace files and generated results are ignored by Git. Run the Slurm scripts
with `DRY_RUN=1` before submission to print the effective models, checkpoints,
positions, and output paths.

