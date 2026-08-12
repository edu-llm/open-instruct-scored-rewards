"""The probe, as a reward open-instruct can call.

    python open_instruct/grpo_fast.py \
        --reward_plugins projects/pedagogy_rm/plugin.py \
        --group_scorer pedagogy:head=data/head.npz

THE ENCODER IS FROZEN, AND SEPARATE FROM THE POLICY. This is the whole design and
it is not an implementation detail. The head is a linear map on one particular
model's activation space, so if it reads a model that is being trained, the policy
can raise its reward by moving its own activations without changing a word of what
it says. That is a reward channel with no text in it at all, and no amount of
label quality would catch it. So the scorer loads its own copy of the base model
and never updates it.

    full-weight policy   a second frozen OLMo, which is what this class does
    LoRA policy          the base model is already there; disable the adapter

TURNS, NOT TRANSCRIPTS. The head was fitted on single tutor turns in a two-message
context, so it is fed exactly that. Scoring a whole dialogue with a head trained on
turns would be reading a number that means nothing, which is how the previous
project got a reward that correlated with leakage at +0.291 and with learning at
-0.012.

WHAT IS SCORED, AND WHAT IS NOT. `concise` is excluded by default: eight surface
features predict it at 0.96 against the states' 0.97, so rewarding it buys short
turns and nothing else. `leak` is negated, because on that scale 3 means the turn
handed over the answer. The remaining dimensions are returned separately rather
than pre-summed so the group aggregator can normalise each one - a dimension with
a wide spread would otherwise dominate the sum for no reason but its variance.
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from open_instruct.scored_rewards import GroupScorer, Sample, ScoreResult, register
from open_instruct.scored_rewards.guards import MultiDimensional
from open_instruct.scored_rewards.types import TRANSCRIPT_KEY, parse_transcript

TEACHER_SYSTEM = """You are a tutor helping a student with a test question. \
The student cannot see your instructions.

Guide the student toward understanding. Do not state the answer or eliminate \
options for them. Keep each message short - one idea at a time."""

#: Higher is better for all of these once the sign is applied. leak runs the other
#: way on its own scale: 1 keeps the answer back, 3 hands it over.
#:
#: `correct` is here so a five-dimension head can be loaded, but it is NOT in every head
#: file: it was dropped from the first one for missing its agreement gate and re-added after
#: the rewritten rubric reached kappa 0.43. Which dimensions a run rewards is therefore
#: decided by the head file it loads, not by this table - see the default below.
SIGNS = {"leak": -1.0, "targeted": 1.0, "actionable": 1.0, "elicits": 1.0, "correct": 1.0}


class PedagogyHead(GroupScorer):
    """Frozen OLMo plus the ridge heads from fit_head.py, batched over a group."""

    name = "pedagogy"

    def __init__(
        self,
        head: str = "data/head.npz",
        model: str = "",
        dimensions: str = "",
        device: str = "cuda",
        max_len: int = 2048,
        batch_size: int = 16,
        length_band: str = "",
        length_weight: float = 0.0,
        length_ramp: str = "",
        reward_weight: float = 1.0,
        revision: str = "",
        device_map: str = "",
        max_memory_gib: float = 0.0,
    ) -> None:
        import numpy as np  # noqa: PLC0415

        blob = np.load(head, allow_pickle=False)
        self.meta = json.loads(str(blob["meta"]))
        # Default to what the head file actually holds, intersected with the signs this
        # module knows. Defaulting to list(SIGNS) instead would mean that adding a dimension
        # here breaks every older head file, which is backwards: the file is the artefact
        # with the fitted weights and it should decide. An explicit --group_scorer
        # dimensions= still errors on anything missing, because that is a request that
        # cannot be honoured rather than a default that can be narrowed.
        asked = [d.strip() for d in dimensions.split(",") if d.strip()]
        wanted = asked or self._default_dimensions()
        missing = [d for d in wanted if d not in self.meta["dimensions"]]
        if missing:
            raise ValueError(f"{head} has no head for {missing}; it holds {sorted(self.meta['dimensions'])}")
        if not wanted:
            raise ValueError(f"{head} holds {sorted(self.meta['dimensions'])}, none of which have a sign here")
        self.dims = wanted
        self.weights = {d: {k: blob[f"{d}/{k}"] for k in ("mean", "scale", "coef", "intercept")} for d in self.dims}
        # Dimensions fitted with length projected out carry the projection and the one length
        # coefficient that was deliberately kept. Both must be applied here or the head is being
        # read on states it was not fitted on. See fit_head.py for why it is done this way.
        self.length = {
            d: (blob[f"{d}/length_proj"], self.meta["dimensions"][d]["length"])
            for d in self.dims
            if f"{d}/length_proj" in blob.files and (self.meta["dimensions"][d].get("length") or {}).get("slope")
        }
        # A FLAT BAND, WHICH IS THE POINT RATHER THAN A SIMPLIFICATION. Every dimension the
        # head scores gets *better* as a turn gets shorter - measured at -0.26 per log-word
        # over the five of them, and the human's own ratings agree at -0.49 - so no
        # reweighting of them can express "this is now too short". Only a term that turns
        # around can, and this is the cheapest one that does.
        #
        # WHY THE LOWER EDGE IS THE TARGET AND NOT THE LIMIT. Inside the band this term is
        # constant, so the head's pull towards brevity slides the policy to the bottom edge
        # and parks it there. That is a feature: the resting length is a number you write
        # down rather than one you infer from two quantities you do not know precisely.
        # Measured over 1131 length judgements, 21-58 words is the 90%-acceptable range, so
        # an edge at 30 leaves room below it and rests where ~90% of turns read as right.
        #
        # A SMOOTH CURVE WAS TRIED AND IS WORSE HERE. Fitting the acceptability curve itself
        # and rewarding it directly rests at 25 words at the measured slope, but drifts to 19
        # if the slope steepens to -0.8 and to 32 if it flattens to -0.15 - and the slope does
        # move as the policy trains. The band does not move at all.
        #
        # WEIGHT 2.0 BECAUSE THE FAILURE IS A CLIFF, NOT A DRIFT. Too small and the band stops
        # binding entirely and the policy collapses to five words; there is no graceful middle.
        # At 1.0 that happens once the slope passes -0.5, at 1.5 past -0.7, at 2.0 past -1.0.
        # The measured slope is -0.26, so 2.0 is roughly a 4x margin on a catastrophic mode.
        self.length_lo, self.length_hi = 0, 10**9
        self.length_zero_lo, self.length_zero_hi = 0, 10**9
        self.length_weight = float(length_weight)
        self.reward_weight = float(reward_weight)
        if length_band:
            lo, _, hi = length_band.partition("-")
            self.length_lo, self.length_hi = int(lo), int(hi)
            if self.length_lo >= self.length_hi:
                raise ValueError(f"length_band wants lo-hi with lo < hi, got {length_band!r}")
            # Dash, not comma: --group_scorer splits its kwargs on commas, so a comma here
            # would be parsed as a second key and silently drop the upper edge.
            zlo, _, zhi = (length_ramp or "").partition("-")
            self.length_zero_lo = int(zlo) if zlo else self.length_lo
            self.length_zero_hi = int(zhi) if zhi else self.length_hi
            if not self.length_zero_lo <= self.length_lo < self.length_hi <= self.length_zero_hi:
                raise ValueError(
                    f"length_ramp must bracket length_band: got ramp {length_ramp!r} around band {length_band!r}"
                )
        elif self.length_weight:
            raise ValueError("length_weight without length_band would reward every turn equally")

        self.model_name = model or self.meta["model"]
        self.revision = revision or self.meta.get("revision") or ""
        self.device_map = device_map
        self.max_memory_gib = float(max_memory_gib)
        self.device, self.max_len, self.batch_size = device, max_len, int(batch_size)
        self._lock = threading.Lock()
        # Annotated because these are loaded on first use, not in __init__: without it ty
        # narrows both to None for the whole class and reports every call on them as a call
        # on None. Any rather than the concrete classes so this module still imports without
        # transformers, which the CPU-only tests rely on.
        self._model: Any = None
        self._tokenizer: Any = None
        self._input_device: Any = None
        # Scoring is synchronous and can spend minutes loading/running a sharded
        # encoder. Keep it off the vLLM asyncio loop, but pin every CUDA call to
        # one worker so concurrent completed groups cannot race the same model.
        self._score_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pedagogy-reward")
        self._score_thread_id: int | None = None

    def _default_dimensions(self) -> list[str]:
        return [dimension for dimension in SIGNS if dimension in self.meta["dimensions"]]

    def _load(self):
        """Loaded once, on first use, inside the actor that will use it."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer  # noqa: PLC0415

            revision = self.revision or None
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=revision)
            load_kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "revision": revision}
            if self.device_map:
                load_kwargs["device_map"] = self.device_map
                if self.max_memory_gib:
                    if not torch.cuda.is_available():
                        raise RuntimeError("a sharded reward encoder requires CUDA")
                    load_kwargs["max_memory"] = {
                        index: f"{self.max_memory_gib:g}GiB" for index in range(torch.cuda.device_count())
                    }
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            except ValueError as causal_error:
                print(f"causal reward encoder unavailable ({causal_error}); trying image-text model", flush=True)
                model = AutoModelForImageTextToText.from_pretrained(self.model_name, **load_kwargs)
            self._truncate(model)
            if not self.device_map:
                device = self.device if torch.cuda.is_available() else "cpu"
                model = model.to(device)  # ty: ignore[invalid-argument-type]  # transformers stubs type .to() as taking a model
            self._model = model.eval()
            self._input_device = next(self._model.get_input_embeddings().parameters()).device
            for p in self._model.parameters():  # the encoder is never trained
                p.requires_grad_(False)

    def _truncate(self, model) -> None:
        """Drop the blocks above the deepest one the active heads read.

        The heads read layers 16 and 20 of 32, and everything above is computed
        and discarded - a third of the forward pass on every completion of every
        group, for the whole run. Dropping the blocks drops their weights too,
        which is most of what the encoder costs in memory beside vLLM.

        ONE BLOCK MORE THAN NEEDED, DELIBERATELY. Transformers appends the final
        norm's output as the LAST entry of hidden_states and leaves every earlier
        entry as the raw block output. Cutting to exactly the deepest layer would
        make that layer the last one, so it would arrive normed - a different
        vector from the one the head was fitted on, still finite, still plausible,
        and wrong. Keeping one spare block leaves the read layers raw.
        """
        deepest = max(self.meta["dimensions"][d]["layer"] for d in self.dims)
        owner = None
        blocks = None
        for path in (("model",), ("language_model",), ("model", "language_model"), ("model", "model")):
            candidate = model
            for name in path:
                candidate = getattr(candidate, name, None)
                if candidate is None:
                    break
            candidate_blocks = getattr(candidate, "layers", None)
            if candidate_blocks is not None:
                owner, blocks = candidate, candidate_blocks
                break
        if blocks is None or owner is None or deepest + 1 >= len(blocks):
            return
        owner.layers = blocks[: deepest + 1]
        self.kept_layers = deepest + 1

    def context(self, sample: Sample) -> tuple[list[dict], str]:
        """The chat context the head was fitted on, and the turn to score.

        Must match extract_hidden.context_messages exactly. If the rollout format
        changes and this does not, the states drift away from the ones the head
        saw and the reward degrades silently rather than failing.
        """
        item = sample.item
        transcript = parse_transcript(sample.env_info.get(TRANSCRIPT_KEY, "")) if sample.env_info else []
        tutor_turns = [t.get("text", "") for t in transcript if t.get("who") in ("policy", "tutor", "assistant")]
        student_turns = [t.get("text", "") for t in transcript if t.get("who") in ("partner", "student", "user")]
        turn = tutor_turns[-1] if tutor_turns else sample.policy_text
        before = student_turns[-1] if student_turns else item.get("student_before", "")
        question = item.get("question") or item.get("prompt") or sample.prompt
        return [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"Question the student is working on:\n{question}"},
            {"role": "user", "content": before},
        ], turn

    def states(self, contexts: list[tuple[list[dict], str]]) -> dict[tuple[str, int], list]:
        """Pooled hidden states for every (pooling, layer) the heads ask for."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self._load()
        cells = {(self.meta["dimensions"][d]["pooling"], self.meta["dimensions"][d]["layer"]) for d in self.dims}
        out: dict[tuple[str, int], list] = {cell: [] for cell in cells}
        for start in range(0, len(contexts), self.batch_size):
            prepared = []
            for messages, turn in contexts[start : start + self.batch_size]:
                prefix = self._tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
                )
                empty = self._tokenizer.apply_chat_template(
                    [*messages, {"role": "assistant", "content": ""}],
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=False,
                )
                whole = self._tokenizer.apply_chat_template(
                    [*messages, {"role": "assistant", "content": turn}],
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=False,
                )
                prefix_len = prefix.shape[1]
                suffix_len = max(1, empty.shape[1] - prefix_len)
                content_stop = whole.shape[1] - suffix_len
                trimmed_left = max(0, whole.shape[1] - self.max_len)
                ids = whole[0, -self.max_len :]
                lo = max(0, min(prefix_len - trimmed_left, ids.shape[0] - 1))
                hi = max(lo + 1, min(content_stop - trimmed_left, ids.shape[0]))
                prepared.append((ids, lo, hi))

            width = max(ids.shape[0] for ids, _, _ in prepared)
            pad = self._tokenizer.pad_token_id
            pad = self._tokenizer.eos_token_id if pad is None else pad
            input_ids = torch.full((len(prepared), width), pad, dtype=prepared[0][0].dtype)
            attention_mask = torch.zeros((len(prepared), width), dtype=torch.long)
            spans = []
            for index, (ids, lo, hi) in enumerate(prepared):
                offset = width - ids.shape[0]
                input_ids[index, offset:] = ids
                attention_mask[index, offset:] = 1
                spans.append((offset + lo, offset + hi))
            with torch.no_grad():
                hidden = self._model(
                    input_ids.to(self._input_device),
                    attention_mask=attention_mask.to(self._input_device),
                    output_hidden_states=True,
                ).hidden_states
            for index, (lo, hi) in enumerate(spans):
                for pooling, layer in cells:
                    h = hidden[layer][index]
                    vec = h[-1] if pooling == "eot" else h[hi - 1] if pooling == "last" else h[lo:hi].mean(0)
                    out[(pooling, layer)].append(vec.float().cpu().numpy().astype(np.float32))
        return out

    def length_fit(self, words: int) -> float:
        """1.0 inside the band, falling linearly to 0 at the ramp edges.

        FLAT INSIDE, SLOPED OUTSIDE, which is the combination the two failures argued for.

        The interior has to be flat because that is what pins the resting length. A smooth peak
        fitted to the acceptability curve rests at 25 words at the measured quality slope but
        drifts to 19 or 32 as that slope moves during training; a flat band rests at its lower
        edge whatever the slope, so the target is a number written down rather than inferred.

        The exterior has to be sloped because a hard band cannot tell 59 words from 200 - both
        score zero - so nothing pushes a runaway turn back. Arm C ended at a median of 46 words
        with the band at 30-58 and its longest turns unpunished relative to its merely-long ones.
        A ramp makes every extra word past the edge cost something.

        AND THE RAMP IS THE ONLY LEVER LEFT ON LENGTH, which is why it matters more than it looks.
        The head's `leak` tracks length at +0.70 against the human's +0.43, and two attempts to
        correct that inside the head failed - a rank-1 projection moved it to 0.67, residualising
        length out before fitting moved it to 0.74. Both were fitted in-distribution and neither
        survived the shift to policy-generated text. So the head pulls towards brevity harder than
        a person would, and this term is what offsets it.
        """
        if words >= self.length_lo and words <= self.length_hi:
            return 1.0
        if words < self.length_lo:
            span = self.length_lo - self.length_zero_lo
            return max(0.0, (words - self.length_zero_lo) / span) if span else 0.0
        span = self.length_zero_hi - self.length_hi
        return max(0.0, (self.length_zero_hi - words) / span) if span else 0.0

    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._score_executor, self._run_score_group_sync, group)

    def _run_score_group_sync(self, group: list[Sample]) -> list[ScoreResult]:
        worker_thread = threading.get_ident()
        if self._score_thread_id is None:
            self._score_thread_id = worker_thread
        elif self._score_thread_id != worker_thread:
            raise RuntimeError("reward scorer changed CUDA worker thread")
        return self._score_group_sync(group)

    def _score_group_sync(self, group: list[Sample]) -> list[ScoreResult]:
        import numpy as np  # noqa: PLC0415

        contexts = [self.context(s) for s in group]
        pooled = self.states(contexts)
        words = np.array([max(len(turn.split()), 1) for _, turn in contexts], dtype=np.float64)
        scored: dict[str, list[float]] = {}
        for dim in self.dims:
            spec = self.meta["dimensions"][dim]
            w = self.weights[dim]
            states = np.stack(pooled[(spec["pooling"], spec["layer"])]).astype(np.float64)
            bias = 0.0
            if dim in self.length:
                proj, meta = self.length[dim]
                centred = np.log(words) - meta["mean"]
                states = states - np.outer(centred, proj)
                bias = meta["slope"] * centred
            x = (states - w["mean"]) / w["scale"]
            raw = x @ w["coef"] + w["intercept"] + bias
            scored[dim] = [float(np.clip(v, spec["lo"], spec["hi"])) for v in raw]

        results = []
        for i, (_, turn) in enumerate(contexts):
            dims = {d: SIGNS[d] * scored[d][i] for d in self.dims}
            score = float(sum(dims.values()) / len(dims))
            # Word count is recorded whether or not length is being rewarded. It used to be written
            # only inside the branch below, which meant the one run that most needs it - an ablation
            # with the length term switched off, where the expected outcome IS a collapse in length -
            # was the one run that would not log it. A diagnostic should not appear and disappear
            # with the objective it happens to be diagnosing.
            n = len(turn.split())
            info = {"raw": {d: scored[d][i] for d in self.dims}, "turn_chars": len(turn), "words": n}
            if self.length_weight:
                fit = self.length_fit(n)
                score += self.length_weight * fit
                dims["length"] = fit
            info["unweighted_score"] = score
            score *= self.reward_weight
            results.append(ScoreResult(score=score, dimensions=dims, info=info))
        return results


def normalized(**kwargs) -> GroupScorer:
    """The same head, with each dimension z-scored inside the group before averaging.

    A SECOND NAME RATHER THAN A REPLACEMENT, because which of the two is right is the
    question and not a detail. `pedagogy` averages the four signed dimensions on their own
    1-3 scales, so a dimension with a wider spread across a group moves the group-relative
    advantage more - and measured on the 600 labelled turns the spreads are not equal:

        elicits     sd 0.72
        actionable  sd 0.71
        leak        sd 0.57
        targeted    sd 0.47

    So the raw mean leans about 1.5x harder on `elicits` than on `targeted`, which is the
    wrong way round for this project. probe.py measured that eight surface features with no
    notion of teaching predict `elicits` at 0.81 and `targeted` at 0.36, against the states'
    0.95 and 0.85 - `targeted` is the dimension carrying something a word counter cannot
    fake, and it is the one the raw mean discounts.

    MultiDimensional z-scores each dimension within the group and averages those, so the
    four contribute equally whatever their scales. It is the aggregation the docstring at
    the top of this file already assumed was happening; nothing was applying it, because
    --group_scorer builds one registered name and composes no wrappers.

    What it costs is the absolute scale. A z-scored reward says only how a turn compares
    with the other seven sampled from the same prompt, so `scores` is no longer readable as
    a rubric number and the [0, 2] bound is gone. For GRPO that is no loss - it centres
    within the group anyway - but it does mean the two runs' reward curves cannot be
    compared to each other directly, only their per-dimension `dim_*` metrics can.
    """
    head = PedagogyHead(**kwargs)
    return MultiDimensional(head, dimensions=head.dims, name="pedagogy_z")


register("pedagogy", PedagogyHead)
register("pedagogy_z", normalized)
