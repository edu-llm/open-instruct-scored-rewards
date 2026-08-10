"""Teacher-forced replay tracer for the learning-science semantic atlas.

One forward pass replays a fixed prompt and a fixed response together, so every
position is scored under the same text in every checkpoint. Four position roles
are recorded because they answer different questions:

- ``prompt_last``: the final prompt token, the decision point before any
  response token exists;
- ``generated_i``: the first ``k`` response tokens, where a strategy is
  committed;
- ``boundary_i``: response tokens that close a sentence or clause, where the
  next move is chosen;
- ``content_last``: the final response token, the summary position the earlier
  pilot used on its own.

Dense OLMo-2 and OLMoE-1B-7B are both supported. For OLMoE the tracer stores
true pre-softmax router logits, their softmax, OLMoE's *native* top-k weights
(unnormalized whenever ``norm_topk_prob`` is false), the selected expert
indices, and per-position expert-path identifiers. Those logits are recomputed
from the router weight rather than read off the gate, so every record checks
that their softmax reproduces the gate's own top-k weights and refuses to write
a trace where it does not.

Every artifact carries fingerprints: the exact replayed text, the token ids, the
router weights, and a sampled parameter digest. ``checkpoint_delta`` refuses to
compare two traces whose text fingerprints differ.

Example:
    python -m projects.learning_science_semantic_atlas.trace \
        --input projects/learning_science_semantic_atlas/data/siblings.jsonl \
        --output projects/learning_science_semantic_atlas/results/base.npz \
        --model allenai/OLMoE-1B-7B-0924-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from projects.pedagogy_mech_interp.modeling import (
    decoder_layers,
    is_moe_block,
    moe_norm_topk_prob,
    moe_top_k,
    selected_expert_contributions,
    verify_moe_decomposition,
)
from projects.pedagogy_mech_interp.trace import (
    TraceCollector,
    choose_layers,
    mean_sequence_logprob,
    model_device,
    read_jsonl,
)

SCHEMA = "semantic-atlas-trace/v1"
SITES = ("residual_in", "residual_out", "mlp_in", "mlp_out")
BOUNDARY_CHARACTERS = ".?!;:\n"
TOKEN_HISTOGRAM_BINS = 512
# Router logits are recomputed from the router weight in float32 while the native
# top-k weights come out of a bfloat16 forward, so the two disagree by roughly a
# percent even when they describe the same router. A router this tracer does not
# model - a bias term, or a gate that is not a plain linear softmax - moves them
# by order one. This threshold sits between the two scales rather than near zero.
ROUTER_FIDELITY_TOLERANCE = 0.05
SURFACE_FEATURES = (
    "response_token_count",
    "word_count",
    "character_count",
    "sentence_count",
    "question_count",
    "mean_word_length",
    "digit_fraction",
    "uppercase_fraction",
)

# Record fields. Only ``response`` and one of ``messages``/``prompt`` are
# required; everything else defaults so that a partially annotated corpus still
# traces and the analysis layer decides what it can support.
RESPONSE_FIELDS = ("response", "tutor_turn", "completion")
ID_FIELDS = ("record_id", "variant_id", "id")


@dataclass(frozen=True)
class SlotPlan:
    """The fixed per-record position layout every trace writes."""

    roles: tuple[str, ...]
    bases: tuple[str, ...]
    ordinals: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.roles)


def slot_plan(first_k: int, max_boundaries: int) -> SlotPlan:
    if first_k < 0 or max_boundaries < 0:
        raise ValueError("first_k and max_boundaries must be non-negative")
    roles = ["prompt_last"]
    bases = ["prompt_last"]
    ordinals = [0]
    for index in range(first_k):
        roles.append(f"generated_{index}")
        bases.append("generated")
        ordinals.append(index)
    for index in range(max_boundaries):
        roles.append(f"boundary_{index}")
        bases.append("boundary")
        ordinals.append(index)
    roles.append("content_last")
    bases.append("content_last")
    ordinals.append(0)
    return SlotPlan(tuple(roles), tuple(bases), tuple(ordinals))


@dataclass
class Replay:
    """One tokenized teacher-forced replay with its resolved position slots."""

    record_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    content_start: int
    content_stop: int
    slot_positions: tuple[int, ...]
    text: str
    response_text: str
    boundary_count: int
    merged_prompt_boundary: bool

    @property
    def valid_slots(self) -> list[int]:
        return [slot for slot, position in enumerate(self.slot_positions) if position >= 0]

    @property
    def unique_positions(self) -> list[int]:
        return sorted({position for position in self.slot_positions if position >= 0})


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(payload: str) -> int:
    """A machine-independent non-negative 63-bit identifier."""
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def expert_set_id(indices: np.ndarray) -> int:
    """Identify a selected-expert set, ignoring the order top-k returns."""
    return stable_id(",".join(str(int(value)) for value in np.sort(np.asarray(indices).reshape(-1))))


def expert_path_id(layer_set_ids: np.ndarray) -> int:
    """Identify one token's route through every traced layer."""
    return stable_id("|".join(str(int(value)) for value in np.asarray(layer_set_ids).reshape(-1)))


def first_present(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field_name in fields:
        if record.get(field_name) not in (None, ""):
            return record[field_name]
    return None


def record_text(record: dict[str, Any], tokenizer) -> tuple[str, str]:
    """Return the prompt prefix and the response, exactly as they will be replayed.

    ``messages`` goes through the chat template with a generation prompt.
    ``prompt`` is replayed verbatim, which is what a base model without a chat
    template needs. The response is concatenated rather than templated so the
    replayed string is fixed by this function alone and cannot drift with a
    template's assistant-turn suffix.
    """
    response = first_present(record, RESPONSE_FIELDS)
    if response is None:
        raise ValueError(f"record {record.get('record_id')} has none of {RESPONSE_FIELDS}")
    messages = record.get("messages")
    if messages:
        prefix = tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
    elif record.get("prompt"):
        prefix = str(record["prompt"])
    else:
        raise ValueError(f"record {record.get('record_id')} has neither 'messages' nor 'prompt'")
    return str(prefix), str(response)


def build_replay(
    tokenizer,
    record: dict[str, Any],
    plan: SlotPlan,
    *,
    max_len: int,
    first_k: int,
    max_boundaries: int,
    boundary_characters: str = BOUNDARY_CHARACTERS,
) -> Replay:
    """Tokenize one record and resolve every slot to an absolute token index."""
    record_id = str(first_present(record, ID_FIELDS) or sha256_text(json.dumps(record, sort_keys=True))[:16])
    prefix, response = record_text(record, tokenizer)
    text = prefix + response
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    except (NotImplementedError, ValueError) as exc:
        raise ValueError("atlas replay requires a fast tokenizer with character offsets") from exc
    offsets = encoded.pop("offset_mapping")[0].tolist()
    response_start = len(prefix)
    content = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > response_start and start < len(text) and stop > start
    ]
    if not content:
        raise ValueError(f"{record_id}: tokenizer found no response tokens")
    merged = offsets[content[0]][0] < response_start
    boundaries = [
        index
        for index in content
        if any(character in text[offsets[index][0] : offsets[index][1]] for character in boundary_characters)
    ]

    total = int(encoded.input_ids.shape[1])
    left_cut = max(0, total - max_len)
    content_start = content[0] - left_cut
    content_stop = content[-1] + 1 - left_cut
    if content_start < 0:
        raise ValueError(f"{record_id}: truncation removed part of the response")
    if content_start == 0:
        raise ValueError(f"{record_id}: no prompt token survives before the response")

    def shift(index: int) -> int:
        moved = index - left_cut
        return moved if 0 <= moved < total - left_cut else -1

    positions = [content_start - 1]
    positions += [shift(content[index]) if index < len(content) else -1 for index in range(first_k)]
    positions += [shift(boundaries[index]) if index < len(boundaries) else -1 for index in range(max_boundaries)]
    positions.append(content_stop - 1)
    if len(positions) != len(plan):
        raise ValueError(f"{record_id}: resolved {len(positions)} positions for {len(plan)} slots")
    return Replay(
        record_id=record_id,
        input_ids=encoded.input_ids[:, left_cut:],
        attention_mask=encoded.attention_mask[:, left_cut:],
        content_start=content_start,
        content_stop=content_stop,
        slot_positions=tuple(positions),
        text=text,
        response_text=response,
        boundary_count=len(boundaries),
        merged_prompt_boundary=merged,
    )


def surface_features(response: str, response_token_count: int) -> np.ndarray:
    words = response.split()
    characters = len(response)
    letters = [character for character in response if character.isalpha()]
    return np.asarray(
        [
            float(response_token_count),
            float(len(words)),
            float(characters),
            float(sum(response.count(character) for character in ".!?")),
            float(response.count("?")),
            float(np.mean([len(word) for word in words])) if words else 0.0,
            float(sum(character.isdigit() for character in response) / characters) if characters else 0.0,
            float(sum(character.isupper() for character in letters) / len(letters)) if letters else 0.0,
        ],
        dtype=np.float32,
    )


def token_histogram(token_ids: np.ndarray, bins: int = TOKEN_HISTOGRAM_BINS) -> np.ndarray:
    """A hashed bag-of-tokens control feature that needs no tokenizer to read.

    Token ids are folded modulo ``bins``. Collisions weaken the control's
    resolution; they cannot make it optimistic, because a real lexical signal
    still lands in some bin.
    """
    counts = np.bincount(np.asarray(token_ids, dtype=np.int64) % bins, minlength=bins).astype(np.float32)
    total = counts.sum()
    return counts / total if total else counts


def moe_blocks(model, layer_indices: list[int]) -> dict[int, Any]:
    layers = decoder_layers(model)
    return {index: layers[index].mlp for index in layer_indices if is_moe_block(layers[index].mlp)}


def router_weights(model, layer_indices: list[int]) -> dict[int, torch.Tensor]:
    weights = {}
    for index, block in moe_blocks(model, layer_indices).items():
        weight = getattr(block.gate, "weight", None)
        if weight is not None:
            weights[index] = weight.detach().float().cpu()
    return weights


def parameter_fingerprint(model, *, max_parameters: int = 24, elements: int = 256) -> dict[str, Any]:
    """A sampled digest that distinguishes two checkpoints of the same architecture.

    Sampling keeps this cheap on a 7B model. It identifies checkpoints; it is
    not a proof of equality, so ``checkpoint_delta`` reports differences it
    measures rather than inferring them from this digest.
    """
    parameters = dict(model.named_parameters())
    names = sorted(parameters)
    if not names:
        return {"sampled": [], "shape_sha256": "", "value_sha256": "", "skipped_meta": 0}
    stride = max(1, len(names) // max_parameters)
    sampled = names[::stride][:max_parameters]
    shapes = hashlib.sha256()
    values = hashlib.sha256()
    skipped = 0
    for name in sampled:
        parameter = parameters[name]
        shapes.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
        if parameter.device.type == "meta":
            skipped += 1
            continue
        flat = parameter.detach().reshape(-1)[:elements].float().cpu().numpy()
        values.update(name.encode())
        values.update(np.ascontiguousarray(flat).tobytes())
    return {
        "sampled": sampled,
        "shape_sha256": shapes.hexdigest(),
        "value_sha256": values.hexdigest(),
        "skipped_meta": skipped,
    }


def collect_record(
    collector: TraceCollector, layer_indices: list[int], sites: tuple[str, ...], *, store_expert_outputs: bool
) -> dict[str, np.ndarray]:
    """Pull one record's captures out of a finished collector as ``[layer, position, ...]``."""
    stores = {
        "residual_in": collector.residual_in,
        "residual_out": collector.residual_out,
        "mlp_in": collector.mlp_in,
        "mlp_out": collector.mlp_out,
    }
    captured = {
        site: np.stack([stores[site][index].float().cpu().numpy() for index in layer_indices]).astype(np.float16)
        for site in sites
    }
    if not collector.routing:
        return captured
    partial = [index for index in layer_indices if index not in collector.routing]
    if partial:
        # An upcycled model can mix dense and sparse layers. Stacking those into
        # one array would silently align dense layers with expert axes.
        raise ValueError(f"layers {partial} have no router while others do; trace sparse and dense layers separately")
    logits, probabilities, weights, indices, contributions = [], [], [], [], []
    for index in layer_indices:
        routing = collector.routing[index]
        layer_logits = routing.logits.float()
        logits.append(layer_logits.cpu().numpy())
        probabilities.append(torch.softmax(layer_logits, dim=-1).cpu().numpy())
        weights.append(routing.weights.float().cpu().numpy())
        indices.append(routing.indices.cpu().numpy())
        if store_expert_outputs:
            block = collector.layers[index].mlp
            selected = selected_expert_contributions(block, collector.mlp_in[index], routing)
            contributions.append(selected.float().cpu().numpy())
    captured["router_logits"] = np.stack(logits).astype(np.float16)
    captured["router_probs"] = np.stack(probabilities).astype(np.float16)
    captured["topk_weights"] = np.stack(weights).astype(np.float16)
    captured["topk_indices"] = np.stack(indices).astype(np.int16)
    if store_expert_outputs:
        captured["expert_outputs"] = np.stack(contributions).astype(np.float16)
    return captured


def router_fidelity(captured: dict[str, np.ndarray], *, norm_topk_prob: bool) -> tuple[float, float]:
    """Do the stored router probabilities explain the stored native top-k weights?

    ``router_logits`` is recomputed from the router weight matrix, while
    ``topk_weights`` and ``topk_indices`` are whatever the model's own gate
    returned. The two are only interchangeable if that recomputation is the
    model's real pre-softmax score. A router with a bias term, or a gate that is
    not a plain linear softmax, breaks the identity, and every downstream metric
    that reads probabilities beside native weights would then be describing two
    different routers in one number.

    Returns the largest absolute weight discrepancy and the fraction of positions
    whose selected set matches the top-k of the recomputed probabilities.
    """
    probabilities = captured["router_probs"].astype(np.float64)
    indices = captured["topk_indices"].astype(np.int64)
    native = captured["topk_weights"].astype(np.float64)
    gathered = np.take_along_axis(probabilities, indices, axis=-1)
    expected = gathered
    if norm_topk_prob:
        expected = gathered / np.clip(gathered.sum(axis=-1, keepdims=True), 1e-12, None)
    recomputed = np.argsort(-probabilities, axis=-1)[..., : indices.shape[-1]]
    agreement = (np.sort(recomputed, axis=-1) == np.sort(indices, axis=-1)).all(axis=-1)
    return float(np.abs(native - expected).max()), float(agreement.mean())


def expand_to_slots(value: np.ndarray, rows: list[int], slots: list[int], n_slots: int) -> np.ndarray:
    """Scatter ``[layer, unique_position, ...]`` captures into the fixed slot layout."""
    shaped = (value.shape[0], n_slots, *value.shape[2:])
    out = np.zeros(shaped, dtype=value.dtype)
    out[:, slots] = value[:, rows]
    return out


def routing_paths(topk_indices: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-layer expert-set ids, a cross-layer path id, and the top-1 expert.

    ``topk_indices`` is ``[layer, slot, top_k]``. Invalid slots keep zero so a
    masked slot never contributes a spurious path.
    """
    n_layers, n_slots, _ = topk_indices.shape
    set_ids = np.zeros((n_layers, n_slots), dtype=np.int64)
    top1 = np.full((n_layers, n_slots), -1, dtype=np.int16)
    for layer in range(n_layers):
        for slot in range(n_slots):
            if not valid[slot]:
                continue
            set_ids[layer, slot] = expert_set_id(topk_indices[layer, slot])
            top1[layer, slot] = int(topk_indices[layer, slot, 0])
    path = np.zeros(n_slots, dtype=np.int64)
    for slot in range(n_slots):
        if valid[slot]:
            path[slot] = expert_path_id(set_ids[:, slot])
    return set_ids, path, top1


def dry_run_report(replays: list[Replay], plan: SlotPlan, layers: str, sites: tuple[str, ...]) -> dict[str, Any]:
    filled = np.zeros(len(plan), dtype=np.int64)
    for replay in replays:
        for slot in replay.valid_slots:
            filled[slot] += 1
    return {
        "records": len(replays),
        "slots": list(plan.roles),
        "slots_filled": {role: int(count) for role, count in zip(plan.roles, filled)},
        "requested_layers": layers or "five spread layers chosen from the loaded model",
        "sites": list(sites),
        "merged_prompt_boundary": sum(replay.merged_prompt_boundary for replay in replays),
        "mean_response_tokens": float(np.mean([replay.content_stop - replay.content_start for replay in replays])),
        "mean_boundaries": float(np.mean([replay.boundary_count for replay in replays])),
        "duplicate_text_fingerprints": len(replays) - len({sha256_text(replay.text) for replay in replays}),
    }


@dataclass
class AtlasTrace:
    """A loaded trace with slot-aware and layer-aware accessors."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def get(self, key: str, default=None):
        return self.arrays.get(key, default)

    @property
    def n_records(self) -> int:
        return int(self.arrays["ids"].shape[0])

    @property
    def layers(self) -> list[int]:
        return [int(value) for value in self.arrays["layers"]]

    @property
    def roles(self) -> list[str]:
        return [str(value) for value in self.arrays["position_roles"]]

    @property
    def has_routing(self) -> bool:
        return "router_logits" in self.arrays

    @property
    def sites(self) -> list[str]:
        return [site for site in SITES if site in self.arrays]

    def slot(self, role: str) -> int:
        roles = self.roles
        if role not in roles:
            raise KeyError(f"role {role!r} not traced; available roles are {roles}")
        return roles.index(role)

    def slots_for(self, base: str) -> list[int]:
        return [index for index, value in enumerate(self.arrays["position_bases"]) if str(value) == base]

    def layer_position(self, layer_index: int) -> int:
        layers = self.layers
        if layer_index not in layers:
            raise KeyError(f"layer {layer_index} not traced; available layers are {layers}")
        return layers.index(layer_index)

    def valid(self, slot: int) -> np.ndarray:
        return self.arrays["position_mask"][:, slot].astype(bool)

    def features(self, site: str, layer_index: int, slot: int) -> np.ndarray:
        """``[records, hidden]`` for one site, one layer, and one slot."""
        if site not in self.arrays:
            raise KeyError(f"site {site!r} not traced; available sites are {self.sites}")
        return self.arrays[site][:, self.layer_position(layer_index), slot].astype(np.float32)

    def routing_probs(self, layer_index: int, slot: int) -> np.ndarray:
        return self.arrays["router_probs"][:, self.layer_position(layer_index), slot].astype(np.float32)


def load_atlas_trace(path: Path | str) -> AtlasTrace:
    with np.load(Path(path), allow_pickle=False) as blob:
        arrays = {key: blob[key] for key in blob.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}, found {metadata.get('schema')!r}")
    return AtlasTrace(arrays=arrays, metadata=metadata)


def parse_sites(value: str) -> tuple[str, ...]:
    if not value:
        return SITES
    chosen = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [site for site in chosen if site not in SITES]
    if unknown:
        raise ValueError(f"unknown sites {unknown}; choose from {list(SITES)}")
    return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--revision", default="")
    parser.add_argument("--adapter", default="", help="PEFT adapter directory or zip applied on top of --model")
    parser.add_argument("--layers", default="", help="comma-separated layer indices; empty spreads five layers")
    parser.add_argument("--sites", default="", help=f"comma-separated subset of {list(SITES)}")
    parser.add_argument("--first-k", type=int, default=8, help="number of leading response tokens to trace")
    parser.add_argument("--max-boundaries", type=int, default=6)
    parser.add_argument("--boundary-characters", default=BOUNDARY_CHARACTERS)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-map", default="")
    parser.add_argument("--store-expert-outputs", action="store_true")
    parser.add_argument("--router-weights", type=Path, default=None, help="sidecar npz for the routing decomposition")
    parser.add_argument("--verify-decomposition", action="store_true")
    parser.add_argument("--require-clean-boundary", action="store_true", help="fail when a token straddles the prompt")
    parser.add_argument("--dry-run", action="store_true", help="tokenize and report the plan without loading weights")
    return parser


def load_tokenizer(model_name: str, revision: str):
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(model_name, revision=revision or None)


@dataclass
class TraceResult:
    """Stacked per-record arrays plus everything the model itself determines."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    router_weights: dict[int, torch.Tensor]


def trace_records(
    model,
    rows: list[dict[str, Any]],
    replays: list[Replay],
    *,
    plan: SlotPlan,
    layer_indices: list[int],
    sites: tuple[str, ...] = SITES,
    store_expert_outputs: bool = False,
    verify_decomposition: bool = False,
    progress_every: int = 25,
    router_fidelity_tolerance: float = ROUTER_FIDELITY_TOLERANCE,
) -> TraceResult:
    """Replay every record once and stack the captures into the slot layout."""
    device = model_device(model)
    blocks = moe_blocks(model, layer_indices)
    normalized = moe_norm_topk_prob(next(iter(blocks.values()))) if blocks else None
    text_columns = (
        "ids",
        "item_ids",
        "sibling_ids",
        "constructs",
        "domains",
        "sources",
        "template_ids",
        "text_sha256",
        "token_sha256",
    )
    numeric_columns = (
        "labels",
        "named_labels",
        "enacted_labels",
        "quality",
        "response_logprob",
        "prompt_tokens",
        "response_tokens",
        "boundary_count",
    )
    stacked: dict[str, list[np.ndarray]] = {}
    columns: dict[str, list[Any]] = {key: [] for key in text_columns}
    numeric: dict[str, list[float]] = {key: [] for key in numeric_columns}
    slot_index = np.full((len(replays), len(plan)), -1, dtype=np.int32)
    slot_mask = np.zeros((len(replays), len(plan)), dtype=bool)
    slot_tokens = np.full((len(replays), len(plan)), -1, dtype=np.int32)
    verified = False
    weight_errors: list[float] = []
    selection_agreements: list[float] = []

    with torch.inference_mode():
        for row_index, (row, replay) in enumerate(zip(rows, replays)):
            input_ids = replay.input_ids.to(device)
            attention_mask = replay.attention_mask.to(device)
            positions = replay.unique_positions
            with TraceCollector(model, layer_indices, positions) as collector:
                output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            captured = collect_record(collector, layer_indices, sites, store_expert_outputs=store_expert_outputs)
            if blocks:
                error, agreement = router_fidelity(captured, norm_topk_prob=bool(normalized))
                if error > router_fidelity_tolerance:
                    raise ValueError(
                        f"{replay.record_id}: the softmax of the recomputed router logits misses this model's "
                        f"own top-k weights by {error:.4g} (> {router_fidelity_tolerance:.4g}). The stored "
                        "logits are therefore not this model's pre-softmax router score, and probabilities "
                        "stored beside native weights would describe two different routers"
                    )
                weight_errors.append(error)
                selection_agreements.append(agreement)
            valid = replay.valid_slots
            rows_for_slots = [positions.index(replay.slot_positions[slot]) for slot in valid]
            for key, value in captured.items():
                stacked.setdefault(key, []).append(expand_to_slots(value, rows_for_slots, valid, len(plan)))

            valid_mask = np.zeros(len(plan), dtype=bool)
            valid_mask[valid] = True
            slot_mask[row_index] = valid_mask
            for slot in valid:
                slot_index[row_index, slot] = replay.slot_positions[slot]
                slot_tokens[row_index, slot] = int(replay.input_ids[0, replay.slot_positions[slot]])

            if blocks:
                set_ids, path, top1 = routing_paths(stacked["topk_indices"][-1], valid_mask)
                stacked.setdefault("expert_set_id", []).append(set_ids)
                stacked.setdefault("expert_path_id", []).append(path)
                stacked.setdefault("top1_expert", []).append(top1)
            if verify_decomposition and blocks and not verified:
                index = next(iter(blocks))
                error = verify_moe_decomposition(blocks[index], collector.mlp_in[index])
                print(f"expert decomposition max error: {error:.6g}")
                verified = True

            response_ids = replay.input_ids[0, replay.content_start : replay.content_stop].cpu().numpy()
            stacked.setdefault("token_histogram", []).append(token_histogram(response_ids))
            stacked.setdefault("surface_features", []).append(
                surface_features(replay.response_text, int(response_ids.size))
            )
            columns["ids"].append(replay.record_id)
            columns["item_ids"].append(str(row.get("item_id", "")))
            columns["sibling_ids"].append(str(row.get("sibling_id", row.get("pair_id", ""))))
            columns["constructs"].append(str(row.get("construct", row.get("concept", ""))))
            columns["domains"].append(str(row.get("domain", "")))
            columns["sources"].append(str(row.get("source", "")))
            columns["template_ids"].append(str(row.get("template_id", "")))
            columns["text_sha256"].append(sha256_text(replay.text))
            columns["token_sha256"].append(
                hashlib.sha256(np.ascontiguousarray(replay.input_ids.cpu().numpy()).tobytes()).hexdigest()
            )
            numeric["labels"].append(int(row.get("label", -1)))
            numeric["named_labels"].append(int(row.get("named_label", -1)))
            numeric["enacted_labels"].append(int(row.get("enacted_label", -1)))
            numeric["quality"].append(float(row.get("quality", float("nan"))))
            logits = getattr(output, "logits", output)
            numeric["response_logprob"].append(
                mean_sequence_logprob(logits, input_ids, replay.content_start, replay.content_stop)
            )
            numeric["prompt_tokens"].append(replay.content_start)
            numeric["response_tokens"].append(replay.content_stop - replay.content_start)
            numeric["boundary_count"].append(replay.boundary_count)
            if progress_every and (row_index + 1) % progress_every == 0:
                print(f"  {row_index + 1}/{len(replays)}")

    weights = router_weights(model, layer_indices)
    arrays: dict[str, np.ndarray] = {
        "layers": np.asarray(layer_indices, dtype=np.int16),
        "position_roles": np.asarray(plan.roles),
        "position_bases": np.asarray(plan.bases),
        "position_ordinals": np.asarray(plan.ordinals, dtype=np.int16),
        "position_index": slot_index,
        "position_mask": slot_mask,
        "position_token_id": slot_tokens,
    }
    arrays.update({key: np.asarray(values) for key, values in columns.items()})
    arrays.update(
        {
            key: np.asarray(values, dtype=np.float32 if key in {"quality", "response_logprob"} else np.int32)
            for key, values in numeric.items()
        }
    )
    arrays.update({key: np.stack(values) for key, values in stacked.items()})
    metadata = {
        "schema": SCHEMA,
        "layers": layer_indices,
        "sites": list(sites),
        "slots": list(plan.roles),
        "token_histogram_bins": TOKEN_HISTOGRAM_BINS,
        "surface_features": list(SURFACE_FEATURES),
        "mixture_of_experts": bool(blocks),
        "top_k": moe_top_k(next(iter(blocks.values()))) if blocks else None,
        "norm_topk_prob": normalized,
        # Only a sparse trace stores top-k weights at all, so a dense trace makes
        # no claim about them rather than an unfalsifiable true one.
        "topk_weights_are_native": True if blocks else None,
        "router_fidelity": {
            "max_native_weight_error": max(weight_errors),
            "mean_selection_agreement": float(np.mean(selection_agreements)),
            "tolerance": router_fidelity_tolerance,
        }
        if weight_errors
        else None,
        "expert_outputs_stored": bool(store_expert_outputs and blocks),
        "device": str(device),
        "torch_version": torch.__version__,
        "parameters": parameter_fingerprint(model),
        "router_weight_sha256": {
            str(index): hashlib.sha256(np.ascontiguousarray(weight.numpy()).tobytes()).hexdigest()
            for index, weight in weights.items()
        },
        "merged_prompt_boundary": sum(replay.merged_prompt_boundary for replay in replays),
    }
    return TraceResult(arrays=arrays, metadata=metadata, router_weights=weights)


def write_trace(path: Path, result: TraceResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=np.asarray(json.dumps(result.metadata)), **result.arrays)


def write_router_weights(path: Path, result: TraceResult, *, model: str, adapter: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(
            json.dumps(
                {
                    "schema": "semantic-atlas-router-weights/v1",
                    "model": model,
                    "adapter": adapter,
                    "layers": sorted(result.router_weights),
                }
            )
        ),
        **{f"layer_{index}": weight.numpy().astype(np.float32) for index, weight in result.router_weights.items()},
    )


def main() -> None:
    args = build_parser().parse_args()
    sites = parse_sites(args.sites)
    plan = slot_plan(args.first_k, args.max_boundaries)
    rows = read_jsonl(args.input, args.limit)
    if not rows:
        raise SystemExit("input contains no records")

    tokenizer = load_tokenizer(args.model, args.revision)
    replays = [
        build_replay(
            tokenizer,
            row,
            plan,
            max_len=args.max_len,
            first_k=args.first_k,
            max_boundaries=args.max_boundaries,
            boundary_characters=args.boundary_characters,
        )
        for row in rows
    ]
    merged = [replay.record_id for replay in replays if replay.merged_prompt_boundary]
    if merged and args.require_clean_boundary:
        raise SystemExit(f"{len(merged)} records merge prompt and response into one token, first: {merged[:3]}")

    if args.dry_run:
        report = dry_run_report(replays, plan, args.layers, sites)
        report["merged_prompt_boundary_ids"] = merged[:10]
        print(json.dumps(report, indent=2))
        return

    from projects.learning_science_semantic_atlas.checkpoint_delta import load_traced_model  # noqa: PLC0415

    model, _ = load_traced_model(
        args.model, revision=args.revision, adapter=args.adapter, device_map=args.device_map, tokenizer=tokenizer
    )
    layer_indices = choose_layers(len(decoder_layers(model)), args.layers)
    print(f"{args.model}: tracing layers {layer_indices} at {len(plan)} slots")

    result = trace_records(
        model,
        rows,
        replays,
        plan=plan,
        layer_indices=layer_indices,
        sites=sites,
        store_expert_outputs=args.store_expert_outputs,
        verify_decomposition=args.verify_decomposition,
    )
    result.metadata.update(
        model=args.model,
        revision=args.revision or None,
        adapter=args.adapter or None,
        input=str(args.input),
        input_sha256=sha256_file(args.input),
        first_k=args.first_k,
        max_boundaries=args.max_boundaries,
        boundary_characters=args.boundary_characters,
        max_len=args.max_len,
    )
    write_trace(args.output, result)
    print(f"wrote {args.output} ({len(replays)} records, {len(plan)} slots)")

    if args.router_weights and result.router_weights:
        write_router_weights(args.router_weights, result, model=args.model, adapter=args.adapter or None)
        print(f"wrote {args.router_weights} ({len(result.router_weights)} router matrices)")


if __name__ == "__main__":
    main()
