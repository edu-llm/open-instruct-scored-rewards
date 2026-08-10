"""Token-aligned hidden-state and OLMoE routing tracer.

The tracer records the final tutor content token by default. It captures
residual inputs/outputs and MLP outputs in both models, plus router logits,
selected experts, normalized weights, and weighted selected-expert outputs in
OLMoE.

Example:
    python -m projects.pedagogy_mech_interp.trace \
        --input projects/pedagogy_mech_interp/data/pairs/discovery_train.jsonl \
        --output projects/pedagogy_mech_interp/results/olmoe_discovery.npz \
        --model allenai/OLMoE-1B-7B-0924-Instruct
"""

from __future__ import annotations

import argparse
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from projects.pedagogy_mech_interp.modeling import (
    Routing,
    decoder_layers,
    is_moe_block,
    routing_from_gate_output,
    selected_expert_contributions,
    tensor_from_output,
    verify_moe_decomposition,
)

TEACHER_SYSTEM = """You are a tutor helping a student with a test question. The student cannot see your instructions.

Guide the student toward understanding. Adapt the amount of help to what the student has shown, keep factual claims
correct, and leave a meaningful next step for the student when they can take one."""


@dataclass
class TokenizedRecord:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    content_start: int
    content_stop: int

    @property
    def last_content_position(self) -> int:
        return self.content_stop - 1


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def context_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": f"Question the student is working on:\n{record['question']}"},
        {"role": "user", "content": str(record["student_before"])},
    ]


def tokenize_record(tokenizer, record: dict[str, Any], max_len: int) -> TokenizedRecord:
    messages = context_messages(record)
    prefix_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    whole_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": record["tutor_turn"]}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not whole_text.startswith(prefix_text):
        raise ValueError(f"{record.get('variant_id')}: complete chat does not preserve generation prefix")
    response_start = len(prefix_text)
    response_stop = response_start + len(str(record["tutor_turn"]))
    try:
        whole = tokenizer(
            whole_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except (NotImplementedError, ValueError) as exc:
        raise ValueError("trace tokenization requires a fast tokenizer with character offsets") from exc
    offsets = whole.pop("offset_mapping")[0].tolist()
    content_positions = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > response_start and start < response_stop and stop > start
    ]
    if not content_positions:
        raise ValueError(f"{record.get('variant_id')}: tokenizer found no tutor content tokens")
    total_length = int(whole.input_ids.shape[1])
    left_cut = max(0, total_length - max_len)
    input_ids = whole.input_ids[:, left_cut:]
    attention_mask = whole.attention_mask[:, left_cut:]
    start = content_positions[0] - left_cut
    stop = content_positions[-1] + 1 - left_cut
    if stop <= start:
        raise ValueError(f"{record.get('variant_id')}: truncation removed the tutor response")
    if start < 0:
        raise ValueError(f"{record.get('variant_id')}: truncation removed part of the tutor response")
    return TokenizedRecord(input_ids=input_ids, attention_mask=attention_mask, content_start=start, content_stop=stop)


def mean_sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor, start: int, stop: int) -> float:
    if start == 0:
        raise ValueError("cannot score content beginning at token zero")
    token_logits = logits[:, start - 1 : stop - 1].float()
    targets = input_ids[:, start:stop]
    values = token_logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(values.mean().item())


class TraceCollector(AbstractContextManager):
    """Hooks one forward and immediately narrows captures to requested positions."""

    def __init__(self, model, layer_indices: list[int], positions: list[int]):
        self.model = model
        self.layers = decoder_layers(model)
        self.layer_indices = layer_indices
        self.positions = positions
        self.handles = []
        self.residual_in: dict[int, torch.Tensor] = {}
        self.residual_out: dict[int, torch.Tensor] = {}
        self.mlp_in: dict[int, torch.Tensor] = {}
        self.mlp_out: dict[int, torch.Tensor] = {}
        self.routing: dict[int, Routing] = {}

    def _slice(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        return value[0, self.positions].detach()

    def __enter__(self):
        for index in self.layer_indices:
            layer = self.layers[index]

            def layer_pre(_module, args, layer_index=index):
                self.residual_in[layer_index] = self._slice(args[0])

            def layer_post(_module, _args, output, layer_index=index):
                self.residual_out[layer_index] = self._slice(tensor_from_output(output))

            def mlp_pre(_module, args, layer_index=index):
                self.mlp_in[layer_index] = self._slice(args[0])

            def mlp_post(_module, _args, output, layer_index=index):
                self.mlp_out[layer_index] = self._slice(tensor_from_output(output))

            self.handles.append(layer.register_forward_pre_hook(layer_pre))
            self.handles.append(layer.register_forward_hook(layer_post))
            self.handles.append(layer.mlp.register_forward_pre_hook(mlp_pre))
            self.handles.append(layer.mlp.register_forward_hook(mlp_post))
            if is_moe_block(layer.mlp):

                def gate_post(_module, gate_args, output, layer_index=index, block=layer.mlp):
                    routing = routing_from_gate_output(block, gate_args[0], output)
                    self.routing[layer_index] = Routing(
                        logits=routing.logits[self.positions].detach(),
                        weights=routing.weights[self.positions].detach(),
                        indices=routing.indices[self.positions].detach(),
                    )

                self.handles.append(layer.mlp.gate.register_forward_hook(gate_post))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def arrays(self) -> dict[str, np.ndarray]:
        def stack(values: dict[int, torch.Tensor], dtype=np.float16) -> np.ndarray:
            return np.stack([values[index].float().cpu().numpy() for index in self.layer_indices]).astype(dtype)

        result = {
            "residual_in": stack(self.residual_in),
            "residual_out": stack(self.residual_out),
            "mlp_in": stack(self.mlp_in),
            "mlp_out": stack(self.mlp_out),
        }
        if self.routing:
            router_logits = []
            topk_weights = []
            topk_indices = []
            contributions = []
            for index in self.layer_indices:
                routing = self.routing[index]
                router_logits.append(routing.logits.float().cpu().numpy())
                topk_weights.append(routing.weights.float().cpu().numpy())
                topk_indices.append(routing.indices.cpu().numpy())
                block = self.layers[index].mlp
                selected = selected_expert_contributions(block, self.mlp_in[index], routing)
                contributions.append(selected.float().cpu().numpy())
            result.update(
                router_logits=np.stack(router_logits).astype(np.float16),
                router_probs=np.stack(
                    [
                        torch.softmax(self.routing[index].logits.float(), dim=-1).cpu().numpy()
                        for index in self.layer_indices
                    ]
                ).astype(np.float16),
                topk_weights=np.stack(topk_weights).astype(np.float16),
                topk_indices=np.stack(topk_indices).astype(np.int16),
                expert_outputs=np.stack(contributions).astype(np.float16),
            )
        return result


def choose_layers(n_layers: int, requested: str) -> list[int]:
    if requested:
        values = sorted({int(value) for value in requested.split(",")})
    else:
        values = sorted({round(fraction * (n_layers - 1)) for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)})
    if not values or values[0] < 0 or values[-1] >= n_layers:
        raise ValueError(f"layers must be in [0, {n_layers - 1}], got {values}")
    return values


def model_device(model) -> torch.device:
    return next(parameter for parameter in model.parameters() if parameter.device.type != "meta").device


def load_model(model_name: str, revision: str, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision or None)
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision or None, **kwargs).eval()
    if not device_map:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--revision", default="")
    parser.add_argument("--layers", default="")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-map", default="", help="for example 'auto'; empty moves the complete model to one device")
    parser.add_argument("--verify-decomposition", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input, args.limit)
    if not rows:
        raise SystemExit("input contains no records")
    model, tokenizer = load_model(args.model, args.revision, args.device_map)
    layers = decoder_layers(model)
    selected_layers = choose_layers(len(layers), args.layers)
    device = model_device(model)
    print(f"{args.model}: {len(layers)} layers; tracing {selected_layers} on {device}")

    arrays: dict[str, list[np.ndarray]] = {}
    ids, pair_ids, item_ids, concepts, domains, template_ids, sources = [], [], [], [], [], [], []
    labels, logprobs, word_counts = [], [], []
    verified = False
    with torch.inference_mode():
        for row_index, row in enumerate(rows):
            tokenized = tokenize_record(tokenizer, row, args.max_len)
            tokenized.input_ids = tokenized.input_ids.to(device)
            tokenized.attention_mask = tokenized.attention_mask.to(device)
            position = tokenized.last_content_position
            with TraceCollector(model, selected_layers, [position]) as collector:
                output = model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask, use_cache=False)
            captured = collector.arrays()
            for key, value in captured.items():
                # Position axis has length one; persist compact [layer, ...].
                arrays.setdefault(key, []).append(value[:, 0])
            if args.verify_decomposition and not verified and collector.routing:
                index = selected_layers[0]
                full_state = collector.mlp_in[index]
                error = verify_moe_decomposition(layers[index].mlp, full_state)
                print(f"expert decomposition max error: {error:.6g}")
                verified = True

            ids.append(row["variant_id"])
            pair_ids.append(row["pair_id"])
            item_ids.append(row["item_id"])
            concepts.append(row["concept"])
            domains.append(row.get("domain", ""))
            template_ids.append(row.get("template_id", ""))
            sources.append(row.get("source", ""))
            labels.append(int(row.get("label", -1)))
            logprobs.append(
                mean_sequence_logprob(
                    output.logits, tokenized.input_ids, tokenized.content_start, tokenized.content_stop
                )
            )
            word_counts.append(int(row.get("word_count", len(str(row["tutor_turn"]).split()))))
            if (row_index + 1) % 25 == 0:
                print(f"  {row_index + 1}/{len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "pedagogy-mech-trace/v1",
        "model": args.model,
        "revision": args.revision or None,
        "input": str(args.input),
        "layers": selected_layers,
        "position": "last_tutor_content_token",
        "max_len": args.max_len,
    }
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids),
        pair_ids=np.asarray(pair_ids),
        item_ids=np.asarray(item_ids),
        concepts=np.asarray(concepts),
        domains=np.asarray(domains),
        template_ids=np.asarray(template_ids),
        sources=np.asarray(sources),
        labels=np.asarray(labels, dtype=np.int8),
        layers=np.asarray(selected_layers, dtype=np.int16),
        response_logprob=np.asarray(logprobs, dtype=np.float32),
        word_count=np.asarray(word_counts, dtype=np.int16),
        metadata=np.asarray(json.dumps(metadata)),
        **{key: np.stack(values) for key, values in arrays.items()},
    )
    print(f"wrote {args.output} ({len(rows)} records)")


if __name__ == "__main__":
    main()
