#!/usr/bin/env python
"""Full-parameter OLMoE continued pretraining on the Dolmino stage-2 mix."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import fsspec
import torch
from accelerate import Accelerator, DataLoaderConfiguration, InitProcessGroupKwargs
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from huggingface_hub import HfApi
from projects.olmoe_full_finetune.s3_checkpoints import S3CheckpointStore
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed

from open_instruct.model_utils import save_with_accelerate
from open_instruct.utils import clean_last_n_checkpoints, get_last_checkpoint

DEFAULT_CONFIGS = ["dclm", "flan", "pes2o", "wiki", "stackexchange", "math"]
# Published 50B-token Dolmino stage-2 mixture. Values sum to 1.0001 because
# the source table is rounded; they are normalized before interleaving.
DEFAULT_PROBABILITIES = [0.472, 0.166, 0.0585, 0.0711, 0.0245, 0.208]


class PackedTokenStream(IterableDataset):
    """Tokenize documents and concatenate them into fixed-length CLM examples."""

    def __init__(self, source: HFIterableDataset, tokenizer: Any, sequence_length: int):
        self.source = source
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.buffer: list[int] = []

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id for document separation")

        for example in self.source:
            text = example.get("text")
            if not isinstance(text, str) or not text:
                continue
            self.buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))
            self.buffer.append(eos_token_id)
            while len(self.buffer) >= self.sequence_length:
                input_ids = torch.tensor(self.buffer[: self.sequence_length], dtype=torch.long)
                del self.buffer[: self.sequence_length]
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids.clone(),
                }

    def state_dict(self) -> dict[str, Any]:
        return {"source": self.source.state_dict(), "buffer": list(self.buffer)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.source.load_state_dict(state_dict["source"])
        self.buffer = list(state_dict["buffer"])


class TokenMixtureStream(IterableDataset):
    """Sample packed sequences so probabilities represent token yield."""

    def __init__(self, streams: list[PackedTokenStream], probabilities: list[float], seed: int):
        self.streams = streams
        probability_sum = sum(probabilities)
        self.probabilities = [probability / probability_sum for probability in probabilities]
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        iterators = [iter(stream) for stream in self.streams]
        active_indices = list(range(len(iterators)))
        while active_indices:
            active_weights = [self.probabilities[index] for index in active_indices]
            index = self.rng.choices(active_indices, weights=active_weights, k=1)[0]
            try:
                yield next(iterators[index])
            except StopIteration:
                active_indices.remove(index)

    def state_dict(self) -> dict[str, Any]:
        return {"streams": [stream.state_dict() for stream in self.streams], "rng_state": self.rng.getstate()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for stream, stream_state in zip(self.streams, state_dict["streams"], strict=True):
            stream.load_state_dict(stream_state)
        self.rng.setstate(state_dict["rng_state"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--model_revision", default="6d84c48581ece794365f2b8e9cfb043c68ade9c5")
    parser.add_argument("--dataset_name", default="allenai/dolmino-mix-1124")
    parser.add_argument("--dataset_revision", default="a319f19eef1e257417b11ea8c30da266ae175557")
    parser.add_argument("--dataset_configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--mix_probabilities", nargs="+", type=float, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--sequence_length", type=int, default=4096)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--max_train_steps", type=int, default=11_931)
    parser.add_argument("--learning_rate", type=float, default=6.1499e-5)
    parser.add_argument("--lr_scheduler_type", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--router_z_loss_coef", type=float, default=0.001)
    parser.add_argument("--checkpointing_steps", type=int, default=20)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=1)
    parser.add_argument(
        "--empty_cache_steps",
        type=int,
        default=1,
        help="Release unused CUDA allocator blocks every N optimizer steps; 0 disables it.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--remote_checkpoint_dir",
        help="Optional s3:// prefix used to restore and persist resumable DeepSpeed checkpoints.",
    )
    parser.add_argument(
        "--remote_output_dir", help="Optional s3:// prefix whose final/ directory receives the exported model."
    )
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--wandb_project", default="olmoe-midtraining")
    parser.add_argument("--wandb_entity", default="eduLLM")
    args = parser.parse_args()

    if len(args.dataset_configs) != len(args.mix_probabilities):
        parser.error("--dataset_configs and --mix_probabilities must have the same length")
    if any(probability <= 0 for probability in args.mix_probabilities):
        parser.error("all mixture probabilities must be positive")
    if args.empty_cache_steps < 0:
        parser.error("--empty_cache_steps must be non-negative")
    return args


def save_checkpoint(
    accelerator: Accelerator,
    stream: TokenMixtureStream,
    output_dir: Path,
    completed_steps: int,
    keep_last_n: int,
    remote_store: S3CheckpointStore | None = None,
) -> None:
    checkpoint_dir = output_dir / f"step_{completed_steps}"
    accelerator.save_state(str(checkpoint_dir))
    torch.save(stream.state_dict(), checkpoint_dir / f"stream_state_rank{accelerator.process_index}.pt")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps({"completed_steps": completed_steps}, indent=2) + "\n"
        )
        (checkpoint_dir / "COMPLETED").write_text("COMPLETED\n")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if remote_store is not None:
            remote_store.upload_step(checkpoint_dir)
        clean_last_n_checkpoints(str(output_dir), keep_last_n)
    accelerator.wait_for_everyone()


def get_last_complete_checkpoint(output_dir: Path) -> Path | None:
    """Return the newest checkpoint that finished on every rank."""
    latest = get_last_checkpoint(str(output_dir))
    if latest is not None and (Path(latest) / "COMPLETED").is_file():
        return Path(latest)

    complete_checkpoints = [
        checkpoint
        for checkpoint in output_dir.glob("step_*")
        if checkpoint.is_dir() and (checkpoint / "COMPLETED").is_file()
    ]
    return max(complete_checkpoints, key=lambda checkpoint: int(checkpoint.name.removeprefix("step_")), default=None)


def stream_jsonl_files(files: list[str], dataset_name: str, dataset_revision: str) -> Iterator[dict[str, str]]:
    """Read only text from heterogeneous JSONL schemas in Dolmino's math tree."""
    for filename in files:
        path = f"hf://datasets/{dataset_name}@{dataset_revision}/{filename}"
        with fsspec.open(path, "rt", compression="infer") as handle:
            for line in handle:
                example = json.loads(line)
                text = example.get("text")
                if isinstance(text, str) and text:
                    yield {"text": text}


def load_source(args: argparse.Namespace, config_name: str) -> HFIterableDataset:
    if config_name != "math":
        return load_dataset(
            args.dataset_name, config_name, split="train", streaming=True, revision=args.dataset_revision
        )

    # The Hub's broad `math` config mixes incompatible metadata schemas. Reading
    # only `text` avoids Arrow attempting to cast those unrelated metadata fields.
    math_files = [
        filename
        for filename in HfApi().list_repo_files(args.dataset_name, repo_type="dataset", revision=args.dataset_revision)
        if filename.startswith("data/math/")
    ]
    if not math_files:
        raise RuntimeError("No Dolmino math files were found at the pinned dataset revision")
    return HFIterableDataset.from_generator(
        stream_jsonl_files,
        gen_kwargs={"files": math_files, "dataset_name": args.dataset_name, "dataset_revision": args.dataset_revision},
    )


def keep_replica_example(_example: dict[str, Any], index: int, replica_count: int, replica_index: int) -> bool:
    return index % replica_count == replica_index


def load_sources(args: argparse.Namespace, accelerator: Accelerator) -> list[HFIterableDataset]:
    streams = []
    for offset, config_name in enumerate(args.dataset_configs):
        stream = load_source(args, config_name)
        stream = stream.shuffle(seed=args.seed + offset, buffer_size=args.shuffle_buffer_size)
        if stream.num_shards >= accelerator.num_processes:
            stream = stream.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
        else:
            # Some sources (currently Wikipedia) have fewer physical files than
            # ranks. Pair ranks on each file, then partition its examples.
            shard_index = accelerator.process_index % stream.num_shards
            replica_index = accelerator.process_index // stream.num_shards
            replica_count = (accelerator.num_processes - shard_index + stream.num_shards - 1) // stream.num_shards
            stream = stream.shard(num_shards=stream.num_shards, index=shard_index)
            stream = stream.filter(
                keep_replica_example,
                with_indices=True,
                fn_kwargs={"replica_count": replica_count, "replica_index": replica_index},
            )
        streams.append(stream)
    return streams


def router_z_loss(router_logits: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
    if not router_logits:
        raise ValueError("OLMoE did not return router logits")
    losses = [torch.logsumexp(logits.float(), dim=-1).square().mean() for logits in router_logits]
    return torch.stack(losses).mean()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb" if args.with_tracking else None,
        # The stream is manually partitioned across ranks. This tells
        # Accelerate that one scheduler step represents one global batch,
        # preventing it from stepping once per process.
        dataloader_config=DataLoaderConfiguration(split_batches=True),
        # Rank zero may spend several minutes transferring a ~97 GB checkpoint
        # while the other ranks wait at a barrier.
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )
    if accelerator.state.deepspeed_plugin is not None:
        deepspeed_config = accelerator.state.deepspeed_plugin.deepspeed_config
        deepspeed_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        deepspeed_config["train_batch_size"] = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes
        )
    set_seed(args.seed)

    remote_checkpoint_store = None
    if accelerator.is_main_process and args.remote_checkpoint_dir:
        remote_checkpoint_store = S3CheckpointStore(args.remote_checkpoint_dir)
        restored = remote_checkpoint_store.download_latest(args.output_dir)
        if restored is not None:
            accelerator.print(f"Restored remote checkpoint {restored.name} from {args.remote_checkpoint_dir}")
    accelerator.wait_for_everyone()

    if args.with_tracking:
        accelerator.init_trackers(
            args.wandb_project,
            config=vars(args) | {"output_dir": str(args.output_dir)},
            init_kwargs={"wandb": {"entity": args.wandb_entity, "name": args.run_name}},
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.model_revision, use_fast=True)
    sources = load_sources(args, accelerator)
    stream = TokenMixtureStream(
        [PackedTokenStream(source, tokenizer, args.sequence_length) for source in sources],
        args.mix_probabilities,
        seed=args.seed,
    )

    resume_checkpoint = get_last_complete_checkpoint(args.output_dir)
    completed_steps = 0
    if resume_checkpoint is not None:
        stream_state_path = resume_checkpoint / f"stream_state_rank{accelerator.process_index}.pt"
        stream.load_state_dict(torch.load(stream_state_path, map_location="cpu", weights_only=False))
        completed_steps = json.loads((resume_checkpoint / "trainer_state.json").read_text())["completed_steps"]

    dataloader = DataLoader(
        stream, batch_size=args.per_device_train_batch_size, num_workers=0, pin_memory=True, drop_last=True
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path, revision=args.model_revision)
    config.output_router_logits = True
    config.router_aux_loss_coef = args.router_aux_loss_coef
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    model.gradient_checkpointing_enable()

    embedding_parameter_ids = {id(parameter) for parameter in model.get_input_embeddings().parameters()}
    decay_parameters = [parameter for parameter in model.parameters() if id(parameter) not in embedding_parameter_ids]
    embedding_parameters = [parameter for parameter in model.parameters() if id(parameter) in embedding_parameter_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": embedding_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Sources are explicitly sharded before packing so their resumable state is
    # rank-local. Do not let Accelerate shard this dataloader a second time.
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    if resume_checkpoint is not None:
        accelerator.print(f"Resuming from {resume_checkpoint} at optimizer step {completed_steps}")
        accelerator.load_state(str(resume_checkpoint))

    accelerator.print(
        f"dataset={args.dataset_name}@{args.dataset_revision} configs={args.dataset_configs} "
        f"probabilities={args.mix_probabilities}"
    )
    accelerator.print(
        f"full_parameters=true scheduler={args.lr_scheduler_type} peak_lr={args.learning_rate} "
        f"global_batch={args.per_device_train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes} "
        f"sequence_length={args.sequence_length} max_steps={args.max_train_steps}"
    )

    model.train()
    optimizer.zero_grad()
    running_loss = torch.zeros((), device=accelerator.device)
    running_aux_loss = torch.zeros((), device=accelerator.device)
    running_z_loss = torch.zeros((), device=accelerator.device)
    micro_steps = 0
    log_started = time.perf_counter()

    while completed_steps < args.max_train_steps:
        for batch in dataloader:
            batch = {key: value.to(accelerator.device, non_blocking=True) for key, value in batch.items()}
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False, output_router_logits=True)
                z_loss = router_z_loss(outputs.router_logits)
                loss = outputs.loss + args.router_z_loss_coef * z_loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().float()
            running_aux_loss += outputs.aux_loss.detach().float()
            running_z_loss += z_loss.detach().float()
            micro_steps += 1
            del batch, loss, outputs, z_loss

            if not accelerator.sync_gradients:
                continue

            completed_steps += 1
            mean_loss = accelerator.gather(running_loss).mean().item() / micro_steps
            mean_aux_loss = accelerator.gather(running_aux_loss).mean().item() / micro_steps
            mean_z_loss = accelerator.gather(running_z_loss).mean().item() / micro_steps
            global_tokens = (
                args.sequence_length
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * accelerator.num_processes
            )
            elapsed = time.perf_counter() - log_started
            metrics = {
                "train_loss": mean_loss,
                "aux_loss": mean_aux_loss,
                "router_z_loss": mean_z_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "tokens": completed_steps * global_tokens,
                "tokens_per_second": global_tokens / elapsed,
            }
            accelerator.print(f"step={completed_steps} metrics={metrics}")
            if args.with_tracking:
                accelerator.log(metrics, step=completed_steps)
            running_loss.zero_()
            running_aux_loss.zero_()
            running_z_loss.zero_()
            micro_steps = 0
            log_started = time.perf_counter()

            if completed_steps % args.checkpointing_steps == 0 or completed_steps == args.max_train_steps:
                save_checkpoint(
                    accelerator,
                    stream,
                    args.output_dir,
                    completed_steps,
                    args.keep_last_n_checkpoints,
                    remote_checkpoint_store,
                )
            if args.empty_cache_steps and completed_steps % args.empty_cache_steps == 0:
                torch.cuda.empty_cache()
            if completed_steps >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_with_accelerate(accelerator, model, tokenizer, str(args.output_dir), chat_template_name=None)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and args.remote_output_dir:
        S3CheckpointStore(args.remote_output_dir).upload_final_export(args.output_dir)
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
