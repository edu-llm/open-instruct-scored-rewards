"""Shared reward-model plumbing for the PRM-vs-ORM experiment.

This module is the *single source of truth* for two things that MUST be identical between
reward-model **training** (``reward_modeling_scored.py``, Stages 3-4) and reward-model
**scoring** at GRPO time (``rm_verifier.py``, Stage 6):

  1. **How a (problem, assistant-text) pair becomes token ids.** SFT, RM training and GRPO
     all use the ``tulu`` chat template over the same dolma2 tokenizer. The rendered string
     for a single user+assistant exchange is exactly::

         <|user|>\\n{problem}\\n<|assistant|>\\n{assistant_text}{eos}

     (see ``CHAT_TEMPLATES["tulu"]`` in ``open_instruct/dataset_transformation.py`` — the
     final assistant turn appends ``eos_token`` and no trailing newline). We build that string
     directly rather than depend on ``apply_chat_template`` being wired identically in every
     stage, then tokenize it with ``add_special_tokens=False`` and optionally prepend BOS to
     match SFT's ``--add_bos`` decision. A GRPO rollout is the policy generating
     ``{assistant_text}`` after the ``<|assistant|>\\n`` prefix and emitting ``eos`` — so the
     RM sees byte-for-byte what SFT produced. Format consistency is a correctness requirement.

  2. **Where the scalar/label lives.** We pool the model's per-token score at the **last real
     token** (the appended ``eos``): a Cobbe-style whole-solution verifier for the ORM, and
     the step-boundary token for each PRM ``(prefix, step)`` example. The attention mask is
     built from *known* padding positions (never from ``input_ids == pad_id``), so pooling is
     correct even when the tokenizer aliases ``pad_token_id`` to ``eos_token_id``.

``--selftest`` (via the trainer) exercises every pure function here with a fake tokenizer and
no torch tensors of real weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Same blank-line step separator ``data_prep.py`` used to join a solution's steps.
STEP_SEP = "\n\n"

#: PRM rating (-1/0/+1) -> class index for 3-way cross-entropy. neg=0, neu=1, pos=2.
#: At scoring time P(step correct) = P(neu) + P(pos) (neutral counts as positive).
RATING_TO_CLASS = {-1: 0, 0: 1, 1: 2}
NUM_PRM_CLASSES = 3
PRM_CORRECT_CLASSES = (1, 2)  # neutral + positive


def tulu_prompt_string(problem: str) -> str:
    """The user half of a tulu exchange, up to (not including) the assistant content."""
    return f"<|user|>\n{problem}\n<|assistant|>\n"


def tulu_exchange_string(problem: str, assistant_text: str, eos_token: str) -> str:
    """Full ``<|user|>..<|assistant|>..{eos}`` string for one problem+response exchange.

    This is exactly the substring the ``tulu`` template emits for a single user turn followed
    by a final assistant turn, so RM training and GRPO scoring tokenize the identical text.
    """
    return tulu_prompt_string(problem) + assistant_text + eos_token


def assistant_text_from_steps(steps: list[str]) -> str:
    """Join step texts with the blank-line separator (how a solution body is assembled)."""
    return STEP_SEP.join(s for s in steps if s)


def encode_exchange(
    tokenizer: Any, problem: str, assistant_text: str, *, add_bos: bool, max_length: int
) -> list[int]:
    """Tokenize one exchange to ids, prepending BOS if requested and left-truncating.

    Left-truncation (dropping the earliest prompt tokens) is deliberate: the scored token is
    the trailing ``eos`` / step boundary, so it must always survive truncation. The pooled
    position is therefore always the true end of ``assistant_text``.
    """
    text = tulu_exchange_string(problem, assistant_text, tokenizer.eos_token)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if add_bos:
        bos = tokenizer.bos_token_id
        if bos is not None:
            ids = [bos] + ids
    if len(ids) > max_length:
        ids = ids[-max_length:]  # keep the end (the boundary token we pool at)
    return ids


def orm_example_ids(tokenizer: Any, row: dict, *, add_bos: bool, max_length: int) -> tuple[list[int], float]:
    """An ``orm.jsonl`` row -> (token ids, outcome label in {0.0, 1.0}).

    Rows are ``{"messages": [user, assistant], "outcome": 0|1}`` (see ``data_prep.py``).
    """
    msgs = row["messages"]
    problem = msgs[0]["content"]
    assistant_text = msgs[1]["content"]
    ids = encode_exchange(tokenizer, problem, assistant_text, add_bos=add_bos, max_length=max_length)
    return ids, float(row["outcome"])


def prm_example_ids(tokenizer: Any, row: dict, *, add_bos: bool, max_length: int) -> tuple[list[int], int]:
    """A ``prm.jsonl`` row -> (token ids, rating class in {0,1,2}).

    Rows are ``{"problem", "prefix": [step,...], "step", "rating": -1|0|1}``. The assistant
    text is the prefix steps plus this step, joined the same way a full solution is, so the
    trailing ``eos`` sits right after ``step`` — exactly the boundary token we pool at.
    """
    assistant_text = assistant_text_from_steps([*row["prefix"], row["step"]])
    ids = encode_exchange(tokenizer, row["problem"], assistant_text, add_bos=add_bos, max_length=max_length)
    return ids, RATING_TO_CLASS[int(row["rating"])]


@dataclass
class ScoredCollator:
    """Right-pad a batch of (ids, label) pairs into tensors for the RM.

    The attention mask is derived from real vs padded positions we control here, never from
    ``input_ids == pad_id``, so it stays correct if ``pad_token_id == eos_token_id``.
    ``label_dtype`` is float for ORM (BCE) and long for PRM (CE).
    """

    pad_token_id: int
    label_dtype: str  # "float" | "long"

    def __call__(self, batch: list[tuple[list[int], Any]]) -> dict[str, Any]:
        import torch  # noqa: PLC0415 -- keep pure helpers importable without torch

        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        dtype = torch.float if self.label_dtype == "float" else torch.long
        labels = torch.tensor([lab for _, lab in batch], dtype=dtype)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def pooled_scores(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    """Per-sequence logits pooled at the last real token: shape ``(batch, num_labels)``.

    Routes through ``model(...)`` (the top-level ``forward``) rather than reaching into
    ``model.base_model_prefix`` + ``model.score`` directly. That distinction is load-bearing under
    data parallelism: plain ``DistributedDataParallel`` does **not** forward attribute access
    (``ddp.base_model_prefix`` raises ``AttributeError``) and only synchronizes gradients when the
    wrapper's own ``forward`` is the autograd entry point. ``open_instruct``'s ``get_reward`` reaches
    into submodules and gets away with it only because it runs under DeepSpeed (whose engine *does*
    forward attribute access); this trainer uses accelerate's DDP, so it must go through ``forward``.

    ``Olmo3ForSequenceClassification.forward`` pools at the last real token using this same
    attention mask (robust to ``pad_id == eos_id``), so ORM (1 logit) and PRM (3 logits at a
    step-boundary prefix) both read the correct position. Works identically on the raw,
    unwrapped model used at inference (bon_eval / rm_verifier).
    """
    import torch  # noqa: PLC0415

    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(input_ids, attention_mask == 0, 0)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        use_cache=False,
    )
    return out.logits  # (batch, num_labels) — forward pools at the last real token by mask


# Documented RoPE base for the olmo3_370M policy in this experiment: olmo3_370M is built from
# olmo2_370M (rope_theta=500_000) with a [4096,4096,4096,-1] sliding-window pattern and NO separate
# local base freq, so sliding-window and full-attention layers share this single theta. Used only as
# a last-resort fallback when a checkpoint's config declares no theta anywhere — never a fresh guess.
OLMO3_370M_ROPE_THETA = 500000.0


def ensure_vllm_rope_parameters(model_dir: Any, *, fallback_theta: float = OLMO3_370M_ROPE_THETA,
                                log: Any = print) -> bool:
    """Guarantee ``config.json``'s ``rope_parameters`` carries ``rope_theta`` for vLLM's olmo2 loader.

    vLLM 0.19.1 routes ``Olmo3ForCausalLM`` to its ``olmo2`` implementation, which on every
    sliding-window layer reads ``config.rope_parameters["rope_theta"]`` (``olmo2.py:144``) and on
    full-attention layers hands the whole ``rope_parameters`` dict to ``get_rope``. Some saved Olmo3
    configs land with a ``rope_parameters`` dict that lacks ``rope_theta`` — transformers loads them
    fine (it also keeps a top-level ``rope_theta``), but vLLM ``KeyError``s at engine init. This
    repairs the on-disk config once so every downstream vLLM consumer (bon_eval / grpo_fast rollouts /
    final eval) loads with a valid, *checkpoint-derived* RoPE base.

    Theta is sourced, in order, from the config's OWN fields — ``rope_parameters.rope_theta`` (already
    fine → no-op) then top-level ``rope_theta`` — and only if neither exists is ``fallback_theta``
    (the documented olmo3_370M preset value) used, so the base is never invented. For olmo3_370M the
    sliding and full layers share one theta, so filling this single key is faithful, not an
    approximation. Idempotent; returns True iff it rewrote the file.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        log(f"[rope-fix] no config.json under {model_dir}; skipping")
        return False
    cfg = json.loads(cfg_path.read_text())
    rp = dict(cfg["rope_parameters"]) if isinstance(cfg.get("rope_parameters"), dict) else {}
    if rp.get("rope_theta") is not None:
        log(f"[rope-fix] rope_parameters.rope_theta already set ({rp['rope_theta']}); no-op")
        return False
    if cfg.get("rope_theta") is not None:
        theta, src = cfg["rope_theta"], "top-level rope_theta"
    else:
        theta, src = fallback_theta, "fallback (olmo3_370M preset)"
    rp.setdefault("rope_type", "default")
    rp["rope_theta"] = theta
    cfg["rope_parameters"] = rp
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log(f"[rope-fix] set rope_parameters.rope_theta={theta} (source: {src}) in {cfg_path}")
    return True


class _FakeTokenizer:
    """Whitespace tokenizer for offline selftests: ids are char lengths of split tokens."""

    eos_token = "<eos>"
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        return {"input_ids": [len(tok) for tok in text.split()]}


def _selftest() -> None:
    tok = _FakeTokenizer()
    # ORM row: outcome broadcast to one scalar; string ends with the eos surface form.
    s = tulu_exchange_string("2+2?", "It is 4.", tok.eos_token)
    assert s == "<|user|>\n2+2?\n<|assistant|>\nIt is 4.<eos>", repr(s)
    orm_ids, y = orm_example_ids(
        tok, {"messages": [{"content": "2+2?"}, {"content": "It is 4."}], "outcome": 1},
        add_bos=True, max_length=999,
    )
    assert y == 1.0 and orm_ids[0] == tok.bos_token_id, (y, orm_ids[:2])
    # PRM row: prefix+step joined by STEP_SEP; rating -1/0/1 -> class 0/1/2.
    a_ids, c = prm_example_ids(
        tok, {"problem": "p", "prefix": ["s1"], "step": "s2", "rating": 0}, add_bos=False, max_length=999
    )
    assert c == 1, c
    assert RATING_TO_CLASS == {-1: 0, 0: 1, 1: 2}
    assert assistant_text_from_steps(["a", "", "b"]) == "a\n\nb"
    # left-truncation keeps the tail (boundary token), never the head
    ids_full = encode_exchange(tok, "p", "a b c d e", add_bos=False, max_length=999)
    ids_trunc = encode_exchange(tok, "p", "a b c d e", add_bos=False, max_length=3)
    assert ids_trunc == ids_full[-3:], (ids_trunc, ids_full)
    # rope_parameters repair for vLLM's olmo2 loader (no torch/network needed).
    import json as _json  # noqa: PLC0415
    import tempfile as _tf  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    _d = _tf.mkdtemp()
    _cfg = _Path(_d) / "config.json"
    _quiet = lambda *a, **k: None  # noqa: E731
    # legacy top-level rope_theta but no rope_parameters -> filled from top-level, idempotent
    _cfg.write_text(_json.dumps({"model_type": "olmo3", "rope_theta": 500000.0}))
    assert ensure_vllm_rope_parameters(_d, log=_quiet) is True
    _got = _json.loads(_cfg.read_text())["rope_parameters"]
    assert _got["rope_theta"] == 500000.0 and _got["rope_type"] == "default", _got
    assert ensure_vllm_rope_parameters(_d, log=_quiet) is False  # second call is a no-op
    # rope_parameters present but missing theta -> sourced from top-level rope_theta
    _cfg.write_text(_json.dumps({"rope_parameters": {"rope_type": "default"}, "rope_theta": 12345.0}))
    assert ensure_vllm_rope_parameters(_d, log=_quiet) is True
    assert _json.loads(_cfg.read_text())["rope_parameters"]["rope_theta"] == 12345.0
    # nothing declared anywhere -> documented fallback, never left unset
    _cfg.write_text(_json.dumps({"model_type": "olmo3"}))
    assert ensure_vllm_rope_parameters(_d, fallback_theta=777.0, log=_quiet) is True
    assert _json.loads(_cfg.read_text())["rope_parameters"]["rope_theta"] == 777.0
    print("RM_COMMON SELFTEST OK: tulu encoding, label maps, truncation, rope-fix verified")


if __name__ == "__main__":
    _selftest()
