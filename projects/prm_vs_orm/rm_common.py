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


def _find_theta(obj: Any) -> Any:
    """Depth-first search for a scalar ``rope_theta``/``theta`` anywhere in a nested config fragment."""
    if isinstance(obj, dict):
        for k in ("rope_theta", "theta"):
            v = obj.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        for v in obj.values():
            found = _find_theta(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_theta(v)
            if found is not None:
                return found
    return None


def _is_scalar(v: Any) -> bool:
    """True for JSON scalars vLLM's ``get_rope`` can put in a hashable cache key (``bool`` is an
    ``int`` subclass and hashable, so it is fine; ``dict``/``list`` are not)."""
    return v is None or isinstance(v, (int, float, str))


def _rope_theta_from_config(cfg: dict, rp: dict, fallback_theta: float) -> tuple[float, str]:
    """Source the RoPE base from the config's OWN fields, never invented until nothing exists.

    Order: a scalar ``rope_parameters.rope_theta`` → top-level ``rope_theta`` → any scalar
    ``rope_theta``/``theta`` nested anywhere under ``rope_parameters`` or ``rope_scaling`` →
    ``fallback_theta`` (the documented olmo3_370M preset).
    """
    theta_in_rp = rp.get("rope_theta")
    if isinstance(theta_in_rp, (int, float)) and not isinstance(theta_in_rp, bool):
        return float(theta_in_rp), "rope_parameters.rope_theta"
    top = cfg.get("rope_theta")
    if isinstance(top, (int, float)) and not isinstance(top, bool):
        return float(top), "top-level rope_theta"
    nested = _find_theta(rp) if rp else None
    if nested is None:
        nested = _find_theta(cfg.get("rope_scaling"))
    if nested is not None:
        return float(nested), "nested rope_theta/theta"
    return float(fallback_theta), "fallback (olmo3_370M preset)"


def _verify_vllm_rope(model_dir: Any, *, log: Any = print) -> None:
    """Reload the on-disk config exactly as vLLM does (transformers ``AutoConfig``) and prove the
    ``get_rope`` cache key is hashable — failing fast in *this* process with the real shape.

    This is the belt to the sanitizer's braces. transformers **folds top-level ``rope_theta`` and
    ``rope_scaling`` into ``config.rope_parameters``** at load (and keeps unrecognised keys), so a
    file edit of ``rope_parameters`` alone can be silently overridden or re-nested. Rather than
    discover that as ``TypeError: unhashable type: 'dict'`` deep inside a vLLM EngineCore subprocess
    (whose traceback the platform log truncates), we reconstruct the exact ``get_rope`` key here and
    raise with the loaded ``rope_parameters`` printed. A transformers/AutoConfig import or load
    failure is not fatal — it only skips the check (the sanitizer already rewrote the file).
    """
    import json  # noqa: PLC0415

    try:
        from transformers import AutoConfig  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001 -- verification is best-effort; never block on infra
        log(f"[rope-fix] verify skipped (transformers import failed: {e})")
        return
    try:
        cfg = AutoConfig.from_pretrained(str(model_dir))
    except Exception as e:  # noqa: BLE001
        log(f"[rope-fix] verify skipped (AutoConfig load failed: {type(e).__name__}: {e})")
        return
    rp = getattr(cfg, "rope_parameters", None)
    log(f"[rope-fix] verify: transformers-loaded rope_parameters={json.dumps(rp, default=str)}")
    if not isinstance(rp, dict):
        return
    # Mirror get_rope's key build: list values -> tuples, everything else left as-is, then hash the
    # whole key tuple (the extra positional slots stand in for head_size/rotary_dim/... which are all
    # hashable scalars; only rope_parameters can smuggle in an unhashable value).
    rp_tuple = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in rp.items())
    try:
        hash((64, 64, 4096, True, rp_tuple, None, None))
    except TypeError as e:
        raise RuntimeError(
            "[rope-fix] rope_parameters is STILL unhashable for vLLM get_rope after sanitize "
            f"(transformers merged/re-nested it): {json.dumps(rp, default=str)} -- {e}"
        ) from e
    log("[rope-fix] verify: get_rope cache key is hashable OK")


def ensure_vllm_rope_parameters(model_dir: Any, *, fallback_theta: float = OLMO3_370M_ROPE_THETA,
                                log: Any = print, verify: bool = True) -> bool:
    """Sanitise ``config.json`` so vLLM's olmo2 loader can build RoPE for every layer.

    vLLM 0.19.1 routes ``Olmo3ForCausalLM`` to its ``olmo2`` implementation. On a **sliding-window**
    layer it reads ``config.rope_parameters["rope_theta"]`` (``olmo2.py:144``) and rebuilds a fresh
    ``{"rope_type": "default", "rope_theta": theta}``; on a **full-attention** layer it hands the
    *entire* ``config.rope_parameters`` to ``get_rope`` (``olmo2.py:142/146``), which builds an
    ``@_ROPE_DICT`` cache key from ``rope_parameters.items()`` — **list values become tuples but dict
    values stay dicts** (``rotary_embedding/__init__.py:41-45,68-77``) — then does ``key in _ROPE_DICT``.
    So any *dict-valued* entry (or a *list containing a dict*) raises ``TypeError: unhashable type:
    'dict'``, and a *missing* ``rope_theta`` raises ``KeyError`` on the sliding branch.

    The trap that defeated a naive per-key edit: **transformers folds the config's top-level
    ``rope_theta`` and ``rope_scaling`` INTO ``config.rope_parameters`` when the checkpoint is
    reloaded** (verified against transformers 5.4 — a top-level ``rope_scaling`` dict's members and a
    top-level ``rope_theta`` appear inside the loaded ``rope_parameters``, overriding the on-disk
    ``rope_parameters`` values), and it *keeps unrecognised keys*. Editing only ``rope_parameters`` on
    disk is therefore not enough — a top-level ``rope_scaling`` can re-introduce a non-scalar. The
    Olmo3 configs saved through the OLMo-core→HF converter can carry exactly these shapes.

    Fix, robust to every observed shape: reduce ``rope_parameters`` to **scalar entries only** plus
    ``{"rope_type": "default", "rope_theta": <theta>}``, set top-level ``rope_scaling`` to ``null``
    (removing the merge source), and set top-level ``rope_theta`` to the same scalar. For olmo3_370M
    the sliding and full layers share one theta with no RoPE scaling, so this is exact, not an
    approximation. Then (``verify``) reload via ``AutoConfig`` and prove the ``get_rope`` key hashes.

    Idempotent: a no-op (returns ``False``) once the config is already in this sanitised form;
    otherwise rewrites and returns ``True``. ``verify`` still runs on a no-op.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        log(f"[rope-fix] no config.json under {model_dir}; skipping")
        return False
    cfg = json.loads(cfg_path.read_text())
    rp_raw = cfg.get("rope_parameters")
    rp = dict(rp_raw) if isinstance(rp_raw, dict) else {}
    log(f"[rope-fix] raw rope_parameters={json.dumps(rp_raw)} top-level rope_theta={cfg.get('rope_theta')!r} "
        f"rope_scaling={json.dumps(cfg.get('rope_scaling'))} layer_types={json.dumps(cfg.get('layer_types'))[:120]}")

    theta, src = _rope_theta_from_config(cfg, rp, fallback_theta)

    # Keep only hashable scalar extras (e.g. partial_rotary_factor); drop every dict/list value that
    # get_rope cannot hash and a "default" rope does not use.
    scalar = {k: v for k, v in rp.items() if _is_scalar(v)}
    scalar["rope_type"] = "default"
    scalar["rope_theta"] = theta

    already_sanitised = (
        isinstance(rp_raw, dict)
        and rp == scalar
        and cfg.get("rope_scaling") is None
        and isinstance(cfg.get("rope_theta"), (int, float)) and not isinstance(cfg.get("rope_theta"), bool)
        and float(cfg["rope_theta"]) == theta
    )
    changed = not already_sanitised
    if changed:
        dropped = sorted(k for k, v in rp.items() if not _is_scalar(v))
        cfg["rope_parameters"] = scalar
        cfg["rope_scaling"] = None
        cfg["rope_theta"] = theta
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log(f"[rope-fix] sanitised rope_parameters -> {json.dumps(scalar)}; rope_scaling->null; "
            f"top-level rope_theta={theta} (theta source: {src}); dropped non-scalar {dropped} in {cfg_path}")
    else:
        log(f"[rope-fix] rope config already sanitised (rope_theta={theta}); no-op")

    if verify:
        _verify_vllm_rope(cfg_path.parent, log=log)
    return changed


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
    # rope_parameters repair for vLLM's olmo2 loader (offline: no torch/network; verify=False so the
    # selftest does not need transformers to AutoConfig-load these minimal configs).
    import json as _json  # noqa: PLC0415
    import tempfile as _tf  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    _d = _tf.mkdtemp()
    _cfg = _Path(_d) / "config.json"
    _quiet = lambda *a, **k: None  # noqa: E731

    def _fix(**kw):
        return ensure_vllm_rope_parameters(_d, log=_quiet, verify=False, **kw)

    def _cfg_now():
        return _json.loads(_cfg.read_text())

    def _sanitised_ok(theta):
        """Every post-fix invariant vLLM's get_rope depends on, in one place."""
        c = _cfg_now()
        rp = c["rope_parameters"]
        assert rp["rope_type"] == "default" and rp["rope_theta"] == theta, rp
        assert not any(isinstance(v, (dict, list)) for v in rp.values()), rp  # hashable get_rope key
        assert c.get("rope_scaling") is None, c.get("rope_scaling")  # merge source neutralised
        assert c.get("rope_theta") == theta, c.get("rope_theta")  # top-level scalar kept consistent

    # legacy top-level rope_theta but no rope_parameters -> filled from top-level, then idempotent
    _cfg.write_text(_json.dumps({"model_type": "olmo3", "rope_theta": 500000.0}))
    assert _fix() is True
    _sanitised_ok(500000.0)
    assert _fix() is False  # now sanitised -> no-op
    # rope_parameters present but missing theta -> sourced from top-level rope_theta
    _cfg.write_text(_json.dumps({"rope_parameters": {"rope_type": "default"}, "rope_theta": 12345.0}))
    assert _fix() is True
    _sanitised_ok(12345.0)
    # crash-2 case: rope_parameters carries a nested DICT (unhashable in get_rope) and no scalar theta
    # at top level -> drop the dict, source theta from the nesting.
    _cfg.write_text(_json.dumps({
        "rope_parameters": {"rope_type": "default", "full_attention": {"rope_theta": 500000.0}},
    }))
    assert _fix() is True
    _sanitised_ok(500000.0)
    assert "full_attention" not in _cfg_now()["rope_parameters"]  # nested dict dropped
    assert _fix() is False  # sanitised -> no-op
    # crash-2b (ac9fbe6 MISS): a LIST-of-dict value. get_rope maps lists->tuples but their dict
    # elements stay dicts, so the key is still unhashable. Must be dropped, scalar extras kept.
    _cfg.write_text(_json.dumps({
        "rope_parameters": {"rope_theta": 500000.0, "partial_rotary_factor": 0.5,
                            "per_layer": [{"rope_type": "default"}, {"rope_type": "default"}]},
    }))
    assert _fix() is True
    _got = _cfg_now()["rope_parameters"]
    assert _got["partial_rotary_factor"] == 0.5 and "per_layer" not in _got, _got  # list-of-dict gone
    _sanitised_ok(500000.0)
    # crash-2c (the real defeat of a naive per-key edit): transformers folds a top-level rope_scaling
    # dict INTO rope_parameters on reload. We neutralise it to null so the merge cannot re-introduce a
    # non-scalar; theta still sourced correctly from rope_parameters.
    _cfg.write_text(_json.dumps({
        "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "mrope_section": [16, 24, 24]},
    }))
    assert _fix() is True
    _sanitised_ok(500000.0)  # asserts rope_scaling is now None
    # nothing declared anywhere -> documented fallback, never left unset
    _cfg.write_text(_json.dumps({"model_type": "olmo3"}))
    assert _fix(fallback_theta=777.0) is True
    _sanitised_ok(777.0)
    print("RM_COMMON SELFTEST OK: tulu encoding, label maps, truncation, "
          "rope sanitise (nested-dict + list-of-dict + rope_scaling-merge + idempotency) verified")


if __name__ == "__main__":
    _selftest()
