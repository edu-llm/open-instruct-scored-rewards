"""Per-token weights that pull SFT toward tokens the frozen base already finds likely.

Ordinary SFT spends the same gradient on every target token. This spends less on the tokens
the base model would have been most surprised by, which are the ones that move the policy
furthest from where it started. The knob is a temperature: ``T -> inf`` recovers ordinary SFT
exactly, and lower ``T`` reweights harder.

For each loss-bearing token ``t`` we compute a scalar distance-from-base ``s_t`` in one of two
ways, then map it to a multiplier through a temperatured softmax of ``-s_t``:

  (a) base-surprise  ``s_t = -log pi_0(y_t | ctx)``          one frozen-base pass
  (b) forward-KL     ``s_t = KL(pi_0(.|ctx) || pi_SFT(.|ctx))``  needs a vanilla SFT adapter

The multiplier is normalised to mean 1 over the reweighted tokens, so the effective learning
rate is unchanged and only the *distribution* of gradient across tokens moves.

WHY THE REFERENCE MODEL IS ONE MODEL AND NOT TWO. Variant b compares two distributions, and the
obvious implementation loads the base and a merged copy of the vanilla SFT side by side. That
costs 2x the weights, which was tolerable at 1B and is not at 20B: gpt-oss-20b dequantises to
about 42GB, so the pair wants 84GB and does not fit on an 80GB card. Since the vanilla SFT is a
LoRA adapter over the same base, both distributions are available from ONE set of weights by
toggling the adapter, which is what ``disable_adapter`` does and what ``grpo_fast`` already does
for its reference logprobs. Same arithmetic, half the memory.

The pass is expensive and its result depends only on (data, variant, models) -- not on the
temperature -- so it is cached and an entire temperature sweep pays for it once.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os

import torch
import torch.nn.functional as F

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

IGNORE_INDEX = -100
NAN = float("nan")

# Cap the transient (positions x vocab) float32 tensors the variant-b KL builds. gpt-oss has a
# 201k vocab, so one position costs ~800KB per distribution and variant b holds two at once;
# at the 1024-position chunk that was fine for a 100k-vocab 1B model this would be 1.6GB.
_CHUNK_BYTES = 256 * 1024 * 1024


def position_chunk(vocab_size: int) -> int:
    """How many positions to score at once, so the intermediate stays near ``_CHUNK_BYTES``."""
    return max(32, _CHUNK_BYTES // max(1, vocab_size * 4))


def _is_reweighted(row) -> bool:
    """Only pedagogy rows are reweighted; general rows keep multiplier 1 and anchor the scale."""
    return row.get("kind", "pedagogy") != "general"


def content_hash(dataset, variant: str, base_model: str, sft_adapter: str | None) -> str:
    """Cache key over exactly the things that change the signal, and nothing else.

    Batch size is deliberately absent: it changes how forward passes are grouped, not what they
    return, so a cache written at one batch size is valid at another.
    """
    h = hashlib.md5()
    h.update(f"{variant}|{base_model}|{sft_adapter}".encode())
    for ids in dataset["input_ids"]:
        h.update(bytes(memoryview(torch.tensor(ids, dtype=torch.int32).numpy())))
    return h.hexdigest()[:16]


def loss_positions(input_ids, labels):
    """``(loss_pos, pred_pos, targets)`` for one row.

    ``labels[i]`` is the target AT index i, so the logit that predicts it is at ``i-1``. Position
    0 therefore can never be a loss position.
    """
    loss_pos = [i for i in range(1, len(input_ids)) if labels[i] != IGNORE_INDEX]
    return loss_pos, [p - 1 for p in loss_pos], [input_ids[p] for p in loss_pos]


def signal_from_logits(base_logits, sft_logits, targets, variant, chunk):
    """Per-token signal at the predicting positions. ``(n, V)`` in, ``(n,)`` out."""
    out = []
    for i in range(0, base_logits.shape[0], chunk):
        b = base_logits[i : i + chunk].float()
        if variant == "a":
            # -log pi_0(y_t), via logsumexp rather than log_softmax: identical value, and it
            # never materialises the normalised (chunk, V) tensor.
            s = b.logsumexp(dim=-1) - b.gather(1, targets[i : i + chunk].unsqueeze(1)).squeeze(1)
        else:
            lp0 = F.log_softmax(b, dim=-1)
            lp1 = F.log_softmax(sft_logits[i : i + chunk].float(), dim=-1)
            s = (lp0.exp() * (lp0 - lp1)).sum(dim=-1)
        out.append(s)
    return torch.cat(out) if out else base_logits.new_zeros(0)


@contextlib.contextmanager
def _adapter_disabled(model, active: bool):
    """``model`` with its LoRA adapter off, giving the frozen base distribution.

    A no-op when ``active`` is false so the caller can write one code path for "base model with
    no adapter loaded" and "base model with the vanilla SFT adapter attached".
    """
    if not active:
        yield
        return
    inner = getattr(model, "module", model)
    with inner.disable_adapter():
        yield


@torch.no_grad()
def compute_signal(dataset, variant, model, *, sft_attached, batch_size=8, pad_token_id=0):
    """Per-row, full-length signal lists. General rows come back all-NaN and are never reweighted.

    ``model`` is the base model, optionally carrying the vanilla SFT LoRA adapter. When
    ``sft_attached`` the adapter is toggled off to read the base distribution, which is what
    makes variant b cost one model rather than two.

    Rows run in LENGTH-SORTED batches to keep padding waste low, and the original order is
    restored before returning. Padding is on the right under a causal mask with an attention
    mask, so trailing pads cannot change any real position's logits and the batched result
    matches the row-at-a-time one.
    """
    if variant not in ("a", "b"):
        raise ValueError(f"unknown variant {variant!r} (expected 'a' or 'b')")
    if variant == "b" and not sft_attached:
        raise ValueError("variant 'b' compares base against a vanilla SFT, so it needs that adapter attached")

    device = next(model.parameters()).device
    chunk = position_chunk(int(getattr(model.config, "vocab_size", 128000)))

    signals, todo = [None] * len(dataset), []
    kinds = []
    for i in range(len(dataset)):
        row = dataset[i]
        kinds.append(row.get("kind", "pedagogy"))
        if _is_reweighted(row):
            todo.append((i, row["input_ids"], row["labels"]))
        else:
            signals[i] = [NAN] * len(row["input_ids"])

    todo.sort(key=lambda t: len(t[1]))
    for start in range(0, len(todo), batch_size):
        group = todo[start : start + batch_size]
        width = max(len(ids) for _, ids, _ in group)
        ids = torch.full((len(group), width), pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(group), width), dtype=torch.long, device=device)
        for r, (_, row_ids, _) in enumerate(group):
            ids[r, : len(row_ids)] = torch.tensor(row_ids, dtype=torch.long, device=device)
            mask[r, : len(row_ids)] = 1

        # The adapter is ON for the SFT distribution and OFF for the base one. Variant a wants
        # only the base, so it takes a single pass with the adapter off.
        sft_logits = model(input_ids=ids, attention_mask=mask).logits if variant == "b" else None
        with _adapter_disabled(model, sft_attached):
            base_logits = model(input_ids=ids, attention_mask=mask).logits

        for r, (idx, row_ids, row_labels) in enumerate(group):
            full = [NAN] * len(row_ids)
            pos, pred, targets = loss_positions(row_ids, row_labels)
            if pos:
                sel = torch.tensor(pred, device=device)
                s = signal_from_logits(
                    base_logits[r].index_select(0, sel),
                    sft_logits[r].index_select(0, sel) if sft_logits is not None else None,
                    torch.tensor(targets, device=device),
                    variant,
                    chunk,
                )
                for p, v in zip(pos, s.tolist()):
                    full[p] = v
            signals[idx] = full

        del base_logits, sft_logits
        if (start // batch_size) % 50 == 0:
            logger.info("signal: %d/%d reweighted rows", min(start + batch_size, len(todo)), len(todo))

    return signals, kinds


def robust_zscore_params(values):
    """``(center, scale)`` from median and MAD, falling back to mean/std on a flat signal.

    Median/MAD rather than mean/std because the signal has a long right tail -- a handful of
    genuinely unpredictable tokens would otherwise set the scale for everything else.
    """
    vals = sorted(v for v in values if v == v)  # NaN != NaN drops the general rows
    if not vals:
        return 0.0, 1.0
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    devs = sorted(abs(v - med) for v in vals)
    mad = devs[n // 2] if n % 2 else 0.5 * (devs[n // 2 - 1] + devs[n // 2])
    scale = 1.4826 * mad  # MAD -> std for a Gaussian
    if scale > 1e-8:
        return med, scale
    # MAD collapses to zero when a majority of the values are identical, and then a scale is
    # still needed. Only the SCALE falls back: the centre stays the median, because the obvious
    # fallback of returning the mean hands the centre to whatever outlier the median just
    # excluded, which is the opposite of what this function is for.
    var = sum((v - med) ** 2 for v in vals) / max(1, n - 1)
    return med, math.sqrt(var) or 1.0


def multipliers_from_signal(signals, kinds, temperature):
    """Per-row multiplier lists, mean 1 over reweighted tokens.

    Because the softmax is taken over the WHOLE corpus rather than per row, a row of uniformly
    surprising tokens is damped relative to other rows instead of being renormalised back up.
    Preserving the mean at 1 is what keeps this comparable to the vanilla run at the same LR.
    """
    ped = [v for row, k in zip(signals, kinds) if k != "general" for v in row if v == v]
    center, scale = robust_zscore_params(ped)

    if math.isinf(temperature):
        return [[1.0] * len(row) for row in signals]

    total, count = 0.0, 0
    for row, k in zip(signals, kinds):
        if k == "general":
            continue
        for v in row:
            if v == v:
                total += math.exp(-((v - center) / scale) / temperature)
                count += 1
    total = total or 1.0

    out = []
    for row, k in zip(signals, kinds):
        w = [1.0] * len(row)
        if k != "general":
            for i, v in enumerate(row):
                if v == v:
                    w[i] = count * math.exp(-((v - center) / scale) / temperature) / total
        out.append(w)
    return out


def load_or_compute(dataset, variant, model, *, cache_dir, base_model, sft_adapter,
                    sft_attached, batch_size=8, pad_token_id=0):
    """The cached form of :func:`compute_signal`, keyed so a temperature sweep pays once."""
    key = content_hash(dataset, variant, base_model, sft_adapter)
    path = os.path.join(cache_dir, f"signal_{variant}_{key}.pt")
    if os.path.exists(path):
        logger.info("reusing cached signal: %s", path)
        blob = torch.load(path)
        return blob["signals"], blob["kinds"]

    logger.info("computing signal (variant=%s) -> %s", variant, path)
    signals, kinds = compute_signal(
        dataset, variant, model, sft_attached=sft_attached,
        batch_size=batch_size, pad_token_id=pad_token_id,
    )
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"signals": signals, "kinds": kinds, "variant": variant}, path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "base_model": base_model, "sft_adapter": sft_adapter,
                   "rows": len(signals)}, f, indent=2)
    return signals, kinds
