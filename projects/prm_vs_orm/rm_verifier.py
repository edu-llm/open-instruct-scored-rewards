"""Stage 6: the learned-RM-as-GRPO-reward bridge for the PRM-vs-ORM experiment.

``open_instruct``'s GRPO reward comes from *ground-truth* verifiers
(``GSM8KVerifier`` / ``MathVerifier`` in ``ground_truth_utils.py``). The whole point of this
experiment is to drive GRPO with a **trained reward model** instead — either the ORM or the
PRM from Stages 3-4. This module adds one ``VerifierFunction`` subclass that loads an RM
checkpoint and returns its score in ``[0, 1]`` as the reward.

**Registration.** ``build_all_verifiers`` discovers verifiers via
``VerifierFunction.__subclasses__()``, so this module only needs to be *imported* into the
GRPO process for the RM verifier to appear in the mapping. The Stage-7 launcher
(``_grpo_launch.py``) imports it before calling ``grpo_fast`` — no edit to ``open_instruct``.

**Configuration without touching core.** ``build_all_verifiers`` instantiates every subclass as
``subclass(verifier_config)`` and threads config off the GRPO ``Args`` via
``VerifierConfig.from_args``. Adding an RM-path field there would edit core, so instead this
verifier reads its checkpoint from the environment (set by the Stage-7 job command):

  * ``PRM_VS_ORM_RM_PATH``  -- local dir of the trained RM (required at call time, not import).
  * ``PRM_VS_ORM_RM_TYPE``  -- ``orm`` | ``prm`` (selects head width + scoring rule).
  * ``PRM_VS_ORM_RM_NAME``  -- verifier name GRPO routes to by dataset (default ``rm``); the
    Stage-7 prompt rows carry ``dataset`` equal to this so the RM (not the GT checker) scores.
  * ``PRM_VS_ORM_RM_DEVICE`` -- ``cuda`` / ``cuda:K`` / ``cpu`` (default: cuda if available).
  * ``PRM_VS_ORM_RM_MAXLEN`` / ``PRM_VS_ORM_RM_ADD_BOS`` -- match RM training (1024 / auto).

**Scoring (identical formatting to training, via ``rm_common``).**

  * **ORM** -- one ``<|user|>..<|assistant|>{solution}{eos}`` sequence, sigmoid of the
    last-token 1-logit -> ``P(correct)``.
  * **PRM** -- split the solution into steps on ``STEP_SEP`` and score each *cumulative prefix*
    as its own sequence ending in ``eos`` (exactly the ``(prefix, step)`` sequences the PRM was
    trained on), softmax the 3-logit, ``P(step correct) = P(neu) + P(pos)``, and return the
    **product** over steps (Lightman et al.'s best aggregator; product slightly penalises
    longer solutions, which is intended).

Import is cheap (torch/transformers are imported lazily inside methods) so ``--selftest``
exercises the pure reward arithmetic with no torch and no checkpoint.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_common  # noqa: E402 -- overlay-local module, added to path above

try:  # the real GRPO image always has these; the fallback only supports offline --selftest
    from open_instruct.ground_truth_utils import VerificationResult, VerifierFunction  # noqa: E402
except ImportError:  # pragma: no cover - exercised only when open_instruct is absent locally

    class VerifierFunction:  # type: ignore[no-redef] - minimal stand-in for pure-helper selftest
        def __init__(self, name: str, weight: float = 1.0, verifier_config: Any | None = None) -> None:
            self.name = name
            self.weight = weight
            self.verifier_config = verifier_config

    class VerificationResult:  # type: ignore[no-redef]
        def __init__(self, score: float, cost: float = 0.0, reasoning: str | None = None) -> None:
            self.score = score
            self.cost = cost
            self.reasoning = reasoning


# --------------------------------------------------------------------------------------------
# Pure reward arithmetic (no torch) — the core of the ORM/PRM contrast, unit-tested offline.
# --------------------------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    # numerically stable logistic
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax3(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def orm_reward(last_logit: float) -> float:
    """ORM reward = P(solution correct) = sigmoid(last-token logit)."""
    return sigmoid(last_logit)


def prm_step_correct_prob(step_logits: list[float]) -> float:
    """P(step correct) = P(neutral) + P(positive) from a step's 3-logit (neg,neu,pos)."""
    p = softmax3(step_logits)
    return sum(p[c] for c in rm_common.PRM_CORRECT_CLASSES)


def prm_reward(step_logits_list: list[list[float]]) -> float:
    """PRM reward = product over steps of P(step correct). Empty -> 0.0 (no scorable step)."""
    if not step_logits_list:
        return 0.0
    out = 1.0
    for sl in step_logits_list:
        out *= prm_step_correct_prob(sl)
    return out


def split_steps(solution: str) -> list[str]:
    """Split a generated solution into steps the same way training assembled them."""
    return [s for s in solution.split(rm_common.STEP_SEP) if s.strip()]


class RMVerifier(VerifierFunction):
    """Score a rollout with a trained ORM/PRM reward model (see module docstring)."""

    def __init__(self, verifier_config: Any | None = None) -> None:
        name = os.environ.get("PRM_VS_ORM_RM_NAME", "rm")
        super().__init__(name, weight=1.0, verifier_config=verifier_config)
        self.rm_type = os.environ.get("PRM_VS_ORM_RM_TYPE", "orm").lower()
        self.rm_path = os.environ.get("PRM_VS_ORM_RM_PATH")
        self.max_length = int(os.environ.get("PRM_VS_ORM_RM_MAXLEN", "1024"))
        self.add_bos_choice = os.environ.get("PRM_VS_ORM_RM_ADD_BOS", "auto")
        self._device_env = os.environ.get("PRM_VS_ORM_RM_DEVICE")
        # Loaded lazily on first call so import / instantiation stays cheap and never needs a GPU.
        self._model = None
        self._tokenizer = None
        self._add_bos = None
        self._device = None

    # -- lazy model load -----------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.rm_path:
            raise RuntimeError(
                "PRM_VS_ORM_RM_PATH is unset; RMVerifier was called without a checkpoint."
            )
        import olmo3_adapter  # noqa: PLC0415 -- overlay-local (path inserted at import time)
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        olmo3_adapter.register()
        num_labels = 1 if self.rm_type == "orm" else rm_common.NUM_PRM_CLASSES
        device = self._device_env or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        tok = AutoTokenizer.from_pretrained(self.rm_path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self._tokenizer = tok
        if self.add_bos_choice == "yes":
            self._add_bos = True
        elif self.add_bos_choice == "no":
            self._add_bos = False
        else:
            self._add_bos = tok.bos_token_id is not None

        model = AutoModelForSequenceClassification.from_pretrained(
            self.rm_path, num_labels=num_labels, torch_dtype=torch.bfloat16
        )
        model.config.pad_token_id = tok.pad_token_id
        model.eval()
        model.to(device)
        self._model = model

    # -- forward a batch of (problem, assistant_text) -> pooled logits (list rows) --------
    def _pooled_logits(self, exchanges: list[tuple[str, str]]) -> list[list[float]]:
        import torch  # noqa: PLC0415

        ids = [
            rm_common.encode_exchange(
                self._tokenizer, problem, assistant, add_bos=self._add_bos, max_length=self.max_length
            )
            for problem, assistant in exchanges
        ]
        collator = rm_common.ScoredCollator(pad_token_id=self._tokenizer.pad_token_id, label_dtype="float")
        batch = collator([(row, 0.0) for row in ids])
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)
        with torch.no_grad():
            pooled = rm_common.pooled_scores(self._model, input_ids, attention_mask)
        return pooled.float().cpu().tolist()  # (batch, num_labels)

    # -- VerifierFunction API ------------------------------------------------------------
    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        self._ensure_loaded()
        problem = query or ""
        if self.rm_type == "orm":
            rows = self._pooled_logits([(problem, prediction)])
            score = orm_reward(rows[0][0])
        else:
            steps = split_steps(prediction)
            if not steps:
                return VerificationResult(score=0.0)
            prefixes = [rm_common.assistant_text_from_steps(steps[: i + 1]) for i in range(len(steps))]
            rows = self._pooled_logits([(problem, pfx) for pfx in prefixes])
            score = prm_reward(rows)
        return VerificationResult(score=float(score))


def _selftest() -> None:
    # sigmoid monotone + midpoint
    assert abs(sigmoid(0.0) - 0.5) < 1e-9
    assert sigmoid(10) > 0.99 and sigmoid(-10) < 0.01
    assert orm_reward(0.0) == 0.5
    # PRM: neutral+positive counts as correct. A logit vector strongly on class 2 (pos) ~ 1.
    p_correct = prm_step_correct_prob([-10.0, -10.0, 10.0])
    assert p_correct > 0.99, p_correct
    # strongly negative step -> ~0
    p_bad = prm_step_correct_prob([10.0, -10.0, -10.0])
    assert p_bad < 0.01, p_bad
    # neutral alone counts as correct
    p_neu = prm_step_correct_prob([-10.0, 10.0, -10.0])
    assert p_neu > 0.99, p_neu
    # product over steps: one bad step tanks the whole solution
    assert prm_reward([[-10, -10, 10], [-10, -10, 10]]) > 0.98
    assert prm_reward([[-10, -10, 10], [10, -10, -10]]) < 0.02
    assert prm_reward([]) == 0.0
    # step splitting mirrors STEP_SEP join
    assert split_steps("a\n\nb\n\nc") == ["a", "b", "c"]
    assert split_steps("only one step") == ["only one step"]
    # PRM_CORRECT_CLASSES is (neu, pos)
    assert rm_common.PRM_CORRECT_CLASSES == (1, 2)
    print("RM_VERIFIER SELFTEST OK: sigmoid/softmax, ORM+PRM aggregation, step split verified")


if __name__ == "__main__":
    _selftest()
