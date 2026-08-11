"""Minimal, torch-only GPU validation of the dtype-align guard (no deepspeed/transformers).

The ``edullm run`` machine is a bare NVIDIA-driver image with no Python framework, so the full
``repro_dtype.py`` stack (deepspeed + transformers Olmo3) would need a slow, version-mismatched
install that could not faithfully reproduce the *image's* live stack anyway. This script instead
isolates the two things the offline selftests cannot prove, using only ``torch``:

  1. **The crash class is real on this GPU.** ``nn.Linear.forward`` calls ``F.linear(input,
     self.weight, self.bias)``; a bf16 input against an fp32 weight raises exactly the
     ``RuntimeError: expected ... same dtype`` that killed both GRPO arms. ``--mode reproduce``
     forces that (fp32 weight, bf16 input) and must exit 7.

  2. **The real guard, delivered the real way, rescues it.** ``--mode guard`` writes the guard via
     ``run_grpo.write_dtype_guard`` (the SAME source the GRPO job ships) and re-execs with the
     overlay dir on ``PYTHONPATH`` and ``PRM_VS_ORM_LINEAR_DTYPE_GUARD=1`` -- so ``site`` imports
     ``sitecustomize`` at interpreter startup, exactly as the DeepSpeed learner Ray actors will.
     sitecustomize (not usercustomize) is the load-bearing delivery: this validator runs inside a
     uv/venv where ENABLE_USER_SITE is false and ``site`` skips usercustomize entirely -- the same
     condition as the image -- so a usercustomize-only guard would never arm here or there. The
     armed child runs the identical fp32-weight/bf16-input call and must exit 0. A clean guard run
     where the un-guarded run crashed proves both that ``sitecustomize`` actually imported and that
     the ``F.linear`` patch fixes the mismatch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_grpo  # noqa: E402 -- overlay-local; use the REAL guard source, not a copy


def _is_dtype_mismatch(e: Exception) -> bool:
    """The F.linear dtype check surfaces differently by device: CUDA says 'expected mat1 and mat2
    to have the same dtype', CPU says 'expected ... scalar type ... but found ...'. Match both."""
    msg = str(e).lower()
    return ("same dtype" in msg or "mat1 and mat2" in msg
            or ("scalar type" in msg and ("bfloat" in msg or "float" in msg)))


def _fp32_weight_bf16_input_linear() -> None:
    """The exact crash condition: bf16 activation x fp32 Linear weight.

    Device-independent: the mismatch is a dtype-consistency check F.linear runs before the matmul,
    so it fires on CPU as well as CUDA. We forced the fp32 weight, so the live DeepSpeed/Ray stack
    that caused the natural upcast is not needed to exercise the crash class or the guard's fix.
    """
    import torch  # noqa: PLC0415

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[min] device={dev} torch={torch.__version__}", flush=True)
    lin = torch.nn.Linear(16, 16).to(dev).to(torch.bfloat16)
    lin.weight.data = lin.weight.data.float()  # fp32 weight, like the upcast that bit both arms
    if lin.bias is not None:
        lin.bias.data = lin.bias.data.float()
    x = torch.randn(4, 16, device=dev, dtype=torch.bfloat16)
    y = lin(x)  # -> F.linear(bf16, fp32) : RuntimeError unless the guard aligns dtypes
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"[min] linear ok out.dtype={y.dtype}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["reproduce", "guard"], required=True)
    ap.add_argument("--_guard-armed", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Arm the guard exactly as run_grpo.main() does, then re-exec so `site` runs sitecustomize at
    # interpreter startup -- validates the real delivery mechanism, not an inline monkeypatch.
    if args.mode == "guard" and not args._guard_armed:
        gd = run_grpo.write_dtype_guard("/tmp/prm_vs_orm_overlay")
        env = dict(os.environ)
        env[run_grpo._DTYPE_GUARD_ENV] = "1"
        env["PYTHONPATH"] = gd + os.pathsep + env.get("PYTHONPATH", "")
        print(f"[min] arming guard via {gd}/sitecustomize.py, re-exec ...", flush=True)
        os.execve(sys.executable,
                  [sys.executable, os.path.abspath(__file__), "--mode", "guard", "--_guard-armed"],
                  env)  # replaces this process; never returns

    guard_env = os.environ.get(run_grpo._DTYPE_GUARD_ENV) == "1"
    print(f"[min] mode={args.mode} guard_env={guard_env}", flush=True)

    import torch  # noqa: PLC0415

    # If armed, confirm `site` actually imported sitecustomize and patched F.linear before we test.
    if args._guard_armed:
        patched = torch.nn.functional.linear.__name__ == "_aligned_linear"
        print(f"[min] F.linear patched by sitecustomize at startup = {patched}", flush=True)
        assert patched, "sitecustomize did not arm the guard -- delivery mechanism FAILED"

    try:
        _fp32_weight_bf16_input_linear()
    except RuntimeError as e:
        if _is_dtype_mismatch(e):
            print(f"[min] RESULT=REPRODUCED dtype-mismatch RuntimeError: {e}", flush=True)
            sys.exit(7)
        raise
    print("[min] RESULT=CLEAN forward ok", flush=True)


if __name__ == "__main__":
    main()
