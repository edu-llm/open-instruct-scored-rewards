"""Diagnostic: why does vLLM's olmo2 loader still see an unhashable (nested) rope_parameters
for the real Olmo3-370M SFT checkpoint, and which on-disk config fix survives vLLM's own
``patch_rope_parameters`` (which calls ``standardize_rope_params`` on every load)?

This reads ONLY the SFT checkpoint's ``config.json`` (via the GPU job role's S3 read grant),
then replays vLLM's exact config path — ``vllm.transformers_utils.config.get_config`` followed
by the olmo2 full-attention branch's ``get_rope(rope_parameters=config.rope_parameters)`` — for
the raw config and several candidate fixes, on a CPU (no model weights are loaded, so ``gpu-1xt4``
with no bf16 is fine). Every finding is printed AFTER the ``=== PROBE RESULTS ===`` sentinel so it
survives the ~50-line ``edullm logs`` tail. No AWS SDK/CLI is called directly — only the sanctioned
``edullm_data.s3.Boto3S3`` job-role client, and only to GET one small object.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_common  # noqa: E402 -- overlay-local module


def _split_uri(uri: str) -> tuple[str, str]:
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def _get_config_json(s3: Any, sft_uri: str) -> dict:
    """Fetch just ``config.json`` from under the SFT checkpoint prefix."""
    bucket, prefix = _split_uri(sft_uri)
    prefix = prefix.rstrip("/") + "/"
    # Prefer the exact key; fall back to listing if the layout nests it.
    try:
        return json.loads(s3.get(bucket, prefix + "config.json").decode("utf-8"))
    except Exception:  # noqa: BLE001
        for obj in s3.list(bucket, prefix):
            if obj["key"].endswith("config.json"):
                return json.loads(s3.get(bucket, obj["key"]).decode("utf-8"))
    raise RuntimeError(f"no config.json under {sft_uri}")


def _write(cfg: dict) -> str:
    d = tempfile.mkdtemp()
    Path(d, "config.json").write_text(json.dumps(cfg, indent=2))
    return d


def _rope_key_hashable(rp: Any) -> tuple[bool, str]:
    """Mirror vLLM get_rope's cache-key build: list->tuple, dict stays dict (unhashable)."""
    if not isinstance(rp, dict):
        return True, f"(rope_parameters is {type(rp).__name__}, not a dict)"
    try:
        rp_tuple = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in rp.items())
        hash((64, 64, 4096, True, rp_tuple, None, None))
        return True, "hashable"
    except TypeError as e:
        return False, f"UNHASHABLE: {e}"


def _via_vllm_get_config(model_dir: str) -> tuple[Any, str]:
    """Run vLLM's real config path (AutoConfig + patch_rope_parameters/standardize)."""
    from vllm.transformers_utils.config import get_config  # noqa: PLC0415

    cfg = get_config(model_dir, trust_remote_code=False)
    return cfg, "ok"


def _try_get_rope(rp: Any) -> str:
    """Actually invoke vLLM get_rope exactly as olmo2.py's full-attention branch does."""
    from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: PLC0415

    try:
        get_rope(64, rotary_dim=64, max_position=4096, rope_parameters=rp)
        return "get_rope OK"
    except Exception as e:  # noqa: BLE001
        return f"get_rope RAISED {type(e).__name__}: {e}"


def _report(name: str, cfg: dict) -> None:
    model_dir = _write(cfg)
    print(f"\n----- candidate: {name} -----")
    print(f"on-disk rope_parameters = {json.dumps(cfg.get('rope_parameters'), default=str)}")
    print(f"on-disk rope_scaling    = {json.dumps(cfg.get('rope_scaling'), default=str)}")
    print(f"on-disk model_type      = {cfg.get('model_type')}  architectures={cfg.get('architectures')}")
    try:
        vcfg, _ = _via_vllm_get_config(model_dir)
    except Exception as e:  # noqa: BLE001
        print(f"vllm.get_config RAISED {type(e).__name__}: {e}")
        return
    rp = getattr(vcfg, "rope_parameters", None)
    ok, why = _rope_key_hashable(rp)
    print(f"vllm-loaded rope_parameters = {json.dumps(rp, default=str)}")
    print(f"hashable = {ok}  ({why})")
    print(f"full-attention branch: {_try_get_rope(rp)}")
    # sliding branch reads rope_parameters['rope_theta']
    if isinstance(rp, dict):
        print(f"sliding branch rope_theta lookup: {rp.get('rope_theta', 'MISSING (KeyError at runtime)')}")


def _fix_current(raw: dict) -> dict:
    """Reproduce rm_common.ensure_vllm_rope_parameters (flat rp, rope_scaling null, top theta)."""
    d = _write(copy.deepcopy(raw))
    rm_common.ensure_vllm_rope_parameters(d, verify=False, log=lambda *_: None)
    return json.loads(Path(d, "config.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-uri", required=True)
    args = ap.parse_args()

    import transformers  # noqa: PLC0415
    import vllm  # noqa: PLC0415
    from edullm_data.s3 import Boto3S3  # noqa: PLC0415

    s3 = Boto3S3.default()
    raw = _get_config_json(s3, args.sft_uri)

    print("=== PROBE RESULTS ===", flush=True)
    print(f"transformers {transformers.__version__}  ({transformers.__file__})")
    print(f"vllm {vllm.__version__}")
    print("\n=== RAW config.json (rope-relevant + full) ===")
    print(json.dumps(raw, indent=2, default=str)[:4000])

    # C0: raw, no fix (should reproduce the crash)
    _report("C0 raw (no fix)", raw)

    # C1: current sanitizer
    c1 = _fix_current(raw)
    _report("C1 current sanitizer (flat rp + rope_scaling null + top theta)", c1)

    # C2: current sanitizer + delete layer_types on disk
    c2 = copy.deepcopy(c1)
    c2.pop("layer_types", None)
    _report("C2 C1 + delete layer_types", c2)

    # C3: current sanitizer + force model_type/architectures to olmo2 (no per-layer rope nesting)
    c3 = copy.deepcopy(c1)
    c3["model_type"] = "olmo2"
    c3["architectures"] = ["Olmo2ForCausalLM"]
    c3.pop("layer_types", None)
    c3.pop("sliding_window", None)
    _report("C3 C1 + model_type=olmo2 + architectures=Olmo2ForCausalLM (drop layer_types/sliding_window)", c3)

    # C4: current sanitizer + recursively null EVERY nested rope container anywhere in the config
    c4 = copy.deepcopy(raw)

    def _scrub(o: Any) -> Any:
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if k in ("rope_scaling",):
                    out[k] = None
                elif k == "rope_parameters":
                    out[k] = {"rope_type": "default", "rope_theta": rm_common.OLMO3_370M_ROPE_THETA}
                else:
                    out[k] = _scrub(v)
            return out
        if isinstance(o, list):
            return [_scrub(v) for v in o]
        return o

    c4 = _scrub(c4)
    c4["rope_theta"] = rm_common.OLMO3_370M_ROPE_THETA
    _report("C4 recursive scrub of every rope_parameters/rope_scaling", c4)

    print("\n=== END PROBE ===", flush=True)


if __name__ == "__main__":
    main()
