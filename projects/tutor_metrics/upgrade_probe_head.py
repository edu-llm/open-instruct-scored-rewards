"""Upgrade a flat linear probe head to the online reward-head contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _scalar(blob: np.lib.npyio.NpzFile, key: str) -> str:
    return str(blob[key].item())


def upgrade_head(
    source: Path,
    destination: Path,
    *,
    prompt_scheme: str,
    results_path: Path | None = None,
    units_path: Path | None = None,
) -> None:
    with np.load(source, allow_pickle=False) as blob:
        if "meta" in blob.files:
            raise ValueError(f"{source} already carries reward-head metadata")
        if _scalar(blob, "schema") != "tutor-metrics/linear-head-v1":
            raise ValueError(f"{source} has unsupported schema {_scalar(blob, 'schema')!r}")
        stored_prompt_scheme = _scalar(blob, "prompt_scheme")
        if stored_prompt_scheme in {"tutor_metrics", "tutor_metrics_neutral"}:
            if stored_prompt_scheme != prompt_scheme:
                raise ValueError(
                    f"{source} uses prompt scheme {stored_prompt_scheme!r}, not requested {prompt_scheme!r}"
                )
        elif stored_prompt_scheme == "stored":
            if units_path is None:
                raise ValueError("a head with prompt_scheme='stored' requires --units to prove its actual context")
            from projects.tutor_metrics.generate import tutor_messages, tutor_messages_neutral  # noqa: PLC0415

            units = json.loads(units_path.read_text())["units"]
            for index, unit in enumerate(units):
                problem = {"question": unit["question"], "choices": unit.get("choices")}
                if prompt_scheme == "tutor_metrics" and not unit.get("style"):
                    raise ValueError(
                        f"{units_path} unit {index} does not prove stored context {prompt_scheme!r}"
                    )
                expected = (
                    tutor_messages(problem, unit["student_before"], unit["style"])
                    if prompt_scheme == "tutor_metrics"
                    else tutor_messages_neutral(problem, unit["student_before"])
                )
                if unit.get("encoder_messages") != expected:
                    raise ValueError(
                        f"{units_path} unit {index} does not prove stored context {prompt_scheme!r}"
                    )
        else:
            raise ValueError(f"{source} has unsupported prompt scheme {stored_prompt_scheme!r}")

        metrics = [str(value) for value in blob["metrics"]]
        if len(metrics) != len(set(metrics)):
            raise ValueError(f"{source} contains duplicate metric names")
        count = len(metrics)
        vectors = {
            "mean": blob["feature_means"],
            "scale": blob["feature_scales"],
            "coef": blob["coefficients"],
        }
        for name, values in vectors.items():
            if values.ndim != 2 or values.shape[0] != count or not np.isfinite(values).all():
                raise ValueError(f"{source} has invalid {name} array shape {values.shape}")
        if np.any(vectors["scale"] <= 0):
            raise ValueError(f"{source} contains nonpositive feature scales")

        one_dimensional = {
            "poolings": blob["poolings"],
            "layers": blob["layers"],
            "intercepts": blob["intercepts"],
            "alphas": blob["alphas"],
            "deployable": blob["deployable"],
        }
        for name, values in one_dimensional.items():
            if values.ndim != 1 or values.shape[0] != count:
                raise ValueError(f"{source} has invalid {name} array shape {values.shape}")

        results = json.loads(results_path.read_text()) if results_path else {}
        result_metrics = results.get("metrics", {})
        status_values = blob["deployment_status"] if "deployment_status" in blob.files else None
        lo = float(blob["label_min"])
        hi = float(blob["label_max"])
        dimensions = {}
        arrays: dict[str, object] = {}
        for index, key in enumerate(metrics):
            result = result_metrics.get(key, {})
            primary = result.get("ridgeNested") or result.get("ridgeBest") or {}
            enabled = bool(blob["deployable"][index])
            dimensions[key] = {
                "pooling": str(blob["poolings"][index]),
                "layer": int(blob["layers"][index]),
                "lo": lo,
                "hi": hi,
                "alpha": float(blob["alphas"][index]),
                "deployable": enabled,
                "deployment_status": (
                    str(status_values[index]) if status_values is not None else ("enabled" if enabled else "disabled")
                ),
                "cv_pearson": float(primary.get("pearson", float("nan"))),
                "cv_mae": float(primary.get("mae", float("nan"))),
            }
            arrays[f"{key}/mean"] = np.asarray(vectors["mean"][index], dtype=np.float32)
            arrays[f"{key}/scale"] = np.asarray(vectors["scale"][index], dtype=np.float32)
            arrays[f"{key}/coef"] = np.asarray(vectors["coef"][index], dtype=np.float32)
            arrays[f"{key}/intercept"] = np.float32(blob["intercepts"][index])

        if not dimensions.get("assistance_level", {}).get("deployable"):
            raise ValueError("assistance_level must be deployable")
        meta = {
            "schema": "tutor-metrics/reward-head-v1",
            "model": _scalar(blob, "model"),
            "revision": _scalar(blob, "revision"),
            "prompt_scheme": prompt_scheme,
            "dimensions": dimensions,
            "conversion": {"source": str(source), "lossless": True},
        }
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-scheme", choices=("tutor_metrics", "tutor_metrics_neutral"), required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--units", type=Path)
    args = parser.parse_args()
    upgrade_head(
        args.source,
        args.out,
        prompt_scheme=args.prompt_scheme,
        results_path=args.results,
        units_path=args.units,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
