"""Write an /api/admin/export bundle to disk in the layout the analysis scripts already read.

    curl -sH "Authorization: Bearer $ADMIN_TOKEN" \\
        "$APP/api/admin/export?batch=b1&min_status=active" > export.json
    python scripts/unpack_export.py export.json --into data/

Then, with no edits to either file:

    python -m projects.pedagogy_rm.agreement --labels 'data/labels/*.json' --dimensions ...
    python -m projects.pedagogy_rm.extract_hidden --units data/label_slices/units.json \\
        --out data/hidden.npz
    python -m projects.pedagogy_rm.fit_head --hidden data/hidden.npz \\
        --labels 'data/labels/*.json' --slices data/label_slices/units.json \\
        --dimensions ... --decouple '' --cells '...'

`covariates/` is deliberately a sibling of `labels/` rather than inside it: fit_head would take a
mean of a non-ordinal key, and a mean of 2.0 on such a scale can be every turn correct or half of
them wrong in each direction.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", help="the JSON written by /api/admin/export")
    parser.add_argument("--into", default="data", help="directory to write under")
    args = parser.parse_args()

    with open(args.bundle) as handle:
        blob = json.load(handle)
    files = blob.get("files") or {}
    root = pathlib.Path(args.into)

    written = 0
    for name, content in sorted(files.items()):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # .jsonl arrives as one string; everything else is a JSON value.
        if isinstance(content, str):
            target.write_text(content)
        else:
            target.write_text(json.dumps(content, indent=1) + "\n")
        written += 1

    manifest = blob.get("manifest") or {}
    counts = manifest.get("counts") or {}
    print(f"wrote {written} files under {root}/")
    for key in ("labels_rows", "checks_excluded", "units", "shared_units", "not_applicable", "flags"):
        if key in counts:
            print(f"  {key:<18} {counts[key]}")
    per_bucket = counts.get("per_bucket") or {}
    if per_bucket:
        print(f"  per bucket         {per_bucket}")

    violations = manifest.get("violations") or []
    if violations:
        print("\n  EXPORT ASSERTIONS FAILED — do not fit on this:")
        for v in violations:
            print(f"    {v}")
        raise SystemExit(1)

    if not counts.get("shared_units"):
        print(
            "\n  No unit was labelled by two raters, so agreement is unmeasurable.\n"
            "  Raise item.target_labels before collecting more; this is the mistake the last round made."
        )


if __name__ == "__main__":
    main()
