#!/usr/bin/env python3
"""Build results/k400_rtx3050_frozen17/final17.csv with EXACTLY 17 rows.

Selects, for every frozen17 model, the single best verifiable RTX3050 row
(same checks as scripts/audit_rtx3050.py: exact checkpoint, frames, subset,
architecture). Fails loudly when any model has no verifiable measurement, so
a partial table can never silently masquerade as the final result.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.registry import ROOT, FROZEN17_KEYS, model_spec
from scripts.audit_rtx3050 import canonical_key, collect_existing_rows, evaluate

OUT_DIR = ROOT / "results" / "k400_rtx3050_frozen17"
OUT = OUT_DIR / "final17.csv"


def main():
    existing = collect_existing_rows()
    selected = {}
    missing = []
    for key in FROZEN17_KEYS:
        spec = model_spec(key)
        verified = [
            row for row in existing
            if canonical_key(row) == key
            and all(evaluate(spec, row).values())
            and (row.get("Status") or "completed").lower() in {"completed", "success", "ok"}
        ]
        if not verified:
            missing.append(key)
            continue
        # Prefer the most recent verifiable run (rows keep file order; the
        # newest results directories sort last in collect_existing_rows).
        selected[key] = verified[-1]

    if missing:
        print("Cannot build final17.csv: no verifiable RTX3050 measurement for:", file=sys.stderr)
        for key in missing:
            print(f"  - {key}", file=sys.stderr)
        print("\nRun scripts/audit_rtx3050.py for the exact fill-missing command.", file=sys.stderr)
        raise SystemExit(1)

    fieldnames = ["canonical_key"] + sorted(
        {name for row in selected.values() for name in row if name != "_source"}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key in FROZEN17_KEYS:
            writer.writerow({"canonical_key": key, **selected[key]})
    print(f"Wrote {OUT} with exactly {len(selected)} rows.")


if __name__ == "__main__":
    main()
