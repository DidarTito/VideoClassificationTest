#!/usr/bin/env python3
"""Frozen-17 checkpoint audit under the canonical report names.

This is the Phase 2 entry point. It reuses the existing auditor in
``scripts/audit_checkpoints.py`` (single implementation, no duplicated
logic) and writes:

    reports/frozen17_checkpoint_audit.csv
    reports/FROZEN17_CHECKPOINT_AUDIT.md

in addition to the historical ``reports/checkpoint_audit.csv`` /
``reports/CHECKPOINT_AUDIT.md`` names that other tooling already reads.
"""
import csv
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_checkpoints import main as run_audit  # noqa: E402
from training.registry import ROOT  # noqa: E402


def main():
    run_audit()
    reports = ROOT / "reports"
    pairs = (
        ("checkpoint_audit.csv", "frozen17_checkpoint_audit.csv"),
        ("CHECKPOINT_AUDIT.md", "FROZEN17_CHECKPOINT_AUDIT.md"),
    )
    for source, destination in pairs:
        shutil.copyfile(reports / source, reports / destination)
        print(f"Wrote {reports / destination}")
    with (reports / "frozen17_checkpoint_audit.csv").open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    if len(rows) != 17:
        raise SystemExit(f"Expected 17 audited models, found {len(rows)}")


if __name__ == "__main__":
    main()
