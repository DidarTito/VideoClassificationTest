"""Kinetics-400 class names in the checkpoint's alphabetical class order.

Generated clip caches live inside the dataset root.  They are implementation
artifacts, not action classes, and must never shift every checkpoint index by
one.  Only visible class directories participate in the label vocabulary.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def kinetics400_classes():
    d = ROOT / "datasets" / "kinetics400"
    if d.is_dir():
        names = sorted(
            p.name for p in d.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        if len(names) == 400:
            return names
        if names:
            raise RuntimeError(
                "Kinetics-400 dataset must contain exactly 400 visible class "
                f"directories; found {len(names)} under {d}"
            )
    return [f"action {i}" for i in range(400)]
