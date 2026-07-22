"""Backward-compatible entry point for the Stage 1 screening figure.

The canonical implementation and candidate data now live in
``scripts/plot_stage1_screening.py`` and ``data/stage1_candidates_k400.csv``.
"""
from scripts.plot_stage1_screening import main


if __name__ == "__main__":
    raise SystemExit(main())
