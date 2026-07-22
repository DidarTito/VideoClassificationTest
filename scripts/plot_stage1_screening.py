"""Generate the reproducible Stage 1 Kinetics-400 screening figure.

The input table contains published literature configurations.  It is kept
separate from device measurements because the exact checkpoint/input recipes
used in Stage 2 can have different accuracy, parameter, and FLOP values.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Allow direct execution from any working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import netscore  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "stage1_candidates_k400.csv"
DEFAULT_OUTPUT_DIR = ROOT / "figures"
DEFAULT_STEM = "stage1_screening_k400"

ACCURACY_THRESHOLD = 78.0
GFLOPS_THRESHOLD = 300.0
PARAMS_THRESHOLD_M = 300.0
EXPECTED_CANDIDATES = 36
EXPECTED_SELECTED = 17

REQUIRED_COLUMNS = {
    "model",
    "dataset",
    "top1_published_pct",
    "params_m",
    "gflops",
    "net_score",
    "selected_stage2",
    "venue",
    "value_scope",
}

# Deterministic offsets in screen points.  Keeping this mapping explicit makes
# the slide layout stable across machines and handles the coincident
# DualFormer-S / DualFormer-B point without altering either data coordinate.
LABEL_OFFSETS: dict[str, tuple[int, int]] = {
    "UniFormer-B": (10, 13),
    "UniFormer-S": (-62, 15),
    "MViT-B, 64x3": (10, -22),
    "MViT-B-24, 32x3": (-76, 18),
    "MViT-B, 32x3": (13, -27),
    "MViT-S": (-55, 23),
    "VideoSwin-T": (12, -22),
    "VideoSwin-S": (-72, -17),
    "VideoSwin-B": (13, 18),
    "VideoSwin-L": (-60, -25),
    "ViViT-S": (10, 20),
    "ViViT-B": (-72, 18),
    "ViViT-L": (10, -24),
    "ViViT-L (320)": (-84, 12),
    "ViViT-H": (-75, -8),
    "TimeSformer-B": (-72, -19),
    "TimeSformer-HR": (12, -28),
    "TimeSformer-L": (-76, 12),
    "ZeroI2V ViT-B/16": (12, -23),
    "ZeroI2V ViT-L/14": (-82, -18),
    "Video-FocalNet-T": (12, -15),
    "Video-FocalNet-S": (-76, 17),
    "Video-FocalNet-B": (12, -22),
    "DualFormer-T": (-78, -24),
    "DualFormer-S": (13, 19),
    "DualFormer-B": (-76, -20),
    "Omnivore-B": (12, 23),
    "ActionCLIP-B": (-90, 22),
    "BIKE-B": (12, 18),
    "VATT-L": (-72, -20),
    "VideoMAE-B": (13, 18),
    "SVT-B": (-72, 17),
    "VTN-B": (12, -20),
    "Motionformer-B": (-84, 12),
    "TokenLearner-L": (-88, 21),
    "Efficient-VDiT": (-70, -12),
}


def selection_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the exact Stage 1 deployment-screening decision."""
    return (
        (frame["top1_published_pct"] >= ACCURACY_THRESHOLD)
        & (frame["gflops"] <= GFLOPS_THRESHOLD)
        & (frame["params_m"] < PARAMS_THRESHOLD_M)
    )


def _parse_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        values = sorted(normalized[invalid].unique())
        raise ValueError(f"selected_stage2 contains invalid values: {values}")
    return normalized.eq("true")


def load_candidates(path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load and fully validate the canonical Stage 1 candidate table."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Stage 1 CSV is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"Expected {EXPECTED_CANDIDATES} candidates, found {len(frame)}"
        )
    if frame["model"].duplicated().any():
        duplicates = frame.loc[frame["model"].duplicated(), "model"].tolist()
        raise ValueError(f"Duplicate Stage 1 models: {duplicates}")
    if set(frame["dataset"]) != {"Kinetics-400"}:
        raise ValueError("Every Stage 1 row must use dataset=Kinetics-400")
    if set(frame["value_scope"]) != {"published_literature_configuration"}:
        raise ValueError("Every row must identify its literature-configuration scope")

    numeric = ["top1_published_pct", "params_m", "gflops", "net_score"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains a non-finite value")
    if (frame[["top1_published_pct", "params_m", "gflops"]] <= 0).any().any():
        raise ValueError("Accuracy, parameters, and GFLOPs must be positive")

    computed_scores = np.array(
        [
            round(netscore(acc, params, gflops), 2)
            for acc, params, gflops in frame[
                ["top1_published_pct", "params_m", "gflops"]
            ].itertuples(index=False, name=None)
        ]
    )
    if not np.array_equal(computed_scores, frame["net_score"].to_numpy()):
        mismatched = frame.loc[
            computed_scores != frame["net_score"].to_numpy(),
            ["model", "net_score"],
        ].copy()
        mismatched["computed_net_score"] = computed_scores[
            computed_scores != frame["net_score"].to_numpy()
        ]
        raise ValueError(
            "Stored NetScore values disagree with metrics.netscore:\n"
            + mismatched.to_string(index=False)
        )

    stored_selection = _parse_boolean(frame["selected_stage2"])
    computed_selection = selection_mask(frame)
    if not stored_selection.equals(computed_selection):
        mismatched_models = frame.loc[
            stored_selection != computed_selection, "model"
        ].tolist()
        raise ValueError(
            "Stored Stage 2 selection disagrees with the thresholds for: "
            + ", ".join(mismatched_models)
        )
    frame["selected_stage2"] = computed_selection
    if int(computed_selection.sum()) != EXPECTED_SELECTED:
        raise ValueError(
            f"Expected {EXPECTED_SELECTED} selected candidates, found "
            f"{int(computed_selection.sum())}"
        )
    if set(frame["model"]) != set(LABEL_OFFSETS):
        missing_offsets = sorted(set(frame["model"]) - set(LABEL_OFFSETS))
        stale_offsets = sorted(set(LABEL_OFFSETS) - set(frame["model"]))
        raise ValueError(
            f"Label-offset mismatch; missing={missing_offsets}, stale={stale_offsets}"
        )
    return frame


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker="*",
            markersize=18,
            markerfacecolor="none",
            markeredgecolor="#d59b00",
            markeredgewidth=2.0,
            linestyle="None",
            label="Selected for Stage 2",
        ),
        Line2D(
            [],
            [],
            color="black",
            linewidth=1.8,
            linestyle="--",
            label="300 GFLOPs threshold",
        ),
        Line2D(
            [],
            [],
            color="black",
            linewidth=1.8,
            linestyle=":",
            label="78% accuracy threshold",
        ),
    ]


def create_figure(frame: pd.DataFrame) -> plt.Figure:
    """Create the deterministic, slide-ready Stage 1 screening figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(18, 11))

    x_min, x_max = 20.0, 4500.0
    y_min, y_max = 74.5, 87.0
    ax.fill_between(
        [x_min, GFLOPS_THRESHOLD],
        ACCURACY_THRESHOLD,
        y_max,
        color="#6cab6c",
        alpha=0.10,
        zorder=0,
    )

    norm = Normalize(vmin=frame["params_m"].min(), vmax=frame["params_m"].max())
    dots = ax.scatter(
        frame["gflops"],
        frame["top1_published_pct"],
        c=frame["params_m"],
        cmap="viridis",
        norm=norm,
        s=105,
        marker="o",
        edgecolors="#262626",
        linewidths=0.75,
        alpha=0.95,
        zorder=2,
    )

    selected = frame[frame["selected_stage2"]]
    # A hollow star retains the parameter-colored dot underneath it.
    ax.scatter(
        selected["gflops"],
        selected["top1_published_pct"],
        s=365,
        marker="*",
        facecolors="none",
        edgecolors="#d59b00",
        linewidths=2.1,
        zorder=4,
    )

    ax.axvline(
        GFLOPS_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.8,
        zorder=1,
    )
    ax.axhline(
        ACCURACY_THRESHOLD,
        color="black",
        linestyle=":",
        linewidth=1.8,
        zorder=1,
    )

    for row in frame.itertuples(index=False):
        dx, dy = LABEL_OFFSETS[row.model]
        ax.annotate(
            f"{row.model}\nNS-{row.net_score:.2f}",
            xy=(row.gflops, row.top1_published_pct),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.6,
            fontweight="bold" if row.selected_stage2 else "normal",
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "#777777",
                "linewidth": 0.6,
                "alpha": 0.91,
            },
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#707070",
                "linewidth": 0.7,
                "mutation_scale": 7,
                "shrinkA": 1,
                "shrinkB": 5,
            },
            zorder=5,
            annotation_clip=False,
        )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Model Complexity (GFLOPs)", fontsize=14)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.30)
    ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.5, alpha=0.16)

    fig.suptitle(
        "Stage 1 Screening of Video Transformer Models",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    ax.set_title(
        "★ Selected models satisfy Stage 1 constraints",
        fontsize=14,
        pad=10,
    )

    colorbar = fig.colorbar(dots, ax=ax, pad=0.025, fraction=0.035)
    colorbar.set_label("Parameters (Millions)", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)

    legend = ax.legend(
        handles=_legend_handles(),
        loc="lower right",
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=10.5,
    )
    legend.set_zorder(10)

    fig.text(
        0.065,
        0.018,
        (
            "Published Kinetics-400 literature configurations; exact local checkpoint "
            "recipes may differ.  NS = 20 log10(A² / √(Params × GFLOPs))."
        ),
        fontsize=8.8,
        color="#404040",
    )
    fig.tight_layout(rect=(0.025, 0.045, 0.985, 0.965))
    return fig


def save_figure(
    figure: plt.Figure,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    stem: str = DEFAULT_STEM,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    title = "Stage 1 Screening of Video Transformer Models"
    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        metadata={"Title": title, "Author": "VideoClassificationTest"},
    )
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Title": title, "Creator": "VideoClassificationTest", "Date": None},
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": title,
            "Author": "VideoClassificationTest",
            "Creator": "VideoClassificationTest",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    return png_path, svg_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_candidates(args.input)
    figure = create_figure(frame)
    try:
        outputs = save_figure(figure, args.output_dir, args.stem)
    finally:
        plt.close(figure)

    selected_names = frame.loc[frame["selected_stage2"], "model"].tolist()
    print(
        f"Validated {len(frame)} published K400 candidates; "
        f"selected {len(selected_names)} for Stage 2."
    )
    print("Selected:", ", ".join(selected_names))
    for output in outputs:
        print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
