import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def clean_numeric(series):
    """Convert column to numeric, tolerating %, commas, spaces, etc."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def pareto_frontier(df, x_col, y_col):
    """
    Pareto objective:
        x -> MINIMIZE  (energy / memory)
        y -> MAXIMIZE  (accuracy)

    A point is Pareto optimal if no other point is:
        x <= current_x
        y >= current_y
    with at least one strict improvement.
    """
    valid = df.dropna(subset=[x_col, y_col]).copy()

    pareto_mask = []

    for idx, row in valid.iterrows():
        x = row[x_col]
        y = row[y_col]

        dominated = (
            (valid[x_col] <= x)
            & (valid[y_col] >= y)
            & (
                (valid[x_col] < x)
                | (valid[y_col] > y)
            )
        ).any()

        pareto_mask.append(not dominated)

    frontier = valid.loc[pareto_mask].copy()

    # Left -> right by cost for drawing frontier line
    frontier = frontier.sort_values(
        [x_col, y_col],
        ascending=[True, True]
    )

    return frontier


def plot_pareto(
    df,
    name_col,
    accuracy_col,
    cost_col,
    xlabel,
    title,
    output_path,
):
    usable = df.dropna(
        subset=[name_col, accuracy_col, cost_col]
    ).copy()

    frontier = pareto_frontier(
        usable,
        cost_col,
        accuracy_col
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # All tested models
    ax.scatter(
        usable[cost_col],
        usable[accuracy_col],
        s=75,
        alpha=0.75,
        label="Tested models",
    )

    # Pareto points
    ax.scatter(
        frontier[cost_col],
        frontier[accuracy_col],
        s=150,
        marker="*",
        label="Pareto frontier",
        zorder=3,
    )

    # Connect Pareto frontier
    if len(frontier) > 1:
        ax.plot(
            frontier[cost_col],
            frontier[accuracy_col],
            linewidth=1.8,
            linestyle="--",
            zorder=2,
        )

    # Annotate every model
    for _, row in usable.iterrows():
        is_pareto = row[name_col] in set(frontier[name_col])

        label = str(row[name_col])

        if is_pareto:
            label += " ★"

        ax.annotate(
            label,
            (row[cost_col], row[accuracy_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Measured Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return frontier


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate accuracy-vs-energy and accuracy-vs-memory "
            "Pareto plots for video-model benchmarks."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV"
    )

    parser.add_argument(
        "--name-col",
        default="Model",
        help="Model-name column"
    )

    parser.add_argument(
        "--accuracy-col",
        default="MeasuredTop1",
        help="Measured accuracy column"
    )

    parser.add_argument(
        "--energy-col",
        default="EnergyPerClip",
        help="Energy per clip column"
    )

    parser.add_argument(
        "--memory-col",
        default="PeakVRAM",
        help="Peak GPU memory column"
    )

    parser.add_argument(
        "--output-dir",
        default="results/pareto",
        help="Output directory"
    )

    parser.add_argument(
        "--title-prefix",
        default="Video Model Benchmark",
        help="Text shown at beginning of plot titles"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    required = [
        args.name_col,
        args.accuracy_col,
        args.energy_col,
        args.memory_col,
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        print("\nERROR: missing columns:")
        for col in missing:
            print("  -", col)

        print("\nAvailable columns:")
        for col in df.columns:
            print("  -", col)

        raise SystemExit(1)

    # Convert numeric columns
    for col in [
        args.accuracy_col,
        args.energy_col,
        args.memory_col,
    ]:
        df[col] = clean_numeric(df[col])

    # -----------------------------
    # ACCURACY vs ENERGY
    # -----------------------------

    energy_frontier = plot_pareto(
        df=df,
        name_col=args.name_col,
        accuracy_col=args.accuracy_col,
        cost_col=args.energy_col,
        xlabel="Energy per Clip (J) — lower is better",
        title=(
            f"{args.title_prefix}\n"
            "Measured Accuracy vs Energy per Clip"
        ),
        output_path=output_dir / "pareto_accuracy_vs_energy.png",
    )

    # -----------------------------
    # ACCURACY vs MEMORY
    # -----------------------------

    memory_frontier = plot_pareto(
        df=df,
        name_col=args.name_col,
        accuracy_col=args.accuracy_col,
        cost_col=args.memory_col,
        xlabel="Peak GPU Memory (MiB) — lower is better",
        title=(
            f"{args.title_prefix}\n"
            "Measured Accuracy vs Peak GPU Memory"
        ),
        output_path=output_dir / "pareto_accuracy_vs_memory.png",
    )

    # Save Pareto model lists
    energy_frontier.to_csv(
        output_dir / "pareto_energy_models.csv",
        index=False
    )

    memory_frontier.to_csv(
        output_dir / "pareto_memory_models.csv",
        index=False
    )

    print("\n======================================")
    print("ACCURACY vs ENERGY PARETO FRONTIER")
    print("======================================")

    print(
        energy_frontier[
            [
                args.name_col,
                args.accuracy_col,
                args.energy_col,
            ]
        ].to_string(index=False)
    )

    print("\n======================================")
    print("ACCURACY vs MEMORY PARETO FRONTIER")
    print("======================================")

    print(
        memory_frontier[
            [
                args.name_col,
                args.accuracy_col,
                args.memory_col,
            ]
        ].to_string(index=False)
    )

    print("\nGenerated:")
    print(
        output_dir / "pareto_accuracy_vs_energy.png"
    )
    print(
        output_dir / "pareto_accuracy_vs_memory.png"
    )
    print(
        output_dir / "pareto_energy_models.csv"
    )
    print(
        output_dir / "pareto_memory_models.csv"
    )


if __name__ == "__main__":
    main()