"""Full Stage 2 orchestration: screening -> measurement -> NetScore variants.

Usage:
  python stage2_pipeline.py --videos-dir datasets/kinetics400 --max-clips 1000 \
      --device-name RTX3050 --output results_stage2/

Outputs (in --output):
  netscore_screening.csv         all candidates ranked by original NetScore
  netscore_filtered_models.csv   candidates passing the screening constraints
  device_metrics.csv             measured device metrics and 1/8 NetScore variants
  ranking_ns_e.csv               candidates ranked by NS-E(1/8)
  ranking_ns_m.csv               candidates ranked by NS-M(1/8)
  ranking_ns_hash.csv            candidates ranked by NS#(1/8)
  netscore_variants.png          NS-E/NS-M/NS# comparison (when matplotlib exists)
  summary_report.md
"""
import argparse
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from data import load_annotation_ids, load_clip, load_clip_set, load_ground_truth
from models import MODEL_REGISTRY
from run_benchmark import benchmark_models
from screen_models import (
    MAX_GFLOPS_PARETO,
    MAX_PARAMS_M,
    MIN_TOP1,
    build_screening_table,
)


DEPLOYMENT_METRICS = ("NS-E(1/8)", "NS-M(1/8)", "NS#(1/8)")
RANKING_OUTPUTS = {
    "NS-E(1/8)": "ranking_ns_e.csv",
    "NS-M(1/8)": "ranking_ns_m.csv",
    "NS#(1/8)": "ranking_ns_hash.csv",
}


def make_plot(df, out_path):
    """Plot the two specialized scores and use the joint score as color."""
    required = list(DEPLOYMENT_METRICS)
    plot_data = df.dropna(subset=required)
    if plot_data.empty:
        print("No complete NS-E/NS-M/NS# rows; skipping plot.")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        plot_data["NS-E(1/8)"],
        plot_data["NS-M(1/8)"],
        c=plot_data["NS#(1/8)"],
        cmap="viridis",
        s=80,
    )
    x_right = plot_data["NS-E(1/8)"].quantile(0.80)
    y_top = plot_data["NS-M(1/8)"].quantile(0.90)
    for _, row in plot_data.iterrows():
        place_left = row["NS-E(1/8)"] >= x_right
        place_below = row["NS-M(1/8)"] >= y_top
        ax.annotate(
            row["Model"],
            (row["NS-E(1/8)"], row["NS-M(1/8)"]),
            fontsize=8,
            xytext=(-4 if place_left else 4, -5 if place_below else 4),
            textcoords="offset points",
            ha="right" if place_left else "left",
            va="top" if place_below else "bottom",
        )
    ax.margins(x=0.08, y=0.10)
    ax.set_xlabel("NS-E (1/8): accuracy, inference time, and power")
    ax.set_ylabel("NS-M (1/8): accuracy and peak VRAM")
    ax.set_title(f"Deployment NetScore variants on {plot_data['Device'].iloc[0]}")
    fig.colorbar(scatter, label="NS# (1/8): joint deployment score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _duration_text(seconds):
    if not math.isfinite(seconds):
        return "unavailable"
    return f"{seconds:.1f} s ({seconds / 60:.2f} min, {seconds / 3600:.3f} h)"


def _ranking_table(metrics, metric):
    columns = [
        "Model",
        "FramesUsed",
        "MetricAccuracy(%)",
        "MetricAccuracySource",
        "Top1Measured(%)",
        "Top1Published(%)",
        "Latency(ms)",
        "AvgPower(W)",
        "Energy(J)",
        "PeakVRAM(MiB)",
        "TotalTime(s)",
        *DEPLOYMENT_METRICS,
    ]
    return metrics.sort_values(metric, ascending=False)[columns]


def write_report(out_dir, screening, filtered, metrics, errors, n_clips, args):
    device = metrics["Device"].iloc[0] if not metrics.empty else (args.device_name or "unknown")
    dataset = metrics["Dataset"].iloc[0] if "Dataset" in metrics else args.dataset
    total_times = pd.to_numeric(metrics["TotalTime(s)"], errors="coerce")
    estimated_all_s = (
        float(total_times.sum()) if total_times.notna().any() else float("nan")
    )

    def md(frame, columns=None):
        selected = frame[columns] if columns else frame
        return selected.to_markdown(index=False)

    hardware_columns = [
        "Model",
        "FramesUsed",
        "Top1Measured(%)",
        "Top1Published(%)",
        "MetricAccuracy(%)",
        "MetricAccuracySource",
        "Latency(ms)",
        "LatencyStd(ms)",
        "FPS",
        "AvgPower(W)",
        "Energy(J)",
        "TotalTime(s)",
        "TotalEnergy(J)",
        "PeakVRAM(MiB)",
        *DEPLOYMENT_METRICS,
    ]
    rankings = {
        metric: metrics.sort_values(metric, ascending=False)
        for metric in DEPLOYMENT_METRICS
    }

    winners = []
    for metric, ranked in rankings.items():
        valid = ranked.dropna(subset=[metric])
        if valid.empty:
            winners.append(f"- **{metric}:** unavailable because a required measurement is missing.")
        else:
            top = valid.iloc[0]
            winners.append(
                f"- **{metric}:** {top['Model']} ranks first at {top[metric]:.2f}."
            )

    lines = [
        "# Stage 2 Summary Report - Energy-Efficient Video ViT Inference",
        f"\nGenerated: {datetime.now():%Y-%m-%d %H:%M} | Device: **{device}** | "
        f"Dataset: **{dataset}** | Clips per model: **{n_clips}** | Batch size: **1**",
        "\nEach row reports the model adapter's actual temporal input in `FramesUsed`; "
        f"clips were decoded at {args.frames} frames and {args.size}x{args.size} before "
        "model-specific preprocessing.",
        "\n## 2.1 Device-Agnostic Screening (NetScore)",
        f"\nConstraints: Top-1 >= {MIN_TOP1}%, Params < {MAX_PARAMS_M}M, "
        f"GFLOPs <= {MAX_GFLOPS_PARETO}. {len(filtered)}/{len(screening)} candidates pass.",
        "\n"
        + md(
            filtered.reset_index(),
            ["Rank", "Model", "Top1(%)", "Params(M)", "GFLOPs", "NetScore"],
        ),
        "\n## 2.2 Hardware Measurements and Deployment Scores",
        "\n" + md(metrics, hardware_columns),
        "\n### Total inference-only time estimate",
        f"\nEstimated inference-only time for all {len(metrics)} measured candidates: "
        f"**{_duration_text(estimated_all_s)}**. This is the sum of each model's "
        "mean latency multiplied by its clip count; dataset decoding, model loading, "
        "warm-up, and power-logger startup are excluded.",
        "\n## 2.3 NetScore Variant Rankings (weight = 1/8)",
    ]
    for metric in DEPLOYMENT_METRICS:
        lines += [f"\n### {metric}", "\n" + md(_ranking_table(metrics, metric))]

    lines += [
        "\n## Intermediate conclusions",
        "\n" + "\n".join(winners),
        "\n- NS-E combines the selected accuracy with total inference time and average "
        "power; NS-M combines it with peak inference VRAM; NS# applies all three "
        "deployment penalties jointly.",
        "- Higher is better for every NetScore variant. Different leaders expose "
        "whether time/power efficiency, memory efficiency, or the joint trade-off "
        "is driving a model's position.",
        "\n## Notes",
        "\n- `Energy(J)` is average power multiplied by mean inference latency for one "
        "clip. `TotalTime(s)` and `TotalEnergy(J)` cover the complete timed clip set.",
        "- The 1/8 exponent is applied independently to each deployment penalty, as "
        "specified for NS-E, NS-M, and NS# in the reference methodology.",
        "- `MetricAccuracy(%)` is measured top-1 when annotations are available and "
        "published top-1 otherwise; `MetricAccuracySource` records that choice.",
        "- Measured top-1 uses one uniformly sampled full-video view and is not "
        "directly comparable to a paper's multi-view evaluation protocol.",
    ]
    if errors:
        lines += [
            "\n## Skipped models",
            "\nFailed candidates are excluded from the aggregate time estimate and rankings.",
            "\n" + "\n".join(f"- `{key}`: {value}" for key, value in errors.items()),
        ]
    not_runnable = screening[
        ~screening["RunnableHere"] & screening["PassesPareto"]
    ]["Model"].tolist()
    if not_runnable:
        lines += [
            "\nCandidates passing screening but without a runnable implementation here: "
            + ", ".join(not_runnable)
            + "."
        ]

    report = out_dir / "summary_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", help="dataset video directory")
    parser.add_argument("--video", help="single video fallback (smoke test)")
    parser.add_argument("--max-clips", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=32, help="frames decoded per clip")
    parser.add_argument("--size", type=int, default=224, help="decoded frame size")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="decode clips without reading or writing the sharded dataset cache",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="runs per model in single-video mode"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_REGISTRY),
        help="override: benchmark these registry keys instead of the screened set",
    )
    parser.add_argument("--dataset", default="k400", choices=["k400", "ssv2"])
    parser.add_argument(
        "--annotations",
        help="SSV2 labeled split annotation (official JSON, semicolon, or vendored numeric list)",
    )
    parser.add_argument("--split-name", help="split label recorded in result rows")
    parser.add_argument(
        "--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:1, mps, ..."
    )
    parser.add_argument(
        "--power",
        default="auto",
        choices=["auto", "nvml", "nvidia-smi", "tegrastats", "none"],
    )
    parser.add_argument("--device-name", default=None)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    if not args.video and not args.videos_dir:
        parser.error("provide --videos-dir or --video")
    if args.dataset == "ssv2" and args.videos_dir and not args.annotations:
        parser.error("--annotations is required for annotation-filtered SSV2 sampling")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 2.1: screening ----
    print("### Stage 2.1: NetScore screening")
    screening = build_screening_table()
    screening_path = out_dir / "netscore_screening.csv"
    filtered_path = out_dir / "netscore_filtered_models.csv"
    screening.to_csv(screening_path)
    # Stage 2 follows the 300-GFLOP deployment threshold used to select the
    # 17 requested candidates, not the broader 600-GFLOP literature table.
    filtered = screening[screening["PassesPareto"]]
    filtered.to_csv(filtered_path)
    print(f"{len(filtered)}/{len(screening)} candidates pass screening.")

    if args.models:
        model_keys = args.models
    else:
        model_keys = [key for key in filtered["RegistryKey"] if key in MODEL_REGISTRY]
    print("Models to benchmark:", " ".join(model_keys))

    # ---- Stage 2.2: measurement ----
    print("\n### Stage 2.2: hardware measurement")
    labels = None
    if args.videos_dir:
        allowed_stems = (
            load_annotation_ids(args.annotations) if args.dataset == "ssv2" else None
        )
        clips, paths = load_clip_set(
            args.videos_dir,
            max_clips=args.max_clips,
            frames=args.frames,
            size=args.size,
            seed=args.seed,
            use_cache=not args.no_cache,
            allowed_stems=allowed_stems,
        )
        labels = load_ground_truth(paths, args.dataset, args.annotations)
    else:
        clips = [load_clip(args.video, args.frames, args.size)] * args.runs

    split_name = args.split_name
    if split_name is None and args.annotations:
        split_name = Path(args.annotations).stem
    metrics, errors = benchmark_models(
        model_keys,
        clips,
        args.power,
        args.device_name,
        labels,
        dataset=args.dataset,
        device=args.device,
        split_name=split_name,
        annotation_file=args.annotations,
    )
    if metrics.empty:
        raise SystemExit("No model produced results; see errors above.")
    metrics_path = out_dir / "device_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # ---- Stage 2.3: deployment NetScore ranking ----
    print("\n### Stage 2.3: NS-E, NS-M, and NS# ranking (weight = 1/8)")
    ranking_paths = []
    for metric, filename in RANKING_OUTPUTS.items():
        ranked = metrics.sort_values(metric, ascending=False)
        ranking_path = out_dir / filename
        ranked.to_csv(ranking_path, index=False)
        ranking_paths.append(ranking_path)
        valid = ranked.dropna(subset=[metric])
        if valid.empty:
            print(f"{metric}: unavailable (a required measurement is missing)")
        else:
            top = valid.iloc[0]
            print(
                f"{metric} best: {top['Model']} ({metric}={top[metric]}, "
                f"accuracy {top['MetricAccuracy(%)']}%)"
            )

    estimated_all_s = float(
        pd.to_numeric(metrics["TotalTime(s)"], errors="coerce").fillna(0).sum()
    )
    print(
        f"Estimated inference-only time for all {len(metrics)} measured candidates: "
        f"{_duration_text(estimated_all_s)}"
    )

    plot = make_plot(metrics, out_dir / "netscore_variants.png")
    report = write_report(out_dir, screening, filtered, metrics, errors, len(clips), args)
    deliverables = [screening_path, filtered_path, metrics_path, *ranking_paths]
    if plot:
        deliverables.append(plot)
    deliverables.append(report)
    print(f"\nDeliverables written to {out_dir.resolve()}:")
    for path in deliverables:
        print("  -", path.name)


if __name__ == "__main__":
    main()
