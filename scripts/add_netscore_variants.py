"""Add the deployment-aware NetScore variants to a Stage-2 metrics CSV.

The input file is read-only.  Results are written to a separate output
directory as an augmented ``device_metrics.csv``, three ranking CSVs, and a
short Markdown report.

Example:
    python scripts/add_netscore_variants.py \
        --input results_stage2_k400_1000/device_metrics.csv \
        --output-dir results_stage2_k400_1000_netscore
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


# Executing this file directly puts ``scripts/`` rather than the repository
# root on sys.path.  Add the root so this post-processor always uses the shared
# metric definitions in metrics.py.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from metrics import netscore_e, netscore_hash, netscore_m  # noqa: E402


WEIGHT = 1 / 8
SCORE_COLUMNS = ("NS-E(1/8)", "NS-M(1/8)", "NS#(1/8)")
RANKING_FILENAMES = {
    "NS-E(1/8)": "ranking_ns_e.csv",
    "NS-M(1/8)": "ranking_ns_m.csv",
    "NS#(1/8)": "ranking_ns_hash.csv",
}
REPORT_FILENAME = "netscore_variants_report.md"
_SAM_COLUMN = re.compile(r"^SAM(?:$|[\d\s(_-])", re.IGNORECASE)


def _same_path(left: Path, right: Path) -> bool:
    """Compare resolved paths using the host's case-normalisation rules."""
    left_text = os.path.normcase(str(left.resolve()))
    right_text = os.path.normcase(str(right.resolve()))
    return left_text == right_text


def _first_column(
    columns: Iterable[str], aliases: Sequence[str], *, required: bool = True
) -> str | None:
    """Return the first present alias, accepting case-only header differences."""
    by_casefold = {str(column).casefold(): str(column) for column in columns}
    for alias in aliases:
        if alias in columns:
            return alias
        match = by_casefold.get(alias.casefold())
        if match is not None:
            return match
    if required:
        raise ValueError(
            "missing required column; expected one of: " + ", ".join(aliases)
        )
    return None


def _row_names(frame: pd.DataFrame, mask: pd.Series) -> str:
    """Return concise row identifiers for a validation error."""
    model_column = _first_column(frame.columns, ("Model", "ModelKey"), required=False)
    labels = []
    for index in frame.index[mask]:
        if model_column is None or pd.isna(frame.at[index, model_column]):
            labels.append(str(index))
        else:
            labels.append(f"{index} ({frame.at[index, model_column]})")
    return ", ".join(labels[:8]) + (" ..." if len(labels) > 8 else "")


def _positive_numeric(
    frame: pd.DataFrame, column: str, *, description: str
) -> pd.Series:
    """Parse and validate one positive, finite metric column."""
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.notna() & values.map(
        lambda value: math.isfinite(float(value)) if pd.notna(value) else False
    ) & (values > 0)
    if not bool(valid.all()):
        bad_rows = _row_names(frame, ~valid)
        raise ValueError(
            f"{description} column {column!r} must contain positive finite "
            f"values; invalid rows: {bad_rows}"
        )
    return values.astype(float)


def _metric_accuracy(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Select measured accuracy per row, falling back to published accuracy."""
    measured_column = _first_column(
        frame.columns, ("Top1Measured(%)",), required=False
    )
    published_column = _first_column(
        frame.columns, ("Top1Published(%)", "Top1(%)"), required=False
    )
    if measured_column is None and published_column is None:
        raise ValueError(
            "missing accuracy columns; expected Top1Measured(%) and/or "
            "Top1Published(%) (legacy Top1(%) is also accepted)"
        )

    missing = pd.Series(float("nan"), index=frame.index, dtype=float)
    measured = (
        pd.to_numeric(frame[measured_column], errors="coerce")
        if measured_column is not None
        else missing
    )
    published = (
        pd.to_numeric(frame[published_column], errors="coerce")
        if published_column is not None
        else missing
    )

    measured_valid = measured.notna() & measured.map(
        lambda value: math.isfinite(float(value)) if pd.notna(value) else False
    ) & (measured > 0) & (measured <= 100)
    published_valid = published.notna() & published.map(
        lambda value: math.isfinite(float(value)) if pd.notna(value) else False
    ) & (published > 0) & (published <= 100)

    accuracy = measured.where(measured_valid, published)
    valid = measured_valid | published_valid
    if not bool(valid.all()):
        bad_rows = _row_names(frame, ~valid)
        raise ValueError(
            "each row needs a positive accuracy no greater than 100 in "
            "Top1Measured(%) or its published fallback; invalid rows: "
            + bad_rows
        )

    sources = pd.Series(
        ["measured" if is_measured else "published" for is_measured in measured_valid],
        index=frame.index,
        dtype="object",
    )
    return accuracy.astype(float), sources


def augment_dataframe(source: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, SAM-free copy with all three 1/8 NetScores."""
    if source.empty:
        raise ValueError("input CSV has no model rows")

    result = source.drop(
        columns=[column for column in source.columns if _SAM_COLUMN.match(str(column))]
    ).copy()

    total_time_column = _first_column(result.columns, ("TotalTime(s)",))
    power_column = _first_column(result.columns, ("AvgPower(W)",))
    memory_column = _first_column(
        result.columns, ("PeakVRAM(MiB)", "GPUMemory(MB)")
    )

    accuracy, accuracy_source = _metric_accuracy(result)
    total_time = _positive_numeric(
        result, total_time_column, description="total inference time"
    )
    average_power = _positive_numeric(
        result, power_column, description="average power"
    )
    peak_vram = _positive_numeric(
        result, memory_column, description="peak VRAM"
    )

    result["MetricAccuracy(%)"] = accuracy.round(6)
    result["MetricAccuracySource"] = accuracy_source
    # Older Stage-2 files stored the same CUDA MiB measurement under the
    # historical GPUMemory(MB) label.  Keep that source column and also emit a
    # canonical header for rankings and downstream reports.
    result["PeakVRAM(MiB)"] = peak_vram.round(6)
    result["NS-E(1/8)"] = [
        round(netscore_e(a, t, p, w=WEIGHT), 2)
        for a, t, p in zip(accuracy, total_time, average_power)
    ]
    result["NS-M(1/8)"] = [
        round(netscore_m(a, memory, w=WEIGHT), 2)
        for a, memory in zip(accuracy, peak_vram)
    ]
    result["NS#(1/8)"] = [
        round(netscore_hash(a, memory, t, p, w=WEIGHT), 2)
        for a, memory, t, p in zip(
            accuracy, peak_vram, total_time, average_power
        )
    ]
    return result


def _deduplicate(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _ranking(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build a compact, presentation-ready ranking for one score."""
    identity_columns = [
        column
        for column in ("Device", "Dataset", "Split", "ModelKey", "Model")
        if column in frame.columns
    ]
    accuracy_columns = [
        column
        for column in ("Top1Published(%)", "Top1Measured(%)", "Top1(%)")
        if column in frame.columns
    ]
    columns = _deduplicate(
        identity_columns
        + accuracy_columns
        + [
            "MetricAccuracy(%)",
            "MetricAccuracySource",
            "TotalTime(s)",
            "AvgPower(W)",
            "PeakVRAM(MiB)",
            metric,
        ]
    )
    ranked = frame.sort_values(metric, ascending=False, kind="stable")[columns].copy()
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked


def _markdown_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without requiring optional packages."""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def _leaders(frame: pd.DataFrame, column: str) -> str:
    model_column = _first_column(frame.columns, ("Model", "ModelKey"))
    best = frame[column].max()
    names = frame.loc[frame[column] == best, model_column].astype(str).tolist()
    return ", ".join(names)


def _minimum_models(frame: pd.DataFrame, column: str) -> str:
    model_column = _first_column(frame.columns, ("Model", "ModelKey"))
    best = pd.to_numeric(frame[column]).min()
    names = frame.loc[pd.to_numeric(frame[column]) == best, model_column].astype(str)
    return ", ".join(names.tolist())


def build_report(frame: pd.DataFrame, source_path: Path) -> str:
    """Create a concise table and evidence-based intermediate conclusions."""
    model_column = _first_column(frame.columns, ("Model", "ModelKey"))
    table = frame.sort_values("NS#(1/8)", ascending=False, kind="stable")[
        [
            model_column,
            "MetricAccuracy(%)",
            "MetricAccuracySource",
            "TotalTime(s)",
            "AvgPower(W)",
            "PeakVRAM(MiB)",
            *SCORE_COLUMNS,
        ]
    ].copy()
    table.insert(0, "Rank by NS#", range(1, len(table) + 1))

    aggregate_seconds = float(pd.to_numeric(frame["TotalTime(s)"]).sum())
    measured_count = int((frame["MetricAccuracySource"] == "measured").sum())
    fallback_count = len(frame) - measured_count
    ns_e_leader = _leaders(frame, "NS-E(1/8)")
    ns_m_leader = _leaders(frame, "NS-M(1/8)")
    ns_hash_leader = _leaders(frame, "NS#(1/8)")
    fastest = _minimum_models(frame, "TotalTime(s)")
    lowest_memory = _minimum_models(frame, "PeakVRAM(MiB)")
    lowest_power = _minimum_models(frame, "AvgPower(W)")

    return "\n".join(
        [
            "# Stage-2 deployment-aware NetScore report",
            "",
            f"Source: `{source_path.name}` ({len(frame)} candidates).",
            "",
            "The scores use the paper's deployment-only 1/8 weighting, measured "
            "top-1 accuracy where valid, and published top-1 as the fallback. "
            "NS-E uses total inference time × average power; NS-M uses peak VRAM; "
            "NS# combines all three deployment costs.",
            "",
            "With accuracy A in percent, total inference time t in seconds, "
            "average power P in watts, peak VRAM r in MiB, and w = 1/8: "
            "`NS-E = 20 log10(A² / (tP)^w)`, "
            "`NS-M = 20 log10(A² / r^w)`, and "
            "`NS# = 20 log10(A² / (rtP)^w)`.",
            "",
            "## Results",
            "",
            _markdown_table(table),
            "",
            "## Intermediate conclusions",
            "",
            f"- **NS-E leader:** {ns_e_leader}; this is the strongest "
            "accuracy–inference-energy balance under the 1/8 weighting.",
            f"- **NS-M leader:** {ns_m_leader}; this is the strongest "
            "accuracy–peak-memory balance.",
            f"- **NS# leader:** {ns_hash_leader}; this is the strongest joint "
            "accuracy, time, power, and memory balance.",
            f"- Resource minima are: total time — {fastest}; average power — "
            f"{lowest_power}; peak VRAM — {lowest_memory}. A resource minimum "
            "does not necessarily imply the best composite score because accuracy "
            "remains the numerator.",
            f"- Accuracy source: {measured_count}/{len(frame)} measured and "
            f"{fallback_count}/{len(frame)} published fallback.",
            f"- Aggregate all-candidate inference time is {aggregate_seconds:.3f} s "
            f"({aggregate_seconds / 60:.2f} min, "
            f"{aggregate_seconds / 3600:.2f} h), computed as the sum of "
            "`TotalTime(s)`.",
            "- Rankings are hardware-, dataset-, and protocol-specific; compare "
            "candidates only when those conditions are held constant.",
            "",
        ]
    )


def process_csv(input_path: Path, output_dir: Path) -> dict[str, Path]:
    """Read, validate, augment, and write all post-processing artifacts."""
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {input_path}")

    device_metrics_path = output_dir / "device_metrics.csv"
    if _same_path(input_path, device_metrics_path):
        raise ValueError(
            "output device_metrics.csv would overwrite the source CSV; choose a "
            "different --output-dir"
        )

    source = pd.read_csv(input_path)
    augmented = augment_dataframe(source)
    report = build_report(augmented, input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(device_metrics_path, index=False)

    outputs: dict[str, Path] = {"device_metrics": device_metrics_path}
    for metric, filename in RANKING_FILENAMES.items():
        path = output_dir / filename
        _ranking(augmented, metric).to_csv(path, index=False)
        outputs[metric] = path

    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(report, encoding="utf-8")
    outputs["report"] = report_path
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add NS-E, NS-M, and NS# (1/8) to an existing Stage-2 metrics CSV "
            "without modifying that source file."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="source CSV")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="separate artifact directory"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = process_csv(args.input, args.output_dir)
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print("NetScore post-processing complete; source CSV was not modified.")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
