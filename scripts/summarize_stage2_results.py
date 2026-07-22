"""Create a bounded K400/SSV2 Stage 2 comparison without running inference.

The utility left-joins two completed ``device_metrics.csv`` files onto the
fixed 15-model K400-runnable roster.  Missing measurements remain blank, and
SSV2 model/checkpoint availability comes from an explicit manifest rather
than from architecture-name guesses or proxy checkpoints.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "stage2_candidate_availability.csv"
DEFAULT_COMBINED_NAME = "stage2_cross_dataset_results.csv"
DEFAULT_REPORT_NAME = "stage2_cross_dataset_report.md"
EXPECTED_CANDIDATES = 15

DATASETS = {
    "k400": "Kinetics-400",
    "ssv2": "Something-Something-v2",
}

MANIFEST_COLUMNS = {
    "candidate_order",
    "model_key",
    "model",
    "k400_runnable",
    "ssv2_pair_supported",
    "ssv2_availability_code",
    "ssv2_availability_reason",
}

COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "model_key": ("ModelKey", "model_key"),
    "model": ("Model", "model"),
    "device": ("Device", "device"),
    "dataset": ("Dataset", "dataset"),
    "split": ("Split", "split"),
    "input_protocol": ("InputProtocol", "input_protocol"),
    "clip_count": ("ClipCount", "NumClips", "Clips", "clip_count"),
    "frames_used": ("FramesUsed", "Frames", "frames_used"),
    "power_backend": ("PowerBackend", "power_backend"),
    "published_top1": (
        "Top1Published(%)",
        "PublishedTop1(%)",
        "Top1Published",
        "Top1(%)",
    ),
    "measured_top1": (
        "Top1Measured(%)",
        "MeasuredTop1(%)",
        "Top1Measured",
    ),
    "metric_accuracy": ("MetricAccuracy(%)", "AccuracyUsedForNetScore(%)"),
    "metric_accuracy_source": (
        "MetricAccuracySource",
        "AccuracySourceForNetScore",
    ),
    "latency_ms": ("Latency(ms)", "Latency(ms/clip)", "LatencyMs"),
    "average_power_w": ("AvgPower(W)", "AveragePower(W)"),
    "energy_per_clip_j": ("Energy(J)", "Energy(J/clip)"),
    "total_time_s": ("TotalTime(s)",),
    "total_energy_j": ("TotalEnergy(J)",),
    "peak_vram_mib": ("PeakVRAM(MiB)", "GPUMemory(MB)", "PeakMemory(MiB)"),
    "ns_e": ("NS-E(1/8)", "NS-E"),
    "ns_m": ("NS-M(1/8)", "NS-M"),
    "ns_hash": ("NS#(1/8)", "NS#"),
}

OUTPUT_COLUMNS = [
    "CandidateOrder",
    "ModelKey",
    "Model",
    "Dataset",
    "DatasetName",
    "DatasetAvailability",
    "AvailabilityReason",
    "ResultStatus",
    "ResultNote",
    "Device",
    "Split",
    "InputProtocol",
    "ClipCount",
    "FramesUsed",
    "PowerBackend",
    "PublishedTop1(%)",
    "PublishedAccuracyScope",
    "MeasuredTop1Local(%)",
    "MeasuredAccuracyScope",
    "AccuracyDeltaLocalMinusPublished(pp)",
    "AccuracyUsedForNetScore(%)",
    "AccuracySourceForNetScore",
    "Latency(ms/clip)",
    "AvgPower(W)",
    "Energy(J/clip)",
    "TotalTime(s)",
    "TotalEnergy(J)",
    "PeakVRAM(MiB)",
    "NS-E(1/8)",
    "NS-M(1/8)",
    "NS#(1/8)",
    "RankNS-EWithinDataset",
    "RankNS-MWithinDataset",
    "RankNS#WithinDataset",
]

NUMERIC_OUTPUTS = [
    "ClipCount",
    "FramesUsed",
    "PublishedTop1(%)",
    "MeasuredTop1Local(%)",
    "AccuracyDeltaLocalMinusPublished(pp)",
    "AccuracyUsedForNetScore(%)",
    "Latency(ms/clip)",
    "AvgPower(W)",
    "Energy(J/clip)",
    "TotalTime(s)",
    "TotalEnergy(J)",
    "PeakVRAM(MiB)",
    "NS-E(1/8)",
    "NS-M(1/8)",
    "NS#(1/8)",
]


def _boolean(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        values = sorted(normalized[invalid].unique())
        raise ValueError(f"{column} contains invalid booleans: {values}")
    return normalized.eq("true")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> pd.DataFrame:
    """Load and validate the fixed 15-model reporting roster."""
    if not path.is_file():
        raise FileNotFoundError(f"availability manifest does not exist: {path}")
    frame = pd.read_csv(path)
    missing = MANIFEST_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"availability manifest is missing: {sorted(missing)}")
    if len(frame) != EXPECTED_CANDIDATES:
        raise ValueError(
            f"availability manifest must contain {EXPECTED_CANDIDATES} rows, "
            f"found {len(frame)}"
        )
    frame["candidate_order"] = pd.to_numeric(
        frame["candidate_order"], errors="raise"
    ).astype(int)
    if sorted(frame["candidate_order"].tolist()) != list(
        range(1, EXPECTED_CANDIDATES + 1)
    ):
        raise ValueError("candidate_order must contain each integer from 1 to 15")
    for column in ("model_key", "model"):
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any() or frame[column].duplicated().any():
            raise ValueError(f"manifest {column} values must be non-empty and unique")
    frame["model_key"] = frame["model_key"].str.lower()
    frame["k400_runnable"] = _boolean(frame["k400_runnable"], "k400_runnable")
    frame["ssv2_pair_supported"] = _boolean(
        frame["ssv2_pair_supported"], "ssv2_pair_supported"
    )
    if not bool(frame["k400_runnable"].all()):
        raise ValueError("every manifest row must be in the K400-runnable roster")
    if frame["ssv2_availability_code"].isna().any() or frame[
        "ssv2_availability_reason"
    ].isna().any():
        raise ValueError("every candidate needs an explicit SSV2 availability reason")
    for row in frame.itertuples(index=False):
        supported_code = row.ssv2_availability_code == "official_pair_supported"
        if bool(row.ssv2_pair_supported) != supported_code:
            raise ValueError(
                f"SSV2 availability code/boolean disagree for {row.model_key}"
            )
        if not str(row.ssv2_availability_reason).strip():
            raise ValueError(f"empty SSV2 reason for {row.model_key}")
    return frame.sort_values("candidate_order", kind="stable").reset_index(drop=True)


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    casefolded = {str(column).casefold(): str(column) for column in frame.columns}
    resolved: dict[str, str | None] = {}
    for field, aliases in COLUMN_ALIASES.items():
        match = next(
            (
                alias
                if alias in frame.columns
                else casefolded.get(alias.casefold())
                for alias in aliases
                if alias in frame.columns or alias.casefold() in casefolded
            ),
            None,
        )
        resolved[field] = match
    if resolved["model_key"] is None:
        raise ValueError(
            "metrics CSV needs ModelKey (or model_key) for bounded roster matching"
        )
    return resolved


def _dataset_matches(value: object, expected: str) -> bool:
    if pd.isna(value) or not str(value).strip():
        return True
    normalized = "".join(character for character in str(value).lower() if character.isalnum())
    accepted = {
        "k400": {"k400", "kinetics400"},
        "ssv2": {"ssv2", "sth2", "sthv2", "somethingsomethingv2"},
    }
    return normalized in accepted[expected]


def _optional_text(value: object) -> object:
    if pd.isna(value) or not str(value).strip():
        return pd.NA
    return str(value).strip()


def _optional_number(
    value: object,
    *,
    field: str,
    row_name: str,
    positive: bool = False,
    percentage: bool = False,
    integer: bool = False,
) -> float | int:
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric for {row_name}: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite for {row_name}: {value!r}")
    if positive and number <= 0:
        raise ValueError(f"{field} must be positive for {row_name}: {number}")
    if percentage and not 0 <= number <= 100:
        raise ValueError(f"{field} must be between 0 and 100 for {row_name}: {number}")
    if integer:
        if number <= 0 or not number.is_integer():
            raise ValueError(f"{field} must be a positive integer for {row_name}")
        return int(number)
    return number


def _source_value(row: pd.Series, column: str | None) -> object:
    return pd.NA if column is None else row[column]


def normalize_metrics(
    path: Path,
    dataset: str,
    manifest: pd.DataFrame,
    *,
    clip_count_override: int | None = None,
) -> dict[str, dict[str, object]]:
    """Normalize supplied rows without filling absent metrics."""
    if not path.is_file():
        raise FileNotFoundError(f"{dataset} metrics CSV does not exist: {path}")
    source = pd.read_csv(path)
    if source.empty:
        raise ValueError(f"{dataset} metrics CSV has no rows: {path}")
    columns = _resolve_columns(source)
    allowed = set(manifest["model_key"])
    normalized: dict[str, dict[str, object]] = {}

    if clip_count_override is not None and clip_count_override <= 0:
        raise ValueError(f"{dataset} clip-count override must be positive")

    for index, source_row in source.iterrows():
        key = str(source_row[columns["model_key"]]).strip().lower()
        row_name = f"{dataset} row {index} ({key or 'empty key'})"
        if not key:
            raise ValueError(f"empty ModelKey in {dataset} row {index}")
        if key not in allowed:
            raise ValueError(f"{row_name} is outside the fixed 15-model roster")
        if key in normalized:
            raise ValueError(f"duplicate ModelKey in {dataset} CSV: {key}")
        if columns["dataset"] is not None and not _dataset_matches(
            source_row[columns["dataset"]], dataset
        ):
            raise ValueError(
                f"dataset column disagrees with --{dataset} input for {row_name}"
            )

        clip_count = _optional_number(
            _source_value(source_row, columns["clip_count"]),
            field="ClipCount",
            row_name=row_name,
            integer=True,
        )
        if clip_count_override is not None:
            if not pd.isna(clip_count) and int(clip_count) != clip_count_override:
                raise ValueError(
                    f"clip-count override {clip_count_override} conflicts with "
                    f"source value {clip_count} for {row_name}"
                )
            clip_count = clip_count_override

        published = _optional_number(
            _source_value(source_row, columns["published_top1"]),
            field="Top1Published(%)",
            row_name=row_name,
            percentage=True,
        )
        measured = _optional_number(
            _source_value(source_row, columns["measured_top1"]),
            field="Top1Measured(%)",
            row_name=row_name,
            percentage=True,
        )
        metric_accuracy = _optional_number(
            _source_value(source_row, columns["metric_accuracy"]),
            field="MetricAccuracy(%)",
            row_name=row_name,
            percentage=True,
        )
        metric_source = _optional_text(
            _source_value(source_row, columns["metric_accuracy_source"])
        )
        if not pd.isna(metric_source):
            metric_source = str(metric_source).lower()
            if metric_source not in {"measured", "published"}:
                raise ValueError(
                    f"MetricAccuracySource must be measured or published for {row_name}"
                )
            if metric_source == "measured" and pd.isna(measured):
                raise ValueError(
                    f"MetricAccuracySource is measured but measured accuracy is blank for {row_name}"
                )
            if metric_source == "published" and pd.isna(published):
                raise ValueError(
                    f"MetricAccuracySource is published but published accuracy is blank for {row_name}"
                )
            expected_accuracy = measured if metric_source == "measured" else published
            if not pd.isna(metric_accuracy) and not math.isclose(
                metric_accuracy, expected_accuracy, rel_tol=0.0, abs_tol=0.011
            ):
                raise ValueError(
                    f"MetricAccuracy does not match its {metric_source} source for {row_name}"
                )

        input_protocol = _optional_text(
            _source_value(source_row, columns["input_protocol"])
        )
        if pd.isna(measured):
            measured_scope: object = pd.NA
        elif pd.isna(clip_count):
            measured_scope = (
                "Local device evaluation; clip count is not encoded in the source CSV. "
                "This is not canonical published accuracy."
            )
        else:
            view = (
                str(input_protocol)
                if not pd.isna(input_protocol)
                else "input-view protocol not encoded"
            )
            measured_scope = (
                f"Local {view} evaluation on {int(clip_count):,} clips; "
                "not canonical published accuracy."
            )

        normalized[key] = {
            "Model": _optional_text(_source_value(source_row, columns["model"])),
            "Device": _optional_text(_source_value(source_row, columns["device"])),
            "Split": _optional_text(_source_value(source_row, columns["split"])),
            "InputProtocol": input_protocol,
            "ClipCount": clip_count,
            "FramesUsed": _optional_number(
                _source_value(source_row, columns["frames_used"]),
                field="FramesUsed",
                row_name=row_name,
                integer=True,
            ),
            "PowerBackend": _optional_text(
                _source_value(source_row, columns["power_backend"])
            ),
            "PublishedTop1(%)": published,
            "PublishedAccuracyScope": (
                "Canonical paper/model-zoo accuracy with its paper-specific evaluation protocol."
                if not pd.isna(published)
                else pd.NA
            ),
            "MeasuredTop1Local(%)": measured,
            "MeasuredAccuracyScope": measured_scope,
            "AccuracyDeltaLocalMinusPublished(pp)": (
                measured - published
                if not pd.isna(measured) and not pd.isna(published)
                else float("nan")
            ),
            "AccuracyUsedForNetScore(%)": metric_accuracy,
            "AccuracySourceForNetScore": metric_source,
            "Latency(ms/clip)": _optional_number(
                _source_value(source_row, columns["latency_ms"]),
                field="Latency(ms)",
                row_name=row_name,
                positive=True,
            ),
            "AvgPower(W)": _optional_number(
                _source_value(source_row, columns["average_power_w"]),
                field="AvgPower(W)",
                row_name=row_name,
                positive=True,
            ),
            "Energy(J/clip)": _optional_number(
                _source_value(source_row, columns["energy_per_clip_j"]),
                field="Energy(J/clip)",
                row_name=row_name,
                positive=True,
            ),
            "TotalTime(s)": _optional_number(
                _source_value(source_row, columns["total_time_s"]),
                field="TotalTime(s)",
                row_name=row_name,
                positive=True,
            ),
            "TotalEnergy(J)": _optional_number(
                _source_value(source_row, columns["total_energy_j"]),
                field="TotalEnergy(J)",
                row_name=row_name,
                positive=True,
            ),
            "PeakVRAM(MiB)": _optional_number(
                _source_value(source_row, columns["peak_vram_mib"]),
                field="PeakVRAM(MiB)",
                row_name=row_name,
                positive=True,
            ),
            "NS-E(1/8)": _optional_number(
                _source_value(source_row, columns["ns_e"]),
                field="NS-E(1/8)",
                row_name=row_name,
            ),
            "NS-M(1/8)": _optional_number(
                _source_value(source_row, columns["ns_m"]),
                field="NS-M(1/8)",
                row_name=row_name,
            ),
            "NS#(1/8)": _optional_number(
                _source_value(source_row, columns["ns_hash"]),
                field="NS#(1/8)",
                row_name=row_name,
            ),
        }
    return normalized


COMPLETE_FIELDS = (
    "PublishedTop1(%)",
    "MeasuredTop1Local(%)",
    "AccuracyUsedForNetScore(%)",
    "AccuracySourceForNetScore",
    "InputProtocol",
    "ClipCount",
    "Latency(ms/clip)",
    "AvgPower(W)",
    "Energy(J/clip)",
    "TotalTime(s)",
    "TotalEnergy(J)",
    "PeakVRAM(MiB)",
    "NS-E(1/8)",
    "NS-M(1/8)",
    "NS#(1/8)",
)


def _blank_metrics() -> dict[str, object]:
    row: dict[str, object] = {column: float("nan") for column in NUMERIC_OUTPUTS}
    for column in (
        "Device",
        "Split",
        "InputProtocol",
        "PowerBackend",
        "PublishedAccuracyScope",
        "MeasuredAccuracyScope",
        "AccuracySourceForNetScore",
    ):
        row[column] = pd.NA
    return row


def _missing_fields(metrics: Mapping[str, object]) -> list[str]:
    return [field for field in COMPLETE_FIELDS if pd.isna(metrics.get(field, pd.NA))]


def build_combined(
    manifest: pd.DataFrame,
    k400_metrics: Mapping[str, Mapping[str, object]],
    ssv2_metrics: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Build the fixed 30-row dataset/candidate matrix and within-dataset ranks."""
    supplied = {"k400": k400_metrics, "ssv2": ssv2_metrics}
    rows: list[dict[str, object]] = []
    for dataset in ("k400", "ssv2"):
        for candidate in manifest.itertuples(index=False):
            if dataset == "k400":
                availability = "runnable"
                availability_reason = (
                    "Candidate belongs to the exact 15-model K400-runnable roster."
                )
                is_available = True
            else:
                availability = str(candidate.ssv2_availability_code)
                availability_reason = str(candidate.ssv2_availability_reason)
                is_available = bool(candidate.ssv2_pair_supported)

            metrics = supplied[dataset].get(candidate.model_key)
            if metrics is not None and not is_available:
                raise ValueError(
                    f"{candidate.model_key} has an SSV2 metrics row but the manifest "
                    "has no exact supervised pair; refusing a possible proxy result"
                )

            base: dict[str, object] = {
                "CandidateOrder": int(candidate.candidate_order),
                "ModelKey": candidate.model_key,
                "Model": candidate.model,
                "Dataset": dataset,
                "DatasetName": DATASETS[dataset],
                "DatasetAvailability": availability,
                "AvailabilityReason": availability_reason,
            }
            if metrics is None:
                base.update(_blank_metrics())
                if is_available:
                    base["ResultStatus"] = "available_not_measured"
                    base["ResultNote"] = (
                        f"No {dataset.upper()} metrics row was supplied; numeric fields "
                        "remain blank."
                    )
                else:
                    base["ResultStatus"] = "not_available"
                    base["ResultNote"] = (
                        "No exact supervised model/dataset pair; numeric fields remain blank."
                    )
            else:
                base.update(metrics)
                # Always retain the canonical manifest display name.  The source
                # model name can differ only cosmetically (e.g. an IN21K suffix).
                base["Model"] = candidate.model
                missing = _missing_fields(metrics)
                if missing:
                    base["ResultStatus"] = "partial_metrics"
                    base["ResultNote"] = (
                        "Supplied row is incomplete; no values were inferred for: "
                        + ", ".join(missing)
                    )
                else:
                    base["ResultStatus"] = "measured_complete"
                    base["ResultNote"] = "Complete supplied device-metrics row."
            rows.append(base)

    combined = pd.DataFrame(rows)
    for column in NUMERIC_OUTPUTS:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    rank_pairs = {
        "NS-E(1/8)": "RankNS-EWithinDataset",
        "NS-M(1/8)": "RankNS-MWithinDataset",
        "NS#(1/8)": "RankNS#WithinDataset",
    }
    for rank_column in rank_pairs.values():
        combined[rank_column] = pd.Series(pd.NA, index=combined.index, dtype="Int64")
    for dataset in DATASETS:
        dataset_mask = combined["Dataset"].eq(dataset)
        for score_column, rank_column in rank_pairs.items():
            scores = combined.loc[dataset_mask, score_column]
            valid = scores.notna()
            ranks = scores[valid].rank(method="min", ascending=False).astype("Int64")
            combined.loc[ranks.index, rank_column] = ranks
    return combined[OUTPUT_COLUMNS]


def _markdown_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def _leader_rows(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        subset = combined[combined["Dataset"].eq(dataset)]
        for score in ("NS-E(1/8)", "NS-M(1/8)", "NS#(1/8)"):
            ranked = subset.dropna(subset=[score]).sort_values(
                score, ascending=False, kind="stable"
            )
            if ranked.empty:
                rows.append([DATASETS[dataset], score, pd.NA, np.nan, 0])
            else:
                best = ranked.iloc[0]
                rows.append(
                    [DATASETS[dataset], score, best["Model"], best[score], len(ranked)]
                )
    return pd.DataFrame(
        rows, columns=["Dataset", "Metric", "Leader", "Score", "Rows ranked"]
    )


def _extreme(combined: pd.DataFrame, dataset: str, column: str, maximize: bool) -> str:
    subset = combined[combined["Dataset"].eq(dataset)].dropna(subset=[column])
    if subset.empty:
        return "not available"
    target = subset[column].max() if maximize else subset[column].min()
    names = subset.loc[subset[column].eq(target), "Model"].tolist()
    return f"{', '.join(names)} ({target:.2f})"


def build_report(combined: pd.DataFrame, manifest_path: Path) -> str:
    """Create the cross-dataset Markdown analysis without cross-dataset ranking."""
    availability = []
    for order in range(1, EXPECTED_CANDIDATES + 1):
        k400 = combined[
            combined["CandidateOrder"].eq(order) & combined["Dataset"].eq("k400")
        ].iloc[0]
        ssv2 = combined[
            combined["CandidateOrder"].eq(order) & combined["Dataset"].eq("ssv2")
        ].iloc[0]
        availability.append(
            [
                order,
                k400["Model"],
                k400["ResultStatus"],
                ssv2["DatasetAvailability"],
                ssv2["ResultStatus"],
                ssv2["AvailabilityReason"],
            ]
        )
    availability_table = pd.DataFrame(
        availability,
        columns=[
            "#",
            "Model",
            "K400 result",
            "SSV2 availability",
            "SSV2 result",
            "SSV2 reason",
        ],
    )

    result_columns = [
        "Model",
        "ResultStatus",
        "PublishedTop1(%)",
        "MeasuredTop1Local(%)",
        "AccuracyDeltaLocalMinusPublished(pp)",
        "Latency(ms/clip)",
        "AvgPower(W)",
        "Energy(J/clip)",
        "PeakVRAM(MiB)",
        "NS-E(1/8)",
        "RankNS-EWithinDataset",
        "NS-M(1/8)",
        "RankNS-MWithinDataset",
        "NS#(1/8)",
        "RankNS#WithinDataset",
    ]
    parts = [
        "# Stage 2 cross-dataset device report",
        "",
        f"Availability manifest: `{manifest_path.name}`.",
        "",
        "> **Accuracy provenance:** `PublishedTop1(%)` is the canonical "
        "paper/model-zoo value under its paper-specific evaluation protocol. "
        "`MeasuredTop1Local(%)` is the local single-view device result on the "
        "clip count recorded in the source CSV. They are deliberately separate.",
        "",
        "> **Missing-data rule:** an em dash means the value was not supplied or "
        "is not applicable. This report never copies K400 values into SSV2 rows, "
        "never substitutes a proxy checkpoint, and never ranks missing values.",
        "",
        "> **Ranking scope:** every rank is within one dataset and only among "
        "rows carrying that score. K400 and SSV2 scores are not ranked against "
        "each other.",
        "",
        "## Candidate and SSV2 availability matrix",
        "",
        _markdown_table(availability_table),
        "",
        "## NetScore leaders",
        "",
        _markdown_table(_leader_rows(combined)),
    ]

    for dataset in ("k400", "ssv2"):
        subset = combined[combined["Dataset"].eq(dataset)].copy()
        complete = int(subset["ResultStatus"].eq("measured_complete").sum())
        partial = int(subset["ResultStatus"].eq("partial_metrics").sum())
        not_measured = int(
            subset["ResultStatus"].isin(["available_not_measured"]).sum()
        )
        unavailable = int(subset["ResultStatus"].eq("not_available").sum())
        clip_counts = sorted(
            {int(value) for value in subset["ClipCount"].dropna().tolist()}
        )
        clip_text = ", ".join(f"{value:,}" for value in clip_counts) or "not supplied"
        parts.extend(
            [
                "",
                f"## {DATASETS[dataset]} results",
                "",
                f"Coverage: {complete} complete, {partial} partial, "
                f"{not_measured} available but not measured, and {unavailable} "
                f"not available. Supplied clip counts: {clip_text}.",
                "",
                _markdown_table(subset[result_columns]),
                "",
                "### Data-bounded observations",
                "",
                f"- Highest local measured accuracy: "
                f"{_extreme(combined, dataset, 'MeasuredTop1Local(%)', True)}.",
                f"- Lowest supplied latency: "
                f"{_extreme(combined, dataset, 'Latency(ms/clip)', False)} ms/clip.",
                f"- Lowest supplied energy: "
                f"{_extreme(combined, dataset, 'Energy(J/clip)', False)} J/clip.",
                f"- Lowest supplied peak VRAM: "
                f"{_extreme(combined, dataset, 'PeakVRAM(MiB)', False)} MiB.",
                "- These observations describe only supplied local rows; absent "
                "candidates are not treated as zero and cannot win a ranking.",
            ]
        )

    missing_ssv2 = combined[
        combined["Dataset"].eq("ssv2")
        & ~combined["ResultStatus"].eq("measured_complete")
    ][
        [
            "Model",
            "DatasetAvailability",
            "ResultStatus",
            "AvailabilityReason",
            "ResultNote",
        ]
    ]
    parts.extend(
        [
            "",
            "## SSV2 missing or incomplete rows",
            "",
            _markdown_table(missing_ssv2),
            "",
        ]
    )
    return "\n".join(parts)


def summarize(
    k400_path: Path,
    ssv2_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    k400_clip_count: int | None = None,
    ssv2_clip_count: int | None = None,
) -> tuple[pd.DataFrame, str]:
    manifest = load_manifest(manifest_path)
    k400 = normalize_metrics(
        k400_path,
        "k400",
        manifest,
        clip_count_override=k400_clip_count,
    )
    ssv2 = normalize_metrics(
        ssv2_path,
        "ssv2",
        manifest,
        clip_count_override=ssv2_clip_count,
    )
    combined = build_combined(manifest, k400, ssv2)
    report = build_report(combined, manifest_path)
    return combined, report


def write_outputs(
    combined: pd.DataFrame,
    report: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / DEFAULT_COMBINED_NAME
    markdown_path = output_dir / DEFAULT_REPORT_NAME
    combined.to_csv(csv_path, index=False)
    markdown_path.write_text(report, encoding="utf-8")
    return csv_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k400", required=True, type=Path, help="K400 device_metrics CSV")
    parser.add_argument("--ssv2", required=True, type=Path, help="SSV2 device_metrics CSV")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--k400-clip-count",
        type=int,
        help="explicit clip count only when the K400 source CSV omits it",
    )
    parser.add_argument(
        "--ssv2-clip-count",
        type=int,
        help="explicit clip count only when the SSV2 source CSV omits it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        combined, report = summarize(
            args.k400,
            args.ssv2,
            args.manifest,
            k400_clip_count=args.k400_clip_count,
            ssv2_clip_count=args.ssv2_clip_count,
        )
        outputs = write_outputs(combined, report, args.output_dir)
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"Combined {len(combined)} dataset/candidate rows without running inference."
    )
    for output in outputs:
        print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
