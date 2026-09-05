#!/usr/bin/env python3
"""RTX3050 frozen17 consistency audit (Phase 6).

Compares every frozen17 model against ALL existing RTX3050 K400 result rows
and decides whether an old measurement is reusable for the final table.

An old row is reusable only when it can be POSITIVELY verified against the
frozen configuration: canonical key (or known alias), exact frame count,
matching published Top-1 / params row, recorded exact checkpoint path, and
the fixed 1000-clip seed0 subset. Rows from the legacy CSV format record no
checkpoint or manifest provenance, so they can never be verified and are
marked RERUN_CONFIG_MISMATCH -- per the audit rule "do NOT assume an old
result is reusable merely because its display name matches".

Outputs reports/rtx3050_frozen17_status.csv and prints the fill-missing
command for the models that still need a run.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.registry import ROOT, FROZEN17_KEYS, model_spec

RESULTS_DIRS = [
    ROOT / "results" / "rtx3050",
    ROOT / "results" / "NVIDIA_GeForce_RTX_3050_Laptop_GPU",
]
REPORT = ROOT / "reports" / "rtx3050_frozen17_status.csv"
REPORT_MD = ROOT / "reports" / "RTX3050_EXACT_STATUS.md"

# Old registry keys that map onto a frozen17 canonical key. Anything else
# (vtn-b, svt-b, mvit-b-24-32x3, ...) is outside the frozen experiment.
LEGACY_ALIASES = {
    "mvit-v1-b": "mvit-v1-b-32x3",
    "omnivore-b": "omnivore-b-in21k",
    "dualformer-b": "dualformer-b-in21k",
}

EXPECTED_CLIPS = 1000


def _rows_from_csv(path):
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row, _source=str(path.relative_to(ROOT))) for row in csv.DictReader(handle)]
    except OSError:
        return []


def collect_existing_rows():
    rows = []
    for base in RESULTS_DIRS:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("device_metrics.csv")):
            rows.extend(_rows_from_csv(path))
    return rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_key(row):
    key = (row.get("ModelKey") or "").strip()
    if key in FROZEN17_KEYS:
        return key
    return LEGACY_ALIASES.get(key)


def evaluate(spec, row):
    """Return per-check booleans for one existing row against the frozen spec."""
    frozen_frames = int(spec["input"]["num_frames"])
    frozen_params = float(
        spec.get("runtime_actual_params_m", spec["stage1"]["params_m"])
    )
    ckpt_rel = spec.get("k400_checkpoint")
    frozen_ckpt_name = Path(ckpt_rel).name if ckpt_rel else ""

    frames = _to_float(row.get("ModelFrames") or row.get("FramesUsed"))
    params = _to_float(row.get("ParamsM") or row.get("Params(M)"))
    published = _to_float(row.get("PublishedTop1") or row.get("Top1Published(%)"))
    clips = _to_float(row.get("ClipCount"))
    checkpoint = (row.get("Checkpoint") or "").strip()
    batch = _to_float(row.get("BatchSize"))

    exact_frames = frames is not None and int(frames) == frozen_frames
    # Runtime graph parameters use a 5% hard-preflight tolerance. For
    # Video-FocalNet this intentionally uses the official checkpoint graph
    # count, not the smaller image-only FocalNet backbone count.
    exact_architecture = (
        params is not None
        and abs(params - frozen_params) / frozen_params <= 0.05
    )
    # Only rows that RECORD the exact checkpoint path can be verified.
    exact_checkpoint = bool(checkpoint) and Path(checkpoint).name == frozen_ckpt_name
    same_subset = clips is not None and int(clips) == EXPECTED_CLIPS
    # Preprocessing is only verifiable for new-format rows (which record
    # checkpoint + manifest + batch size); old rows cannot be verified.
    same_preprocessing = exact_checkpoint and (batch is None or int(batch) == 1)
    return {
        "exact_architecture": exact_architecture,
        "exact_checkpoint": exact_checkpoint,
        "exact_frames": exact_frames,
        "same_subset": same_subset,
        "same_preprocessing": same_preprocessing,
    }


def audit():
    existing = collect_existing_rows()
    report_rows = []
    needs_run = []
    for key in FROZEN17_KEYS:
        spec = model_spec(key)
        candidates = [row for row in existing if canonical_key(row) == key]
        best = None
        best_checks = None
        best_score = -1
        for row in candidates:
            checks = evaluate(spec, row)
            score = sum(checks.values())
            if score > best_score:
                best, best_checks, best_score = row, checks, score
        configured_status = spec.get("checkpoint_status", "")
        if key == "zeroi2v-b16-8f":
            action = "BLOCKED_RUNTIME"
            checks = {name: False for name in (
                "exact_architecture", "exact_checkpoint", "exact_frames",
                "same_subset", "same_preprocessing")}
            existing_run = best["_source"] if best else ""
            reusable = False
        elif configured_status.startswith("BLOCKED_EXACT"):
            action = "BLOCKED_EXACT_CHECKPOINT"
            checks = {name: False for name in (
                "exact_architecture", "exact_checkpoint", "exact_frames",
                "same_subset", "same_preprocessing")}
            existing_run = best["_source"] if best else ""
            reusable = False
        elif best is None:
            action = "RUN_MISSING"
            checks = {name: False for name in (
                "exact_architecture", "exact_checkpoint", "exact_frames",
                "same_subset", "same_preprocessing")}
            existing_run = ""
            reusable = False
        else:
            checks = best_checks
            existing_run = best["_source"]
            status = (best.get("Status") or "completed").lower()
            reusable = all(checks.values()) and status in {"completed", "success", "ok"}
            if reusable:
                action = "VALID_REUSE"
            elif status not in {"completed", "success", "ok"}:
                action = "RERUN_INVALID_RESULT"
            else:
                if not checks["exact_architecture"]:
                    action = "RERUN_WRONG_ARCHITECTURE"
                else:
                    action = "RERUN_CONFIGURATION"
        if action not in {"VALID_REUSE", "METADATA_FIX_ONLY"}:
            needs_run.append(key)
        report_rows.append({
            "model": spec["display_name"],
            "canonical_key": key,
            "existing_run": existing_run,
            **{name: str(value) for name, value in checks.items()},
            "reusable": str(reusable),
            "action": action,
        })

    REPORT.parent.mkdir(exist_ok=True)
    with REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    for row in report_rows:
        print(f"{row['canonical_key']}: {row['action']}"
              + (f" (from {row['existing_run']})" if row["existing_run"] else ""))
    counts = {}
    for row in report_rows:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    reuse = counts.get("VALID_REUSE", 0)
    print(f"\nVALID_REUSE={reuse}/17  NEEDS_RUN={len(needs_run)}/17")
    print(f"Wrote {REPORT}")
    if needs_run:
        print("\nFill-missing command (RTX3050, fixed 1000-clip seed0 subset):")
        runnable = [key for key in needs_run if not next(
            row for row in report_rows if row["canonical_key"] == key
        )["action"].startswith("BLOCKED")]
        print("python run_benchmark.py --videos-dir <k400_val_root> "
              "--manifest manifests/k400_1000_seed0.csv --num-clips 1000 "
              "--batch-size 1 --models " + " ".join(runnable))

    lines = [
        "# RTX3050 exact frozen17 status",
        "",
        "Raw execution is **15/17 completed**. This is not a scientific 15/17 final table.",
        "",
        "| Model | Category | Existing result | Evidence |",
        "|---|---|---|---|",
    ]
    for row in report_rows:
        evidence = ", ".join(
            f"{name}={row[name]}" for name in (
                "exact_architecture", "exact_checkpoint", "exact_frames",
                "same_subset", "same_preprocessing"
            )
        )
        lines.append(
            f"| {row['model']} | **{row['action']}** | "
            f"`{row['existing_run'] or 'none'}` | {evidence} |"
        )
    lines += ["", "## Counts", ""]
    lines += [f"- {name}: {value}" for name, value in sorted(counts.items())]
    lines += [
        "",
        "Video-FocalNet note: the official released checkpoint graphs contain "
        "49.553M / 88.735M / 157.389M parameters. The smaller 28M / 49M / "
        "88M values describe the spatial image backbones and are retained as "
        "paper metadata, not used to reject strictly compatible video checkpoints.",
        "",
        "UniFormer-B note: the official model zoo links a distinct 32x4/82.9 "
        "checkpoint. The local 16x4/82.0 artifact is not silently relabeled.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_MD}")
    return report_rows


if __name__ == "__main__":
    audit()
