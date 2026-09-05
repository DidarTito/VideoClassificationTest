#!/usr/bin/env python3
"""Generate reports/PROGRESS_FOR_PROFESSOR.md from the actual artifacts.

Every number is read from a produced report/result file; nothing is invented.
Sections for which no artifact exists yet say so explicitly.
"""

import csv
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.registry import ROOT, FROZEN17_KEYS

OUT = ROOT / "reports" / "PROGRESS_FOR_PROFESSOR.md"


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_float(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def best_row(rows, column):
    scored = [(float(r[column]), r) for r in rows if r.get(column) not in (None, "", "nan")]
    return max(scored, key=lambda item: item[0]) if scored else None


def recent_work():
    try:
        out = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except Exception:
        return []
    return [line.strip() for line in out if line.strip()]


def main():
    lines = [
        "# Project progress -- frozen17 energy-efficiency study",
        "",
        f"_Auto-generated {date.today().isoformat()} by `scripts/generate_progress_report.py`; "
        "all numbers come from artifacts in `reports/` and `results/`._",
        "",
    ]

    # ---------------- FROZEN17 STATUS ----------------
    audit = read_csv(ROOT / "reports" / "frozen17_checkpoint_audit.csv")
    lines += ["## Frozen17 checkpoint status", ""]
    if audit:
        ready = [r["model"] for r in audit if r["status"] == "READY_EXACT"]
        verify = [r["model"] for r in audit if r["status"] == "READY_NEEDS_CONFIG_VALIDATION"]
        blocked = [r["model"] for r in audit if r["status"].startswith(("BLOCKED", "MISSING"))]
        lines += [
            f"- 17 models total (frozen list; thresholds never re-run)",
            f"- {len(ready)}/17 exact K400 checkpoints ready: {', '.join(ready)}",
            f"- {len(verify)} need backend config validation: {', '.join(verify) or 'none'}",
            f"- {len(blocked)} blocked/missing: {', '.join(blocked) or 'none'}",
        ]
    else:
        lines.append("- checkpoint audit not yet run (`python scripts/audit_frozen17.py`)")
    lines.append("")

    # ---------------- RTX3050 ----------------
    status = read_csv(ROOT / "reports" / "rtx3050_frozen17_status.csv")
    final17 = read_csv(ROOT / "results" / "k400_rtx3050_frozen17" / "final17.csv")
    lines += ["## RTX3050 (Experiment A)", ""]
    if status:
        reuse = [r["canonical_key"] for r in status if r["action"] in {"VALID_REUSE", "METADATA_FIX_ONLY"}]
        rerun = [r["canonical_key"] for r in status if r["action"].startswith("RERUN")]
        missing = [r["canonical_key"] for r in status if r["action"] == "RUN_MISSING"]
        blocked_3050 = [r["canonical_key"] for r in status if r["action"].startswith("BLOCKED")]
        lines += [
            f"- verified reusable measurements: {len(reuse)}/17"
            + (f" ({', '.join(reuse)})" if reuse else ""),
            f"- need rerun (config/provenance mismatch): {len(rerun)}"
            + (f" ({', '.join(rerun)})" if rerun else ""),
            f"- never measured: {len(missing)}" + (f" ({', '.join(missing)})" if missing else ""),
            f"- blocked: {len(blocked_3050)}" + (f" ({', '.join(blocked_3050)})" if blocked_3050 else ""),
        ]
    else:
        lines.append("- consistency audit not yet run (`python scripts/audit_rtx3050.py`)")
    if final17:
        lines.append(f"- final table: {len(final17)}/17 rows in `results/k400_rtx3050_frozen17/final17.csv`")
        for metric in ("NS-E", "NS-M", "NS#"):
            top = best_row(final17, metric)
            if top:
                lines.append(f"- best {metric}: {top[1].get('Model', top[1].get('canonical_key'))} ({fmt_float(top[0])})")
    else:
        lines.append("- final 17-row table not built yet (needs verified measurements for all 17)")
    lines.append("")

    # ---------------- RTX5070 ----------------
    lines += ["## RTX5070", ""]
    if audit:
        ready_keys = [r["model"] for r in audit if r["status"] == "READY_EXACT"]
        lines.append(
            f"- tomorrow READY_EXACT benchmark set: {len(ready_keys)}/17 "
            f"({', '.join(ready_keys)})"
        )
    preflight_md = ROOT / "reports" / "RTX5070_PREFLIGHT.md"
    if preflight_md.is_file():
        match = re.search(r"model preflight: (\d+)/(\d+) OK", preflight_md.read_text(encoding="utf-8"))
        if match:
            stale = preflight_md.stat().st_mtime < (ROOT / "configs" / "frozen17.yaml").stat().st_mtime
            label = "previous/stale load preflight" if stale else "current load preflight"
            lines.append(f"- {label}: {match.group(1)}/{match.group(2)} models OK "
                         "(see `reports/RTX5070_PREFLIGHT.md` for its host and date)")
    else:
        lines.append("- preflight not yet run (`python scripts/preflight_rtx5070.py`)")
    bench = read_csv(ROOT / "results" / "k400_rtx5070_frozen17" / "device_metrics.csv")
    if bench:
        done = [r for r in bench if (r.get("Status") or "").lower() in {"completed", "success", "ok"}]
        lines.append(f"- K400 benchmark: {len(done)}/17 models completed")
    else:
        lines.append("- K400 benchmark not yet run (step 10 of `run_project_steps.sh`)")
    lines.append("")

    # ---------------- SSV2 FINE-TUNING ----------------
    lines += ["## SSV2 fine-tuning (Experiment B)", ""]
    ssv2_audit = ROOT / "reports" / "SSV2_DATASET_AUDIT.md"
    lines.append(
        "- dataset validation: "
        + ("report exists (`reports/SSV2_DATASET_AUDIT.md`)" if ssv2_audit.is_file()
           else "not yet run (`python scripts/validate_ssv2.py`)")
    )
    ft = read_csv(ROOT / "reports" / "FINETUNE_PREFLIGHT.csv")
    if ft:
        ok = [r["model"] for r in ft if r.get("status") in {"PASS", "OK"}]
        lines.append(f"- fine-tune preflight: {len(ok)}/{len(ft)} models pass "
                     f"({', '.join(ok) or 'none'})")
    else:
        lines.append("- fine-tune preflight not yet run (`python scripts/preflight_finetune.py`)")
    runs_root = ROOT / "runs" / "ssv2"
    trained = []
    if runs_root.is_dir():
        for model_dir in sorted(runs_root.iterdir()):
            metas = sorted(model_dir.glob("*/best_metadata.json"))
            if metas:
                meta = json.loads(metas[-1].read_text(encoding="utf-8"))
                trained.append((model_dir.name, meta))
    if trained:
        lines.append(f"- completed trainings: {len(trained)}")
        for name, meta in trained:
            top1 = meta.get("val_top1", meta.get("top1", "n/a"))
            lines.append(f"  - {name}: best val Top-1 {top1} (`runs/ssv2/{name}/.../best.pth`)")
    else:
        lines.append("- completed trainings: 0 (no `runs/ssv2/<model>/<ts>/best_metadata.json` yet)")
    lines.append("")

    # ---------------- RECENT WORK / NEXT ----------------
    changes = recent_work()
    lines += ["## Recent concrete work", ""]
    lines.append(f"- {len(changes)} files added/modified in the working tree (uncommitted, see `git status`)")
    lines += [
        "- frozen17 registry + `configs/frozen17.yaml` as single source of truth",
        "- exact MViTv1 16x4 and MViTv1 32x3 integrations",
        "- exact DualFormer-S integration",
        "- checkpoint audit, SSV2 quarantine, safe strict checkpoint loading",
        "- corrected official VideoMAE-B 1600-epoch K400 classifier",
        "- explicit DualFormer-B IN21K and UniFormer-B 32x4 blockers",
        "- RTX3050 consistency audit and 17-row final-table builder",
        "- `run_project_steps.sh` master runner (tomorrow / rtx3050 modes)",
        "",
        "## Next 3 actions",
        "",
        "1. On the RTX5070 box: `bash run_project_steps.sh tomorrow` (env check, audits, "
        "17-model preflight, K400 benchmark, two fine-tune smoke tests).",
        "2. Resolve remaining blockers (UniFormer-B 32x4 artifact; DualFormer-B "
        "IN21K checkpoint; ZeroI2V Linux runtime preflight).",
        "3. Fill RTX3050 gaps: `bash run_project_steps.sh rtx3050` after mounting the K400 subset.",
        "",
    ]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
