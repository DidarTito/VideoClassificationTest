#!/usr/bin/env python3
"""RTX5070 preflight (Phase 7).

Checks the Linux RTX5070 environment and then, for every frozen17 model:
instantiate -> load exact K400 checkpoint -> one forward -> confirm 400
logits -> record peak VRAM. Writes reports/RTX5070_PREFLIGHT.md.

The script never fakes success: any failed environment check or model step
is recorded verbatim in the report. It can also run on a non-target host
(e.g. the Windows RTX3050 laptop) for a dry check; environment rows will
simply show FAIL for the Linux/RTX5070-specific requirements.

Usage:
    python scripts/preflight_rtx5070.py                # all 17
    python scripts/preflight_rtx5070.py --models uniformer-s videomae-b
    python scripts/preflight_rtx5070.py --device cpu   # environment-only hosts
"""

import argparse
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from training.registry import ROOT, model_spec, resolve_models
from training.frozen_preflight import validate_frozen_model

REPORT = ROOT / "reports" / "RTX5070_PREFLIGHT.md"


def check_environment(device):
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), str(detail)))

    add("Linux", platform.system() == "Linux", f"{platform.system()} {platform.release()}")
    add("Python >= 3.10", sys.version_info >= (3, 10), platform.python_version())
    add("torch", True, torch.__version__)
    try:
        import torchvision
        add("torchvision", True, torchvision.__version__)
    except Exception as exc:  # pragma: no cover - environment specific
        add("torchvision", False, exc)

    cuda_ok = torch.cuda.is_available()
    add("CUDA available", cuda_ok, torch.version.cuda or "n/a")
    gpu_name = ""
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        add("RTX5070 detected", "5070" in gpu_name, gpu_name)
        add("Compute capability >= 8.9", capability >= (8, 9), f"{capability[0]}.{capability[1]}")
        add("BF16 support", torch.cuda.is_bf16_supported(), "torch.cuda.is_bf16_supported()")
    else:
        add("RTX5070 detected", False, "CUDA unavailable")
        add("BF16 support", False, "CUDA unavailable")
    try:
        import pynvml
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        add("NVML / driver", True, driver)
        pynvml.nvmlShutdown()
    except Exception as exc:
        add("NVML / driver", False, exc)
    try:
        import decord
        add("decord", True, decord.__version__)
    except Exception as exc:
        add("decord", False, exc)
    add("ffmpeg on PATH", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found")

    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / 1e9
    add("Disk free >= 50 GB", free_gb >= 50, f"{free_gb:.1f} GB free")

    for label, env_key in (("K400 videos", "K400_VAL_ROOT"), ("SSV2 videos", "SSV2_ROOT")):
        import os
        value = os.environ.get(env_key, "")
        path_ok = bool(value) and Path(value).is_dir()
        add(f"{label} ({env_key})", path_ok, value or f"set ${env_key} to the dataset root")

    add("Requested device", True, device)
    return checks


def preflight_model(key, device):
    """Instantiate + exact checkpoint + forward + 400 logits + peak VRAM."""
    from models import MODEL_REGISTRY

    spec = model_spec(key)
    row = {
        "key": key,
        "display_name": spec["display_name"],
        "checkpoint": spec.get("k400_checkpoint") or "none",
        "status": "FAIL",
        "logits": "",
        "peak_vram_mib": "",
        "load_s": "",
        "forward_s": "",
        "params_m": "",
        "error": "",
    }
    if spec["checkpoint_status"].startswith(("MISSING", "BLOCKED")):
        row["status"] = "BLOCKED"
        row["error"] = spec["checkpoint_status"]
        return row
    ckpt_rel = spec.get("k400_checkpoint")
    if not ckpt_rel or not (ROOT / ckpt_rel).is_file():
        row["status"] = "BLOCKED"
        row["error"] = f"checkpoint not on disk: {ckpt_rel}"
        return row
    try:
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model = MODEL_REGISTRY[key](device=device)
        row["load_s"] = f"{time.perf_counter() - start:.1f}"
        contract = validate_frozen_model(
            key,
            model,
            dataset="k400",
            checkpoint=ROOT / spec["k400_checkpoint"],
        )
        row["params_m"] = f"{contract['actual_params_m']:.3f}"
        # Some adapters expose their frame count via info; the frozen manifest
        # is authoritative either way.
        frames = int(model.info.get("frames") or spec["input"]["num_frames"])
        clip = torch.rand(1, frames, 3, 224, 224)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(clip)
        row["forward_s"] = f"{time.perf_counter() - start:.2f}"
        row["logits"] = str(tuple(logits.shape))
        if tuple(logits.shape) != (1, 400):
            raise RuntimeError(f"expected (1, 400) K400 logits, got {tuple(logits.shape)}")
        if use_cuda:
            row["peak_vram_mib"] = f"{torch.cuda.max_memory_allocated() / (1 << 20):.0f}"
        row["status"] = "OK"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["frozen17"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-only", action="store_true",
                        help="run only the environment checks, skip model loads")
    args = parser.parse_args()

    keys = resolve_models("frozen17" if args.models == ["frozen17"] else args.models)
    env_checks = check_environment(args.device)
    for name, ok, detail in env_checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    if args.env_only:
        failed = [name for name, ok, _ in env_checks if not ok]
        print(f"\nEnvironment checks failed: {len(failed)}" + (f" ({', '.join(failed)})" if failed else ""))
        return

    rows = []
    for key in keys:
        print(f"\n=== preflight {key} ===", flush=True)
        row = preflight_model(key, args.device)
        print(f"{key}: {row['status']}" + (f" ({row['error']})" if row["error"] else ""))
        rows.append(row)

    ok_count = sum(row["status"] == "OK" for row in rows)
    lines = [
        "# RTX5070 preflight",
        "",
        f"Host: {platform.node()} ({platform.system()} {platform.release()}), "
        f"torch {torch.__version__}, generated by `scripts/preflight_rtx5070.py`.",
        "",
        "## Environment",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {'OK' if ok else 'FAIL'} | {detail} |" for name, ok, detail in env_checks]
    lines += [
        "",
        f"## Frozen17 model preflight: {ok_count}/{len(rows)} OK",
        "",
        "| Model | Checkpoint | Status | Params (M) | Logits | Peak VRAM (MiB) | Load (s) | Forward (s) | Error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | `{row['checkpoint']}` | **{row['status']}** | "
            f"{row['params_m']} | {row['logits']} | {row['peak_vram_mib']} | {row['load_s']} | "
            f"{row['forward_s']} | {row['error']} |"
        )
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPreflight OK: {ok_count}/{len(rows)}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
