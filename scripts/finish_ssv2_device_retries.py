"""Run the two repaired VideoSwin loaders only after fresh accuracy inference ends."""
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
while any("run_benchmark.py" in (p.info["cmdline"] or []) and "accuracy-verification" in p.info["cmdline"]
          for p in psutil.process_iter(["cmdline"])):
    time.sleep(5)
command = [sys.executable, "-u", "run_benchmark.py", "--mode", "device-only", "--dataset", "ssv2",
           "--models", "videoswin-t", "videoswin-s", "--manifest", "manifests/ssv2_1000_seed0.csv",
           "--annotations", "third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_validation_split.txt",
           "--num-clips", "1000", "--batch-size", "1", "--precision", "fp32", "--seed", "0", "--warmup", "2",
           "--repeats", "1", "--power", "nvml", "--power-sample-ms", "20", "--device", "cuda:0", "--allow-skips",
           "--output", "results/ssv2_device_only_videoswin_retry.csv"]
with (ROOT / "logs/ssv2_device_only_videoswin_retry.log").open("w", encoding="utf-8") as stream:
    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
print("VideoSwin device retries exit code:", result.returncode, flush=True)
subprocess.run([sys.executable, "scripts/generate_ssv2_full17_device.py"], cwd=ROOT, check=True)
