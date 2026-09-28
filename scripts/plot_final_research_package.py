"""Export static scientific plots from the validated master CSV."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.generate_ssv2_full17_device import read

SHORT = {"uniformer": "UF", "dualformer": "DF", "video-focalnet": "FN", "videoswin": "VS", "mvit-v1": "MViT", "omnivore": "Omni", "timesformer": "TS", "videomae": "MAE"}

def main():
    rows = read(ROOT / "results/final_k400_ssv2_master.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for y, dataset in enumerate(("k400", "ssv2")):
        selected = [r for r in rows if r["Dataset"] == dataset and r["ExactFrozenRecipe"] == "TRUE"]
        for x, (metric, label) in enumerate((("LatencyMs", "Latency (ms/clip)"), ("EnergyPerClipJ", "Energy (J/clip)"), ("PeakVRAMMiB", "Peak allocated VRAM (MiB)"))):
            ax = axes[y, x]
            for i, r in enumerate(selected):
                key = r["Model"]
                for family, short in SHORT.items():
                    if key.startswith(family):
                        key = key.replace(family, short, 1)
                        break
                vx, vy = float(r[metric]), float(r["MeasuredTop1"])
                ax.scatter(vx, vy, color=colors(i % 10), s=36, edgecolor="white", linewidth=0.4)
                ax.annotate(key, (vx, vy), xytext=(4, 4 if i % 2 else -11), textcoords="offset points", fontsize=7)
            ax.set_xlabel(label)
            ax.set_ylabel("Measured Top-1 (%)")
            ax.set_title("K400 exact compatible recipes" if dataset == "k400" else "SSV2 genuine checkpoints")
            ax.grid(alpha=0.2)
            ax.margins(x=0.2, y=0.18)
    fig.suptitle("Accuracy and device costs on fixed 1,000-clip subsets\nRTX 3050 Laptop GPU · FP32 · batch 1 · single view", fontsize=14)
    target = ROOT / "figures/final_accuracy_device_tradeoffs"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(target.with_suffix("." + extension), dpi=250)
    plt.close(fig)
    print(target)

if __name__ == "__main__":
    main()
