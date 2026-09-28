from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = [
    ["UniFormer-S", 92.2, 4.396, 216],
    ["Video-FocalNet-T", 95.9, 4.701, 745],
    ["UniFormer-B", 92.0, 9.344, 324],
    ["VideoMAE-B", 91.3, 9.161, 412],
    ["ZeroI2V ViT-B/16 (8f)", 90.5, 7.448, 475.52],
    ["DualFormer-T", 89.9, 6.858, 517],
    ["MViT-B, 16x4", 85.4, 4.410, 354.9],
    ["TimeSformer-B", 92.7, 9.445, 755],
    ["DualFormer-S", 91.9, 12.641, 815.55],
    ["MViT-B, 32x3", 90.8, 12.373, 855],
    ["VideoSwin-S", 93.3, 16.712, 1004],
    ["VideoSwin-T", 89.7, 10.108, 909],
    ["DualFormer-B (IN1K)", 93.1, 18.905, 1157.22],
    ["Video-FocalNet-B", 88.6, 12.539, 1267],
    ["VideoSwin-B", 91.7, 23.818, 1465],
    ["Video-FocalNet-S", 80.7, 8.234, 831],
    ["Omnivore-B (IN21K)", 89.8, 33.602, 2143],
]
df = pd.DataFrame(DATA, columns=["Model", "MeasuredTop1", "EnergyPerClip", "PeakVRAM"])

def pareto_frontier(df, x_col, y_col):
    valid = df.dropna(subset=[x_col, y_col]).copy()
    keep = []
    for _, row in valid.iterrows():
        dominated = (
            (valid[x_col] <= row[x_col]) &
            (valid[y_col] >= row[y_col]) &
            ((valid[x_col] < row[x_col]) | (valid[y_col] > row[y_col]))
        ).any()
        keep.append(not dominated)
    return valid.loc[keep].sort_values(x_col)

def plot(cost_col, xlabel, title, out_path):
    frontier = pareto_frontier(df, cost_col, "MeasuredTop1")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df[cost_col], df["MeasuredTop1"], s=70, alpha=0.8, label="All models")
    ax.scatter(frontier[cost_col], frontier["MeasuredTop1"], s=180, marker="*", label="Pareto-optimal", zorder=3)
    if len(frontier) > 1:
        ax.plot(frontier[cost_col], frontier["MeasuredTop1"], linestyle="--", linewidth=1.8)
    pnames = set(frontier["Model"])
    for _, row in df.iterrows():
        suffix = " *" if row["Model"] in pnames else ""
        ax.annotate(row["Model"] + suffix, (row[cost_col], row["MeasuredTop1"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Measured Top-1 Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return frontier

out = Path("results/pareto/rtx3050_k400")
out.mkdir(parents=True, exist_ok=True)

energy = plot(
    "EnergyPerClip",
    "Energy per Clip (J) - lower is better",
    "RTX 3050 Laptop GPU - Kinetics-400\nMeasured Accuracy vs Energy per Clip",
    out / "pareto_accuracy_vs_energy.png",
)
memory = plot(
    "PeakVRAM",
    "Peak GPU Memory (MiB) - lower is better",
    "RTX 3050 Laptop GPU - Kinetics-400\nMeasured Accuracy vs Peak GPU Memory",
    out / "pareto_accuracy_vs_memory.png",
)

df.to_csv(out / "k400_3050_17_manual.csv", index=False)
energy.to_csv(out / "pareto_energy_models.csv", index=False)
memory.to_csv(out / "pareto_memory_models.csv", index=False)

print("\nENERGY PARETO:")
print(energy[["Model", "MeasuredTop1", "EnergyPerClip"]].to_string(index=False))
print("\nMEMORY PARETO:")
print(memory[["Model", "MeasuredTop1", "PeakVRAM"]].to_string(index=False))
print("\nSaved to:", out)
