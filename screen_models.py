# """Stage 2.1 - Device-agnostic screening with NetScore.

# Uses published Top-1 accuracy / params / GFLOPs from the literature table,
# ranks all candidate video ViTs by NetScore, and applies the screening
# criteria (Top-1 >= 78%, Params < 100M, GFLOPs < 600) to select the pool
# for hardware measurement.
# """
# import pandas as pd

# from metrics import netscore
# from models import MODEL_REGISTRY

# # Published numbers (Kinetics-400) from the respective papers.
# LITERATURE = [
#     # name,                 top1,  params_m, gflops, venue,                registry key (if runnable here)
#     ("UniFormer-S",          80.8,  22,    42,   "ICLR 2022",     None),
#     ("UniFormer-B",          83.0,  50,   259,   "ICLR 2022",     None),
#     ("MViT-S",               76.0,  26,    33,   "ICCV 2021",     None),
#     ("MViT-B (32x3)",        80.2,  37,   170,   "ICCV 2021",     "mvit-v1-b"),
#     ("MViT-B (64x3)",        81.2,  37,   455,   "ICCV 2021",     None),
#     ("MViTv2-S",             81.0,  34,    64,   "CVPR 2022",     "mvit-v2-s"),
#     ("VideoSwin-T",          78.8,  28,    88,   "CVPR 2022",     "videoswin-t"),
#     ("VideoSwin-S",          80.6,  50,   166,   "CVPR 2022",     "videoswin-s"),
#     ("VideoSwin-B",          80.6,  88,   282,   "CVPR 2022",     "videoswin-b"),
#     ("TimeSformer-B",        78.0, 121,   196,   "ICML 2021",     "timesformer-b"),
#     ("TimeSformer-HR",       79.7, 121,  1703,   "ICML 2021",     "timesformer-hr"),
#     ("TimeSformer-L",        80.7, 121,  2380,   "ICML 2021",     None),
#     ("VideoMAE-B (800ep)",   82.9,  86,   180,   "NeurIPS 2022",  None),
#     ("VideoMAE-B (1600ep)",  83.4,  86,   180,   "NeurIPS 2022",  "videomae-b1600"),
#     ("VideoMAE-L",           84.3, 304,   597,   "NeurIPS 2022",  "videomae-l"),
#     ("Video-FocalNet-T",     79.8,  28,    63,   "ICCV 2023",     None),
#     ("Video-FocalNet-S",     81.4,  49,   124,   "ICCV 2023",     None),
#     ("Video-FocalNet-B",     83.6,  88,   149,   "ICCV 2023",     None),
#     ("Omnivore-B (Swin)",    84.0,  88,   282,   "CVPR 2022",     None),
#     ("ActionCLIP ViT-B/8",   82.1,  86,   280,   "arXiv 2021",    "actionclip-b8"),
#     ("ActionCLIP ViT-B/16",  83.8,  86,   560,   "arXiv 2021",    "actionclip-b16"),
#     ("ActionCLIP ViT-L",     85.2, 307,  1200,   "arXiv 2021",    None),
#     ("ViViT-S",              78.5,  31.8,  55,   "ICCV 2021",     None),
#     ("ViViT-B",              80.0,  87.9, 455.2, "ICCV 2021",     "vivit-b"),
#     ("VTN-B",                78.0, 114,   218,   "ICCV-W 2021",   "vtn-b"),
#     ("SVT-B",                78.1,  86,   180,   "CVPR 2022",     "svt-b"),
#     ("ZeroI2V-B",            83.0,  86,   422,   "ECCV 2024",     "zeroi2v-b"),
# ]

# MIN_TOP1 = 78.0
# MAX_PARAMS_M = 300
# MAX_GFLOPS = 600


# def build_screening_table() -> pd.DataFrame:
#     rows = []
#     for name, top1, params_m, gflops, venue, key in LITERATURE:
#         passes = top1 >= MIN_TOP1 and params_m < MAX_PARAMS_M and gflops < MAX_GFLOPS
#         rows.append({
#             "Model": name,
#             "Venue": venue,
#             "Top1(%)": top1,
#             "Params(M)": params_m,
#             "GFLOPs": gflops,
#             "NetScore": round(netscore(top1, params_m, gflops), 2),
#             "PassesScreening": passes,
#             "RegistryKey": key or "",
#             "RunnableHere": bool(key and key in MODEL_REGISTRY),
#         })
#     df = pd.DataFrame(rows).sort_values("NetScore", ascending=False).reset_index(drop=True)
#     df.index += 1
#     df.index.name = "Rank"
#     return df


# def main():
#     df = build_screening_table()
#     pd.set_option("display.width", 160)
#     print(df.to_string())
#     df.to_csv("netscore_screening.csv")
#     selected = df[df["PassesScreening"]]
#     print(f"\n{len(selected)}/{len(df)} models pass screening "
#           f"(Top-1 >= {MIN_TOP1}%, Params < {MAX_PARAMS_M}M, GFLOPs < {MAX_GFLOPS}).")
#     runnable = selected[selected["RunnableHere"]]["RegistryKey"].tolist()
#     print("Screened models runnable on this machine:", " ".join(runnable))
#     print("Saved: netscore_screening.csv")


# if __name__ == "__main__":
#     main()

"""Stage 2.1 - Device-agnostic screening with NetScore.

Uses published Top-1 accuracy / params / GFLOPs from the literature table,
ranks all candidate video ViTs by NetScore, and applies the screening
criteria (Top-1 >= 78%, Params < 300M, GFLOPs < 600) to select the pool
for hardware measurement. Also marks models that satisfy the stricter
Pareto threshold (GFLOPs < 300) for the Pareto plot.
"""
import pandas as pd
import numpy as np

from metrics import netscore
from models import MODEL_REGISTRY
from check_models import inspect_requested_models

# Published numbers (Kinetics-400) from the respective papers.
# Includes all 36 models from your list.
LITERATURE = [
    # Core Video Transformer Models
    ("UniFormer-B",          83.0,  50.3, 259,  "ICLR 2022",     "uniformer-b"),
    ("UniFormer-S",          80.8,  22.0,  42,  "ICLR 2022",     "uniformer-s"),
    ("MViT-B, 64x3",         81.2,  36.6, 455,  "ICCV 2021",     None),
    ("MViT-B-24, 32x3",      81.2,  52.9, 236,  "ICCV 2021",     "mvit-b-24-32x3"),
    ("MViT-B, 32x3",         80.2,  36.6, 170,  "ICCV 2021",     "mvit-v1-b"),
    ("MViT-S",               76.0,  26.1,  32.9,"ICCV 2021",     None),
    ("VideoSwin-T",          78.8,  28.2,  88,  "CVPR 2022",     "videoswin-t"),
    ("VideoSwin-S",          80.6,  49.8, 166,  "CVPR 2022",     "videoswin-s"),
    ("VideoSwin-B",          80.6,  88.0, 282,  "CVPR 2022",     "videoswin-b"),
    ("VideoSwin-L",          83.1, 197,   604,  "CVPR 2022",     None),
    # ViViT Family
    ("ViViT-S",              78.5,  31.8,  55,  "ICCV 2021",     "vivit-s"),
    ("ViViT-B",              80.0,  87.9, 455.2,"ICCV 2021",     "vivit-b"),
    ("ViViT-L",              80.6, 310.8,1446,  "ICCV 2021",     None),
    ("ViViT-L (320)",        81.3, 310.8,3992,  "ICCV 2021",     None),
    ("ViViT-H",              84.9, 647.5,3981,  "ICCV 2021",     None),
    # TimeSformer Family
    ("TimeSformer-B",        78.0, 121.4, 196,  "ICML 2021",     "timesformer-b"),
    ("TimeSformer-HR",       79.7, 121.4,1703,  "ICML 2021",     "timesformer-hr"),
    ("TimeSformer-L",        80.7, 121.4,2380,  "ICML 2021",     None),
    # Zero-Cost Image-to-Video
    ("ZeroI2V ViT-B/16",     83.0,  86.0, 422,  "arXiv 2023",    "zeroi2v-b"),
    ("ZeroI2V ViT-L/14",     86.3, 304,  1946,  "arXiv 2023",    None),
    # Video-FocalNet Family
    ("Video-FocalNet-T",     79.8,  28.0,  63,  "CVPR 2022",     "video-focalnet-t"),
    ("Video-FocalNet-S",     81.4,  49.0, 124,  "CVPR 2022",     "video-focalnet-s"),
    ("Video-FocalNet-B",     83.6,  88.0, 149,  "CVPR 2022",     "video-focalnet-b"),
    # DualFormer Family
    ("DualFormer-T",         79.5,  21.8, 240,  "NeurIPS 2021",  "dualformer-t"),
    ("DualFormer-S",         82.9,  65.0,1000,  "NeurIPS 2021",  None),
    ("DualFormer-B",         82.9,  65.0,1000,  "NeurIPS 2021",  None),
    # Multi-Modal & Foundation
    ("Omnivore-B",           84.0,  88.0, 282,  "CVPR 2022",     "omnivore-b"),
    ("ActionCLIP-B",         83.8, 141.7, 560,  "arXiv 2021",    "actionclip-b16"),
    ("BIKE-B",               83.5, 230,   830,  "CVPR 2023",     None),
    ("VATT-L",               82.1, 306,  2980,  "NeurIPS 2021",  None),
    # Other Video Transformers
    ("VideoMAE-B",           80.9,  86.2, 180,  "NeurIPS 2022",  "videomae-b1600"),
    ("SVT-B",                78.1,  86.0, 180,  "CVPR 2022",     "svt-b"),
    ("VTN-B",                78.6, 114.0, 218,  "ICCV 2021",     "vtn-b"),
    ("Motionformer-B",       79.7, 109.1, 369.5,"NeurIPS 2021",  None),
    ("TokenLearner-L",       85.4, 450,  4076,  "NeurIPS 2021",  None),
    ("Efficient-VDiT",       75.3,  36.5,  25.96,"CVPR 2024",    None),
]

MIN_TOP1 = 78.0
MAX_PARAMS_M = 300
MAX_GFLOPS_SCREEN = 600   # for the screening table
MAX_GFLOPS_PARETO = 300   # for the Pareto plot (17 models)
MAX_GFLOPS = MAX_GFLOPS_SCREEN   # for backward compatibility with stage2_pipeline.py

def build_screening_table() -> pd.DataFrame:
    _, requested_results = inspect_requested_models()
    requested_ready = {result.spec.key: result.ready for result in requested_results}
    rows = []
    for name, top1, params_m, gflops, venue, key in LITERATURE:
        passes_screen = top1 >= MIN_TOP1 and params_m < MAX_PARAMS_M and gflops < MAX_GFLOPS_SCREEN
        passes_pareto = top1 >= MIN_TOP1 and params_m < MAX_PARAMS_M and gflops <= MAX_GFLOPS_PARETO
        rows.append({
            "Model": name,
            "Venue": venue,
            "Top1(%)": top1,
            "Params(M)": params_m,
            "GFLOPs": gflops,
            "NetScore": round(netscore(top1, params_m, gflops), 2),
            "PassesScreening": passes_screen,
            "PassesPareto": passes_pareto,   # new column for Pareto plot
            "RegistryKey": key or "",
            "RunnableHere": bool(
                key
                and key in MODEL_REGISTRY
                and requested_ready.get(key, True)
            ),
        })
    df = pd.DataFrame(rows).sort_values("NetScore", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"
    return df

def main():
    df = build_screening_table()
    pd.set_option("display.width", 160)
    print(df.to_string())
    df.to_csv("netscore_screening.csv")
    selected = df[df["PassesScreening"]]
    print(f"\nScreening ({MAX_GFLOPS_SCREEN} GFLOPs): {len(selected)}/{len(df)} models pass "
          f"(Top-1 >= {MIN_TOP1}%, Params < {MAX_PARAMS_M}M, GFLOPs < {MAX_GFLOPS_SCREEN}).")
    pareto = df[df["PassesPareto"]]
    print(f"Pareto ({MAX_GFLOPS_PARETO} GFLOPs): {len(pareto)}/{len(df)} models pass "
          f"(Acc >= {MIN_TOP1}%, GFLOPs <= {MAX_GFLOPS_PARETO}).")
    runnable = selected[selected["RunnableHere"]]["RegistryKey"].tolist()
    print("Screened models runnable on this machine:", " ".join(runnable))
    print("Saved: netscore_screening.csv")

if __name__ == "__main__":
    main()
