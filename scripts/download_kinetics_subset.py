import os
import random
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------

TRAIN_CSV = "annotations/train.csv"
VAL_CSV = "annotations/val.csv"

OUTPUT = Path("datasets/kinetics400_subset")

TRAIN_DIR = OUTPUT / "train"
VAL_DIR = OUTPUT / "val"

VIDEOS_PER_CLASS = 6          # 5-8 recommended
NUM_CLASSES = 50

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ----------------------------

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)


def clean_label(label):
    return (
        label.strip()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(",", "")
        .replace("'", "")
    )


def make_url(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"


def download_clip(
    youtube_id,
    start,
    end,
    out_file,
):
    duration = float(end) - float(start)

    url = make_url(youtube_id)

    command = [
        "yt-dlp",
        "-f",
        "mp4",
        "--download-sections",
        f"*{start}-{end}",
        "-o",
        str(out_file),
        url,
    ]

    try:

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return True

    except Exception:
        return False


def process_split(csv_path, output_dir):

    df = pd.read_csv(csv_path)

    label_column = None

    for c in df.columns:

        if "label" in c.lower():
            label_column = c
            break

    if label_column is None:
        raise RuntimeError("Cannot locate label column.")

    classes = sorted(df[label_column].unique())

    random.shuffle(classes)

    classes = classes[:NUM_CLASSES]

    print(f"\nUsing {len(classes)} classes\n")

    successful = 0
    failed = 0
    for cls in classes:

        cls_df = df[df[label_column] == cls]

        cls_df = cls_df.sample(
            min(VIDEOS_PER_CLASS * 2, len(cls_df)),
            random_state=RANDOM_SEED,
        )

        cls_name = clean_label(cls)

        cls_dir = output_dir / cls_name
        cls_dir.mkdir(exist_ok=True)

        downloaded = 0

        print(f"{cls_name:35s}", end=" ")

        for _, row in cls_df.iterrows():

            if downloaded >= VIDEOS_PER_CLASS:
                break

            youtube_id = row["youtube_id"]

            start = row["time_start"]

            end = row["time_end"]

            outfile = cls_dir / f"{youtube_id}.mp4"

            if outfile.exists():

                downloaded += 1

                continue

            ok = download_clip(
                youtube_id,
                start,
                end,
                outfile,
            )

            if ok:

                downloaded += 1
                successful += 1

            else:

                failed += 1

        print(f"{downloaded}/{VIDEOS_PER_CLASS}")

    print("\n-----------------------------------")
    print(f"Downloaded : {successful}")
    print(f"Failed     : {failed}")
    print("-----------------------------------")