from datasets import load_dataset
import os
import shutil
import random

OUTPUT = "datasets/kinetics400_subset"

VIDEOS_PER_CLASS = 6
NUM_CLASSES = 50

dataset = load_dataset(
    "liuhuanjim013/kinetics400",
    split="train"
)

classes = {}

for sample in dataset:

    label = sample["label"]

    if label not in classes:
        classes[label] = []

    classes[label].append(sample)

selected = random.sample(list(classes.keys()), NUM_CLASSES)

for label in selected:

    folder = os.path.join(OUTPUT, label)
    os.makedirs(folder, exist_ok=True)

    vids = random.sample(
        classes[label],
        min(VIDEOS_PER_CLASS, len(classes[label]))
    )

    for i, video in enumerate(vids):

        shutil.copy(
            video["video_path"],
            os.path.join(folder, f"{i}.mp4")
        )

print("Finished.")