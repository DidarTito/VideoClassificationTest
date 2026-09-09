from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from decord import VideoReader, cpu
except ImportError as exc:
    raise ImportError(
        "SSV2Dataset requires decord. Install it with: pip install decord"
    ) from exc


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(3, 1, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(3, 1, 1, 1)


def _as_path(value) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _load_labels(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"SSV2 label mapping not found: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, dict):
        classes = len(obj)
    elif isinstance(obj, list):
        classes = len(obj)
    else:
        raise ValueError(
            f"Unsupported labels.json structure: {type(obj).__name__}"
        )

    if classes != 174:
        raise ValueError(
            f"Expected 174 SSV2 classes, found {classes} in {path}"
        )

    return obj


def _video_id_from_token(token: str) -> str:
    token = token.strip()
    token = token.replace("\\", "/")
    name = token.rsplit("/", 1)[-1]

    # Accept IDs with or without an extension.
    stem = Path(name).stem
    return stem


def _parse_split(path: Path) -> list[tuple[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"SSV2 annotation file not found: {path}")

    samples: list[tuple[str, int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            # UniFormer-style lists are whitespace separated.  Taking the
            # first and last fields also tolerates raw-frame lists containing
            # an intermediate frame-count column.
            fields = line.replace(",", " ").split()

            if len(fields) < 2:
                raise ValueError(
                    f"Malformed SSV2 annotation at {path}:{lineno}: {raw!r}"
                )

            video_id = _video_id_from_token(fields[0])

            try:
                label = int(fields[-1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid class ID at {path}:{lineno}: {fields[-1]!r}"
                ) from exc

            if not 0 <= label < 174:
                raise ValueError(
                    f"Class ID outside [0, 173] at "
                    f"{path}:{lineno}: {label}"
                )

            samples.append((video_id, label))

    if not samples:
        raise ValueError(f"No SSV2 samples found in {path}")

    return samples


def _annotation_identity(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()

    for path in paths:
        path = _as_path(path)
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")

        if path.is_file():
            with path.open("rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)

        digest.update(b"\0")

    return digest.hexdigest()


def _check_video_decode(path: Path) -> bool:
    try:
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)

        if len(vr) <= 0:
            return False

        # Actually decode a frame rather than only opening the container.
        _ = vr[0].asnumpy()
        return True
    except Exception:
        return False


def audit_dataset(
    root,
    labels,
    train_annotations,
    val_annotations,
    check_decode: bool = False,
):
    root = _as_path(root)
    labels = _as_path(labels)
    train_annotations = _as_path(train_annotations)
    val_annotations = _as_path(val_annotations)

    label_mapping = _load_labels(labels)
    train = _parse_split(train_annotations)
    val = _parse_split(val_annotations)

    train_ids = [video_id for video_id, _ in train]
    val_ids = [video_id for video_id, _ in val]

    train_set = set(train_ids)
    val_set = set(val_ids)

    overlap = sorted(train_set & val_set)

    missing_train = [
        video_id
        for video_id in train_ids
        if not (root / f"{video_id}.webm").is_file()
    ]

    missing_val = [
        video_id
        for video_id in val_ids
        if not (root / f"{video_id}.webm").is_file()
    ]

    corrupt_train: list[str] = []
    corrupt_val: list[str] = []

    if check_decode:
        for video_id in train_ids:
            path = root / f"{video_id}.webm"
            if path.is_file() and not _check_video_decode(path):
                corrupt_train.append(video_id)

        for video_id in val_ids:
            path = root / f"{video_id}.webm"
            if path.is_file() and not _check_video_decode(path):
                corrupt_val.append(video_id)

    return {
        "classes": len(label_mapping),
        "train_count": len(train),
        "val_count": len(val),
        "overlap": overlap,
        "missing": {
            "train": missing_train,
            "val": missing_val,
        },
        "corrupt": {
            "train": corrupt_train,
            "val": corrupt_val,
        },
        "annotation_hash": _annotation_identity(
            [labels, train_annotations, val_annotations]
        ),
    }


def _sample_indices(length: int, num_frames: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot sample frames from an empty video")

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    # Uniform sampling over the entire clip.  linspace also naturally repeats
    # frames for very short clips.
    indices = np.linspace(
        0,
        length - 1,
        num=num_frames,
        dtype=np.int64,
    )

    return indices


def _resize_center_crop(
    frames: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    """
    frames: [T, C, H, W], float in [0, 1]
    """
    if frames.ndim != 4:
        raise ValueError(
            f"Expected [T,C,H,W] frames, got {tuple(frames.shape)}"
        )

    _, _, height, width = frames.shape

    if height <= 0 or width <= 0:
        raise ValueError(
            f"Invalid decoded resolution: {height}x{width}"
        )

    # Resize shortest side, preserve aspect ratio, then center crop.
    scale = float(resolution) / float(min(height, width))

    new_h = max(resolution, int(round(height * scale)))
    new_w = max(resolution, int(round(width * scale)))

    frames = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )

    top = max(0, (new_h - resolution) // 2)
    left = max(0, (new_w - resolution) // 2)

    frames = frames[
        :,
        :,
        top : top + resolution,
        left : left + resolution,
    ]

    return frames


class SSV2Dataset(Dataset):
    """
    Something-Something V2 video dataset.

    Returns:
        video: FloatTensor [C, T, H, W]
        label: int
    """

    def __init__(
        self,
        videos,
        annotations,
        labels,
        num_frames: int,
        resolution: int,
    ):
        self.videos = _as_path(videos)
        self.annotations = _as_path(annotations)
        self.labels_path = _as_path(labels)

        self.num_frames = int(num_frames)
        self.resolution = int(resolution)

        _load_labels(self.labels_path)
        self.samples = _parse_split(self.annotations)

        if not self.videos.is_dir():
            raise FileNotFoundError(
                f"SSV2 video directory not found: {self.videos}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_id, label = self.samples[index]
        path = self.videos / f"{video_id}.webm"

        if not path.is_file():
            raise FileNotFoundError(f"Missing SSV2 video: {path}")

        try:
            vr = VideoReader(
                str(path),
                ctx=cpu(0),
                num_threads=1,
            )

            indices = _sample_indices(
                len(vr),
                self.num_frames,
            )

            array = vr.get_batch(indices).asnumpy()

        except Exception as exc:
            raise RuntimeError(
                f"Failed to decode SSV2 video {path}: {exc}"
            ) from exc

        # decord -> [T, H, W, C], uint8
        frames = torch.from_numpy(array).float().div_(255.0)
        frames = frames.permute(0, 3, 1, 2).contiguous()

        frames = _resize_center_crop(
            frames,
            self.resolution,
        )

        # Keep [T,C,H,W].
        # training/adapters/runtime.py performs model-specific
        # normalization and converts to the backend layout.
        return frames.contiguous(), int(label)
