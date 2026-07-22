"""Clip loading, annotation parsing, and bounded-memory benchmark caching.

Decoded clips are stored as sharded ``uint8`` tensors and loaded in small
chunks.  A 1,000-clip, 32-frame benchmark is roughly 4.5 GiB even in uint8, so
materialising the complete set in RAM is unsafe on the 16-GiB measurement
machine.  The sharded cache also makes an interrupted decode resumable.
"""
import hashlib
import json
import random
from pathlib import Path

import torch
from decord import VideoReader, cpu
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
CACHE_VERSION = 4  # v4 also uses deterministic single-thread WebM decoding.


class ShardedClipSet:
    """A reusable list-like clip set backed by individual tensor files."""

    def __init__(self, cache_dir, entries):
        self.cache_dir = Path(cache_dir)
        self.entries = list(entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        entry = self.entries[index]
        return torch.load(self.cache_dir / entry["cache"], weights_only=True)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __mul__(self, repeats):
        repeats = int(repeats)
        if repeats < 1:
            return ShardedClipSet(self.cache_dir, [])
        return ShardedClipSet(self.cache_dir, self.entries * repeats)

    __rmul__ = __mul__

    def iter_chunks(self, chunk_size=32):
        """Load at most ``chunk_size`` clips into host memory at once."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for start in range(0, len(self), chunk_size):
            stop = min(start + chunk_size, len(self))
            yield [self[index] for index in range(start, stop)]


def load_clip(path, frames=32, size=224):
    """Decodes one video to a (T, C, H, W) uint8 tensor."""
    # Decord's Windows FFmpeg backend frequently raises EAGAIN from its
    # multi-threaded packet worker on SSV2 WebM files.  A single decoder thread
    # is deterministic and succeeded across the same files in preflight.
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    idx = torch.linspace(0, len(vr) - 1, frames).round().long()
    video = torch.tensor(vr.get_batch(idx.tolist()).asnumpy())  # (T, H, W, C) uint8
    video = video.permute(0, 3, 1, 2).float()
    height, width = video.shape[-2:]
    scale = size / min(height, width)
    resized_height = max(size, round(height * scale))
    resized_width = max(size, round(width * scale))
    video = torch.nn.functional.interpolate(
        video,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    top = (resized_height - size) // 2
    left = (resized_width - size) // 2
    video = video[:, :, top : top + size, left : left + size]
    return video.round_().clamp_(0, 255).to(torch.uint8)


def clip_to_input(clip_u8):
    """(T, C, H, W) uint8 -> (1, T, C, H, W) float in [0, 1] as wrappers expect."""
    return clip_u8.float().div_(255.0).unsqueeze(0)


def _filter_fingerprint(allowed_stems):
    if allowed_stems is None:
        return "all"
    digest = hashlib.sha256()
    for stem in sorted(str(value) for value in allowed_stems):
        digest.update(stem.encode("utf-8"))
        digest.update(b"\n")
    return f"ids{len(allowed_stems)}_{digest.hexdigest()[:12]}"


def _source_key(path, videos_dir):
    """Return a cache-stable source path relative to the dataset root."""
    return Path(path).relative_to(Path(videos_dir)).as_posix()


def _normalise_cached_source(source, videos_dir):
    """Migrate legacy absolute/CWD-relative manifest sources to stable keys."""
    source_path = Path(source)
    videos_dir = Path(videos_dir)
    if source_path.is_absolute():
        try:
            return source_path.resolve().relative_to(videos_dir.resolve()).as_posix()
        except ValueError:
            pass

    # Legacy manifests stored strings such as ``datasets/.../videos/id.webm``
    # or ``../datasets/.../videos/id.webm``.  Recover the suffix after the
    # dataset-root directory so the manifest is independent of the launch CWD.
    root_name = videos_dir.name.casefold()
    parts = source_path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].casefold() == root_name and index + 1 < len(parts):
            return Path(*parts[index + 1 :]).as_posix()
    return source_path.as_posix()


def _source_paths(videos_dir, entries):
    return [str(Path(videos_dir) / Path(entry["source"])) for entry in entries]


def _write_manifest(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_clip_set(
    videos_dir,
    max_clips=1000,
    frames=32,
    size=224,
    seed=0,
    use_cache=True,
    allowed_stems=None,
):
    """Randomly samples up to max_clips videos (seeded, reproducible) from a
    directory tree and decodes them. Returns (clips, paths); corrupt videos
    are skipped with a count.  When ``allowed_stems`` is supplied, sampling is
    restricted to videos represented in the chosen annotation split.

    With caching enabled, ``clips`` is a :class:`ShardedClipSet`; callers can
    iterate normally or use ``iter_chunks`` to bound host-memory use.
    """
    videos_dir = Path(videos_dir)
    allowed_stems = None if allowed_stems is None else {str(value) for value in allowed_stems}
    paths = sorted(p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not paths:
        raise SystemExit(f"No video files found in {videos_dir}")
    if allowed_stems is not None:
        paths = [path for path in paths if path.stem in allowed_stems]
        if not paths:
            raise SystemExit(
                f"No videos in {videos_dir} match the selected annotation split"
            )
        print(f"Annotation filter retained {len(paths)} local videos.")
    random.Random(seed).shuffle(paths)

    target = min(max_clips, len(paths))
    if use_cache:
        fingerprint = _filter_fingerprint(allowed_stems)
        cache_dir = videos_dir / (
            f".clip_cache_v{CACHE_VERSION}_{frames}f_{size}px_"
            f"{max_clips}c_seed{seed}_{fingerprint}"
        )
        manifest_path = cache_dir / "manifest.json"
        entries = []
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != CACHE_VERSION:
                raise RuntimeError(f"Unexpected clip-cache version in {manifest_path}")
            entries = []
            seen_sources = set()
            for raw_entry in manifest.get("entries", []):
                if not (cache_dir / raw_entry["cache"]).is_file():
                    continue
                entry = dict(raw_entry)
                entry["source"] = _normalise_cached_source(
                    entry["source"], videos_dir
                )
                if entry["source"] in seen_sources:
                    continue
                seen_sources.add(entry["source"])
                entries.append(entry)
            if len(entries) >= target:
                entries = entries[:target]
                print(f"Loading {len(entries)} sharded cached clips from {cache_dir} ...")
                return ShardedClipSet(cache_dir, entries), _source_paths(videos_dir, entries)
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)

        known_sources = {entry["source"] for entry in entries}
        # Never derive a new shard name from ``len(entries)``.  During recovery
        # a missing early shard can be filtered from the manifest while a later
        # shard with that compacted index still exists; reusing the index would
        # overwrite the surviving clip and silently corrupt source/label
        # alignment.  Start after every shard name already present on disk.
        cache_index = 0
        for existing in cache_dir.glob("clip_*.pt"):
            try:
                cache_index = max(cache_index, int(existing.stem.rsplit("_", 1)[1]) + 1)
            except (IndexError, ValueError):
                continue
        skipped = 0
        pbar = tqdm(total=target, initial=len(entries), desc="Decoding clips", unit="clip")
        for path in paths:
            if len(entries) >= target:
                break
            source = _source_key(path, videos_dir)
            if source in known_sources:
                continue
            try:
                clip = load_clip(path, frames, size)
                cache_name = f"clip_{cache_index:05d}.pt"
                cache_index += 1
                torch.save(clip, cache_dir / cache_name)
                entries.append({"source": source, "cache": cache_name})
                known_sources.add(source)
                _write_manifest(
                    manifest_path,
                    {
                        "version": CACHE_VERSION,
                        "frames": frames,
                        "size": size,
                        "seed": seed,
                        "filter": fingerprint,
                        "source_format": "relative-v1",
                        "entries": entries,
                    },
                )
                pbar.update(1)
            except Exception:
                skipped += 1
                pbar.set_postfix(skipped=skipped)
        pbar.close()
        if skipped:
            print(f"Skipped {skipped} unreadable videos.")
        if not entries:
            raise RuntimeError("No readable videos were found for the selected split")
        if len(entries) != target:
            raise RuntimeError(
                f"Requested {target} readable clips but only {len(entries)} could be decoded"
            )
        print(f"Using {len(entries)} sharded cached clips from {cache_dir}.")
        return ShardedClipSet(cache_dir, entries), _source_paths(videos_dir, entries)

    clips, kept, skipped = [], [], 0
    pbar = tqdm(paths, desc="Decoding clips", unit="clip", total=target)
    for p in paths:
        if len(clips) >= max_clips:
            break
        try:
            clips.append(load_clip(p, frames, size))
            kept.append(str(p))
            pbar.update(1)
        except Exception:
            skipped += 1
            pbar.set_postfix(skipped=skipped)
    pbar.close()
    if skipped:
        print(f"Skipped {skipped} unreadable videos.")

    if len(clips) != target:
        raise RuntimeError(
            f"Requested {target} readable clips but only {len(clips)} could be decoded"
        )

    return clips, kept


def _ssv2_label_map(annotation_path):
    label_candidates = [
        annotation_path.with_name("something-something-v2-labels.json"),
        annotation_path.with_name("labels.json"),
    ]
    label_file = next((path for path in label_candidates if path.is_file()), None)
    if label_file is None:
        raise FileNotFoundError(
            "SSV2 label map not found beside the split annotation; expected "
            "something-something-v2-labels.json or labels.json"
        )
    raw = json.loads(label_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"SSV2 label map must be a JSON object: {label_file}")
    result = {str(template).replace("[", "").replace("]", ""): int(index)
              for template, index in raw.items()}
    indices = set(result.values())
    if (not result or len(indices) != len(result) or min(indices) < 0
            or max(indices) >= 174):
        raise ValueError(f"SSV2 label map must contain unique indices in [0, 173]: {label_file}")
    return result


def load_ssv2_annotation_map(annotations):
    """Return ``video id -> class index`` for supported SSV2 split formats.

    Official SSV2 JSON records contain a free-form instance description in
    ``label`` and the class-bearing text in ``template``.  The latter must be
    mapped through the neighboring 174-class label map.  Vendored UniFormer
    split lists use ``id frame_count class_index`` and already carry the exact
    numeric class identity.  Semicolon ``id;template`` answer files are also
    accepted.
    """
    annotation_path = Path(annotations)
    text = annotation_path.read_text(encoding="utf-8-sig")
    by_id = {}

    try:
        records = json.loads(text)
    except json.JSONDecodeError:
        records = None

    if isinstance(records, list):
        label_map = None
        for record in records:
            video_id = str(record["id"])
            if "template" in record:
                if label_map is None:
                    label_map = _ssv2_label_map(annotation_path)
                template = str(record["template"]).replace("[", "").replace("]", "")
                if template not in label_map:
                    raise ValueError(
                        f"Unknown SSV2 template {record['template']!r} in {annotation_path}"
                    )
                label = label_map[template]
            elif isinstance(record.get("label"), int) or str(record.get("label", "")).isdigit():
                label = int(record["label"])
            else:
                raise ValueError(
                    f"SSV2 record {video_id} has no class-bearing template/label"
                )
            by_id[video_id] = label
    elif records is not None:
        raise ValueError(f"SSV2 split annotations must be a JSON list: {annotation_path}")
    else:
        label_map = None
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            if ";" in line:
                video_id, template = line.split(";", 1)
                if label_map is None:
                    label_map = _ssv2_label_map(annotation_path)
                template = template.strip().replace("[", "").replace("]", "")
                if template not in label_map:
                    raise ValueError(
                        f"Unknown SSV2 template {template!r} at "
                        f"{annotation_path}:{line_number}"
                    )
                by_id[video_id.strip()] = label_map[template]
                continue

            fields = line.split()
            if len(fields) >= 2 and fields[-1].isdigit():
                class_index = int(fields[-1])
                if not 0 <= class_index < 174:
                    raise ValueError(
                        f"SSV2 class index outside [0, 173] at "
                        f"{annotation_path}:{line_number}"
                    )
                # The vendored numeric split was generated from the official
                # labels.json mapping.  category.txt has a different display
                # order and must not be used to reinterpret this index.
                by_id[fields[0]] = class_index
                continue

            raise ValueError(
                f"Malformed SSV2 annotation at {annotation_path}:{line_number}"
            )

    if not by_id:
        raise ValueError(f"SSV2 annotation file is empty: {annotation_path}")
    return by_id


def load_annotation_ids(annotations):
    """Return video IDs represented by one labeled SSV2 annotation file."""
    return set(load_ssv2_annotation_map(annotations))


def load_ground_truth(paths, dataset="k400", annotations=None):
    """Resolve K400 folder labels or standard SSV2 JSON annotations.

    SSV2 validation/test videos are commonly stored flat as ``<id>.webm``.
    ``annotations`` should point to the split JSON; the neighboring
    ``something-something-v2-labels.json`` supplies template-to-index order.
    """
    if dataset == "k400":
        return [Path(path).parent.name for path in paths]
    if annotations is None:
        # Also supports a user-created class-directory layout.
        return [Path(path).parent.name for path in paths]
    annotations = Path(annotations)
    by_id = load_ssv2_annotation_map(annotations)
    missing = [Path(path).stem for path in paths if Path(path).stem not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} selected SSV2 videos are absent from {annotations}")
    return [by_id[Path(path).stem] for path in paths]
