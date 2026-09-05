from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from data import (
    CACHE_VERSION,
    SampleManifestError,
    ShardedClipSet,
    load_clip_set,
    manifest_sha256,
    read_sample_manifest,
    write_sample_manifest,
)


class StableSampleManifestTests(unittest.TestCase):
    def test_ordered_round_trip_and_file_hash_are_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            sources = ["class_b/b.mp4", "class_a\\a.webm"]
            first = root / "first.csv"
            second = root / "second.csv"

            write_sample_manifest(
                first,
                sources,
                videos,
                dataset="k400",
                seed=0,
                frames=32,
                size=224,
            )
            write_sample_manifest(
                second,
                sources,
                videos,
                dataset="k400",
                seed=0,
                frames=32,
                size=224,
            )

            expected = (
                "ClipIndex,RelativePath,Label,Dataset,Seed,DecodedFrames,Resolution\n"
                "0,class_b/b.mp4,class_b,k400,0,32,224x224\n"
                "1,class_a/a.webm,class_a,k400,0,32,224x224\n"
            ).encode("utf-8")
            self.assertEqual(first.read_bytes(), expected)
            self.assertEqual(second.read_bytes(), expected)
            self.assertEqual(
                read_sample_manifest(first, videos),
                ["class_b/b.mp4", "class_a/a.webm"],
            )
            self.assertEqual(manifest_sha256(first), manifest_sha256(second))
            self.assertEqual(
                manifest_sha256(first), hashlib.sha256(expected).hexdigest()
            )

    def test_read_and_write_reject_paths_that_escape_dataset_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            manifest = root / "traversal.csv"
            manifest.write_text(
                "RelativePath\n../outside.mp4\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(SampleManifestError, "escapes"):
                read_sample_manifest(manifest, videos)
            with self.assertRaisesRegex(SampleManifestError, "escapes"):
                write_sample_manifest(
                    root / "output.csv",
                    ["../outside.mp4"],
                    videos,
                    dataset="k400",
                    seed=0,
                )

    def test_read_and_write_reject_normalized_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            manifest = root / "duplicates.csv"
            manifest.write_text(
                "RelativePath\nclass_a/a.mp4\nclass_a\\a.mp4\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SampleManifestError, "duplicate"):
                read_sample_manifest(manifest, videos)
            with self.assertRaisesRegex(SampleManifestError, "duplicate"):
                write_sample_manifest(
                    root / "output.csv",
                    ["class_a/a.mp4", "class_a\\a.mp4"],
                    videos,
                    dataset="k400",
                    seed=0,
                )

    def test_scientific_metadata_and_clip_indices_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            manifest = root / "sample.csv"
            write_sample_manifest(
                manifest,
                ["class_a/a.mp4"],
                videos,
                dataset="k400",
                seed=0,
                frames=32,
                size=224,
            )

            self.assertEqual(
                read_sample_manifest(
                    manifest,
                    videos,
                    expected_dataset="k400",
                    expected_seed=0,
                    expected_frames=32,
                    expected_size=224,
                ),
                ["class_a/a.mp4"],
            )
            with self.assertRaisesRegex(SampleManifestError, "Seed"):
                read_sample_manifest(manifest, videos, expected_seed=7)

            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "0,class_a/a.mp4", "1,class_a/a.mp4"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SampleManifestError, "ClipIndex"):
                read_sample_manifest(manifest, videos)


class CompatibleShardedCacheTests(unittest.TestCase):
    def test_manifest_prefix_reuses_larger_compatible_cache_without_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            sources = [
                "class_a/a.mp4",
                "class_b/b.mp4",
                "class_c/c.mp4",
            ]
            for source in sources:
                path = videos / Path(source)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            sample_manifest = root / "k400_3_seed0.csv"
            write_sample_manifest(
                sample_manifest,
                sources,
                videos,
                dataset="k400",
                seed=0,
                frames=32,
                size=224,
            )

            cache_dir = (
                videos
                / f".clip_cache_v{CACHE_VERSION}_32f_224px_3c_seed0_all"
            )
            cache_dir.mkdir()
            entries = []
            for index, source in enumerate(sources):
                cache_name = f"clip_{index:05d}.pt"
                torch.save(torch.tensor([index], dtype=torch.uint8), cache_dir / cache_name)
                entries.append({"source": source, "cache": cache_name})
            (cache_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": CACHE_VERSION,
                        "frames": 32,
                        "size": 224,
                        "seed": 0,
                        "filter": "all",
                        "source_format": "relative-v1",
                        "entries": entries,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "data.load_clip",
                side_effect=AssertionError("compatible cache should avoid decoding"),
            ) as decode:
                clips, selected_paths = load_clip_set(
                    videos,
                    max_clips=2,
                    frames=32,
                    size=224,
                    seed=0,
                    use_cache=True,
                    manifest=sample_manifest,
                )

            decode.assert_not_called()
            self.assertIsInstance(clips, ShardedClipSet)
            self.assertEqual(clips.cache_dir, cache_dir)
            self.assertEqual(
                [entry["source"] for entry in clips.entries], sources[:2]
            )
            self.assertEqual(
                selected_paths,
                [str(videos / Path(source)) for source in sources[:2]],
            )
            self.assertTrue(torch.equal(clips[0], torch.tensor([0], dtype=torch.uint8)))
            self.assertTrue(torch.equal(clips[1], torch.tensor([1], dtype=torch.uint8)))

    def test_mismatched_same_key_cache_is_rebuilt_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            desired_sources = ["class_a/a.mp4", "class_b/b.mp4"]
            stale_sources = ["class_c/c.mp4", "class_b/b.mp4"]
            for source in set(desired_sources + stale_sources):
                path = videos / Path(source)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            sample_manifest = root / "k400_2_seed0.csv"
            write_sample_manifest(
                sample_manifest,
                desired_sources,
                videos,
                dataset="k400",
                seed=0,
                frames=32,
                size=224,
            )

            cache_dir = (
                videos
                / f".clip_cache_v{CACHE_VERSION}_32f_224px_2c_seed0_all"
            )
            cache_dir.mkdir()
            stale_entries = []
            for index, source in enumerate(stale_sources):
                cache_name = f"clip_{index:05d}.pt"
                torch.save(
                    torch.tensor([100 + index], dtype=torch.uint8),
                    cache_dir / cache_name,
                )
                stale_entries.append({"source": source, "cache": cache_name})
            (cache_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": CACHE_VERSION,
                        "frames": 32,
                        "size": 224,
                        "seed": 0,
                        "filter": "all",
                        "source_format": "relative-v1",
                        "entries": stale_entries,
                    }
                ),
                encoding="utf-8",
            )

            decoded = {
                "a.mp4": torch.tensor([1], dtype=torch.uint8),
                "b.mp4": torch.tensor([2], dtype=torch.uint8),
            }
            with mock.patch(
                "data.load_clip",
                side_effect=lambda path, frames, size: decoded[Path(path).name],
            ) as decode:
                clips, selected_paths = load_clip_set(
                    videos,
                    max_clips=2,
                    frames=32,
                    size=224,
                    seed=0,
                    use_cache=True,
                    manifest=sample_manifest,
                )

            self.assertEqual(decode.call_count, 2)
            self.assertEqual(
                [entry["source"] for entry in clips.entries], desired_sources
            )
            self.assertEqual(
                selected_paths,
                [str(videos / Path(source)) for source in desired_sources],
            )
            self.assertTrue(torch.equal(clips[0], decoded["a.mp4"]))
            self.assertTrue(torch.equal(clips[1], decoded["b.mp4"]))


if __name__ == "__main__":
    unittest.main()
