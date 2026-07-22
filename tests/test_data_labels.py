import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from data import (
    CACHE_VERSION,
    _filter_fingerprint,
    load_annotation_ids,
    load_clip_set,
    load_ground_truth,
)


class GroundTruthTests(unittest.TestCase):
    def test_official_json_uses_template_instead_of_free_form_text_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = {"Moving [something] up": "7"}
            split = [
                {
                    "id": "12345",
                    "label": "Moving a blue pen up",
                    "template": "Moving [something] up",
                }
            ]
            (root / "something-something-v2-labels.json").write_text(
                json.dumps(labels), encoding="utf-8"
            )
            annotation = root / "something-something-v2-validation.json"
            annotation.write_text(json.dumps(split), encoding="utf-8")
            result = load_ground_truth(
                [root / "videos" / "12345.webm"], "ssv2", annotation
            )
            self.assertEqual(result, [7])

    def test_semicolon_annotation_maps_template_through_official_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "labels.json").write_text(
                json.dumps({"Putting [something] on a surface": "13"}),
                encoding="utf-8",
            )
            annotation = root / "validation.txt"
            annotation.write_text(
                "12345; Putting [something] on a surface\n", encoding="utf-8"
            )

            result = load_ground_truth(
                [root / "videos" / "12345.webm"], "ssv2", annotation
            )

            self.assertEqual(result, [13])

    def test_vendored_numeric_annotation_uses_final_class_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "validation.csv"
            annotation.write_text(
                "12345 48 91\n98765 36 5\n", encoding="utf-8"
            )

            result = load_ground_truth(
                [
                    root / "videos" / "98765.webm",
                    root / "videos" / "12345.webm",
                ],
                "ssv2",
                annotation,
            )

            self.assertEqual(result, [5, 91])


class AnnotationFilteringTests(unittest.TestCase):
    def test_annotation_ids_filter_videos_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            keep = videos / "keep.webm"
            reject = videos / "reject.webm"
            keep.touch()
            reject.touch()
            annotation = root / "validation.txt"
            annotation.write_text("keep 48 12\n", encoding="utf-8")
            allowed_stems = load_annotation_ids(annotation)

            with mock.patch(
                "data.load_clip", side_effect=lambda path, frames, size: path.stem
            ) as decode:
                clips, paths = load_clip_set(
                    videos,
                    max_clips=10,
                    frames=8,
                    size=112,
                    use_cache=False,
                    allowed_stems=allowed_stems,
                )

            self.assertEqual(clips, ["keep"])
            self.assertEqual(paths, [str(keep)])
            decode.assert_called_once_with(keep, 8, 112)

    def test_filter_fingerprint_is_order_independent_and_split_specific(self):
        first = _filter_fingerprint({"video-b", "video-a"})
        reordered = _filter_fingerprint(["video-a", "video-b"])
        different_split = _filter_fingerprint({"video-a", "video-c"})

        self.assertEqual(first, reordered)
        self.assertNotEqual(first, different_split)
        self.assertNotEqual(first, _filter_fingerprint(None))

    def test_partial_cache_recovery_migrates_legacy_cwd_relative_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            videos = Path(directory) / "videos"
            videos.mkdir()
            sources = [videos / f"video-{name}.webm" for name in "abc"]
            for source in sources:
                source.touch()

            cache_dir = videos / (
                f".clip_cache_v{CACHE_VERSION}_8f_112px_3c_seed0_all"
            )
            cache_dir.mkdir()
            torch.save(torch.tensor([10]), cache_dir / "clip_00000.pt")
            # clip_00001.pt is deliberately absent, while the later shard exists.
            torch.save(torch.tensor([30]), cache_dir / "clip_00002.pt")
            legacy_sources = [
                str(
                    Path("..")
                    / "old-working-directory"
                    / "dataset"
                    / "videos"
                    / source.name
                )
                for source in sources
            ]
            manifest = {
                "version": CACHE_VERSION,
                "frames": 8,
                "size": 112,
                "seed": 0,
                "filter": "all",
                "entries": [
                    {"source": legacy_sources[0], "cache": "clip_00000.pt"},
                    {"source": legacy_sources[1], "cache": "clip_00001.pt"},
                    {"source": legacy_sources[2], "cache": "clip_00002.pt"},
                ],
            }
            (cache_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            recovered_clip = torch.tensor([20])
            with mock.patch("data.load_clip", return_value=recovered_clip) as decode:
                clips, paths = load_clip_set(
                    videos, max_clips=3, frames=8, size=112, seed=0
                )

            decode.assert_called_once_with(sources[1], 8, 112)
            self.assertEqual(
                [entry["cache"] for entry in clips.entries],
                ["clip_00000.pt", "clip_00002.pt", "clip_00003.pt"],
            )
            self.assertEqual(paths, [str(sources[0]), str(sources[2]), str(sources[1])])
            self.assertTrue(torch.equal(clips[1], torch.tensor([30])))
            self.assertTrue(torch.equal(clips[2], recovered_clip))
            migrated = json.loads(
                (cache_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [entry["source"] for entry in migrated["entries"]],
                ["video-a.webm", "video-c.webm", "video-b.webm"],
            )

    def test_decode_underfill_raises_in_cached_and_in_memory_modes(self):
        for use_cache in (True, False):
            with self.subTest(use_cache=use_cache):
                with tempfile.TemporaryDirectory() as directory:
                    videos = Path(directory) / "videos"
                    videos.mkdir()
                    for name in ("good-a", "bad", "good-b"):
                        (videos / f"{name}.webm").touch()

                    def decode(path, _frames, _size):
                        if path.stem == "bad":
                            raise ValueError("corrupt video")
                        return torch.tensor([1])

                    with mock.patch("data.load_clip", side_effect=decode):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "Requested 3 readable clips but only 2 could be decoded",
                        ):
                            load_clip_set(
                                videos,
                                max_clips=3,
                                frames=8,
                                size=112,
                                use_cache=use_cache,
                            )


if __name__ == "__main__":
    unittest.main()
