"""Contract tests for the authoritative requested-model metadata."""

from pathlib import PurePosixPath
import unittest

from models import (
    DATASET_MODEL_KEYS,
    MODEL_DATASET_AVAILABILITY,
    MODEL_REGISTRY,
    REQUESTED_MODEL_KEYS,
    REQUESTED_MODEL_SPECS,
)


class RequestedModelMetadataTests(unittest.TestCase):
    def test_metadata_is_authoritative_for_legacy_views(self):
        self.assertEqual(tuple(REQUESTED_MODEL_SPECS), REQUESTED_MODEL_KEYS)
        expected_availability = {
            key: {
                dataset: REQUESTED_MODEL_SPECS[key].datasets[dataset].available
                for dataset in ("k400", "ssv2")
            }
            for key in REQUESTED_MODEL_KEYS
        }
        self.assertEqual(MODEL_DATASET_AVAILABILITY, expected_availability)
        for dataset in ("k400", "ssv2"):
            expected_keys = tuple(
                key
                for key in REQUESTED_MODEL_KEYS
                if REQUESTED_MODEL_SPECS[key].datasets[dataset].available
            )
            self.assertEqual(DATASET_MODEL_KEYS[dataset], expected_keys)

    def test_requested_registry_factories_are_the_spec_builders(self):
        for key, spec in REQUESTED_MODEL_SPECS.items():
            with self.subTest(model=key):
                self.assertIs(MODEL_REGISTRY[key], spec.builder)

    def test_available_pairs_have_complete_portable_identity(self):
        counts = {"k400": 0, "ssv2": 0}
        for key, spec in REQUESTED_MODEL_SPECS.items():
            self.assertTrue(spec.display_name)
            self.assertGreater(spec.screening.top1_percent, 0)
            self.assertGreater(spec.screening.params_m, 0)
            self.assertGreater(spec.screening.gflops, 0)
            for dataset, pair in spec.datasets.items():
                with self.subTest(model=key, dataset=dataset):
                    self.assertEqual(pair.dataset, dataset)
                    self.assertEqual(pair.num_classes, 400 if dataset == "k400" else 174)
                    self.assertTrue(pair.availability_reason)
                    if not pair.available:
                        self.assertIsNone(pair.checkpoint)
                        self.assertIsNone(pair.preprocessing)
                        self.assertIsNone(pair.published)
                        continue
                    counts[dataset] += 1
                    self.assertIsNotNone(pair.checkpoint)
                    checkpoint = PurePosixPath(pair.checkpoint)
                    self.assertFalse(checkpoint.is_absolute())
                    self.assertEqual(checkpoint.parts[0], "checkpoints")
                    self.assertIsNotNone(pair.preprocessing)
                    self.assertEqual(pair.preprocessing.decoded_frames, 32)
                    self.assertEqual(pair.preprocessing.resolution, (224, 224))
                    self.assertGreater(pair.preprocessing.frames, 0)
                    self.assertIsNotNone(pair.published)
                    self.assertGreater(pair.published.top1_percent, 0)
                    self.assertGreater(pair.published.params_m, 0)
                    self.assertGreater(pair.published.gflops, 0)
        self.assertEqual(counts, {"k400": 15, "ssv2": 6})

    def test_checkpoint_paths_and_temporal_recipes_are_exact(self):
        expected = {
            ("uniformer-s", "k400"): (
                "checkpoints/uniformer/uniformer_small_k400_16x4.pth", 16
            ),
            ("uniformer-b", "k400"): (
                "checkpoints/uniformer/uniformer_base_k400_32x4.pth", 32
            ),
            ("mvit-v1-b", "k400"): (
                "checkpoints/mvit/MVIT_B_32x3_f294077834.pyth", 32
            ),
            ("timesformer-b", "k400"): (
                "checkpoints/timesformer/TimeSformer_divST_8x32_224_K400.pyth", 8
            ),
            ("videomae-b", "k400"): (
                "checkpoints/videomae/videomae_vit_b_k400_1600e_ft.pth", 16
            ),
            ("videoswin-b", "ssv2"): (
                "checkpoints/legacy_ssv2_released/swin_base_patch244_window1677_sthv2.pth", 32
            ),
            ("videomae-b", "ssv2"): (
                "checkpoints/legacy_ssv2_released/videomae_vit_b_ssv2_2400e.pth", 16
            ),
        }
        for (key, dataset), (checkpoint, frames) in expected.items():
            with self.subTest(model=key, dataset=dataset):
                pair = REQUESTED_MODEL_SPECS[key].for_dataset(dataset)
                self.assertEqual(pair.checkpoint, checkpoint)
                self.assertEqual(pair.preprocessing.frames, frames)

    def test_unavailable_variants_never_point_to_proxy_checkpoints(self):
        vivit = REQUESTED_MODEL_SPECS["vivit-s"].for_dataset("k400")
        svt = REQUESTED_MODEL_SPECS["svt-b"].for_dataset("k400")
        self.assertFalse(vivit.available)
        self.assertIsNone(vivit.checkpoint)
        self.assertIn("vit-b", vivit.incompatible_artifact.lower())
        self.assertFalse(svt.available)
        self.assertIsNone(svt.checkpoint)
        self.assertIn("self-supervised", svt.availability_reason)

    def test_screening_and_exact_checkpoint_publications_stay_separate(self):
        uniformer_b = REQUESTED_MODEL_SPECS["uniformer-b"]
        self.assertEqual(uniformer_b.screening.top1_percent, 82.9)
        self.assertEqual(
            uniformer_b.for_dataset("k400").published.top1_percent, 82.9
        )
        self.assertEqual(uniformer_b.screening.gflops, 259.0)
        self.assertEqual(uniformer_b.for_dataset("k400").published.gflops, 259.0)

    def test_dataset_aliases_resolve_without_changing_identity(self):
        spec = REQUESTED_MODEL_SPECS["videomae-b"]
        self.assertIs(spec.for_dataset("kinetics-400"), spec.datasets["k400"])
        self.assertIs(spec.for_dataset("sth_v2"), spec.datasets["ssv2"])
        with self.assertRaises(ValueError):
            spec.for_dataset("ucf101")


if __name__ == "__main__":
    unittest.main()
