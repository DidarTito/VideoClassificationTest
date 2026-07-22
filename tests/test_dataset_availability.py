"""Dataset/model availability and new SSV2 adapter contract tests."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from check_dataset_models import inspect_dataset_models, main as preflight_main
from models import (
    DATASET_MODEL_KEYS,
    MODEL_DATASET_AVAILABILITY,
    MODEL_REGISTRY,
    REQUESTED_MODEL_KEYS,
)


ROOT = Path(__file__).resolve().parents[1]


class DatasetAvailabilityTests(unittest.TestCase):
    K400 = (
        "uniformer-b",
        "uniformer-s",
        "mvit-b-24-32x3",
        "mvit-v1-b",
        "videoswin-t",
        "videoswin-s",
        "videoswin-b",
        "timesformer-b",
        "video-focalnet-t",
        "video-focalnet-s",
        "video-focalnet-b",
        "dualformer-t",
        "omnivore-b",
        "videomae-b",
        "vtn-b",
    )
    SSV2 = (
        "uniformer-b",
        "uniformer-s",
        "videoswin-b",
        "timesformer-b",
        "video-focalnet-b",
        "videomae-b",
    )

    def test_explicit_matrix_covers_every_stage1_candidate(self):
        self.assertEqual(tuple(MODEL_DATASET_AVAILABILITY), REQUESTED_MODEL_KEYS)
        for row in MODEL_DATASET_AVAILABILITY.values():
            self.assertEqual(set(row), {"k400", "ssv2"})
            self.assertTrue(all(isinstance(value, bool) for value in row.values()))

    def test_exact_dataset_launcher_lists_are_15_and_6(self):
        self.assertEqual(DATASET_MODEL_KEYS["k400"], self.K400)
        self.assertEqual(DATASET_MODEL_KEYS["ssv2"], self.SSV2)
        for keys in DATASET_MODEL_KEYS.values():
            self.assertTrue(set(keys).issubset(MODEL_REGISTRY))

    def test_preflight_list_models_is_machine_readable(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = preflight_main(["--dataset", "ssv2", "--list-models"])
        self.assertEqual(status, 0)
        self.assertEqual(tuple(output.getvalue().split()), self.SSV2)

    def test_mvit_v1_default_checkpoint_is_workspace_local(self):
        from models.mvit_slowfast import MViTSlowFastModel

        checkpoint = MViTSlowFastModel.MODEL_ZOO["B-32x3"]["checkpoint"]
        self.assertEqual(
            checkpoint,
            ROOT / "checkpoints" / "mvit" / "MVIT_B_32x3_f294077834.pyth",
        )

    def test_installed_ssv2_assets_pass_read_only_preflight(self):
        report = inspect_dataset_models("ssv2", ROOT)
        self.assertTrue(report["all_ready"], report)
        self.assertEqual(report["ready_count"], 6)


class TimeSformerCheckpointTests(unittest.TestCase):
    def test_strict_state_extraction_removes_one_wrapper_prefix(self):
        from models.timesformer import TimeSformerModel

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "tiny.pyth"
            torch.save(
                {
                    "model_state": {
                        "model.head.weight": torch.ones(2, 3),
                        "module.head.bias": torch.zeros(2),
                    }
                },
                checkpoint,
            )
            state = TimeSformerModel._strict_checkpoint_state(checkpoint)
        self.assertEqual(set(state), {"head.weight", "head.bias"})

    def test_strict_state_extraction_rejects_non_classifier_payload(self):
        from models.timesformer import TimeSformerModel

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "tiny.pyth"
            torch.save({"state_dict": {}}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Unexpected TimeSformer"):
                TimeSformerModel._strict_checkpoint_state(checkpoint)


class VideoMAESSV2AdapterTests(unittest.TestCase):
    def test_ssv2_path_uses_official_graph_and_174_way_logits(self):
        from models.videomae import VideoMAEModel

        class FakeOfficialModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(1, 174)
                self.input_shape = None

            def forward(self, clip):
                self.input_shape = tuple(clip.shape)
                return torch.zeros((clip.shape[0], 174), device=clip.device)

        fake = FakeOfficialModel()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "ssv2.pth"
            checkpoint.touch()
            info = {
                "checkpoint": checkpoint,
                "frames": 4,
                "accuracy": 70.8,
                "gflops": 180.0,
            }
            with mock.patch.dict(
                VideoMAEModel.SSV2_MODEL_ZOO, {"B1600": info}, clear=False
            ), mock.patch(
                "models.videomae._load_official_ssv2_model", return_value=fake
            ) as loader:
                adapter = VideoMAEModel("B1600", device="cpu", dataset="ssv2")
                logits = adapter(torch.zeros((1, 7, 3, 8, 8)))

        loader.assert_called_once_with(checkpoint)
        self.assertEqual(tuple(logits.shape), (1, 174))
        self.assertEqual(fake.input_shape, (1, 3, 4, 8, 8))
        self.assertEqual(adapter.num_classes, 174)


if __name__ == "__main__":
    unittest.main()
