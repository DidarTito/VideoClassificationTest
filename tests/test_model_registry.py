"""Tests that model selection is lazy and exact-variant failures stay honest."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from models import LazyFactory, MODEL_REGISTRY, REQUESTED_MODEL_KEYS
from models.common import ModelUnavailableError


ROOT = Path(__file__).resolve().parents[1]


class RequestedRegistryTests(unittest.TestCase):
    EXPECTED = (
        "uniformer-b",
        "uniformer-s",
        "mvit-b-24-32x3",
        "mvit-v1-b",
        "videoswin-t",
        "videoswin-s",
        "videoswin-b",
        "vivit-s",
        "timesformer-b",
        "video-focalnet-t",
        "video-focalnet-s",
        "video-focalnet-b",
        "dualformer-t",
        "omnivore-b",
        "videomae-b",
        "svt-b",
        "vtn-b",
    )

    def test_requested_registry_has_exact_seventeen_keys(self):
        self.assertEqual(REQUESTED_MODEL_KEYS, self.EXPECTED)
        self.assertEqual(len(REQUESTED_MODEL_KEYS), 17)
        self.assertTrue(set(REQUESTED_MODEL_KEYS).issubset(MODEL_REGISTRY))
        for key in REQUESTED_MODEL_KEYS:
            self.assertIsInstance(MODEL_REGISTRY[key], LazyFactory)

    def test_registry_import_does_not_import_adapter_implementations(self):
        # Use a fresh interpreter so test ordering cannot contaminate sys.modules.
        code = """
import json
import sys
import models
modules = sorted({factory.module for factory in models.MODEL_REGISTRY.values()})
print(json.dumps({
    "all_lazy": all(isinstance(v, models.LazyFactory) for v in models.MODEL_REGISTRY.values()),
    "eager": [name for name in modules if name in sys.modules],
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["all_lazy"])
        self.assertEqual(result["eager"], [])

    def test_requested_factories_target_exact_adapters(self):
        expected = {
            "uniformer-b": ("models.uniformer", "UniFormerModel", ("B",)),
            "uniformer-s": ("models.uniformer", "UniFormerModel", ("S",)),
            "mvit-b-24-32x3": (
                "models.mvit_slowfast",
                "MViTSlowFastModel",
                ("B-24-32x3",),
            ),
            "mvit-v1-b": (
                "models.mvit_slowfast",
                "MViTSlowFastModel",
                ("B-32x3",),
            ),
            "vivit-s": ("models.vivit", "ViViTModel", ("S",)),
            "videoswin-t": ("models.videoswin", "VideoSwinModel", ("T",)),
            "videoswin-s": ("models.videoswin", "VideoSwinModel", ("S",)),
            "videoswin-b": ("models.videoswin", "VideoSwinModel", ("B",)),
            "timesformer-b": ("models.timesformer", "TimeSformerModel", ("B",)),
            "video-focalnet-t": (
                "models.videofocalnet",
                "VideoFocalNetModel",
                ("T",),
            ),
            "video-focalnet-s": (
                "models.videofocalnet",
                "VideoFocalNetModel",
                ("S",),
            ),
            "video-focalnet-b": (
                "models.videofocalnet",
                "VideoFocalNetModel",
                ("B",),
            ),
            "dualformer-t": ("models.dualformer", "DualFormerModel", ("T",)),
            "omnivore-b": ("models.omnivore", "OmnivoreModel", ("B-IN21K",)),
            "videomae-b": ("models.videomae", "VideoMAEModel", ("B1600",)),
            "svt-b": ("models.svt", "SVTModel", ("B",)),
            "vtn-b": ("models.vtn", "VTNModel", ("B",)),
        }
        observed = {
            key: (factory.module, factory.class_name, factory.args)
            for key, factory in MODEL_REGISTRY.items()
            if key in REQUESTED_MODEL_KEYS
        }
        self.assertEqual(observed, expected)


class CheckpointLabelOrderTests(unittest.TestCase):
    def test_uniformer_uses_its_official_non_alphabetical_k400_order(self):
        from models.uniformer import _checkpoint_class_names

        names = _checkpoint_class_names()
        self.assertEqual(len(names), 400)
        self.assertEqual(len(set(names)), 400)
        self.assertEqual(names[169], "slacklining")
        self.assertEqual(names[210], "snorkeling")
        self.assertNotEqual(names, sorted(names))


class ExactVariantFailureTests(unittest.TestCase):
    def assert_unavailable(self, constructor, *message_fragments):
        with self.assertRaises(ModelUnavailableError) as caught:
            constructor()
        message = str(caught.exception).lower()
        for fragment in message_fragments:
            self.assertIn(fragment.lower(), message)

    def test_vivit_s_fails_before_transformers_or_vivit_b_can_load(self):
        from models.vivit import ViViTModel

        fake_transformers = mock.Mock()
        fake_transformers.VivitForVideoClassification.from_pretrained.side_effect = (
            AssertionError("ViViT-B must not be loaded for ViViT-S")
        )
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            self.assert_unavailable(
                lambda: ViViTModel("S", device="cpu"),
                "vivit-s",
                "b/l",
                "not substituted",
            )
        fake_transformers.VivitForVideoClassification.from_pretrained.assert_not_called()

    def test_uniformer_b_rejects_missing_exact_k400_checkpoint(self):
        from models.uniformer import UniFormerModel

        # This keeps the test independent of any checkpoints added later and
        # ensures it never reaches construction of the large vendored model.
        with mock.patch("models.uniformer.Path.is_file", return_value=False):
            self.assert_unavailable(
                lambda: UniFormerModel("B", device="cpu"),
                "uniformer-b",
                "400-class",
                "missing",
            )

    def test_dualformer_t_has_a_clear_missing_checkpoint_error(self):
        from models.dualformer import DualFormerModel

        missing = ROOT / "checkpoints" / "dualformer" / "definitely_missing.pth"
        with mock.patch.dict(
            DualFormerModel.MODEL_ZOO["T"], {"checkpoint": missing}, clear=False
        ):
            self.assert_unavailable(
                lambda: DualFormerModel("T", device="cpu"),
                "dualformer-t",
                "missing",
                "randomly initialized",
            )

    def test_svt_b_rejects_ssl_backbone_as_classifier(self):
        from models.svt import SVTModel

        self.assert_unavailable(
            lambda: SVTModel("B", device="cpu"),
            "svt-b",
            "self-supervised",
            "no trained 400-class head",
            "not substituted",
        )

if __name__ == "__main__":
    unittest.main()
