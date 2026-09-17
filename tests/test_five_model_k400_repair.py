"""Regression coverage for the exact five-model K400 checkpoint repair."""

import gc
from pathlib import Path
import unittest

import torch

import check_models
from models import REQUESTED_MODEL_SPECS
from training.registry import ROOT, model_spec


EXPECTED_UNIFORMER_B_SHA = (
    "b432450566cdc4f50222e4851b5d304bba4444a699cdbb43fa529ceb18ae1f8e"
)


def _asset_result(key):
    specs = {spec.key: spec for spec in check_models.REQUESTED_MODELS}
    context = check_models.InspectionContext(
        root=ROOT,
        torch_checkpoint_dirs=check_models._torch_checkpoint_dirs(ROOT),
        registry_keys=check_models._registry_keys(ROOT / "models/__init__.py"),
    )
    return specs[key].inspect(context, specs[key])


class FiveModelK400RepairTests(unittest.TestCase):
    def test_uniformer_s_uses_distributed_official_16x4_checkpoint(self):
        expected = "checkpoints/uniformer/uniformer_small_k400_16x4.pth"
        frozen = model_spec("uniformer-s")
        requested = REQUESTED_MODEL_SPECS["uniformer-s"].for_dataset("k400")

        self.assertEqual(frozen["k400_checkpoint"], expected)
        self.assertEqual(requested.checkpoint, expected)
        self.assertEqual(frozen["input"]["num_frames"], 16)
        self.assertEqual(frozen["input"]["sampling_rate"], 4)
        self.assertEqual(requested.preprocessing.frames, 16)

        result = _asset_result("uniformer-s")
        self.assertTrue(result.ready, result.blockers)
        exact = [asset.path for asset in result.assets if asset.compatibility == "exact"]
        self.assertIn(ROOT / expected, exact)
        self.assertNotIn(
            ROOT / "checkpoints/uniformer/uniformer_small_k400_16x8.pth", exact
        )

    def test_uniformer_b_accepts_exact_32x4_sha(self):
        frozen = model_spec("uniformer-b")
        requested = REQUESTED_MODEL_SPECS["uniformer-b"].for_dataset("k400")
        self.assertEqual(frozen["checkpoint_sha256"], EXPECTED_UNIFORMER_B_SHA)
        self.assertEqual(
            frozen["k400_checkpoint"],
            "checkpoints/uniformer/uniformer_base_k400_32x4.pth",
        )
        self.assertEqual(requested.checkpoint, frozen["k400_checkpoint"])
        self.assertEqual(requested.preprocessing.frames, 32)

        result = _asset_result("uniformer-b")
        self.assertTrue(result.ready, result.blockers)
        exact = [asset for asset in result.assets if asset.compatibility == "exact"]
        self.assertTrue(any(asset.path.name == "uniformer_base_k400_32x4.pth" for asset in exact))

    def test_videomae_official_pth_is_400_class_classifier_and_strict_loads(self):
        frozen = model_spec("videomae-b")
        checkpoint = ROOT / frozen["k400_checkpoint"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.assertEqual(list(payload), ["module"])
        state = payload["module"]
        self.assertEqual(tuple(state["head.weight"].shape), (400, 768))
        self.assertEqual(tuple(state["patch_embed.proj.weight"].shape), (768, 3, 2, 16, 16))
        self.assertFalse(any("decoder" in key.split(".") for key in state))
        self.assertFalse(any("mask_token" in key for key in state))
        self.assertEqual(sum(tensor.numel() for tensor in state.values()), 86_534_800)
        del state, payload

        from models.videomae import VideoMAEModel

        adapter = VideoMAEModel("B1600", device="cpu", dataset="k400")
        self.assertEqual(adapter.info["checkpoint"], checkpoint)
        self.assertEqual(adapter.info["frames"], 16)
        self.assertEqual(adapter.model.head.out_features, 400)
        self.assertEqual(len(adapter.model.blocks), 12)
        del adapter
        gc.collect()


if __name__ == "__main__":
    unittest.main()
