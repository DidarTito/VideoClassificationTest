import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from training.checkpoint_utils import load_k400_backbone_strict, replace_classifier
from training.registry import FROZEN17_KEYS, ROOT, load_manifest


class FrozenRegistryTests(unittest.TestCase):
    def test_exact_frozen_keys(self):
        self.assertEqual(len(FROZEN17_KEYS), 17)
        self.assertEqual(len(set(FROZEN17_KEYS)), 17)
        banned={"vivit-s","vtn-b","svt-b","videomae-s","mvit-b-24-32x3","mvit-v2-s"}
        self.assertFalse(banned & set(FROZEN17_KEYS))
    def test_metadata_and_inputs(self):
        for item in load_manifest()["models"]:
            self.assertEqual(item["ssv2_classes"],174)
            self.assertIn("checkpoint_status",item)
            for field in ("num_frames","resolution","mean","std","tensor_layout"):self.assertIn(field,item["input"])
            self.assertTrue((ROOT/item["finetune"]["config"]).is_file())


class StrictLoaderTests(unittest.TestCase):
    class Tiny(nn.Module):
        def __init__(self):super().__init__();self.backbone=nn.Linear(3,4);self.head=nn.Linear(4,174)
    def test_rejects_backbone_mismatch(self):
        model=self.Tiny();state={"backbone.weight":torch.randn(5,3),"backbone.bias":torch.randn(4),"head.weight":torch.randn(400,4),"head.bias":torch.randn(400)}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.pth";torch.save(state,p)
            with self.assertRaisesRegex(RuntimeError,"Backbone mismatch"):load_k400_backbone_strict(model,p,"head")
    def test_replace_head(self):
        model=nn.Module();model.head=nn.Linear(4,400);replace_classifier(model,"head");self.assertEqual(model.head(torch.randn(2,4)).shape,(2,174))


class OutputIsolationTests(unittest.TestCase):
    def test_final_and_legacy_paths_are_distinct(self):
        final=ROOT/"results/finetuned/ssv2_rtx5070";legacy=ROOT/"results/legacy"
        self.assertNotEqual(final.resolve(),legacy.resolve())
    def test_deployment_contract_is_batch_one(self):
        text=(ROOT/"scripts/benchmark_finetuned_ssv2.py").read_text(encoding="utf-8")
        self.assertIn("batch size = 1", text.lower())

if __name__=="__main__":unittest.main()
