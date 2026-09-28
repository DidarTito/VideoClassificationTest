"""Regressions for K400 accuracy validity and actual checkpoint paper metadata."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import run_benchmark
from tests.test_run_configuration import _SuccessfulModel, _config, _cpu_device_info


class K400ResultMetadataTests(unittest.TestCase):
    def test_k400_accuracy_is_valid_and_base_paper_value_matches_loaded_release(self):
        key = "dualformer-b-in21k"  # Existing registry key routes the IN1K release.
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            with mock.patch.dict(run_benchmark.MODEL_REGISTRY, {key: lambda **_: _SuccessfulModel()}):
                with mock.patch("run_benchmark.torch.cuda.is_available", return_value=False):
                    frame, errors = run_benchmark.benchmark_models(
                        [key], [torch.zeros(2, 3, 4, 4, dtype=torch.uint8)],
                        "none", labels=[0], device_info=_cpu_device_info(),
                        config=config, warmup=0,
                    )
        self.assertFalse(errors)
        row = frame.iloc[0]
        self.assertEqual(row["Status"], "completed")
        self.assertTrue(row["AccuracyValid"])
        self.assertEqual(row["ClassifierClasses"], 400)
        self.assertEqual(row["MeasuredTop1"], 100.0)
        self.assertEqual(row["PublishedTop1"], 81.1)
        self.assertEqual(row["CheckpointPublishedTop1"], 81.1)
        self.assertEqual(row["Stage1PublishedTop1"], 82.9)


if __name__ == "__main__":
    unittest.main()
