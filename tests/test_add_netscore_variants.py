"""Focused tests for the standalone Stage-2 NetScore post-processor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from metrics import netscore_e, netscore_hash, netscore_m
from scripts.add_netscore_variants import augment_dataframe, process_csv


class AddNetScoreVariantsTests(unittest.TestCase):
    def test_legacy_csv_is_augmented_ranked_and_reported_without_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source" / "device_metrics.csv"
            source_path.parent.mkdir()
            source_path.write_text(
                "Model,Top1Published(%),Top1Measured(%),TotalTime(s),"
                "AvgPower(W),GPUMemory(MB),SAM1,SAM5\n"
                "MeasuredModel,70,80,2,50,100,0.1,0.2\n"
                "FallbackModel,90,,4,25,200,0.3,0.4\n",
                encoding="utf-8",
            )
            original_bytes = source_path.read_bytes()

            outputs = process_csv(source_path, root / "output")

            self.assertEqual(source_path.read_bytes(), original_bytes)
            augmented = pd.read_csv(outputs["device_metrics"])
            self.assertNotIn("SAM1", augmented.columns)
            self.assertNotIn("SAM5", augmented.columns)
            self.assertIn("GPUMemory(MB)", augmented.columns)
            self.assertIn("PeakVRAM(MiB)", augmented.columns)
            self.assertEqual(
                augmented["MetricAccuracySource"].tolist(),
                ["measured", "published"],
            )
            self.assertEqual(augmented["MetricAccuracy(%)"].tolist(), [80.0, 90.0])

            measured = augmented.iloc[0]
            self.assertAlmostEqual(
                measured["NS-E(1/8)"], round(netscore_e(80, 2, 50), 2)
            )
            self.assertAlmostEqual(
                measured["NS-M(1/8)"], round(netscore_m(80, 100), 2)
            )
            self.assertAlmostEqual(
                measured["NS#(1/8)"], round(netscore_hash(80, 100, 2, 50), 2)
            )

            for filename in (
                "ranking_ns_e.csv",
                "ranking_ns_m.csv",
                "ranking_ns_hash.csv",
            ):
                ranking = pd.read_csv(root / "output" / filename)
                self.assertEqual(ranking["Rank"].tolist(), [1, 2])
                self.assertEqual(ranking.iloc[0]["Model"], "FallbackModel")

            report = outputs["report"].read_text(encoding="utf-8")
            self.assertIn("## Intermediate conclusions", report)
            self.assertIn("Aggregate all-candidate inference time is 6.000 s", report)
            self.assertIn("1/2 measured and 1/2 published fallback", report)

    def test_new_peak_vram_header_and_zero_measured_accuracy_use_fallback(self):
        source = pd.DataFrame(
            {
                "Model": ["NewSchema"],
                "Top1Published(%)": [78.5],
                "Top1Measured(%)": [0.0],
                "TotalTime(s)": [10.0],
                "AvgPower(W)": [40.0],
                "PeakVRAM(MiB)": [512.0],
                "SAM_custom": [123.0],
            }
        )

        augmented = augment_dataframe(source)

        self.assertEqual(augmented.at[0, "MetricAccuracy(%)"], 78.5)
        self.assertEqual(augmented.at[0, "MetricAccuracySource"], "published")
        self.assertNotIn("SAM_custom", augmented.columns)
        self.assertEqual(augmented.at[0, "PeakVRAM(MiB)"], 512.0)

    def test_perfect_measured_accuracy_is_valid(self):
        source = pd.DataFrame(
            {
                "Model": ["TinyFixture"],
                "Top1Published(%)": [75.0],
                "Top1Measured(%)": [100.0],
                "TotalTime(s)": [1.0],
                "AvgPower(W)": [10.0],
                "PeakVRAM(MiB)": [64.0],
            }
        )

        augmented = augment_dataframe(source)

        self.assertEqual(augmented.at[0, "MetricAccuracy(%)"], 100.0)
        self.assertEqual(augmented.at[0, "MetricAccuracySource"], "measured")

    def test_nonpositive_device_metric_is_rejected_before_writing(self):
        source = pd.DataFrame(
            {
                "Model": ["NoPower"],
                "Top1Published(%)": [80.0],
                "TotalTime(s)": [1.0],
                "AvgPower(W)": [0.0],
                "PeakVRAM(MiB)": [100.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "average power"):
            augment_dataframe(source)

    def test_refuses_to_overwrite_source_device_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            source_path = output_dir / "device_metrics.csv"
            source_path.write_text("Model\nExample\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "overwrite the source"):
                process_csv(source_path, output_dir)


if __name__ == "__main__":
    unittest.main()
