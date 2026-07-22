"""Tests for bounded, missing-data-safe cross-dataset reporting."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.summarize_stage2_results import (
    DEFAULT_MANIFEST,
    NUMERIC_OUTPUTS,
    build_combined,
    load_manifest,
    normalize_metrics,
    summarize,
    write_outputs,
)


def metric_row(
    key: str,
    model: str,
    dataset: str,
    published: float,
    measured: float,
    *,
    ns_e: float,
    ns_m: float,
    ns_hash: float,
) -> dict[str, object]:
    return {
        "Device": "Test GPU",
        "Dataset": dataset,
        "Split": "validation",
        "InputProtocol": "single uniformly sampled full-video view",
        "ClipCount": 1000,
        "FramesUsed": 16,
        "ModelKey": key,
        "Model": model,
        "Top1Published(%)": published,
        "Top1Measured(%)": measured,
        "MetricAccuracy(%)": measured,
        "MetricAccuracySource": "measured",
        "Latency(ms)": 100.0,
        "AvgPower(W)": 50.0,
        "PowerBackend": "pynvml/NVML",
        "TotalTime(s)": 100.0,
        "Energy(J)": 5.0,
        "TotalEnergy(J)": 5000.0,
        "PeakVRAM(MiB)": 400.0,
        "NS-E(1/8)": ns_e,
        "NS-M(1/8)": ns_m,
        "NS#(1/8)": ns_hash,
    }


class CrossDatasetSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.k400_path = self.root / "k400.csv"
        self.ssv2_path = self.root / "ssv2.csv"
        pd.DataFrame(
            [
                metric_row(
                    "uniformer-b",
                    "UniFormer-B local name",
                    "k400",
                    83.0,
                    80.0,
                    ns_e=60.0,
                    ns_m=61.0,
                    ns_hash=55.0,
                ),
                metric_row(
                    "uniformer-s",
                    "UniFormer-S",
                    "Kinetics-400",
                    80.8,
                    79.0,
                    ns_e=58.0,
                    ns_m=60.0,
                    ns_hash=53.0,
                ),
            ]
        ).to_csv(self.k400_path, index=False)
        pd.DataFrame(
            [
                metric_row(
                    "uniformer-b",
                    "UniFormer-B",
                    "Something-Something-v2",
                    70.4,
                    68.0,
                    ns_e=50.0,
                    ns_m=51.0,
                    ns_hash=45.0,
                )
            ]
        ).to_csv(self.ssv2_path, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_is_exact_15_by_6_availability_matrix(self):
        manifest = load_manifest(DEFAULT_MANIFEST)
        self.assertEqual(len(manifest), 15)
        self.assertTrue(manifest["k400_runnable"].all())
        self.assertEqual(int(manifest["ssv2_pair_supported"].sum()), 6)
        self.assertTrue(manifest["ssv2_availability_reason"].str.len().gt(0).all())

    def test_combined_matrix_preserves_missing_values_and_explicit_reasons(self):
        combined, report = summarize(self.k400_path, self.ssv2_path)
        self.assertEqual(len(combined), 30)
        self.assertEqual(combined.groupby("Dataset").size().to_dict(), {"k400": 15, "ssv2": 15})

        unsupported = combined.set_index(["Dataset", "ModelKey"]).loc[
            ("ssv2", "mvit-b-24-32x3")
        ]
        self.assertEqual(unsupported["ResultStatus"], "not_available")
        self.assertIn("MViTv2 is a different model", unsupported["AvailabilityReason"])
        self.assertTrue(unsupported[NUMERIC_OUTPUTS].isna().all())

        supported_missing = combined.set_index(["Dataset", "ModelKey"]).loc[
            ("ssv2", "uniformer-s")
        ]
        self.assertEqual(supported_missing["ResultStatus"], "available_not_measured")
        self.assertTrue(supported_missing[NUMERIC_OUTPUTS].isna().all())
        self.assertIn("never copies K400 values into SSV2", report)

    def test_accuracy_provenance_and_canonical_model_name_are_explicit(self):
        combined, _ = summarize(self.k400_path, self.ssv2_path)
        row = combined.set_index(["Dataset", "ModelKey"]).loc[("k400", "uniformer-b")]
        self.assertEqual(row["Model"], "UniFormer-B")
        self.assertEqual(row["PublishedTop1(%)"], 83.0)
        self.assertEqual(row["MeasuredTop1Local(%)"], 80.0)
        self.assertEqual(row["AccuracyDeltaLocalMinusPublished(pp)"], -3.0)
        self.assertIn("1,000 clips", row["MeasuredAccuracyScope"])
        self.assertIn("Canonical", row["PublishedAccuracyScope"])

    def test_ranks_are_within_dataset_and_missing_scores_are_unranked(self):
        combined, _ = summarize(self.k400_path, self.ssv2_path)
        indexed = combined.set_index(["Dataset", "ModelKey"])
        self.assertEqual(indexed.loc[("k400", "uniformer-b"), "RankNS-EWithinDataset"], 1)
        self.assertEqual(indexed.loc[("k400", "uniformer-s"), "RankNS-EWithinDataset"], 2)
        self.assertEqual(indexed.loc[("ssv2", "uniformer-b"), "RankNS-EWithinDataset"], 1)
        self.assertTrue(pd.isna(indexed.loc[("ssv2", "uniformer-s"), "RankNS-EWithinDataset"]))

    def test_partial_rows_remain_partial_without_metric_invention(self):
        partial = pd.read_csv(self.k400_path)
        partial.loc[0, "Energy(J)"] = float("nan")
        partial.loc[0, "NS#(1/8)"] = float("nan")
        partial.to_csv(self.k400_path, index=False)
        combined, _ = summarize(self.k400_path, self.ssv2_path)
        row = combined.set_index(["Dataset", "ModelKey"]).loc[("k400", "uniformer-b")]
        self.assertEqual(row["ResultStatus"], "partial_metrics")
        self.assertTrue(pd.isna(row["Energy(J/clip)"]))
        self.assertTrue(pd.isna(row["NS#(1/8)"]))
        self.assertTrue(pd.isna(row["RankNS#WithinDataset"]))

    def test_unsupported_ssv2_metrics_row_is_rejected_as_possible_proxy(self):
        row = metric_row(
            "mvit-v1-b",
            "MViT-B",
            "ssv2",
            60.0,
            59.0,
            ns_e=40.0,
            ns_m=41.0,
            ns_hash=35.0,
        )
        pd.DataFrame([row]).to_csv(self.ssv2_path, index=False)
        with self.assertRaisesRegex(ValueError, "possible proxy"):
            summarize(self.k400_path, self.ssv2_path)

    def test_explicit_clip_count_cannot_conflict_with_source(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            summarize(
                self.k400_path,
                self.ssv2_path,
                k400_clip_count=999,
            )

    def test_metric_accuracy_source_mismatch_is_rejected(self):
        frame = pd.read_csv(self.k400_path)
        frame.loc[0, "MetricAccuracy(%)"] = 83.0
        frame.to_csv(self.k400_path, index=False)
        with self.assertRaisesRegex(ValueError, "does not match"):
            summarize(self.k400_path, self.ssv2_path)

    def test_outputs_include_combined_csv_and_markdown_analysis(self):
        combined, report = summarize(self.k400_path, self.ssv2_path)
        csv_path, markdown_path = write_outputs(combined, report, self.root / "output")
        self.assertTrue(csv_path.is_file())
        self.assertTrue(markdown_path.is_file())
        roundtrip = pd.read_csv(csv_path)
        self.assertEqual(len(roundtrip), 30)
        text = markdown_path.read_text(encoding="utf-8")
        self.assertIn("Accuracy provenance", text)
        self.assertIn("Candidate and SSV2 availability matrix", text)
        self.assertIn("NetScore leaders", text)


if __name__ == "__main__":
    unittest.main()
