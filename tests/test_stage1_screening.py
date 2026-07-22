"""Regression tests for the canonical Stage 1 screening figure data."""
from pathlib import Path
import unittest

import numpy as np

from metrics import netscore
from scripts.plot_stage1_screening import (
    ACCURACY_THRESHOLD,
    DEFAULT_INPUT,
    EXPECTED_CANDIDATES,
    EXPECTED_SELECTED,
    GFLOPS_THRESHOLD,
    PARAMS_THRESHOLD_M,
    create_figure,
    load_candidates,
)


class Stage1CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_candidates(DEFAULT_INPUT)

    def test_exact_candidate_and_selected_counts(self):
        self.assertEqual(len(self.frame), EXPECTED_CANDIDATES)
        self.assertEqual(int(self.frame["selected_stage2"].sum()), EXPECTED_SELECTED)

    def test_selected_rows_obey_all_three_thresholds(self):
        selected = self.frame[self.frame["selected_stage2"]]
        self.assertTrue((selected["top1_published_pct"] >= ACCURACY_THRESHOLD).all())
        self.assertTrue((selected["gflops"] <= GFLOPS_THRESHOLD).all())
        self.assertTrue((selected["params_m"] < PARAMS_THRESHOLD_M).all())

    def test_netscores_are_computed_by_the_shared_metric(self):
        expected = np.array(
            [
                round(netscore(a, p, f), 2)
                for a, p, f in self.frame[
                    ["top1_published_pct", "params_m", "gflops"]
                ].itertuples(index=False, name=None)
            ]
        )
        np.testing.assert_array_equal(expected, self.frame["net_score"].to_numpy())

    def test_known_legacy_label_is_corrected(self):
        uniformer_b = self.frame.set_index("model").loc["UniFormer-B"]
        self.assertEqual(uniformer_b["net_score"], 35.61)
        self.assertNotEqual(uniformer_b["net_score"], 48.28)

    def test_dualformer_coincident_rows_are_both_preserved(self):
        rows = self.frame[self.frame["model"].isin(["DualFormer-S", "DualFormer-B"])]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows["gflops"].nunique(), 1)
        self.assertEqual(rows["top1_published_pct"].nunique(), 1)

    def test_figure_contains_all_model_labels(self):
        figure = create_figure(self.frame)
        try:
            labels = {
                text.get_text().split("\n", 1)[0]
                for text in figure.axes[0].texts
                if "\nNS-" in text.get_text()
            }
            self.assertEqual(labels, set(self.frame["model"]))
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
