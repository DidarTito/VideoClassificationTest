"""Focused tests for the NetScore deployment variants."""

import unittest

from metrics import netscore_e, netscore_hash, netscore_m


class DeploymentNetScoreTests(unittest.TestCase):
    def test_matches_paper_vit_s_example(self):
        # ViT-S Full FT, CIFAR-100: A=89.7%, t=21.3 s, P=144.2 W,
        # and peak inference VRAM=1,084 MiB (paper Table V).
        self.assertAlmostEqual(netscore_e(89.7, 21.3, 144.2), 69.39, places=2)
        self.assertAlmostEqual(netscore_m(89.7, 1084), 70.52, places=2)
        self.assertAlmostEqual(
            netscore_hash(89.7, 1084, 21.3, 144.2), 61.81, places=2
        )

    def test_input_units_are_percent_seconds_watts_and_mib(self):
        score = netscore_e(89.7, 21_300 / 1000, 144.2)
        self.assertAlmostEqual(score, 69.3933355622)

        # The function deliberately does not reinterpret an accuracy fraction
        # or milliseconds; callers must convert to the documented paper units.
        self.assertAlmostEqual(netscore_e(0.897, 21.3, 144.2), score - 80)
        self.assertAlmostEqual(netscore_e(89.7, 21_300, 144.2), score - 7.5)

        peak_bytes = 1084 * 1024**2
        self.assertAlmostEqual(netscore_m(89.7, peak_bytes / 1024**2), 70.5241245163)

    def test_rejects_nonpositive_metric_inputs(self):
        calls = (
            (netscore_e, (0, 21.3, 144.2)),
            (netscore_e, (-89.7, 21.3, 144.2)),
            (netscore_e, (89.7, 0, 144.2)),
            (netscore_e, (89.7, 21.3, -144.2)),
            (netscore_m, (89.7, 0)),
            (netscore_m, (89.7, -1084)),
            (netscore_hash, (89.7, 1084, 0, 144.2)),
            (netscore_hash, (89.7, 1084, 21.3, 0)),
        )
        for function, args in calls:
            with self.subTest(function=function.__name__, args=args):
                with self.assertRaises(ValueError):
                    function(*args)

    def test_rejects_nonpositive_weight(self):
        for weight in (0, -1 / 8):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    netscore_e(89.7, 21.3, 144.2, w=weight)


if __name__ == "__main__":
    unittest.main()
