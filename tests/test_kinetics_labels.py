import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.kinetics_labels import kinetics400_classes


class KineticsLabelTests(unittest.TestCase):
    def test_hidden_clip_cache_is_not_a_401st_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "datasets" / "kinetics400"
            dataset.mkdir(parents=True)
            for index in range(400):
                (dataset / f"class_{index:03d}").mkdir()
            (dataset / ".clip_cache_v4_32f_224px_1000c_seed0_all").mkdir()

            with patch("models.kinetics_labels.ROOT", root):
                labels = kinetics400_classes()

        self.assertEqual(len(labels), 400)
        self.assertEqual(labels[0], "class_000")
        self.assertEqual(labels[-1], "class_399")

    def test_visible_incomplete_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "datasets" / "kinetics400"
            dataset.mkdir(parents=True)
            (dataset / "only_one_class").mkdir()

            with patch("models.kinetics_labels.ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "exactly 400"):
                    kinetics400_classes()


if __name__ == "__main__":
    unittest.main()
