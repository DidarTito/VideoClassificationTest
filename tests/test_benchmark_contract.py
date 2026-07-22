"""Tests for inference-output validation and measurement cleanup."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import sys
import unittest
from unittest import mock

import torch

from run_benchmark import _total_energy_joules, _validate_logits, benchmark, main


class LogitValidationTests(unittest.TestCase):
    def test_accepts_one_finite_k400_vector(self):
        output = torch.zeros(1, 400)
        self.assertIs(_validate_logits(output), output)

    def test_accepts_one_finite_ssv2_vector(self):
        output = torch.zeros(1, 174)
        self.assertIs(_validate_logits(output, 174), output)

    def test_rejects_non_tensor(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            _validate_logits([[0.0] * 400])

    def test_rejects_wrong_shape(self):
        for output in (torch.zeros(400), torch.zeros(1, 399), torch.zeros(2, 400)):
            with self.subTest(shape=tuple(output.shape)):
                with self.assertRaisesRegex(ValueError, r"shape \(1, 400\)"):
                    _validate_logits(output)

    def test_rejects_nonfinite_values(self):
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            output = torch.zeros(1, 400)
            output[0, 0] = bad_value
            with self.subTest(value=bad_value):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    _validate_logits(output)


class EnergyUnitTests(unittest.TestCase):
    def test_total_millijoules_are_converted_to_joules(self):
        # 50 W for 20 ms is 1000 mJ (1 J) per clip.
        self.assertEqual(_total_energy_joules(50.0 * 20.0, 10), 10.0)


class SSV2CliValidationTests(unittest.TestCase):
    def test_directory_mode_requires_annotations_before_loading_clips(self):
        stderr = io.StringIO()
        argv = [
            "run_benchmark.py",
            "--videos-dir",
            "unused-videos",
            "--dataset",
            "ssv2",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("run_benchmark.load_clip_set") as load_clip_set_mock:
                with redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        main()

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--annotations is required", stderr.getvalue())
        load_clip_set_mock.assert_not_called()


class _FakePowerLogger:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1
        return [10.0]


class PowerLoggerCleanupTests(unittest.TestCase):
    def test_logger_is_stopped_when_model_output_validation_fails(self):
        logger = _FakePowerLogger()

        def wrong_shape_model(_clip):
            return torch.zeros(1, 399)

        clip = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
        with mock.patch("run_benchmark.time.sleep"):
            with mock.patch("run_benchmark.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, r"shape \(1, 400\)"):
                    benchmark(wrong_shape_model, [clip], logger, warmup=0)

        self.assertEqual(logger.start_calls, 1)
        self.assertEqual(logger.stop_calls, 1)

    def test_plain_list_is_processed_in_bounded_chunks(self):
        logger = _FakePowerLogger()
        model_calls = []

        def valid_model(clip):
            model_calls.append(clip)
            return torch.zeros(1, 400)

        clips = [torch.zeros(2, 3, 4, 4, dtype=torch.uint8) for _ in range(5)]
        with mock.patch("run_benchmark.time.sleep"):
            with mock.patch("run_benchmark.torch.cuda.is_available", return_value=False):
                _latency, _std, power, samples, predictions = benchmark(
                    valid_model,
                    clips,
                    logger,
                    warmup=0,
                    chunk_size=2,
                )

        self.assertEqual(len(model_calls), 5)
        self.assertEqual(predictions, [0, 0, 0, 0, 0])
        self.assertEqual(logger.start_calls, 3)
        self.assertEqual(logger.stop_calls, 3)
        self.assertEqual(samples, 3)
        self.assertEqual(power, 10.0)

    def test_logger_is_stopped_once_after_success(self):
        logger = _FakePowerLogger()

        def valid_model(_clip):
            return torch.zeros(1, 400)

        clip = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
        with mock.patch("run_benchmark.time.sleep"):
            with mock.patch("run_benchmark.torch.cuda.is_available", return_value=False):
                latency, _std, power, samples, predictions = benchmark(
                    valid_model, [clip], logger, warmup=0
                )

        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(power, 10.0)
        self.assertEqual(samples, 1)
        self.assertEqual(predictions, [0])
        self.assertEqual(logger.start_calls, 1)
        self.assertEqual(logger.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
