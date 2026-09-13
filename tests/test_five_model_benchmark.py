"""Focused checks for final SSV2 routing and progress-measurement separation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import run_benchmark
from scripts import benchmark_finetuned_ssv2 as final_ssv2
from tests.test_run_configuration import _config, _cpu_device_info


class FinalCheckpointTests(unittest.TestCase):
    def test_only_complete_non_smoke_best_checkpoint_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs/ssv2/uniformer-s/run-1"
            run.mkdir(parents=True)
            torch.save({
                "model_key": "uniformer-s", "model": {"raw.head.weight": torch.zeros(174, 4)},
                "epoch": 2, "best_top1": 50.0,
            }, run / "best.pth")
            (run / "complete.json").write_text(json.dumps({"status": "completed"}))
            (run / "best_metadata.json").write_text(json.dumps({
                "selection_rule": "highest_val_top1_then_earlier_epoch", "epoch": 2, "top1": 50.0
            }))
            status = run / "protocol_status.json"
            status.write_text(json.dumps({"final_result": False, "protocol_deviation": True}))
            with mock.patch.object(final_ssv2, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "smoke or partial"):
                    final_ssv2.validate_final_checkpoint("uniformer-s", run / "best.pth")
                status.write_text(json.dumps({"final_result": True, "protocol_deviation": False}))
                self.assertEqual(len(final_ssv2.validate_final_checkpoint("uniformer-s", run / "best.pth")), 64)
                with self.assertRaisesRegex(ValueError, "expected runs/ssv2"):
                    final_ssv2.validate_final_checkpoint("videomae-b", run / "best.pth")

    def test_final_factory_runs_an_unavailable_released_ssv2_pair(self):
        class Model:
            num_classes = 174
            accuracy = float("nan")
            parameters = 1000
            gflops = 1.0
            name = "Final tiny"
            device = torch.device("cpu")
            info = {}

            def __call__(self, _clip):
                return torch.zeros(1, 174)

        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary), model_count=1)
            config = run_benchmark.ExperimentConfig(**{
                **vars(config), "dataset": "ssv2", "power_kind": "none"
            })
            clip = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
            with mock.patch("run_benchmark.torch.cuda.is_available", return_value=False):
                frame, errors = run_benchmark.benchmark_models(
                    ["video-focalnet-t"], [clip], "none", labels=[0],
                    dataset="ssv2", device_info=_cpu_device_info(), config=config,
                    warmup=0,
                    final_ssv2_factories={"video-focalnet-t": Model},
                    final_ssv2_checkpoints={"video-focalnet-t": Path(temporary) / "best.pth"},
                )
        self.assertFalse(errors)
        self.assertEqual(frame.iloc[0]["Status"], "completed")
        self.assertEqual(frame.iloc[0]["MeasuredTop1"], 100.0)
        self.assertEqual(frame.iloc[0]["ModelFrames"], 8)


class ProgressScopeTests(unittest.TestCase):
    def test_progress_updates_after_power_sampling_stops(self):
        active = False
        updates = []

        class Logger:
            def start(self):
                nonlocal active
                active = True

            def stop(self):
                nonlocal active
                active = False
                return [10.0]

        class Bar:
            def __init__(self, **_kwargs):
                pass

            def update(self, count):
                self.check(count)

            def close(self):
                self.check(0)

            def check(self, count):
                self_test.assertFalse(active)
                updates.append(count)

        self_test = self
        clip = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
        with mock.patch("run_benchmark.tqdm", Bar), mock.patch("run_benchmark.time.sleep"):
            with mock.patch("run_benchmark.torch.cuda.is_available", return_value=False):
                run_benchmark.benchmark(lambda _: torch.zeros(1, 400), [clip, clip], Logger(), warmup=0, chunk_size=1)
        self.assertEqual(updates, [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
