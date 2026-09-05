from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

import run_benchmark
from run_benchmark import (
    CANONICAL_RESULT_COLUMNS,
    ExperimentConfig,
    ResultStore,
    _output_csv,
    benchmark_models,
    parse_args,
    parse_model_keys,
)
from utils.device import DeviceInfo


def _cpu_device_info() -> DeviceInfo:
    return DeviceInfo(
        torch_device=torch.device("cpu"),
        type="cpu",
        index=None,
        name="CPU",
        total_vram_mib=None,
        compute_capability=None,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        driver_version=None,
        precision="fp32",
    )


def _config(root: Path, *, model_count: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        dataset="k400",
        dataset_root=root / "videos",
        checkpoint_dir=root / "checkpoints",
        manifest=None,
        output_csv=root / "results" / "device_metrics.csv",
        num_clips=model_count,
        seed=0,
        batch_size=1,
        precision="fp32",
        decoded_frames=2,
        resolution=4,
        warmup=0,
        power_kind="auto",
        power_sample_ms=20,
        split_name=None,
        annotations=None,
        use_cache=False,
        mode="smoke",
        cooldown_temp=None,
        cooldown_timeout=0,
        repeats=1,
    )


class _SuccessfulModel:
    num_classes = 400
    accuracy = 80.0
    parameters = 1_000
    gflops = 1.0
    name = "Successful fake model"
    device = torch.device("cpu")
    info = {"checkpoint": "adapter-only-name.pth"}

    def __call__(self, clip):
        logits = torch.zeros((1, self.num_classes), dtype=torch.float32)
        logits[0, 0] = 1.0
        return logits


class _FailingModel(_SuccessfulModel):
    name = "Failing fake model"

    def __call__(self, clip):
        raise RuntimeError("intentional inference failure")


class CliModelSelectionTests(unittest.TestCase):
    def test_all_and_comma_separated_models_follow_cli_contract(self):
        all_args = parse_args(["--dataset", "k400", "--models", "all"])
        self.assertEqual(
            all_args.model_keys,
            parse_model_keys(["all"], "k400"),
        )
        self.assertNotIn("vivit-s", all_args.model_keys)
        self.assertNotIn("svt-b", all_args.model_keys)

        comma_args = parse_args(
            [
                "--dataset",
                "k400",
                "--models",
                "uniformer-s,video-focalnet-t",
            ]
        )
        self.assertEqual(
            comma_args.model_keys,
            ["uniformer-s", "video-focalnet-t"],
        )

    def test_omitted_models_use_full_default_or_smoke_default(self):
        full_args = parse_args(["--dataset", "k400"])
        self.assertEqual(
            full_args.model_keys,
            parse_model_keys(None, "k400", smoke_test=False),
        )
        self.assertEqual(full_args.num_clips, 1000)

        smoke_args = parse_args(["--dataset", "k400", "--smoke-test"])
        self.assertEqual(smoke_args.model_keys, ["uniformer-s"])
        self.assertEqual(smoke_args.num_clips, 5)

    def test_non_unit_batch_size_is_rejected(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            parse_args(["--batch-size", "2"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --batch-size 1", stderr.getvalue())

    def test_legacy_single_video_defaults_clip_count_to_runs(self):
        args = parse_args(["--video", "example.mp4", "--runs", "7"])
        self.assertEqual(args.num_clips, 7)

    def test_canonical_result_schema_contains_required_portability_fields(self):
        required = {
            "Device",
            "DeviceName",
            "DeviceIndex",
            "TotalVRAM",
            "ComputeCapability",
            "TorchVersion",
            "CUDAVersion",
            "DriverVersion",
            "Dataset",
            "Model",
            "Checkpoint",
            "InputProtocol",
            "Frames",
            "Resolution",
            "BatchSize",
            "Precision",
            "Seed",
            "ClipCount",
            "ManifestSHA256",
            "PublishedTop1",
            "MeasuredTop1",
            "Params",
            "GFLOPs",
            "LatencyMean",
            "LatencyStd",
            "FPS",
            "AvgPower",
            "PowerSamples",
            "MeasurementBackend",
            "EnergyPerClip",
            "PeakVRAM",
            "GPUStartTemp",
            "GPUEndTemp",
            "NS-E",
            "NS-M",
            "NS#",
        }
        self.assertTrue(required.issubset(CANONICAL_RESULT_COLUMNS))


class DynamicOutputPathTests(unittest.TestCase):
    def test_output_path_uses_sanitized_device_dataset_mode_count_and_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "results"
            args = SimpleNamespace(
                output=None,
                output_dir=output_root,
                dataset="k400",
                num_clips=100,
                seed=7,
            )
            device_info = DeviceInfo(
                torch_device=torch.device("cuda:1"),
                type="cuda",
                index=1,
                name="NVIDIA GeForce RTX 5070 / Laptop",
                total_vram_mib=8_192.0,
                compute_capability="12.0",
                torch_version="test",
                cuda_version="test",
                driver_version="test",
                precision="fp32",
            )

            output = _output_csv(args, device_info, "quick")

            self.assertEqual(
                output,
                output_root.resolve()
                / "NVIDIA_GeForce_RTX_5070_Laptop"
                / "k400"
                / "quick_100_seed7"
                / "device_metrics.csv",
            )


class ResultStoreTests(unittest.TestCase):
    def test_resume_skips_success_and_retries_then_replaces_failed_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "device_metrics.csv"
            store = ResultStore(csv_path)
            store.upsert(
                {"ModelKey": "uniformer-s", "Status": "completed", "Error": ""}
            )
            store.upsert(
                {
                    "ModelKey": "uniformer-b",
                    "Status": "failed",
                    "Error": "RuntimeError: out of memory",
                }
            )

            with self.assertRaises(FileExistsError):
                ResultStore(csv_path)

            resumed = ResultStore(csv_path, resume=True)
            self.assertEqual(resumed.successful_keys, {"uniformer-s"})
            requested = ["uniformer-s", "uniformer-b"]
            self.assertEqual(
                [key for key in requested if key not in resumed.successful_keys],
                ["uniformer-b"],
            )

            resumed.upsert(
                {"ModelKey": "uniformer-b", "Status": "completed", "Error": ""}
            )
            reloaded = ResultStore(csv_path, resume=True)
            self.assertEqual(
                reloaded.successful_keys,
                {"uniformer-s", "uniformer-b"},
            )
            rows = [
                row
                for row in reloaded.rows
                if str(row.get("ModelKey")) == "uniformer-b"
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Status"], "completed")
            self.assertTrue(math.isnan(float(rows[0]["Error"])))

    def test_resume_rejects_a_changed_experiment_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "device_metrics.csv"
            expected = {
                column: "same" for column in run_benchmark.RESUME_IDENTITY_COLUMNS
            }
            expected["DriverVersion"] = "576.80"
            expected["AnnotationFile"] = ""
            expected.update({"ModelKey": "uniformer-s", "Status": "completed"})
            store = ResultStore(csv_path)
            store.upsert(expected)

            resumed = ResultStore(csv_path, resume=True)
            changed = dict(expected)
            changed["Precision"] = "fp16"
            with self.assertRaisesRegex(ValueError, "different or legacy run"):
                resumed.validate_resume({"uniformer-s": changed})

            resumed.validate_resume({"uniformer-s": expected})


class ModelIsolationTests(unittest.TestCase):
    def test_inference_failure_is_recorded_and_callback_runs_before_next_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clips = [torch.zeros((2, 3, 4, 4), dtype=torch.uint8)]
            callbacks = []
            factories = {
                "uniformer-s": mock.Mock(return_value=_FailingModel()),
                "uniformer-b": mock.Mock(return_value=_SuccessfulModel()),
            }

            with (
                mock.patch.dict(run_benchmark.MODEL_REGISTRY, factories),
                mock.patch("run_benchmark.make_power_logger") as make_logger,
                redirect_stdout(io.StringIO()),
            ):
                frame, errors = benchmark_models(
                    ["uniformer-s", "uniformer-b"],
                    clips,
                    labels=[0],
                    device_info=_cpu_device_info(),
                    config=_config(root),
                    warmup=0,
                    result_callback=lambda row: callbacks.append(dict(row)),
                )

            make_logger.assert_not_called()
            self.assertEqual(frame["ModelKey"].tolist(), ["uniformer-s", "uniformer-b"])
            self.assertEqual(frame["Status"].tolist(), ["failed", "completed"])
            self.assertIn("intentional inference failure", frame.iloc[0]["Error"])
            self.assertEqual(set(errors), {"uniformer-s"})
            self.assertEqual(
                [(row["ModelKey"], row["Status"]) for row in callbacks],
                [("uniformer-s", "failed"), ("uniformer-b", "completed")],
            )

    def test_cpu_uses_no_power_logger_and_records_gpu_metrics_as_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clips = [torch.zeros((2, 3, 4, 4), dtype=torch.uint8)]

            with (
                mock.patch.dict(
                    run_benchmark.MODEL_REGISTRY,
                    {"uniformer-s": mock.Mock(return_value=_SuccessfulModel())},
                ),
                mock.patch("run_benchmark.make_power_logger") as make_logger,
                redirect_stdout(io.StringIO()),
            ):
                frame, errors = benchmark_models(
                    ["uniformer-s"],
                    clips,
                    labels=[0],
                    device_info=_cpu_device_info(),
                    config=_config(root),
                    warmup=0,
                )

            make_logger.assert_not_called()
            self.assertFalse(errors)
            row = frame.iloc[0]
            self.assertEqual(row["Status"], "completed")
            self.assertEqual(row["PowerSamples"], 0)
            self.assertEqual(row["MeasurementBackend"], "unavailable")
            self.assertEqual(row["PeakVRAMMethod"], "unavailable on CPU")
            self.assertEqual(
                row["Checkpoint"],
                str(
                    root
                    / "checkpoints"
                    / "uniformer"
                    / "uniformer_small_k400_16x4.pth"
                ),
            )
            for column in (
                "AvgPower",
                "EnergyPerClip",
                "TotalEnergy",
                "PeakVRAM",
                "GPUStartTemp",
                "GPUEndTemp",
                "NS-E",
                "NS-M",
                "NS#",
            ):
                self.assertTrue(math.isnan(float(row[column])), column)
            self.assertTrue(math.isfinite(float(row["LatencyMean"])))
            self.assertTrue(math.isfinite(float(row["TotalTime"])))


if __name__ == "__main__":
    unittest.main()
