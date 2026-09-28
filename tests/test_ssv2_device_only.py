"""Device-only routing must never score K400 logits as SSV2 predictions."""
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest import mock
import math
import torch

import run_benchmark as runner
from tests.test_run_configuration import _config, _cpu_device_info


def test_device_only_loads_k400_and_never_scores_accuracy():
    class Model:
        num_classes = 400
        parameters = 1000
        name = "K400 test model"
        info = {}
        device = torch.device("cpu")

        def __call__(self, clip):
            return torch.zeros(1, 400)

    with tempfile.TemporaryDirectory() as tmp:
        config = replace(_config(Path(tmp), model_count=1), dataset="ssv2", mode="device-only")
        factory = mock.Mock(return_value=Model())
        with mock.patch.dict(runner.MODEL_REGISTRY, {"video-focalnet-t": factory}), \
             mock.patch.object(runner, "measured_accuracy", side_effect=AssertionError("must not score")), \
             mock.patch.object(runner, "netscore", side_effect=AssertionError("must not calculate NetScore")), \
             mock.patch.object(runner, "netscore_e", side_effect=AssertionError("must not calculate NS-E")), \
             mock.patch.object(runner, "netscore_m", side_effect=AssertionError("must not calculate NS-M")), \
             mock.patch.object(runner, "netscore_hash", side_effect=AssertionError("must not calculate NS#")), \
             mock.patch.object(runner, "_profile_device_gflops", return_value=(1.0, "test")):
            frame, errors = runner.benchmark_models(
                ["video-focalnet-t"], [torch.zeros(2, 3, 4, 4, dtype=torch.uint8)], "none",
                labels=[0], dataset="ssv2", device_info=_cpu_device_info(), config=config, warmup=0)
        assert not errors
        factory.assert_called_once_with(dataset="k400", device="cpu")
        row = frame.iloc[0]
        assert row.CheckpointProtocol == "K400_ON_SSV2_INPUT"
        assert not row.AccuracyValid
        assert row.ModelFrames == 8
        for field in ("MeasuredTop1", "Top5", "NetScore", "NS-E", "NS-M", "NS#"):
            assert math.isnan(row[field]), field


def test_device_only_never_loads_ground_truth():
    args = runner.parse_args(["--mode", "device-only", "--dataset", "ssv2", "--manifest",
                              "manifests/ssv2_1000_seed0.csv", "--num-clips", "1"])
    with mock.patch.object(runner, "read_sample_manifest"), \
         mock.patch.object(runner, "load_clip_set", return_value=([object()], [Path("1.webm")])), \
         mock.patch.object(runner, "load_ground_truth", side_effect=AssertionError("must not read labels")):
        clips, labels, _ = runner._load_dataset(args, args.manifest)
    assert len(clips) == 1
    assert labels is None


def test_true_ssv2_shape_validation_still_rejects_k400():
    import pytest
    with pytest.raises(ValueError, match="174"):
        runner._validate_logits(torch.zeros(1, 400), 174)


def test_top5_collection_excludes_warmups_and_keeps_ranked_indices():
    class Model:
        num_classes = 174
        device = torch.device("cpu")

        def __call__(self, clip):
            return torch.arange(174, dtype=torch.float32).reshape(1, 174)

    top5 = []
    clip = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
    result = runner.benchmark(Model(), [clip, clip], None, warmup=2, top5_predictions=top5)
    assert result[-1] == [173, 173]
    assert top5 == [[173, 172, 171, 170, 169]] * 2
