"""Measured K400 -> SSV2 linear probes on explicitly partial, disjoint data.

Strictly load each K400 checkpoint, freeze it, fit a new 174-class linear
head using train features, select ridge strength on validation features, and
evaluate once on the fixed 1000-clip manifest. Also score a seed-0 untrained
head using the same features. This is neither zero-shot nor full fine-tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import clip_to_input, load_clip_set, load_ssv2_annotation_map
from models import MODEL_REGISTRY
from training.checkpoint_utils import replace_classifier
from training.frozen_preflight import validate_frozen_model
from training.registry import model_spec

KEYS = ["dualformer-t", "video-focalnet-t", "videoswin-t", "mvit-v1-b-16x4",
        "video-focalnet-s", "mvit-v1-b-32x3", "dualformer-s", "videoswin-s",
        "zeroi2v-b16-8f", "dualformer-b-in21k", "omnivore-b-in21k"]
VIDEOS = ROOT / "datasets/ssv2/videos"
LABELS = ROOT / "datasets/ssv2/labels"
SPLITS = ROOT / "third_party/UniFormer/video_classification/data_list/sthv2"
TRAIN_ANN = SPLITS / "somesomev2_rgb_train_split.txt"
VAL_ANN = SPLITS / "somesomev2_rgb_validation_split.txt"
EVAL = ROOT / "manifests/ssv2_1000_seed0.csv"
EVAL_SHA = "c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02"
OUT = ROOT / "results/ssv2_head_transfer"
REPORT = ROOT / "reports/SSV2_HEAD_TRANSFER.md"
PROTOCOL = "K400_FROZEN_BACKBONE_SSV2_LINEAR_PROBE_4_PER_CLASS"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(value, encoding="utf-8", newline="\n")
    temp.replace(path)


def save_json(path, value):
    atomic(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, stream.getvalue())


def prepare():
    if sha(EVAL) != EVAL_SHA:
        raise RuntimeError("Evaluation manifest checksum changed")
    train = load_ssv2_annotation_map(TRAIN_ANN)
    val = load_ssv2_annotation_map(VAL_ANN)
    if set(train) & set(val):
        raise RuntimeError("Official train/validation overlap")
    with EVAL.open(newline="", encoding="utf-8") as stream:
        eval_ids = [Path(r["RelativePath"]).stem for r in csv.DictReader(stream)]
    assert len(eval_ids) == len(set(eval_ids)) == 1000
    assert set(eval_ids) <= set(val)
    available = {p.stem for p in VIDEOS.glob("*.webm")}
    assert set(eval_ids) <= available
    selected = {"eval": [(k, val[k]) for k in eval_ids]}
    for split, source, per_class in [("train", train, 4), ("select", val, 1)]:
        rng = random.Random(0)
        samples = []
        for label in range(174):
            pool = sorted(k for k, v in source.items()
                          if v == label and k in available and k not in eval_ids)
            rng.shuffle(pool)
            if len(pool) < per_class:
                raise RuntimeError(f"Too few available {split} clips for class {label}")
            samples.extend((k, label) for k in pool[:per_class])
        rng.shuffle(samples)
        selected[split] = samples
    for a, b in [("train", "select"), ("train", "eval"), ("select", "eval")]:
        assert not ({k for k, _ in selected[a]} & {k for k, _ in selected[b]})
    for split, samples in selected.items():
        rows = [dict(ClipIndex=i, RelativePath=f"{k}.webm", Label=y,
                     Dataset="ssv2", Seed=0, DecodedFrames=32, Resolution="224x224")
                for i, (k, y) in enumerate(samples)]
        path = OUT / f"{split}_manifest.csv"
        if path.exists():
            old = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
            assert [(r["RelativePath"], int(r["Label"])) for r in old] == [
                (r["RelativePath"], r["Label"]) for r in rows], "Saved selection changed"
        else:
            save_csv(path, rows)
    info = dict(protocol=PROTOCOL, seed=0, train_clips=696, selection_clips=174,
                eval_clips=1000, train_per_class=4, selection_per_class=1,
                available_train=sum(k in available for k in train),
                official_train=len(train), available_val=sum(k in available for k in val),
                official_val=len(val), split_overlap=0, classes=174,
                eval_sha256=EVAL_SHA,
                manifest_sha256={s: sha(OUT / f"{s}_manifest.csv") for s in selected},
                annotation_sha256={"train": sha(TRAIN_ANN), "validation": sha(VAL_ANN),
                                   "labels": sha(LABELS / "labels.json")},
                train_annotation=str(TRAIN_ANN.relative_to(ROOT)),
                val_annotation=str(VAL_ANN.relative_to(ROOT)),
                preprocessing="Existing benchmark: 32 uniform full-video frames, resize short side 224, center crop; exact adapter frame selection and normalization",
                ridge_grid=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                precision="FP32; TF32 disabled", batch_size=1)
    save_json(OUT / "protocol.json", info)
    print(json.dumps(info), flush=True)
    return selected


def clips_for(split):
    path = OUT / f"{split}_manifest.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    annotations = TRAIN_ANN if split == "train" else VAL_ANN
    allowed = set(load_ssv2_annotation_map(annotations))
    clips, paths = load_clip_set(VIDEOS, max_clips=len(rows), frames=32, size=224,
                                seed=0, manifest=path, allowed_stems=allowed)
    assert len(clips) == len(rows)
    assert [Path(p).name for p in paths] == [r["RelativePath"] for r in rows]
    return clips, torch.tensor([int(r["Label"]) for r in rows]), rows


def features(wrapper, head, split, directory, identity):
    path = directory / f"{split}_features.pt"
    if path.is_file():
        payload = torch.load(path, weights_only=True)
        assert payload["identity"] == identity
        print(f"Reusing verified {split} features", flush=True)
        return payload["x"], payload["y"]
    clips, labels, _ = clips_for(split)
    captured = []
    handle = head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    values = []
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for index, raw in enumerate(clips):
                captured.clear()
                output = wrapper(clip_to_input(raw))
                assert output.shape == (1, 400)
                assert len(captured) == 1 and captured[0].shape == (1, head.in_features)
                values.append(captured[0][0].float().cpu().clone())
                if (index + 1) % 50 == 0 or index + 1 == len(clips):
                    print(f"{split} {index + 1}/{len(clips)} in {time.monotonic()-start:.1f}s", flush=True)
    finally:
        handle.remove()
    result = torch.stack(values)
    assert torch.isfinite(result).all()
    torch.save(dict(x=result, y=labels, identity=identity), path)
    return result, labels


def score(logits, y):
    top = logits.topk(5, dim=-1).indices
    n1 = int((top[:, 0] == y).sum())
    n5 = int((top == y[:, None]).any(1).sum())
    return n1, n5, 100 * n1 / len(y), 100 * n5 / len(y)


def fit_head(x, y, vx, vy):
    # Standardization statistics and regression targets come only from train.
    x, vx = x.double(), vx.double()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    z = (x - mean) / scale
    targets = torch.nn.functional.one_hot(y, 174).double()
    intercept = targets.mean(0)
    centered = targets - intercept
    gram = z @ z.T
    eye = torch.eye(len(x), dtype=torch.float64)
    choices = []
    best = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        coef = z.T @ torch.linalg.solve(gram + alpha * eye, centered)
        weight = (coef.T / scale).float()
        bias = (intercept - mean @ weight.double().T).float()
        logits = vx.float() @ weight.T + bias
        correct, _, acc, _ = score(logits, vy)
        ce = float(torch.nn.functional.cross_entropy(logits, vy))
        choices.append(dict(alpha=alpha, selection_top1=acc, selection_cross_entropy=ce))
        # Best selection accuracy, then CE, then earlier grid entry. Never eval.
        rank = (correct, -ce)
        if best is None or rank > best[0]:
            best = (rank, weight, bias, alpha)
    return best[1], best[2], best[3], choices


def evaluate(wrapper, head, random_head, directory):
    from power import PynvmlLogger
    clips, labels, rows = clips_for("eval")
    captured = []
    hook = head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    times, powers, trained_logits, random_logits = [], [], [], []
    logger = PynvmlLogger(interval_ms=20)
    try:
        with torch.inference_mode():
            for _ in range(2):
                captured.clear()
                assert wrapper(clip_to_input(clips[0])).shape == (1, 174)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_all = time.monotonic()
            for start in range(0, len(clips), 8):
                inputs = [clip_to_input(clips[j]) for j in range(start, min(start+8, len(clips)))]
                logger.start()
                try:
                    for clip in inputs:
                        captured.clear()
                        torch.cuda.synchronize()
                        start_time = time.perf_counter()
                        logits = wrapper(clip)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - start_time) * 1000)
                        assert len(captured) == 1
                        trained_logits.append(logits[0].cpu().clone())
                        random_logits.append(random_head(captured[0].cpu())[0].clone())
                finally:
                    powers.extend(logger.stop())
                if len(times) % 80 == 0 or len(times) == len(clips):
                    print(f"eval {len(times)}/{len(clips)} in {time.monotonic()-start_all:.1f}s", flush=True)
    finally:
        hook.remove()
    assert len(times) == 1000 and powers and all(math.isfinite(v) and v > 0 for v in powers)
    logits, random_logits = torch.stack(trained_logits), torch.stack(random_logits)
    assert torch.isfinite(logits).all() and torch.isfinite(random_logits).all()
    torch.save(dict(logits=logits, untrained_logits=random_logits, labels=labels), directory / "eval_logits.pt")
    prediction_rows = []
    for i, r in enumerate(rows):
        prediction_rows.append(dict(ClipIndex=i, Video=r["RelativePath"], Label=int(labels[i]),
            Prediction=int(logits[i].argmax()), Top5=" ".join(map(str, logits[i].topk(5).indices.tolist())),
            UntrainedPrediction=int(random_logits[i].argmax()), LatencyMs=times[i]))
    save_csv(directory / "predictions.csv", prediction_rows)
    save_json(directory / "power_samples.json", powers)
    c1, c5, a1, a5 = score(logits, labels)
    r1, r5, ra1, ra5 = score(random_logits, labels)
    latency, power = statistics.mean(times), statistics.mean(powers)
    return dict(MeasuredTop1=a1, MeasuredTop5=a5, Top1Correct=c1, Top5Correct=c5,
                UntrainedHeadTop1=ra1, UntrainedHeadTop5=ra5,
                UntrainedTop1Correct=r1, UntrainedTop5Correct=r5,
                EvaluatedClips=len(labels), LatencyMs=latency,
                LatencyStdMs=statistics.stdev(times), FPS=1000/latency,
                AveragePowerW=power, EnergyPerClipJ=power*latency/1000,
                TotalEnergyJ=power*sum(times)/1000, TotalTimedSeconds=sum(times)/1000,
                PeakVRAMMiB=torch.cuda.max_memory_allocated()/2**20, PowerSamples=len(powers))


def run_model(key, smoke=False):
    from run_benchmark import _profile_device_gflops
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    spec = model_spec(key)
    directory = OUT / key
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol=sha(OUT / "protocol.json"), source=spec["checkpoint_sha256"])
    result_path = directory / "result.json"
    if not smoke and result_path.exists():
        existing = json.loads(result_path.read_text())
        assert existing["Identity"] == identity
        print(f"Already completed {key}", flush=True)
        return
    print(f"Loading exact K400 checkpoint: {key}", flush=True)
    wrapper = MODEL_REGISTRY[key](device="cuda", dataset="k400")
    preflight = validate_frozen_model(key, wrapper, dataset="k400", checkpoint=ROOT/spec["k400_checkpoint"])
    wrapper.model.eval().requires_grad_(False)
    head_path = spec["classifier"]["path"]
    old_head = wrapper.model.get_submodule(head_path)
    assert isinstance(old_head, torch.nn.Linear) and old_head.out_features == 400
    if smoke:
        clips, _, _ = clips_for("eval")
        with torch.inference_mode():
            assert wrapper(clip_to_input(clips[0])).shape == (1, 400)
        torch.manual_seed(0)
        head = replace_classifier(wrapper.model, head_path).to("cuda").eval()
        wrapper.dataset = "ssv2"
        with torch.inference_mode():
            output = wrapper(clip_to_input(clips[0]))
        assert output.shape == (1, 174) and torch.isfinite(output).all()
        print(f"SMOKE PASSED {key}: strict K400 -> 174-class forward", flush=True)
        return
    x, y = features(wrapper, old_head, "train", directory, identity)
    vx, vy = features(wrapper, old_head, "select", directory, identity)
    weight, bias, alpha, choices = fit_head(x, y, vx, vy)
    save_json(directory / "head_selection.json", choices)
    torch.manual_seed(0)
    head = replace_classifier(wrapper.model, head_path).to("cuda").eval()
    random_head = torch.nn.Linear(head.in_features, 174).eval()
    random_head.load_state_dict(head.state_dict())
    del old_head
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    wrapper.dataset = "ssv2"
    wrapper.model.eval().requires_grad_(False)
    checkpoint = directory / "ssv2_linear_head.pt"
    torch.save(dict(state_dict={k: v.cpu() for k, v in head.state_dict().items()},
                    classifier_path=head_path, model=key, source=spec["k400_checkpoint"],
                    identity=identity, alpha=alpha, protocol=PROTOCOL), checkpoint)
    print(f"Head selected with alpha={alpha}; now evaluating fixed 1000 clips", flush=True)
    measured = evaluate(wrapper, head, random_head, directory)
    clips, _, _ = clips_for("eval")
    gflops, flop_notes = _profile_device_gflops(wrapper, clips[0])
    from metrics import netscore, netscore_e, netscore_m, netscore_hash
    a, p = measured["MeasuredTop1"], wrapper.parameters/1e6
    t, power, mem = measured["TotalTimedSeconds"], measured["AveragePowerW"], measured["PeakVRAMMiB"]
    # A measured zero is retained; log scores have an explicit mathematical limit.
    scores = (dict(NetScore=netscore(a,p,gflops), NSE=netscore_e(a,t,power),
                   NSM=netscore_m(a,mem), NSJoint=netscore_hash(a,mem,t,power))
              if a > 0 else dict(NetScore="-Infinity", NSE="-Infinity", NSM="-Infinity", NSJoint="-Infinity"))
    result = dict(Model=key, Protocol=PROTOCOL, **measured, GFLOPs=gflops,
                  ParamsM=p, **scores, CheckpointSource=spec["k400_checkpoint"],
                  CheckpointSHA256=preflight["checkpoint_sha256"], HeadCheckpointSHA256=sha(checkpoint),
                  HeadCheckpoint=str(checkpoint.relative_to(ROOT)), TrainClips=len(y),
                  SelectionClips=len(vy), RidgeAlpha=alpha, Status="COMPLETED",
                  AccuracyValid=True, ClassifierClasses=174,
                  AccuracyMethod="Supervised SSV2 linear probe; frozen K400 backbone; partial train subset",
                  ModelFrames=spec["input"]["num_frames"], DecodedFrames=32,
                  GPU=torch.cuda.get_device_name(0), Precision="fp32", Seed=0,
                  TorchVersion=str(torch.__version__), CUDAVersion=torch.version.cuda,
                  Pretraining=spec["pretraining"], FLOPNotes=flop_notes, Identity=identity,
                  FinishedUTC=datetime.now(timezone.utc).isoformat())
    save_json(result_path, result)
    print(json.dumps(result), flush=True)


def report():
    rows = []
    for key in KEYS:
        path = OUT / key / "result.json"
        if path.exists():
            row = json.loads(path.read_text())
            row.pop("Identity")
            rows.append(row)
    if not rows:
        return
    save_csv(OUT / "results.csv", rows)
    lines = ["# K400 to SSV2: measured replacement-head baselines", "",
        f"Completed {len(rows)}/11 models on the fixed 1000-clip evaluation manifest.", "",
        "The trained column is a supervised linear probe: frozen K400 backbone, new 174-class head, "
        "696 training clips (4/class), and 174 separate validation clips (1/class) for ridge selection. "
        "It is not full fine-tuning or zero-shot. The untrained head uses fixed seed 0, with no SSV2 training.", "",
        "All three subsets are disjoint. No evaluation labels enter fitting or head selection. "
        "The existing six released-checkpoint results are a different protocol and are not rerun here.", "",
        "RTX 3050 Laptop GPU; batch 1; FP32 with TF32 disabled; 2 warmups; 20 ms NVML sampling. "
        "GFLOPs use fvcore's one-MAC convention and omit unsupported operations listed in each result. "
        "Latency includes adapter preprocessing/transfer/forward and excludes decoding and CPU input conversion. "
        "Power is sampled over inference chunks; energy is average board power times timed inference duration. "
        "Per-clip logits, predictions, latencies, and power samples are retained.", "",
        "| Model | Untrained head Top-1 (%) | Linear probe Top-1 (%) | Top-5 (%) | GFLOPs | NetScore | Latency (ms) | Power (W) | J/clip | J/1000 | Peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        vals = [r["Model"]] + [f"{r[k]:.3f}" if isinstance(r[k],(float,int)) else str(r[k]) for k in
                ["UntrainedHeadTop1","MeasuredTop1","MeasuredTop5","GFLOPs","NetScore","LatencyMs",
                 "AveragePowerW","EnergyPerClipJ","TotalEnergyJ","PeakVRAMMiB"]]
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "DualFormer-B uses the available IN1K -> K400 Base release; its historical registry key "
              "contains `in21k` but that is not the checkpoint provenance.", "",
              "Resume: `venv/Scripts/python.exe -u scripts/run_ssv2_head_transfer.py --run`", ""]
    atomic(REPORT, "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model", choices=KEYS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.model:
        run_model(args.model, args.smoke)
        return
    prepare()
    if not (args.run or args.smoke):
        return
    failed = []
    for key in KEYS:
        log = OUT / key / ("smoke.log" if args.smoke else "run.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable,"-u",str(Path(__file__).resolve()),"--model",key]
        if args.smoke:
            command.append("--smoke")
        with log.open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = child.wait()
        if code:
            failed.append(key)
            print(f"FAILED {key}; see {log}", flush=True)
        if not args.smoke:
            report()
    if failed:
        raise SystemExit(f"Failed models: {failed}")


if __name__ == "__main__":
    main()
