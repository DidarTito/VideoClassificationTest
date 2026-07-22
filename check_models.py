#!/usr/bin/env python3
"""Read-only preflight for the 12 requested video-classification models.

This script deliberately does not import ``models`` or ``torch``.  Importing the
registry can download weights and allocate a model, while a preflight should
only answer whether the exact source/checkpoint assets needed for the requested
published row are present.

Exit status is 0 when every requested model is asset-ready, and 1 when one or
more exact assets are blocked.  ``--allow-missing`` keeps the report unchanged
but returns 0 so a best-effort benchmark can be launched.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RequestedModel:
    key: str
    name: str
    params_m: float
    gflops: float
    top1: float
    inspect: Callable[["InspectionContext", "RequestedModel"], "ModelResult"]


@dataclass(frozen=True)
class Asset:
    path: Path
    kind: str
    compatibility: str = "exact"
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "compatibility": self.compatibility,
            "path": str(self.path),
            "detail": self.detail,
        }


@dataclass
class ModelResult:
    spec: RequestedModel
    assets: list[Asset] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.spec.key,
            "name": self.spec.name,
            "published": {
                "params_m": self.spec.params_m,
                "gflops": self.spec.gflops,
                "top1_percent": self.spec.top1,
            },
            "status": "ready" if self.ready else "blocked",
            "assets": [asset.as_dict() for asset in self.assets],
            "blockers": self.blockers,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class InspectionContext:
    root: Path
    torch_checkpoint_dirs: tuple[Path, ...]
    registry_keys: frozenset[str]

    def workspace(self, relative: str) -> Path:
        return self.root / Path(relative)


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = os.path.normcase(os.path.abspath(os.fspath(path)))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return tuple(result)


def _torch_checkpoint_dirs(root: Path) -> tuple[Path, ...]:
    """Return likely torch.hub checkpoint directories without creating them."""
    candidates: list[Path] = [root / "checkpoints" / "mvit"]
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        candidates.append(Path(torch_home).expanduser() / "hub" / "checkpoints")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(Path(xdg_cache).expanduser() / "torch" / "hub" / "checkpoints")
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "torch" / "hub" / "checkpoints")
    return _unique_paths(candidates)


def _files(directories: Iterable[Path], recursive: bool = False) -> list[Path]:
    found: list[Path] = []
    for directory in _unique_paths(directories):
        if not directory.is_dir():
            continue
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        try:
            found.extend(path for path in iterator if path.is_file())
        except OSError:
            # A cache directory may disappear or become inaccessible mid-scan.
            continue
    return list(_unique_paths(found))


def _matches(files: Iterable[Path], patterns: Sequence[str]) -> list[Path]:
    lowered = tuple(pattern.lower() for pattern in patterns)
    return [
        path
        for path in files
        if any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in lowered)
    ]


def _existing_files(root: Path, relatives: Iterable[str]) -> list[Path]:
    return [path for relative in relatives if (path := root / relative).is_file()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(directories: Iterable[Path]) -> list[Path]:
    return [path for path in _files(directories, recursive=True) if path.suffix.lower() == ".py"]


def _registry_keys(init_file: Path) -> frozenset[str]:
    """Read active MODEL_REGISTRY dict/update keys without importing the package."""
    try:
        tree = ast.parse(init_file.read_text(encoding="utf-8"), filename=str(init_file))
    except (OSError, SyntaxError, UnicodeError):
        return frozenset()

    keys: set[str] = set()

    def add_dict(node: ast.AST) -> None:
        if not isinstance(node, ast.Dict):
            return
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "MODEL_REGISTRY" for target in targets):
                add_dict(node.value)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "update"
                and isinstance(func.value, ast.Name)
                and func.value.id == "MODEL_REGISTRY"
                and node.value.args
            ):
                add_dict(node.value.args[0])
    return frozenset(keys)


def _base_result(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = ModelResult(spec)
    if spec.key not in ctx.registry_keys:
        result.blockers.append(f"Registry key {spec.key!r} is not active in models/__init__.py.")
    return result


def _add_files(result: ModelResult, paths: Iterable[Path], kind: str, detail: str,
               compatibility: str = "exact") -> None:
    for path in _unique_paths(paths):
        result.assets.append(Asset(path, kind, compatibility, detail))


def _uniformer_s(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    result.notes.append(
        "The official 16x8 checkpoint is 21.4M parameters and 167 GFLOPs; "
        "the requested row's 42 GFLOPs does not describe this exact recipe."
    )
    checkpoint = ctx.workspace("checkpoints/uniformer/uniformer_small_k400_16x8.pth")
    incompatible = ctx.workspace("checkpoints/uniformer/uniformer_small_k400_8x8.pth")
    source = ctx.workspace(
        "third_party/UniFormer/video_classification/slowfast/models/uniformer.py"
    )
    expected_sha = "d5fd7b0c49ee6a5422ef5d0c884d962c742003bfbd900747485eb99fa269d0db"
    if checkpoint.is_file() and _sha256(checkpoint) == expected_sha:
        _add_files(result, [checkpoint], "checkpoint", "Kinetics-400 UniFormer-S 16x8")
    elif checkpoint.is_file():
        _add_files(
            result,
            [checkpoint],
            "checkpoint",
            "checksum mismatch/incomplete download",
            "incompatible",
        )
        result.blockers.append("UniFormer-S checkpoint failed its official SHA256 check.")
    else:
        result.blockers.append(
            "Missing the requested 80.8% UniFormer-S Kinetics-400 checkpoint "
            "(uniformer_small_k400_16x8.pth)."
        )
    if incompatible.is_file():
        _add_files(
            result,
            [incompatible],
            "checkpoint",
            "8-frame 78.4% recipe; not the requested 16-frame 80.8% recipe",
            "incompatible",
        )
    if source.is_file():
        _add_files(result, [source], "source", "official UniFormer video model")
    else:
        result.blockers.append("Missing official UniFormer video source tree.")
    return result


def _uniformer_b(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    result.notes.append(
        "The official K400 16x4 checkpoint reports 49.8M parameters, 387 GFLOPs, "
        "and 82.0% top-1, not the requested row's 259 GFLOPs and 83.0%."
    )
    directory = ctx.workspace("checkpoints/uniformer")
    files = _files([directory])
    exact_candidates = _matches(files, ["uniformer_base_k400_*.pth"])
    expected_sha = "a36e08ecff0a2b11020d6e88046353f80fdeb10b40fe167f2fbc0c1d976fb873"
    exact = [path for path in exact_candidates if _sha256(path) == expected_sha]
    corrupt = [path for path in exact_candidates if path not in exact]
    incompatible = _matches(files, ["uniformer_base_sthv2_*.pth"])
    _add_files(result, exact, "checkpoint", "400-way Kinetics UniFormer-B")
    _add_files(result, corrupt, "checkpoint", "checksum mismatch/incomplete download", "incompatible")
    _add_files(
        result,
        incompatible,
        "checkpoint",
        "Something-Something-v2 174-way checkpoint; not valid for Kinetics-400",
        "incompatible",
    )
    source = ctx.workspace(
        "third_party/UniFormer/video_classification/slowfast/models/uniformer.py"
    )
    if source.is_file():
        _add_files(result, [source], "source", "official UniFormer video model")
    else:
        result.blockers.append("Missing official UniFormer video source tree.")
    if not exact:
        if corrupt:
            result.blockers.append("UniFormer-B checkpoint failed its official SHA256 check.")
        elif incompatible:
            result.blockers.append(
                "Only a 174-way Something-Something-v2 UniFormer-B checkpoint is present; "
                "it cannot substitute for the requested Kinetics-400 model."
            )
        else:
            result.blockers.append("Missing a 400-way Kinetics-400 UniFormer-B checkpoint.")
    return result


def _mvit_b24(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    result.notes.append(
        "The current official SlowFast model zoo reports 80.4% top-1 for this "
        "checkpoint; the strict-loaded graph has 53.115M parameters."
    )
    checkpoint = ctx.workspace("checkpoints/mvit/MVIT_B_24_32x3.pyth")
    candidates = [checkpoint] if checkpoint.is_file() else []
    expected_sha = "3aa6523906feda027db6048267a254371f00205af494eb933d1f527f56c08f26"
    exact = [
        path
        for path in candidates
        if path.stat().st_size == 637_806_132 and _sha256(path) == expected_sha
    ]
    partial = [path for path in candidates if path not in exact]
    _add_files(result, exact, "checkpoint", "MViT-B-24 32x3 Kinetics-400")
    _add_files(
        result,
        partial,
        "checkpoint",
        "incomplete or checksum-mismatched official download",
        "incompatible",
    )
    config = ctx.workspace("third_party/SlowFast/configs/Kinetics/MVIT_B_32x3_CONV.yaml")
    if config.is_file():
        _add_files(result, [config], "config", "SlowFast 32x3 MViT configuration")
    else:
        result.blockers.append("Missing SlowFast MViT 32x3 model configuration/source.")
    if not exact:
        result.blockers.append(
            "Missing the distinct MViT-B-24 32x3 checkpoint; MViT-B and MViTv2-B "
            "checkpoints are not substitutes."
        )
    return result


def _mvit_b32(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    checkpoint = ctx.workspace(
        "checkpoints/mvit/MVIT_B_32x3_f294077834.pyth"
    )
    candidates = [checkpoint] if checkpoint.is_file() else []
    expected_sha = "ee62dd4c41f4cf0622a690c55bb3fea5c5685443cf0b4cb444327961047e5064"
    exact = [
        path for path in candidates
        if "mvitv2" not in path.name.lower()
        and "b_24" not in path.name.lower()
        and "b24" not in path.name.lower()
        and path.stat().st_size == 439_671_342
        and _sha256(path) == expected_sha
    ]
    _add_files(
        result,
        exact,
        "checkpoint",
        "MViTv1-B 32x3 Kinetics-400 workspace checkpoint",
    )
    config = ctx.workspace("third_party/SlowFast/configs/Kinetics/MVIT_B_32x3_CONV.yaml")
    if config.is_file():
        _add_files(result, [config], "config", "SlowFast MViTv1-B 32x3")
    else:
        result.blockers.append("Missing SlowFast MViTv1-B 32x3 model configuration/source.")
    if not exact:
        result.blockers.append(
            "Missing or checksum-mismatched workspace MViTv1-B 32x3 checkpoint."
        )
    return result


def _vivit_s(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    result.notes.append(
        "The official Scenic ViViT release publishes B/16x2 and L/16x2 variants, "
        "but no ViViT-S Kinetics-400 model or checkpoint."
    )
    directory = ctx.workspace("checkpoints/vivit")
    files = _files([directory])
    purported_small = [
        path for path in files
        if "vivit" in path.name.lower()
        and ("-s-" in path.name.lower() or "_s_" in path.name.lower())
        and path.suffix.lower() in {".pt", ".pth", ".pyth"}
    ]
    incompatible = [
        path for path in files
        if ("vit-b" in path.name.lower() or "vivit-b" in path.name.lower())
        and path.suffix.lower() in {".pt", ".pth", ".pyth"}
    ]
    _add_files(
        result,
        purported_small,
        "checkpoint",
        "unverified ViViT-S filename; no official ViViT-S release exists",
        "incompatible",
    )
    _add_files(
        result,
        incompatible,
        "checkpoint",
        "ViViT/ViT-B asset; parameterization does not match ViViT-S",
        "incompatible",
    )
    result.blockers.append(
        "The requested ViViT-S Kinetics-400 variant has no official architecture/checkpoint "
        "release; ViViT-B and filename-only substitutions are intentionally rejected."
    )
    return result


def _focalnet(ctx: InspectionContext, spec: RequestedModel, variant: str) -> ModelResult:
    result = _base_result(ctx, spec)
    loaded_params = {"tiny": 49.553, "small": 88.735, "base": 157.389}
    result.notes.append(
        f"The exact strict-loaded video graph has {loaded_params[variant]:.3f}M "
        "parameters; the requested row's parameter count is not this checkpoint graph."
    )
    names = {
        "tiny": "video-focalnet_tiny_kinetics400.pth",
        "small": "video-focalnet_small_kinetics400.pth",
        "base": "video-focalnet_base_kinetics400.pth",
    }
    checkpoint = ctx.workspace(f"checkpoints/focalnet/{names[variant]}")
    source = ctx.workspace("third_party/Video-FocalNets/classification/videofocalnet.py")
    if checkpoint.is_file():
        _add_files(result, [checkpoint], "checkpoint", f"Kinetics-400 Video-FocalNet-{variant}")
    else:
        result.blockers.append(f"Missing Video-FocalNet-{variant} Kinetics-400 checkpoint.")
    if source.is_file():
        _add_files(result, [source], "source", "official Video-FocalNets implementation")
    else:
        result.blockers.append("Missing official Video-FocalNets source tree.")
    return result


def _focalnet_t(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    return _focalnet(ctx, spec, "tiny")


def _focalnet_s(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    return _focalnet(ctx, spec, "small")


def _focalnet_b(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    return _focalnet(ctx, spec, "base")


def _dualformer_t(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    source = ctx.workspace(
        "third_party/DualFormer/mmaction/models/backbones/dualformer.py"
    )
    adapter = ctx.workspace("models/dualformer.py")
    checkpoint = ctx.workspace(
        "checkpoints/dualformer/dualformer_tiny_patch244_window877.pth"
    )
    expected_sha = "f27c1c1996e4d27bc5c71f2bf3f690d933d8ea378030fb9e9e93c816872deb46"
    exact_checkpoint = (
        checkpoint.is_file()
        and checkpoint.stat().st_size == 101_764_429
        and _sha256(checkpoint) == expected_sha
    )
    if source.is_file():
        _add_files(result, [source], "source", "official DualFormer backbone")
    else:
        result.blockers.append("Official DualFormer source repository is absent.")
    if adapter.is_file():
        _add_files(
            result,
            [adapter],
            "source",
            "isolated strict inference adapter for the legacy MMAction graph",
        )
    else:
        result.blockers.append("DualFormer standalone adapter is absent.")
    if exact_checkpoint:
        _add_files(result, [checkpoint], "checkpoint", "DualFormer-T Kinetics-400")
    elif checkpoint.is_file():
        _add_files(
            result,
            [checkpoint],
            "checkpoint",
            "incomplete or checksum-mismatched official download",
            "incompatible",
        )
        result.blockers.append("DualFormer-T checkpoint failed its SHA256/size check.")
    else:
        result.blockers.append("DualFormer-T Kinetics-400 checkpoint is absent.")
    result.notes.append(
        "The released checkpoint includes a training-only token-label head. The adapter "
        "strict-loads all 314 keys, then benchmarks the official 21.973M-parameter test graph."
    )
    return result


def _omnivore_b(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    result.notes.append(
        "The strict-loaded multimodal model (trunk plus all published heads) has "
        "90.115M parameters; benchmark output reports that actual count."
    )
    workspace_dir = ctx.workspace("checkpoints/omnivore")
    files = _files([workspace_dir])
    expected_path = workspace_dir / "swinB_In21k_checkpoint.torch"
    expected_sha = "55707dba9353d1e2be1ae67bf989e468493f487e2f3ad44f951c249e1bd8d2e8"
    exact = [
        expected_path
        for path in [expected_path]
        if path.is_file()
        and path.stat().st_size == 478_602_543
        and _sha256(path) == expected_sha
    ]
    corrupt_exact = [expected_path] if expected_path.is_file() and not exact else []
    standard = [
        path for path in files
        if "swinb" in path.name.lower()
        and "in21k" not in path.name.lower()
        and path.suffix.lower() in {".torch", ".pth", ".pt"}
    ]
    source = ctx.workspace("third_party/omnivore/omnivore/models/omnivore_model.py")
    _add_files(result, exact, "checkpoint", "Omnivore Swin-B IN21K variant (84.0% K400)")
    _add_files(
        result,
        corrupt_exact,
        "checkpoint",
        "incomplete or checksum-mismatched IN21K download",
        "incompatible",
    )
    _add_files(
        result,
        standard,
        "checkpoint",
        "standard Omnivore Swin-B (83.3% K400), not the requested 84.0% IN21K variant",
        "incompatible",
    )
    if source.is_file():
        _add_files(result, [source], "source", "official Omnivore implementation")
    else:
        result.blockers.append("Official Omnivore source tree is absent.")
    if not exact:
        result.blockers.append(
            "The standard Omnivore-B checkpoint is available, but the requested IN21K "
            "Omnivore-B checkpoint that yields 84.0% K400 top-1 is absent."
        )
    return result


def _svt_b(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    result.notes.append(
        "The official SVT release is a self-supervised backbone/projection model; "
        "it requires training a downstream classifier and is not a 400-class model."
    )
    checkpoint_dirs = [
        ctx.workspace("checkpoints/svt"),
        ctx.workspace("checkpoints/others/svt"),
    ]
    files = _files(checkpoint_dirs, recursive=True)
    ssl = [path for path in files if "ssl" in path.name.lower()]
    purported_classifier = [
        path for path in files
        if "ssl" not in path.name.lower()
        and any(token in path.name.lower() for token in ("finetun", "linear", "classifier", "k400"))
        and path.suffix.lower() in {".pth", ".pyth", ".pt"}
    ]
    _add_files(
        result,
        purported_classifier,
        "checkpoint",
        "unverified downstream classifier; no official SVT 400-way head was released",
        "incompatible",
    )
    _add_files(
        result,
        ssl,
        "checkpoint",
        "self-supervised backbone only; contains no trained 400-way classifier",
        "incompatible",
    )
    result.blockers.append(
        "The official SVT artifact is self-supervised and has no released trained 400-way "
        "Kinetics head; a random or unrelated classifier is intentionally rejected."
    )
    return result


def _vtn_b(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    checkpoint = ctx.workspace("checkpoints/vtn/VTN_VIT_B_KINETICS.pyth")
    adapter = ctx.workspace("models/vtn.py")
    expected_sha = "779682d33e7afc32c3eff910e151eab30e7aed375acf7743cfe3041a83dcbf3a"
    if checkpoint.is_file() and _sha256(checkpoint) == expected_sha:
        _add_files(result, [checkpoint], "checkpoint", "VTN-B Kinetics checkpoint")
    elif checkpoint.is_file():
        _add_files(result, [checkpoint], "checkpoint", "checksum mismatch", "incompatible")
        result.blockers.append("VTN-B checkpoint failed its official SHA256 check.")
    else:
        result.blockers.append("VTN-B Kinetics checkpoint is absent.")
    if adapter.is_file():
        _add_files(result, [adapter], "source", "standalone port of official VTN/Longformer graph")
    else:
        result.blockers.append("VTN standalone adapter is absent.")
    result.notes.append(
        "The exact checkpoint's official model-zoo result is 77.72%; the full-video "
        "250-frame protocol is 4218 GFLOPs, not the requested row's 218 GFLOPs."
    )
    return result


def _simple_local(relative_checkpoint: str, relative_source: str | None = None):
    """Build a read-only exact-path inspector for locally wired adapters."""
    def inspect(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
        result = _base_result(ctx, spec)
        checkpoint = ctx.workspace(relative_checkpoint)
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            _add_files(result, [checkpoint], "checkpoint", f"local {spec.name} checkpoint")
        else:
            result.blockers.append(f"Missing checkpoint: {relative_checkpoint}")
        if relative_source:
            source = ctx.workspace(relative_source)
            if source.is_file():
                _add_files(result, [source], "source", f"{spec.name} implementation")
            else:
                result.blockers.append(f"Missing source: {relative_source}")
        return result
    return inspect


def _videomae_b(ctx: InspectionContext, spec: RequestedModel) -> ModelResult:
    result = _base_result(ctx, spec)
    local = ctx.workspace("checkpoints/videomae/base-finetuned-kinetics/model.safetensors")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--MCG-NJU--videomae-base-finetuned-kinetics" / "snapshots"
    cached = list(cache_root.glob("*/model.safetensors")) if cache_root.is_dir() else []
    candidates = ([local] if local.is_file() else []) + cached
    _add_files(result, candidates, "checkpoint", "fine-tuned offline K400 classifier")
    pretraining = ctx.workspace("checkpoints/videomae/VideoMAE-ViT-B_16x5x3_1600epoch.pth")
    if pretraining.is_file():
        _add_files(result, [pretraining], "checkpoint",
                   "pretraining encoder/decoder; no K400 classification head", "incompatible")
    if not candidates:
        result.blockers.append(
            "Missing an offline fine-tuned VideoMAE-B K400 Transformers checkpoint; "
            "the 1600epoch .pth file is pretraining-only."
        )
    return result


REQUESTED_MODELS: tuple[RequestedModel, ...] = (
    RequestedModel("uniformer-b", "UniFormer-B", 50.3, 259, 83.0, _uniformer_b),
    RequestedModel("uniformer-s", "UniFormer-S", 22.0, 42, 80.8, _uniformer_s),
    RequestedModel("mvit-b-24-32x3", "MViT-B-24, 32x3", 52.9, 236, 81.2, _mvit_b24),
    RequestedModel("mvit-v1-b", "MViT-B, 32x3", 36.6, 170, 80.2, _mvit_b32),
    RequestedModel("videoswin-t", "VideoSwin-T", 28.2, 88, 78.8,
                   _simple_local("checkpoints/videoswin/swin_tiny_patch244_window877_kinetics400_1k.pth", "models/videoswin.py")),
    RequestedModel("videoswin-s", "VideoSwin-S", 49.8, 166, 80.6,
                   _simple_local("checkpoints/videoswin/swin_small_patch244_window877_kinetics400_1k.pth", "models/videoswin.py")),
    RequestedModel("videoswin-b", "VideoSwin-B", 88.0, 282, 80.6,
                   _simple_local("checkpoints/videoswin/swin_base_patch244_window877_kinetics400_1k.pth", "models/videoswin.py")),
    RequestedModel("vivit-s", "ViViT-S", 31.8, 55, 78.5, _vivit_s),
    RequestedModel("timesformer-b", "TimeSformer-B", 121.4, 196, 78.0,
                   _simple_local("checkpoints/timesformer/TimeSformer_divST_8x32_224_K400.pyth", "models/timesformer.py")),
    RequestedModel("video-focalnet-t", "Video-FocalNet-T", 28.0, 63, 79.8, _focalnet_t),
    RequestedModel("video-focalnet-s", "Video-FocalNet-S", 49.0, 124, 81.4, _focalnet_s),
    RequestedModel("video-focalnet-b", "Video-FocalNet-B", 88.0, 149, 83.6, _focalnet_b),
    RequestedModel("dualformer-t", "DualFormer-T", 21.8, 240, 79.5, _dualformer_t),
    RequestedModel("omnivore-b", "Omnivore-B (IN21K)", 88.0, 282, 84.0, _omnivore_b),
    RequestedModel("videomae-b", "VideoMAE-B", 86.2, 180, 80.9,
                   _videomae_b),
    RequestedModel("svt-b", "SVT-B", 86.0, 180, 78.1, _svt_b),
    RequestedModel("vtn-b", "VTN-B", 114.0, 218, 78.6, _vtn_b),
)


def inspect_requested_models(root: Path = ROOT) -> tuple[InspectionContext, list[ModelResult]]:
    root = root.resolve()
    context = InspectionContext(
        root=root,
        torch_checkpoint_dirs=_torch_checkpoint_dirs(root),
        registry_keys=_registry_keys(root / "models" / "__init__.py"),
    )
    return context, [spec.inspect(context, spec) for spec in REQUESTED_MODELS]


def _report_dict(context: InspectionContext, results: Sequence[ModelResult]) -> dict[str, object]:
    ready_count = sum(result.ready for result in results)
    return {
        "schema_version": 1,
        "mode": "read-only asset preflight",
        "workspace": str(context.root),
        "torch_checkpoint_dirs": [str(path) for path in context.torch_checkpoint_dirs],
        "requested_count": len(results),
        "ready_count": ready_count,
        "blocked_count": len(results) - ready_count,
        "all_ready": ready_count == len(results),
        "models": [result.as_dict() for result in results],
    }


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _print_human(context: InspectionContext, results: Sequence[ModelResult]) -> None:
    ready_count = sum(result.ready for result in results)
    print("Requested-model asset preflight (read-only; no models are imported)")
    print(f"Workspace: {context.root}")
    print(f"Result: {ready_count}/{len(results)} exact requested models asset-ready")
    print()
    for result in results:
        label = "READY" if result.ready else "BLOCKED"
        spec = result.spec
        print(
            f"[{label:7}] {spec.key:22} {spec.name} "
            f"({spec.params_m:g}M, {spec.gflops:g} GFLOPs, {spec.top1:g}% top-1)"
        )
        for asset in result.assets:
            marker = "asset" if asset.compatibility == "exact" else "incompatible"
            print(f"           {marker}: {_display_path(asset.path, context.root)}")
            if asset.detail:
                print(f"             {asset.detail}")
        for blocker in result.blockers:
            print(f"           blocker: {blocker}")
        for note in result.notes:
            print(f"           note: {note}")
    if ready_count != len(results):
        print()
        print("Preflight failed. Add the exact blocked assets, or use --allow-missing for a best-effort run.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="return success even when exact requested assets are blocked",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    context, results = inspect_requested_models(args.root)
    report = _report_dict(context, results)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(context, results)
    return 0 if report["all_ready"] or args.allow_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
