"""Hard, manifest-driven checks for a frozen17 K400 execution."""

from functools import lru_cache
import hashlib
from pathlib import Path

from .registry import FROZEN17_KEYS, ROOT, load_manifest, model_spec


PARAMETER_TOLERANCE = 0.05


@lru_cache(maxsize=None)
def sha256_file(path_text):
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_manifest(dataset, num_clips, seed):
    key = f"{dataset}_{num_clips}_seed{seed}"
    return load_manifest().get("benchmark_manifests", {}).get(key)


def validate_frozen_model(
    key,
    model,
    *,
    dataset,
    checkpoint,
    manifest=None,
    num_clips=None,
    seed=None,
):
    """Reject a wrong graph/checkpoint/input contract before timed inference."""
    if key not in FROZEN17_KEYS:
        raise RuntimeError(f"{key!r} is not a canonical frozen17 model key")
    if dataset != "k400":
        raise RuntimeError("frozen17 device preflight requires the K400 source task")

    spec = model_spec(key)
    if spec.get("checkpoint_status") != "READY_EXACT":
        raise RuntimeError(
            f"{key} is not READY_EXACT: {spec.get('checkpoint_status')}"
        )
    expected_rel = spec.get("k400_checkpoint")
    if not expected_rel:
        raise RuntimeError(f"{key} has no exact checkpoint configured")
    expected_path = (ROOT / expected_rel).resolve()
    actual_path = Path(checkpoint).resolve()
    if actual_path != expected_path:
        raise RuntimeError(
            f"checkpoint path mismatch for {key}: {actual_path} != {expected_path}"
        )
    if not actual_path.is_file():
        raise RuntimeError(f"checkpoint missing for {key}: {actual_path}")
    model_info = getattr(model, "info", None)
    loaded_checkpoint = (
        Path(model_info.get("checkpoint")).resolve()
        if isinstance(model_info, dict) and model_info.get("checkpoint")
        else None
    )
    if loaded_checkpoint is not None and loaded_checkpoint != expected_path:
        raise RuntimeError(
            f"adapter loaded the wrong checkpoint for {key}: "
            f"{loaded_checkpoint} != {expected_path}"
        )
    expected_hash = spec.get("checkpoint_sha256")
    if not expected_hash:
        raise RuntimeError(f"checkpoint provenance hash is missing for {key}")
    actual_hash = sha256_file(str(actual_path))
    if actual_hash.lower() != str(expected_hash).lower():
        raise RuntimeError(
            f"checkpoint SHA256 mismatch for {key}: {actual_hash}"
        )
    if not spec.get("checkpoint_origin") or not spec.get("source_commit"):
        raise RuntimeError(f"checkpoint/source provenance is incomplete for {key}")

    expected_frames = int(spec["input"]["num_frames"])
    actual_frames = getattr(model, "frames", None)
    if actual_frames is None and isinstance(getattr(model, "info", None), dict):
        actual_frames = model.info.get("frames")
    if int(actual_frames or -1) != expected_frames:
        raise RuntimeError(
            f"frame-count mismatch for {key}: actual={actual_frames}, "
            f"expected={expected_frames}"
        )

    expected_classes = int(spec["k400_classes"])
    actual_classes = int(getattr(model, "num_classes", -1))
    if actual_classes != expected_classes:
        raise RuntimeError(
            f"class-count mismatch for {key}: actual={actual_classes}, "
            f"expected={expected_classes}"
        )

    actual_params_m = float(model.parameters) / 1e6
    stage1 = spec["stage1"]
    expected_params_m = float(
        spec.get("runtime_actual_params_m", stage1["params_m"])
    )
    relative_error = abs(actual_params_m - expected_params_m) / expected_params_m
    if relative_error > PARAMETER_TOLERANCE:
        raise RuntimeError(
            f"parameter-count mismatch for {key}: actual={actual_params_m:.6f}M, "
            f"expected={expected_params_m:.6f}M, tolerance=5%"
        )

    manifest_spec = expected_manifest(dataset, num_clips, seed)
    if manifest_spec:
        manifest_path = Path(manifest).resolve() if manifest else None
        wanted_path = (ROOT / manifest_spec["path"]).resolve()
        if manifest_path != wanted_path:
            raise RuntimeError(
                f"dataset manifest mismatch: {manifest_path} != {wanted_path}"
            )
        actual_manifest_hash = sha256_file(str(manifest_path))
        if actual_manifest_hash.lower() != manifest_spec["sha256"].lower():
            raise RuntimeError(
                f"dataset manifest SHA256 mismatch: {actual_manifest_hash}"
            )

    return {
        "actual_params_m": actual_params_m,
        "expected_params_m": expected_params_m,
        "frames": expected_frames,
        "classes": expected_classes,
        "checkpoint_sha256": actual_hash,
    }
