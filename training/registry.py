from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "frozen17.yaml"


@lru_cache(maxsize=1)
def load_manifest():
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    models = data.get("models", [])
    keys = [item.get("key") for item in models]
    if len(models) != 17 or len(set(keys)) != 17:
        raise RuntimeError("configs/frozen17.yaml must contain exactly 17 unique models")
    if data.get("dataset", {}).get("num_classes") != 174:
        raise RuntimeError("Frozen experiment must use exactly 174 SSV2 classes")
    return data


FROZEN17_KEYS = tuple(item["key"] for item in load_manifest()["models"])


def model_spec(key):
    for item in load_manifest()["models"]:
        if item["key"] == key:
            return item
    raise KeyError(f"Unknown frozen17 model: {key}")


def resolve_models(value):
    if value in (None, "frozen17"):
        return list(FROZEN17_KEYS)
    if isinstance(value, str):
        value = [value]
    unknown = sorted(set(value) - set(FROZEN17_KEYS))
    if unknown:
        raise ValueError(f"Models are not in frozen17: {', '.join(unknown)}")
    return list(value)
