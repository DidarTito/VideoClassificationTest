"""Scientifically strict K400 -> SSV2 fine-tuning support."""

from .registry import FROZEN17_KEYS, load_manifest, model_spec

__all__ = ["FROZEN17_KEYS", "load_manifest", "model_spec"]
