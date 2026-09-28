"""Accuracy counts, linear-head fitting, and the persisted split boundary."""
import csv

import torch

from scripts.run_ssv2_head_transfer import OUT, fit_head, score


def test_accuracy_keeps_real_zero_and_counts_top5():
    logits = torch.arange(174).float().repeat(2, 1)
    assert score(logits, torch.tensor([0, 172])) == (0, 1, 0.0, 50.0)


def test_fitted_head_learns_train_features_with_nonzero_mean_and_scale():
    torch.manual_seed(0)
    x = torch.eye(6).repeat_interleave(4, 0) * 7 + 3
    y = torch.arange(6).repeat_interleave(4)
    vx = torch.eye(6) * 7 + 3
    vy = torch.arange(6)
    weight, bias, alpha, candidates = fit_head(x, y, vx, vy)
    logits = vx @ weight.T + bias
    assert logits.shape == (6, 174)
    assert score(logits, vy)[0] == 6
    assert alpha in [r["alpha"] for r in candidates]
    assert torch.isfinite(weight).all() and torch.isfinite(bias).all()


def test_saved_selection_is_disjoint_and_balanced():
    from collections import Counter
    import pytest
    if not (OUT / "train_manifest.csv").exists():
        pytest.skip("Run the local dataset preparation before the artifact audit")
    groups = {}
    for split, count, per_class in [("train", 696, 4), ("select", 174, 1), ("eval", 1000, None)]:
        with (OUT / f"{split}_manifest.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        groups[split] = {r["RelativePath"] for r in rows}
        assert len(rows) == len(groups[split]) == count
        if per_class:
            counts = Counter(int(r["Label"]) for r in rows)
            assert set(counts) == set(range(174))
            assert set(counts.values()) == {per_class}
    assert groups["train"].isdisjoint(groups["select"] | groups["eval"])
    assert groups["select"].isdisjoint(groups["eval"])
