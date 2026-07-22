"""Device-agnostic and deployment-aware efficiency metrics.

Follows the two-stage methodology of "Energy-Efficient Vision Transformer
Inference for Edge-AI Deployment" (Amanzhol & Park, 2025) and the NetScore
variants from Toktassyn & Park (2026).
"""
import math


DEFAULT_DEPLOYMENT_WEIGHT = 1 / 8


def _validate_positive(**values: float) -> None:
    """Reject values outside the positive, finite domain of a log metric."""
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value, got {value!r}")


def _deployment_netscore(
    top1_acc_percent: float, *penalties: float, w: float
) -> float:
    """Evaluate ``20 log10(A^2 / product(x^w))`` robustly."""
    _validate_positive(top1_acc_percent=top1_acc_percent, w=w)
    for index, penalty in enumerate(penalties):
        _validate_positive(**{f"penalty_{index}": penalty})

    # Computing in log space avoids overflow for otherwise valid large inputs.
    return 20 * (
        2 * math.log10(top1_acc_percent)
        - w * sum(math.log10(penalty) for penalty in penalties)
    )


def netscore(top1_acc_percent: float, params_m: float, gflops: float) -> float:
    """NetScore = 20 * log10( Acc^2 / (sqrt(Params_M) * sqrt(GFLOPs)) ).

    Accuracy in percent (e.g. 80.6), parameters in millions.
    """
    return 20 * math.log10(
        top1_acc_percent ** 2 / (math.sqrt(params_m) * math.sqrt(gflops))
    )


def netscore_e(
    top1_acc_percent: float,
    inference_time_s: float,
    average_power_w: float,
    w: float = DEFAULT_DEPLOYMENT_WEIGHT,
) -> float:
    """Return the paper's inference-energy NetScore (NS-E).

    ``top1_acc_percent`` is a percentage (89.7, not 0.897),
    ``inference_time_s`` is seconds, and ``average_power_w`` is watts:

    ``NS-E = 20 log10(A^2 / (t^w P^w))``.
    """
    return _deployment_netscore(
        top1_acc_percent, inference_time_s, average_power_w, w=w
    )


def netscore_m(
    top1_acc_percent: float,
    peak_vram_mib: float,
    w: float = DEFAULT_DEPLOYMENT_WEIGHT,
) -> float:
    """Return the paper's peak-memory NetScore (NS-M).

    ``top1_acc_percent`` is a percentage (89.7, not 0.897), and
    ``peak_vram_mib`` is peak inference VRAM in mebibytes:

    ``NS-M = 20 log10(A^2 / r^w)``.
    """
    return _deployment_netscore(top1_acc_percent, peak_vram_mib, w=w)


def netscore_hash(
    top1_acc_percent: float,
    peak_vram_mib: float,
    inference_time_s: float,
    average_power_w: float,
    w: float = DEFAULT_DEPLOYMENT_WEIGHT,
) -> float:
    """Return the paper's joint deployment NetScore (NS#).

    Accuracy is a percentage, VRAM is MiB, inference time is seconds, and
    average power is watts:

    ``NS# = 20 log10(A^2 / (r^w t^w P^w))``.
    """
    return _deployment_netscore(
        top1_acc_percent,
        peak_vram_mib,
        inference_time_s,
        average_power_w,
        w=w,
    )


def sam(top1_acc_percent: float, energy_mj: float, a: float = 1, b: float = 1) -> float:
    """SAM = (b * Acc^a) / log10(Energy).

    Accuracy as a fraction (0-1). Energy is taken in millijoules so that
    log10(Energy) stays positive for fast models (per-inference energy on a
    desktop GPU can drop below 1 J, which would flip the metric's sign).
    """
    if energy_mj <= 1:
        return float("nan")
    acc = top1_acc_percent / 100.0
    return (b * acc ** a) / math.log10(energy_mj)
