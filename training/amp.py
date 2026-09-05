from contextlib import nullcontext

import torch


def amp_policy(device, requested="bf16_if_supported"):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        if requested == "bf16_if_supported" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, None
        return torch.float16, torch.amp.GradScaler("cuda")
    return None, None


def autocast(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
