import csv
import math
from pathlib import Path

import torch


def train_one_epoch(model, loader, optimizer, device, autocast_context, scaler=None, accumulation=1, clip_grad=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = samples = 0
    for step, batch in enumerate(loader, start=1):
        clips, labels = batch[:2]
        clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast_context():
            logits = model(clips)
            if logits.shape[1] != 174:
                raise RuntimeError(f"Expected 174 logits, got {tuple(logits.shape)}")
            loss = torch.nn.functional.cross_entropy(logits, labels) / accumulation
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % accumulation == 0 or step == len(loader):
            if scaler:
                scaler.unscale_(optimizer)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.item() * accumulation * labels.numel()
        samples += labels.numel()
    return loss_sum / max(samples, 1)


def append_metrics(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
