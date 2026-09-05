import torch


@torch.no_grad()
def validate(model, loader, device, autocast_context):
    model.eval()
    total = top1 = top5 = 0
    for clips, labels, *_ in loader:
        clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast_context():
            logits = model(clips)
        if logits.ndim != 2 or logits.shape[1] != 174 or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid validation logits: {tuple(logits.shape)}")
        k = min(5, logits.shape[1])
        pred = logits.topk(k, dim=1).indices
        total += labels.numel()
        top1 += pred[:, 0].eq(labels).sum().item()
        top5 += pred.eq(labels[:, None]).any(dim=1).sum().item()
    return {"top1": 100.0 * top1 / total, "top5": 100.0 * top5 / total, "samples": total}
