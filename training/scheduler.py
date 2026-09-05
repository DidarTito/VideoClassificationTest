import torch


def cosine_with_warmup(optimizer, epochs, warmup_epochs=5):
    def factor(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
