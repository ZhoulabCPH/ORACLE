import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.softmax(logits, dim=1)[
            torch.arange(len(target), device=logits.device), target
        ].clamp_(1e-6, 1 - 1e-6)
        loss = (1 - pt).pow(self.gamma) * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()
