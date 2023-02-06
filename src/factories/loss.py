import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, reduction="mean", alpha=1, gamma=2):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        print(self.gamma)

    def forward(self, inputs, targets):
        loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-loss)
        loss = self.alpha * (1.0 - pt) ** self.gamma * loss
        if self.reduction == "none":
            loss = loss
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


class SmoothFocalLoss(nn.Module):
    def __init__(self, reduction="mean", alpha=1, gamma=2, smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.focal_loss = FocalLoss("none", alpha, gamma)
        self.smoothing = smoothing

    @staticmethod
    def _smooth(targets: torch.Tensor, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothFocalLoss._smooth(targets, self.smoothing)
        loss = self.focal_loss(inputs, targets)
        if self.reduction == "none":
            loss = loss
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


def get_loss(loss_class, params):
    print("loss class:", loss_class)
    if "." in loss_class:
        obj = eval(loss_class.split(".")[0])
        attr = loss_class.split(".")[1]
        f = getattr(obj, attr)
    else:
        f = globals().get(loss_class)
    return f(**params)
