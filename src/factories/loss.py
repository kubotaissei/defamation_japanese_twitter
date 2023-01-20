import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, reduction="none", alpha=1, gamma=2):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss = F.cross_entropy(inputs, targets.to(torch.int64))
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
    def __init__(self, reduction="none", alpha=1, gamma=2, smoothing=0.1):
        super().__init__()
        self.reduction = reduction
        self.focal_loss = FocalLoss(reduction, alpha, gamma)
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


def get_loss(config_loss):
    print("loss class:", config_loss.class_name)
    if "." in config_loss.class_name:
        obj = eval(config_loss.class_name.split(".")[0])
        attr = config_loss.class_name.split(".")[1]
        f = getattr(obj, attr)
    else:
        f = globals().get(config_loss.class_name)
    return f(**config_loss.params)
