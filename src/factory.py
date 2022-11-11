import yaml
from addict import Dict
import torch
from model import *
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
)


def read_yaml(fpath="./configs/sample.yaml"):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


def get_optimizer(cfg_optim):
    optimizer_cls = getattr(torch.optim, cfg_optim.optimizer.name)

    if hasattr(cfg_optim, "lr_scheduler"):
        scheduler_cls = getattr(torch.optim.lr_scheduler, cfg_optim.lr_scheduler.name)
    else:
        scheduler_cls = None
    return optimizer_cls, scheduler_cls


def get_loss(cfg_loss):
    return getattr(torch.nn, cfg_loss.name)(**cfg_loss.params)


def get_model(cfg_model):
    return eval(cfg_model.name)(cfg_model.params)


def get_metrics(labels, logits, mode="val"):
    softmax = torch.nn.Softmax(dim=1)
    logits = softmax(logits)
    d = {}
    d[f"{mode}_accuracy"] = accuracy_score(labels, logits.argmax(dim=1))
    d[f"{mode}_precision"] = precision_score(
        labels, logits.argmax(dim=1), average="macro", zero_division=0
    )
    d[f"{mode}_recall"] = recall_score(
        labels, logits.argmax(dim=1), average="macro", zero_division=0
    )
    d[f"{mode}_macro_f1"] = f1_score(labels, logits.argmax(dim=1), average="macro")
    try:
        d[f"{mode}_roc_auc_ovo"] = roc_auc_score(
            labels, logits, multi_class="ovo", average="macro"
        )
        d[f"{mode}_roc_auc_ovr"] = roc_auc_score(
            labels, logits, multi_class="ovr", average="macro"
        )
    except:
        pass
    return d
