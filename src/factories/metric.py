import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(labels, logits, mode="val"):
    d = {}
    d[f"{mode}_accuracy"] = accuracy_score(labels, logits.argmax(dim=1))
    d[f"{mode}_precision"] = precision_score(
        labels, logits.argmax(dim=1), average="macro", zero_division=0
    )
    d[f"{mode}_recall"] = recall_score(
        labels, logits.argmax(dim=1), average="macro", zero_division=0
    )
    d[f"{mode}_macro_f1"] = f1_score(labels, logits.argmax(dim=1), average="macro")
    # d[f"{mode}_roc_auc_ovo"] = roc_auc_score(
    #     labels, logits, multi_class="ovo", average="macro"
    # )
    d[f"{mode}_roc_auc"] = roc_auc_score(
        labels, logits, multi_class="ovr", average="macro"
    )
    return d
