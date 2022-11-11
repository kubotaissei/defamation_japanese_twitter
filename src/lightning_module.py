import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from gensim.models import KeyedVectors
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
import torch.nn as nn
from factory import get_optimizer, get_loss, get_model, get_metrics


class CustomLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.net = get_model(cfg.Model)
        self.criterion = get_loss(cfg.Loss)

    # 順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.net(input_ids, attention_mask)
        loss = 0 if labels is None else self.criterion(logits, labels)
        return loss, logits

    # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # validation、testでもtrain_stepと同じ処理を行う
    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def predict_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return preds

    def validation_epoch_end(self, outputs, mode="val"):
        # loss計算
        epoch_preds = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True, prog_bar=True)

        d = get_metrics(epoch_labels.cpu(), epoch_preds.cpu(), mode)
        self.log_dict(d, prog_bar=True)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "test")

    def configure_optimizers(self):
        cfg_optim = self.cfg.Optimizer
        optimizer_cls, scheduler_cls = get_optimizer(cfg_optim)

        optimizer = optimizer_cls(
            self.parameters(), lr=cfg_optim.optimizer.lr, **cfg_optim.params
        )
        if scheduler_cls is None:
            return [optimizer]
        else:
            scheduler = scheduler_cls(optimizer, **cfg_optim.lr_scheduler.params)
        return [optimizer], [scheduler]
