import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from factories import (
    get_model,
    get_scheduler,
    get_optimizer,
    get_loss,
    HatespeechDataset,
)


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config, train_df, test_df, fold):
        super().__init__()
        self.config = config
        self.fold = fold
        self.train_df = train_df
        self.test_df = test_df

    def setup(self, stage=None) -> None:
        self.train_folds = self.train_df[
            self.train_df["fold"] != self.fold
        ].reset_index(drop=True)
        self.valid_folds = self.train_df[
            self.train_df["fold"] == self.fold
        ].reset_index(drop=True)
        self.config.model.num_train_steps = int(
            len(self.train_folds)
            / self.config.train.batch_size
            * self.config.train.epoch
            / len(self.config.base.gpu_id)
        )

    def train_dataloader(self):
        return DataLoader(
            HatespeechDataset(self.config.data, self.train_folds),
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.base.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            HatespeechDataset(self.config.data, self.valid_folds),
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.base.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            HatespeechDataset(self.config.data, self.test_df, "test"),
            batch_size=self.config.test.batch_size,
            shuffle=False,
            num_workers=self.config.base.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class CustomLitModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(config.model)
        self.criterion = get_loss(loss_class=config.base.loss_class)
        self.config = config

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.model(input_ids, attention_mask)
        loss = 0 if labels is None else self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def predict_step(self, batch, batch_idx):
        _, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        preds = F.softmax(preds, dim=-1)
        return preds

    def validation_epoch_end(self, outputs, mode="val"):
        # loss計算
        epoch_preds = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True, prog_bar=True)
        self.log(
            f"{mode}_f1",
            f1_score(epoch_labels.cpu(), epoch_preds.cpu().argmax(dim=1)),
            logger=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        if self.config.base.use_transformer_parameter:
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "lr": self.config.model.encoder_lr,
                    "weight_decay": self.config.model.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "lr": self.config.model.encoder_lr,
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if "model" not in n
                    ],
                    "lr": self.config.model.decoder_lr,
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_parameters = self.model.parameters()
        optimizer = get_optimizer(
            opt_class=self.config.base.opt_class,
            params=optimizer_parameters,
            lr=self.config.model.encoder_lr,
            eps=self.config.model.eps,
            betas=self.config.model.betas,
            weight_decay=self.config.model.weight_decay,
        )
        scheduler = {
            "scheduler": get_scheduler(
                self.config.model,
                optimizer,
                self.config.model.num_train_steps,
            ),
            "interval": "step" if self.config.model.batch_scheduler else "epoch",
        }

        return [optimizer], [scheduler]
