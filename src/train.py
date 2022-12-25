# ====================================================
# Library
# ====================================================

import os
import warnings

warnings.filterwarnings("ignore")

import gc
import hashlib
import os
import re
import shutil
from glob import glob

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


# ====================================================
# Dataset
# ====================================================
class HatespeechDataset(Dataset):
    def __init__(self, cfg_data, df, is_test=False):
        self.cfg_data = cfg_data
        self.texts = df[cfg_data.text_col].values
        self.labels = df[cfg_data.label_col].values if not is_test else None
        self.tokenizer = AutoTokenizer.from_pretrained(cfg_data.tokenizer)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            self.texts[item],
            add_special_tokens=True,
            max_length=self.cfg_data.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        inputs = dict(
            text=self.texts[item],
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
        )
        if self.labels is not None:
            inputs["labels"] = torch.tensor(self.labels[item])
        return inputs


# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg_model, config_path=None, pretrained=False):
        super().__init__()
        self.cfg_model = cfg_model
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                cfg_model.pretrained, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(
                cfg_model.pretrained, config=self.config
            )
        else:
            self.model = AutoModel(self.config)
        if cfg_model.rnn == "LSTM":
            self.rnn = nn.LSTM(
                self.config.hidden_size,
                self.config.hidden_size,
                # bidirectional=True,
                batch_first=True,
            )
        elif cfg_model.rnn == "GRU":
            self.rnn = nn.GRU(
                self.config.hidden_size,
                self.config.hidden_size,
                # bidirectional=True,
                batch_first=True,
            )
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(self.cfg_model.multi_sample_dropout)
                for _ in range(self.cfg_model.n_msd)
            ]
        )
        self.fc = nn.Linear(self.config.hidden_size, 2)
        if self.cfg_model.reinit_layers != "None":
            for layer in self.model.encoder.layer[self.cfg_model.reinit_layers :]:
                for module in layer.modules():
                    self._init_weights(module)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        all_hidden_states = torch.stack(outputs["hidden_states"])
        rnn_out = (
            self.rnn(outputs["last_hidden_state"], None)[0]
            if self.cfg_model.rnn != "None"
            else outputs["last_hidden_state"]
        )
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(rnn_out.size()).float()
        )
        if self.cfg_model.pooling == "mean_old":
            sequence_output = rnn_out.mean(axis=1)
        elif self.cfg_model.pooling == "max_old":
            sequence_output, _ = rnn_out.max(1)
        elif self.cfg_model.pooling == "mean":
            sum_embeddings = torch.sum(rnn_out * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            sequence_output = sum_embeddings / sum_mask
        elif self.cfg_model.pooling == "max":
            rnn_out[
                input_mask_expanded == 0
            ] = -1e9  # Set padding tokens to large negative value
            sequence_output = torch.max(rnn_out, 1)[0]
        else:
            sequence_output = rnn_out[:, -1, :]
        output = (
            sum([self.fc(dropout(sequence_output)) for dropout in self.dropouts])
            / self.cfg_model.n_msd
        )
        return output


class CustomDataModule(pl.LightningDataModule):
    """
    DataFrameからモデリング時に使用するDataModuleを作成
    """

    def __init__(self, cfg, train_df, fold):
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.train_df = train_df

    def setup(self, stage=None):
        self.train_folds = self.train_df[
            self.train_df["fold"] != self.fold
        ].reset_index(drop=True)
        self.valid_folds = self.train_df[
            self.train_df["fold"] == self.fold
        ].reset_index(drop=True)
        self.cfg.model.num_train_steps = int(
            len(self.train_folds) / self.cfg.train.batch_size * self.cfg.train.epoch
        )

    def train_dataloader(self):
        return DataLoader(
            HatespeechDataset(self.cfg.data, self.train_folds),
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.base.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            HatespeechDataset(self.cfg.data, self.valid_folds),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.base.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class CustomLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.model = CustomModel(cfg.model, config_path=None, pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.cfg = cfg

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

        sch = self.lr_schedulers()
        if self.cfg.model.batch_scheduler:
            sch.step()
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
        def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "lr": encoder_lr,
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in param_optimizer if "model" not in n],
                    "lr": decoder_lr,
                    "weight_decay": 0.0,
                },
            ]
            return optimizer_parameters

        optimizer_parameters = get_optimizer_params(
            self.model,
            encoder_lr=self.cfg.model.encoder_lr,
            decoder_lr=self.cfg.model.decoder_lr,
            weight_decay=self.cfg.model.weight_decay,
        )
        optimizer = AdamW(
            optimizer_parameters,
            lr=self.cfg.model.encoder_lr,
            eps=self.cfg.model.eps,
            betas=self.cfg.model.betas,
        )
        # ====================================================
        # scheduler
        # ====================================================
        def get_scheduler(cfg, optimizer, num_train_steps):
            if cfg.scheduler == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=cfg.num_warmup_steps,
                    num_training_steps=num_train_steps,
                )
            elif cfg.scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=cfg.num_warmup_steps,
                    num_training_steps=num_train_steps,
                    num_cycles=cfg.num_cycles,
                )
            return scheduler

        scheduler = {
            "scheduler": get_scheduler(
                self.cfg.model,
                optimizer,
                self.cfg.model.num_train_steps,
            ),
            "interval": "step" if self.cfg.model.batch_scheduler else "epoch",
        }

        return [optimizer], [scheduler]


def prepair_dir(config: DictConfig):
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        if (
            os.path.exists(path)
            and config.train.warm_start is False
            and config.data.is_train
        ):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def set_up(config: DictConfig):
    prepair_dir(config)
    pl.seed_everything(config.data.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.base.gpu_id)


@hydra.main(version_base=None, config_path="config", config_name="baseline")
def main(config: DictConfig) -> int:
    # ====================================================
    # Data Loading
    # ====================================================
    config.store.workdir = os.getcwd()
    os.chdir(config.store.workdir)
    set_up(config)
    train = pd.read_csv(config.data.train_path)
    test = pd.read_csv(config.data.test_path)

    def clean_text(text):
        return (
            text.replace(" ", "")
            .replace("　", "")
            .replace("__BR__", "\n")
            .replace("\xa0", "")
            .replace("\r", "")
            .lstrip("\n")
        )

    train["text"] = train["text"].apply(clean_text)
    test["text"] = test["text"].apply(clean_text)

    print(f"train.shape: {train.shape}")
    print(f"test.shape: {test.shape}")

    # ====================================================
    # CV split
    # ====================================================
    Fold = StratifiedKFold(
        n_splits=config.data.n_fold, shuffle=True, random_state=config.data.seed
    )
    for n, (train_index, val_index) in enumerate(
        Fold.split(train, train[config.data.label_col].astype(int))
    ):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)

    hparams = {}
    for key, value in config.items():
        if isinstance(value, DictConfig):
            hparams.update(value)
        else:
            hparams.update({key: value})

    preds = []
    results = []
    test_dataloader = DataLoader(
        HatespeechDataset(config.data, test, True),
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.base.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    for fold in config.train.trn_fold:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.store.model_path,
            filename=config.store.model_name + f"-fold{fold}" + "-{epoch}-{val_f1}",
            monitor=config.train.callbacks.monitor_metric,
            verbose=True,
            save_top_k=1,
            mode=config.train.callbacks.mode,
            save_weights_only=False,
        )
        if config.store.wandb_project is not None and config.data.is_train:
            logger = WandbLogger(
                name=config.store.model_name + f"_fold{fold}",
                save_dir=config.store.log_path,
                project=config.store.wandb_project,
                version=hashlib.sha224(
                    bytes(str(hparams) + str(fold), "utf8")
                ).hexdigest()[:4],
                anonymous=True,
                group=config.model.pretrained,
                tags=[config.model.rnn, config.model.pooling],
            )
        else:
            logger = None

        early_stop_callback = EarlyStopping(
            monitor=config.train.callbacks.monitor_metric,
            patience=config.train.callbacks.patience,
            verbose=True,
            mode=config.train.callbacks.mode,
        )

        backend = "ddp" if len(config.base.gpu_id) > 1 else None
        if config.train.warm_start:
            checkpoint_path = sorted(
                glob(config.store.model_path + "/*epoch*"),
                key=lambda path: int(re.split("[=.]", path)[-2]),
            )[-1]
            print(checkpoint_path)
        else:
            checkpoint_path = None
        params = {
            "logger": logger,
            "max_epochs": config.train.epoch,
            # "callbacks": [early_stop_callback, checkpoint_callback],
            "callbacks": [checkpoint_callback],
            "accumulate_grad_batches": config.train.gradient_accumulation_steps,
            "precision": 16,
            "devices": len(config.base.gpu_id),
            "accelerator": "gpu",
            "strategy": backend,
            "limit_train_batches": 1.0,
            "check_val_every_n_epoch": 1,
            "limit_val_batches": 1.0,
            "limit_test_batches": 0.0,
            "num_sanity_val_steps": 5,
            "num_nodes": 1,
            "gradient_clip_val": 0.5,
            "log_every_n_steps": 10,
            "deterministic": False,
            "resume_from_checkpoint": checkpoint_path,
        }
        model = CustomLitModule(config)
        datamodule = CustomDataModule(config, train, fold)
        if config.data.is_train:
            trainer = Trainer(**params)
            trainer.fit(model, datamodule=datamodule)
            results += trainer.validate(
                model=model, dataloaders=datamodule.val_dataloader()
            )
        else:
            state_dict = torch.load(
                sorted(glob(config.store.model_path + "/*.ckpt"))[fold]
            )["state_dict"]
            model.load_state_dict(state_dict)
            params.update(
                {
                    "devices": 1,
                    "limit_train_batches": 0.0,
                    "limit_val_batches": 0.0,
                    "limit_test_batches": 1.0,
                }
            )
            trainer = Trainer(**params)
        logits = trainer.predict(model=model, dataloaders=test_dataloader)
        pred = torch.cat(logits)
        test["label"] = pred.argmax(1)
        test["pred"] = pred[:, 1]
        test[["id", "label"]].to_csv(
            config.store.result_path + f"/submission_fold{fold}.csv", index=None
        )
        test.to_csv(config.store.result_path + f"/pred_fold{fold}.csv", index=None)
        print(test[["id", "label"]].groupby("label").count())
        preds.append(pred)
        if config.store.wandb_project is not None and config.data.is_train:
            wandb.finish()
        del trainer, datamodule, model, logger
        gc.collect()

    if config.data.is_train:
        result_df = pd.DataFrame(results)
        print(result_df)
        result_df.to_csv(config.store.result_path + "/result_cv.csv")
    test["label"] = (sum(preds) / len(preds)).argmax(1)
    test["pred"] = (sum(preds) / len(preds))[:, 1]
    test[["id", "label"]].to_csv(
        config.store.result_path + "/submission_ave.csv", index=None
    )
    test.to_csv(config.store.result_path + "/pred_ave.csv", index=None)


if __name__ == "__main__":
    main()
