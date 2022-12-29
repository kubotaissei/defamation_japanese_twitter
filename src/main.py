import gc
import hashlib
import os
import re
import shutil
from glob import glob

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold

from runner import CustomDataModule, CustomLitModule


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
    # Setup
    prepair_dir(config)
    pl.seed_everything(config.data.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.base.gpu_id)


@hydra.main(config_path="yamls", config_name="baseline.yaml", version_base=None)
def main(config: DictConfig):
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

    train[config.data.text_col] = train[config.data.text_col].apply(clean_text)
    test[config.data.text_col] = test[config.data.text_col].apply(clean_text)

    print(f"train.shape: {train.shape}")
    print(f"test.shape: {test.shape}")

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
    for fold in config.train.trn_fold:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.store.model_path,
            filename=config.store.model_name + f"-fold{fold}" + "-{epoch}-{val_f1:4f}",
            monitor=config.train.callbacks.monitor_metric,
            verbose=True,
            save_top_k=1,
            mode=config.train.callbacks.mode,
            save_weights_only=True,
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
            "deterministic": True,
            "resume_from_checkpoint": checkpoint_path,
        }
        model = CustomLitModule(config)
        datamodule = CustomDataModule(config, train, test, fold)
        if config.data.is_train:
            trainer = pl.Trainer(**params)
            trainer.fit(model, datamodule=datamodule)
            results += trainer.validate(model=model, datamodule=datamodule)
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
            trainer = pl.Trainer(**params)
        logits = trainer.predict(model=model, datamodule=datamodule)
        pred = torch.cat(logits)
        test["label"] = pred.argmax(1)
        test[f"pred_fold{fold}"] = pred[:, 1]
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
        config.store.result_path + "/submission_cv.csv", index=None
    )
    test.to_csv(config.store.result_path + "/pred_cv.csv", index=None)


if __name__ == "__main__":
    main()
