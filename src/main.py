import gc
import hashlib
import os
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import pandas as pd
import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

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
            and config.base.do_train
        ):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def set_up(config: DictConfig):
    # Setup
    prepair_dir(config)
    pl.seed_everything(config.data.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.base.gpu_id)


@hydra.main(config_path="yamls", config_name="research.yaml", version_base=None)
def main(config: DictConfig):
    os.chdir(config.store.workdir)
    set_up(config)
    hparams = {}
    for key, value in config.items():
        if isinstance(value, DictConfig):
            hparams.update(value)
        else:
            hparams.update({key: value})

    if config.debug:
        config.train.trn_fold = [0]
        config.train.epoch = 1
    for fold in config.train.trn_fold:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.store.model_path,
            filename=f"fold{fold}" + "-{epoch}-{val_macro_f1:.4f}",
            monitor=config.train.callbacks.monitor_metric,
            verbose=True,
            save_top_k=0,
            mode=config.train.callbacks.mode,
            save_weights_only=True,
        )
        if (
            config.store.wandb_project is not None
            and config.base.do_train
            and not config.debug
        ):
            logger = WandbLogger(
                name=f"fold{fold}",
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
            "limit_test_batches": 1.0,
            "num_sanity_val_steps": 5,
            "num_nodes": 1,
            "gradient_clip_val": 0.5,
            "log_every_n_steps": 10,
            "deterministic": True,
        }
        model = CustomLitModule(config)
        datamodule = CustomDataModule(config, fold)
        trainer = pl.Trainer(**params)
        trainer.fit(model, datamodule=datamodule)
        if config.store.wandb_project is not None and config.base.do_train:
            wandb.finish()
        del trainer, datamodule, model, logger
        gc.collect()


if __name__ == "__main__":
    main()
