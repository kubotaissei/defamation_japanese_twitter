import gc
import hashlib
import os
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

from factories import get_metrics
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


@hydra.main(config_path="yamls", config_name="baseline.yaml", version_base=None)
def main(config: DictConfig):
    os.chdir(config.store.workdir)
    set_up(config)
    test = pd.read_pickle(config.data.test_path)
    hparams = {}
    for key, value in config.items():
        if isinstance(value, DictConfig):
            hparams.update(value)
        else:
            hparams.update({key: value})
    preds = []
    test_result = []
    if config.data.binary:
        config.model.num_classes = 2
    if config.debug:
        config.train.epoch = 1
        config.train.n_fold = 2
        config.store.wandb_project = None
    for fold in range(config.train.n_fold):
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.store.model_path,
            filename=f"fold{fold}" + "-{epoch}-{val_macro_f1:.4f}",
            monitor=config.train.callbacks.monitor_metric,
            verbose=True,
            save_top_k=config.train.callbacks.save_top_k,
            mode=config.train.callbacks.mode,
            save_weights_only=True,
        )
        if config.store.wandb_project is not None and config.base.do_train:
            logger = WandbLogger(
                name=f"fold{fold}",
                save_dir=config.store.log_path,
                project=config.store.wandb_project,
                version=hashlib.sha224(
                    bytes(str(hparams) + str(fold), "utf8")
                ).hexdigest()[:4],
                anonymous=True,
                group=config.data.type,
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
            "limit_train_batches": 1.0 if not config.debug else 0.1,
            "check_val_every_n_epoch": 1,
            "limit_val_batches": 1.0 if not config.debug else 0.5,
            "limit_test_batches": 1.0 if not config.debug else 0.5,
            "num_sanity_val_steps": 5,
            "num_nodes": 1,
            "gradient_clip_val": 0.5,
            "log_every_n_steps": 10,
            "deterministic": True,
        }
        model = CustomLitModule(config)
        datamodule = CustomDataModule(config, fold)
        if config.base.do_train:
            trainer = pl.Trainer(**params)
            trainer.fit(model, datamodule)
        else:
            ckpt_path = sorted(glob(config.store.model_path + "/*.ckpt"))[fold]
            print(ckpt_path)
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
            params.update(
                {
                    "devices": 1,
                    "logger": None,
                    "limit_train_batches": 0.0,
                    "limit_val_batches": 0.0,
                    "limit_test_batches": 1.0,
                    "accelerator": "gpu",
                }
            )
            trainer = pl.Trainer(**params)
        if config.train.n_fold != 1 and config.base.do_test:
            test_result += trainer.test(model, datamodule)
            logits = trainer.predict(model, datamodule)
            pred = torch.cat(logits)
            test[f"logits_{config.data.hard_label_col}_fold{fold}"] = pred.tolist()
            test[f"pred_{config.data.hard_label_col}_fold{fold}"] = pred.argmax(1)
            preds.append(pred)
        if (
            config.store.wandb_project is not None
            and config.base.do_train
            and (fold != config.train.n_fold - 1 or config.train.n_fold == 1)
        ):
            wandb.finish()
        del trainer, datamodule, model, logger
        gc.collect()

    if config.train.n_fold != 1 and config.base.do_test:
        test[f"logits_{config.data.hard_label_col}_ensemble"] = (
            sum(preds) / len(preds)
        ).tolist()
        test[f"pred_{config.data.hard_label_col}_ensemble"] = (
            sum(preds) / len(preds)
        ).tolist()
        if config.data.binary:
            label = torch.Tensor(
                (test[config.data.hard_label_col] != 0).astype(int).to_list()
            )
        else:
            label = torch.Tensor(test[config.data.hard_label_col].to_list())
        result = get_metrics(
            label,
            sum(preds) / len(preds),
            "test_ensemble",
        )
        test_result.append(result)
        pd.DataFrame(test_result).to_csv(config.store.result_path + "/result.csv")
        test.to_pickle(config.store.result_path + "/pred.pkl")
        if config.store.wandb_project is not None and config.base.do_train:
            wandb.log(result)
            wandb.log(
                {
                    hashlib.sha224(bytes(str(hparams) + str(fold), "utf8")).hexdigest()[
                        :4
                    ]: test
                }
            )
            wandb.finish()


if __name__ == "__main__":
    main()
