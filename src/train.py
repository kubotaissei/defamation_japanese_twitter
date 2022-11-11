import argparse
import os
from pathlib import Path
from typing import Dict
import pandas as pd

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

# from pytorch_lightning.plugins import DDPPlugin

from dataset import CustomDataModule
from factory import read_yaml
from lightning_module import CustomLitModule


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--debug", action="store_true", help="debug")
    arg("--config", default="configs/sample.yaml", type=str, help="config path")
    arg("--gpus", default="0", type=str, help="gpu numbers")
    arg("--fold", default="0", type=str, help="fold number")
    return parser


def train(cfg_name: str, cfg: Dict, output_path: Path) -> None:
    seed_everything(cfg.General.seed)
    debug = cfg.General.debug
    fold = cfg.Data.dataset.fold

    logger = CSVLogger(save_dir=str(output_path), name=f"fold_{fold}")
    # wandb_logger = WandbLogger(
    #     name=f"{cfg_name}_{fold}", project=cfg.General.project, offline=debug
    # )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.05, patience=3, mode="min"
    )
    # 学習済重みを保存するために必要
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_path),
        filename=f"{cfg_name}_fold_{fold}",
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = Trainer(
        max_epochs=5 if debug else cfg.General.epoch,
        accelerator="gpu" if cfg.General.n_gpus else "cpu",
        devices=cfg.General.n_gpus,
        # distributed_backend=cfg.General.multi_gpu_mode,
        amp_backend="native",
        deterministic=True,
        auto_select_gpus=False,
        benchmark=False,
        default_root_dir=os.getcwd(),
        limit_train_batches=0.02 if debug else 1.0,
        limit_val_batches=0.05 if debug else 1.0,
        callbacks=[checkpoint_callback, early_stop_callback],
        # logger=[logger, wandb_logger],
        logger=[logger],
        # For fast https://pytorch-lightning.readthedocs.io/en/1.3.3/benchmarking/performance.html#
    )

    # Lightning module and start training
    model = CustomLitModule(cfg)
    datamodule = CustomDataModule(cfg)
    trainer.fit(model, datamodule=datamodule)


def main():
    args = make_parse().parse_args()

    # Read config
    cfg = read_yaml(fpath=args.config)
    cfg.General.debug = args.debug
    cfg.General.gpus = list(map(int, args.gpus.split(",")))
    cfg.Data.dataset.fold = args.fold

    output_path = f"../output"
    # Start train
    train(cfg_name=Path(args.config).stem, cfg=cfg, output_path=output_path)


if __name__ == "__main__":
    main()
