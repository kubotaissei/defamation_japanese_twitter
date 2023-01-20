import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from glob import glob

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from factories import HatespeechDataset
from runner import CustomLitModule


@hydra.main(version_base=None, config_path="yamls", config_name="research")
def main(config: DictConfig) -> int:
    # ====================================================
    # Data Loading
    # ====================================================
    os.chdir(config.store.workdir)
    pl.seed_everything(config.data.seed)
    test = pd.read_csv(config.data.test_path)
    print(f"test.shape: {test.shape}")

    hparams = {}
    for key, value in config.items():
        if isinstance(value, DictConfig):
            hparams.update(value)
        else:
            hparams.update({key: value})

    preds = []
    test_dataloader = DataLoader(
        HatespeechDataset(config.data, test, "test"),
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.base.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    for fold in config.train.trn_fold:
        backend = "ddp" if len(config.base.gpu_id) > 1 else None
        params = {
            "accumulate_grad_batches": config.train.gradient_accumulation_steps,
            "precision": 16,
            "devices": len(config.base.gpu_id),
            "accelerator": "gpu",
            "strategy": backend,
            "limit_train_batches": 0.0,
            "limit_val_batches": 0.0,
            "limit_test_batches": 1.0,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 5,
            "num_nodes": 1,
            "gradient_clip_val": 0.5,
            "log_every_n_steps": 10,
            "deterministic": False,
        }
        model = CustomLitModule(config)
        ckpt_path = sorted(glob(config.store.model_path + "/**/*.ckpt"))[fold]
        print(ckpt_path)
        state_dict = torch.load(ckpt_path)["state_dict"]
        model.load_state_dict(state_dict)
        trainer = pl.Trainer(**params)
        logits = trainer.predict(model=model, dataloaders=test_dataloader)
        pred = torch.cat(logits)
        test["pred_label"] = pred.argmax(1)
        test[f"pred_fold{fold}"] = pred.tolist()
        print(test.groupby("pred_label").count())
        preds.append(pred)
        del trainer, model
        gc.collect()

    test["pred_label"] = (sum(preds) / len(preds)).argmax(1)
    test["pred_cv"] = (sum(preds) / len(preds)).tolist()
    test.to_csv(
        config.store.result_path + f"/pred_{config.dataset.test_file}",
        encoding="utf-8-sig",
        index=False,
    )
    test.to_pickle(config.store.result_path + "/pred_cv.pkl")


if __name__ == "__main__":
    main()
