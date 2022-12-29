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


@hydra.main(version_base=None, config_path="yamls", config_name="baseline")
def main(config: DictConfig) -> int:
    # ====================================================
    # Data Loading
    # ====================================================
    os.chdir(config.store.workdir)
    pl.seed_everything(config.data.seed)
    test = pd.read_csv(config.data.test_path)
    # test = pd.read_pickle(config.data.test_path)

    def clean_text(text):
        try:
            return (
                text.replace(" ", "")
                .replace("　", "")
                .replace("__BR__", "\n")
                .replace("\xa0", "")
                .replace("\r", "")
                .lstrip("\n")
            )
        except:
            return "あ"

    test[config.data.text_col] = test[config.data.text_col].apply(clean_text)
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
            "limit_train_batches": 1.0,
            "check_val_every_n_epoch": 1,
            "limit_val_batches": 1.0,
            "limit_test_batches": 0.0,
            "num_sanity_val_steps": 5,
            "num_nodes": 1,
            "gradient_clip_val": 0.5,
            "log_every_n_steps": 10,
            "deterministic": False,
        }
        model = CustomLitModule(config)
        ckpt_path = sorted(glob(config.store.model_path + "/*.ckpt"))[fold]
        print(ckpt_path)
        state_dict = torch.load(ckpt_path)["state_dict"]
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
        logits = trainer.predict(model=model, dataloaders=test_dataloader)
        pred = torch.cat(logits)
        test["label"] = pred.argmax(1)
        test[f"pred_fold{fold}"] = pred[:, 1]
        print(test[["id", "label"]].groupby("label").count())
        preds.append(pred)
        del trainer, model
        gc.collect()

    test["label"] = (sum(preds) / len(preds)).argmax(1)
    test["pred"] = (sum(preds) / len(preds))[:, 1]

    print(test.groupby("label").count())

    # test.to_pickle(
    #     config.store.result_path + f"/{Path(config.data.test_path).stem}_pred.pkl",
    # )

    test.to_csv(
        config.store.result_path + f"/pred_{config.dataset.test_file}",
        encoding="utf-8-sig",
        index=False,
    )


if __name__ == "__main__":
    main()
