import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import BertJapaneseTokenizer


class CreateDataset(Dataset):
    """
    DataFrameを下記のitemを保持するDatasetに変換。
    text(原文)、input_ids(tokenizeされた文章)、attention_mask、labels(ラベル)
    """

    def __init__(self, data, tokenizer, cfg):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = cfg.max_token_len
        self.text_col = cfg.text_col
        self.target_col = cfg.target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row[self.text_col]
        labels = data_row[self.target_col]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels),
        )


class CustomDataModule(pl.LightningDataModule):
    """
    DataFrameからモデリング時に使用するDataModuleを作成
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg_dataset = cfg.Data.dataset
        self.cfg_dataloader = cfg.Data.dataloader

        self.test_df = None
        self.train_df = None
        self.valid_df = None

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            cfg.Model.params.pretrained_model
        )

    def get_test_df(self):
        return pd.read_pickle(self.cfg_dataset.test_df)

    def split_train_valid_df(self):
        df = pd.read_pickle(self.cfg_dataset.train_df)

        # Split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for n, (train_index, val_index) in enumerate(
            skf.split(df, df[self.cfg_dataset.target_col])
        ):
            df.loc[val_index, "fold"] = int(n)
        df["fold"] = df["fold"].astype(int)

        fold = int(self.cfg_dataset.fold)
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        return train_df, valid_df

    def setup(self, stage=None):
        self.test_df = self.get_test_df()
        train_df, valid_df = self.split_train_valid_df()
        self.train_df = train_df
        self.valid_df = valid_df

    def get_dataframe(self, phase):
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            return self.test_df

    def get_loader(self, phase):
        dataset = CreateDataset(
            self.get_dataframe(phase),
            self.tokenizer,
            self.cfg_dataset,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg_dataloader.batch_size,
            shuffle=True if phase == "train" else False,
            num_workers=self.cfg_dataloader.num_workers,
            drop_last=True if phase == "train" else False,
        )

    def train_dataloader(self):
        return self.get_loader(phase="train")

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")


if __name__ == "__main__":
    from factory import read_yaml

    cfg = read_yaml(fpath="configs/sample.yaml")
    datamodule = CustomDataModule(cfg)
    datamodule.setup()
    print(datamodule.get_dataframe("train").head())
