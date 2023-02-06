import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def clean_text(text):
    return (
        text.replace(" ", "")
        .replace("@user", "")
        .replace("　", "")
        .replace("__BR__", "\n")
        .replace("\xa0", "")
        .replace("\r", "")
        .lstrip("\n")
    )


class HatespeechDataset(Dataset):
    def __init__(self, data_config: DictConfig, df: pd.DataFrame):
        self.data_config = data_config
        self.texts = df[data_config.text_col].apply(clean_text).values
        self.tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer)
        self.labels = df[data_config.soft_label_col].values
        # if data_config.label_type == "soft" and mode != "test":
        #     self.labels = df[data_config.soft_label_col].values
        # else:
        #     self.labels = df[data_config.hard_label_col].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer.encode_plus(
            self.texts[item],
            add_special_tokens=True,
            max_length=self.data_config.max_len,
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
