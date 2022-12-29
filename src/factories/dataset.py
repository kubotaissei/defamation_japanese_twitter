import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HatespeechDataset(Dataset):
    def __init__(self, data_config: DictConfig, df: pd.DataFrame, mode: str = "train"):
        self.data_config = data_config
        self.texts = df[data_config.text_col].values
        self.labels = df[data_config.label_col].values if mode != "test" else None
        self.tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer)

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
