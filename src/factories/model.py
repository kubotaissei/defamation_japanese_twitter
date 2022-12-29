import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .pooling import *


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
                batch_first=True,
            )
        elif cfg_model.rnn == "GRU":
            self.rnn = nn.GRU(
                self.config.hidden_size,
                self.config.hidden_size,
                batch_first=True,
            )
        if cfg_model.pooling == "mean":
            self.pool = MeanPooling()
        elif cfg_model.pooling == "max":
            self.pool = MaxPooling()
        elif cfg_model.pooling == "min":
            self.pool = MinPooling()
        elif cfg_model.pooling == "weighted":
            self.pool = WeightedLayerPooling()
        elif cfg_model.pooling == "attention":
            self.pool = AttentionPooling(self.config.hidden_size)
        else:
            self.pool = None

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

    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]
        if self.cfg_model.rnn != "None":
            last_hidden_states = self.rnn(last_hidden_states, None)[0]
        feature = (
            self.pool(last_hidden_states, attention_mask)
            if self.pool is not None
            else last_hidden_states[:, -1, :]
        )
        return feature

    def forward(self, input_ids, attention_mask, labels=None):
        feature = self.feature(input_ids, attention_mask)
        output = (
            sum([self.fc(dropout(feature)) for dropout in self.dropouts])
            / self.cfg_model.n_msd
        )
        return output


def get_model(config_model):
    return CustomModel(config_model, config_path=None, pretrained=True)
