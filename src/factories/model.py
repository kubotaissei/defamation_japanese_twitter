import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd.function import InplaceFunction
from torch.nn import Parameter
from torch.nn import functional as F
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
                self.config.hidden_size, self.config.hidden_size, batch_first=True
            )
        elif cfg_model.rnn == "GRU":
            self.rnn = nn.GRU(
                self.config.hidden_size, self.config.hidden_size, batch_first=True
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
            [nn.Dropout(self.cfg_model.dropout) for _ in range(self.cfg_model.n_msd)]
        )
        self.fc = nn.Linear(self.config.hidden_size, self.cfg_model.num_classes)
        for i in range(self.cfg_model.num_reinit_layers):
            self.model.encoder.layer[-(1 + i)].apply(self._init_weights)
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


class Mixout(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "A mix probability of mixout has to be between 0 and 1,"
                " but got {}".format(p)
            )
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = (
                (1 - ctx.noise) * target + ctx.noise * output - ctx.p * target
            ) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)


class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.target = target
        self.p = p

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(
            input, mixout(self.weight, self.target, self.p, self.training), self.bias
        )

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out",
            self.p,
            self.in_features,
            self.out_features,
            self.bias is not None,
        )


def replace_mixout(model, mixout):
    for sup_module in model.modules():
        for name, module in sup_module.named_children():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.Linear):
                target_state_dict = module.state_dict()
                bias = True if module.bias is not None else False
                new_module = MixLinear(
                    module.in_features,
                    module.out_features,
                    bias,
                    target_state_dict["weight"],
                    mixout,
                )
                new_module.load_state_dict(target_state_dict)
                setattr(sup_module, name, new_module)
    return model


def get_model(config_model):
    model = CustomModel(config_model, config_path=None, pretrained=True)
    if config_model.mixout == 0:
        return model
    else:
        return replace_mixout(model, config_model.mixout)
