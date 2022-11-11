import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            cfg_model.pretrained_model, return_dict=True
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size, cfg_model.n_classes)

        # BertLayerモジュールの最後を勾配計算ありに変更
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):

        output = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.classifier(output.pooler_output)

        return logits

    def get_optim_parameters(self):
        return [self.bert.encoder.layer[-1].parameters(), self.classifier.parameters()]
