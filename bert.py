"""
UDA model in torch version
Remain to be updated for more flexibility
"""
import os
import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    Bert based Classifier
    """
    def __init__(self, bert_type: str, n_labels: int, drop_prob: float):
        super(BertClassifier, self).__init__()
        self.drop = nn.Dropout(drop_prob)
        self.linear_layer = nn.Linear(768, n_labels)
        self.bert_layer = BertModel.from_pretrained(bert_type)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Dim of pooled_h: [batch_size, embedding_dim]
        _, pooled_h = self.bert_layer(input_ids, attention_mask, token_type_ids)
        y = self.linear_layer(self.drop(pooled_h))
        return y


class FTBertClassifier(nn.Module):
    """
    Fine Tuned Bert based Classifier
    """
    def __init__(self, ft_bert_file: str, n_labels: int, drop_prob: float):
        super(FTBertClassifier, self).__init__()
        self.drop = nn.Dropout(drop_prob)
        self.linear_layer = nn.Linear(768, n_labels)
        self.bert_layer = torch.load(ft_bert_file)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Dim of pooled_h: [batch_size, embedding_dim]
        _, pooled_h = self.bert_layer(input_ids, attention_mask, token_type_ids)
        y = self.linear_layer(self.drop(pooled_h))
        return y


def get_bert_classifier(n_labels: int = None,
                        drop_prob: float = None,
                        bert_type: str = None,
                        ft_bert_file: str = None,
                        ):
    """
    Args:
        mode: "bert-base-chinese", "bert-base-uncased", or "fine-tuned"
    Return:
        nn.Module: Bert classifier
        :param ft_bert_file:
        :param bert_type:
        :param drop_prob:
        :param para_file:
        :param n_labels:
    """
    if bert_type:
        print("****** Load bert from {} ******".format(bert_type))
        bert_classifier = BertClassifier(bert_type=bert_type, n_labels=n_labels, drop_prob=drop_prob)
    elif ft_bert_file:
        print("****** Load bert from {} ******".format(bert_type))
        assert os.path.exists(ft_bert_file)
        bert_classifier = FTBertClassifier(ft_bert_file=ft_bert_file, n_labels=n_labels, drop_prob=drop_prob)
    else:
        raise NotImplementedError

    return bert_classifier
