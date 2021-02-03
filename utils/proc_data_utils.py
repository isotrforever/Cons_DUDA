"""
Transform InputExample into torch based form;
Dump data into file;
1 - Read raw data;
2 - Data augmentation;
3 - Transform data into torch form;
4 - Dump;
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert_uda.utils.raw_data_utils import *
from bert_uda.augmentation.word_level_augment import *
from bert_uda.augmentation.sent_level_augment import *
from bert_uda.configuration import CFG
import os
import copy
import torch
from tqdm import tqdm
from transformers import BertTokenizer


# Path to the files of task
task_path = {
    "imdb": "../data/raw_data/IMDB_raw",
    "dpp": "../data/raw_data/DPP_raw",
}


class TorchExample:
    """ A standard item fot the input of UDA model in torch version """

    def __init__(self, guid: str,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 token_type_ids: torch.Tensor,
                 label: torch.Tensor = None,
                 ):
        self.guid = guid
        self.label = label
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids


def data_to_torch(task_name: str,
                  bert_type: str,
                  sub_set: str,
                  input_length: int,
                  aug_ops: str = None,
                  back_translation_file: str = None,
                  ):
    """
    Purpose:
        1 - Get raw data from csv;
        2 - Augment as back translation or tf-idf replacement;
        3 - translate
    File_structure:
        /task_raw/
               |_______csv/
               |        |________"train.csv", "test.csv", "pred.csv", "unsup_ext.csv"
               |         sub_set: train/unsup_in,   test,       pred,      unsup_ext
               |
               |_______unsup/aug_ops/
               |            |____"tokenized.pt", "unsup_bert_input.pt"
               |
               |_______sup/
               |         |_______"train_bert_input.pt", "test_bert_input"
               |
               |_______pred/
               |          |______"pred_bert_input.pt"
               |
               |_______"vocab.pt"
    Args:
        task_name: "imdb" or "dpp";
        sub_set: The source of raw data;
                 "train", "test", "pred", unsup_ext, or "unsup_in";
        bert_type: "bert-base-chinese" or "bert-base-uncased";
        aug_ops: None, "bt-0.9", "unif-0.9" or "tf_idf-0.9";
    Return:
        List [TorchExample]
        :param input_length:
        :param task_name:
        :param bert_type:
        :param sub_set:
        :param back_translation_file:
        :param aug_ops:
    """
    ############################
    # Initialize BertTokenizer #
    ############################
    global vocab
    global aug_examples
    global aug_examples_bert
    tokenizer = BertTokenizer.from_pretrained(bert_type)

    ################################
    # Load raw data from CSV model #
    ################################
    assert task_name in ["imdb", "dpp"]
    processor = get_processor(task_name=task_name)

    if sub_set == "train":
        raw_examples = processor.get_train_examples()
    elif sub_set == "test":
        raw_examples = processor.get_test_examples()
    elif sub_set == "pred":
        raw_examples = processor.get_pred_examples()
    elif sub_set == "unsup_in":
        assert aug_ops.split("-")[0] in ["bt", "unif", "tf_idf"]
        raw_examples = processor.get_unsup_examples(unsup_set=sub_set)
    elif sub_set == "unsup_ext":
        assert aug_ops.split("-")[0] in ["bt", "unif", "tf_idf"]
        raw_examples = processor.get_unsup_examples(unsup_set=sub_set)
    else:
        raw_examples = []
    print("******** Raw examples have been loaded. ********")

    ###################################
    # Get word_list_a and word_list_b #
    ###################################
    iter_bar = tqdm(raw_examples)
    for raw_example in iter_bar:
        raw_example.word_list_a = tokenizer.tokenize(raw_example.text_a)
        if raw_example.text_b:
            raw_example.word_list_b = tokenizer.tokenize(raw_example.text_b)
    print("******** Raw examples have been tokenized. ********")

    ###########################
    # Get vocab from examples #
    ###########################
    def build_vocab(examples):
        vocab_dict = []

        def add_to_vocab(word_list):
            for word in word_list:
                if word not in vocab_dict:
                    vocab_dict.append(word)

        bar = tqdm(range(len(examples)))
        for i in bar:
            add_to_vocab(examples[i].word_list_a)
            if examples[i].text_b:
                add_to_vocab(examples[i].word_list_b)

        return vocab_dict

    ###########################
    ###########################
    # Get and save vocabulary #
    ###########################
    # Vocab_file is defined by it's task name and relevant augmentation methods
    if "unsup" in sub_set:
        vocab_file = os.path.join(task_path[task_name], "vocab.pt")
        if os.path.exists(vocab_file):
            vocab = torch.load(vocab_file)
        else:
            vocab = build_vocab(examples=raw_examples)
            torch.save(vocab, vocab_file)
        print("******** Vocabulary has been built successfully. ********")

    ###################
    # Augment if need #
    ###################
    if "unsup" in sub_set:
        # Deep copy raw_examples in case of raw_examples being changed in augmentation
        tmp_examples = copy.deepcopy(raw_examples)
        if aug_ops[:2] == "bt":
            # ?????????????????????????????????????????????? #
            # Back translation function is not available now #
            # ?????????????????????????????????????????????? #
            assert back_translation_file is not None
            aug_examples = sent_augment(examples=tmp_examples,
                                        aug_ops=aug_ops,
                                        back_translation_file=back_translation_file,
                                        )
        elif aug_ops[:4] == "unif":
            data_stats = get_data_stats(tmp_examples)
            aug_examples = word_augment(examples=tmp_examples, aug_ops=aug_ops, vocab=vocab, data_stats=data_stats)
        elif aug_ops[:6] == "tf_idf":
            data_stats = get_data_stats(tmp_examples)
            aug_examples = word_augment(examples=tmp_examples, aug_ops=aug_ops, vocab=vocab, data_stats=data_stats)
        else:
            raise ValueError

    ###############################
    ###############################
    # Save raw and augmented data #
    ###############################
    # Save tokenized examples, the file name is defined by task name
    # "train", "test", "pred", unsup_ext, "unsup_ext" or "unsup_in"
    if sub_set == "train":
        if not os.path.exists(os.path.join(task_path[task_name], "sup")):
            os.mkdir(os.path.join(task_path[task_name], "sup"))
        tokenized_file = os.path.join(task_path[task_name], "sup", "train_tokenized.pt")
        torch.save(raw_examples, tokenized_file)
    elif sub_set == "test":
        if not os.path.exists(os.path.join(task_path[task_name], "sup")):
            os.mkdir(os.path.join(task_path[task_name], "sup"))
        tokenized_file = os.path.join(task_path[task_name], "sup", "test_tokenized.pt")
        torch.save(raw_examples, tokenized_file)
    elif sub_set == "pred":
        if not os.path.exists(os.path.join(task_path[task_name], "pred")):
            os.mkdir(os.path.join(task_path[task_name], "pred"))
        tokenized_file = os.path.join(task_path[task_name], "pred", "pred_tokenized.pt")
        torch.save(raw_examples, tokenized_file)
    elif "unsup" in sub_set:
        if not os.path.exists(os.path.join(task_path[task_name], "unsup", aug_ops)):
            os.mkdir(os.path.join(task_path[task_name], "unsup", aug_ops))
        tokenized_file = os.path.join(task_path[task_name], "unsup", aug_ops, "unsup_tokenized.pt")
        unsup_pairs = list(zip(raw_examples, aug_examples))
        torch.save(unsup_pairs, tokenized_file)
    else:
        tokenized_file = os.path.join(task_path[task_name], "unsup", aug_ops, "unsup_tokenized.pt")
        torch.save(raw_examples, tokenized_file)
    print("******** Transformed examples have been stored. ********")

    #################################
    # Change text(s) into Bert Type #
    #################################
    # Token raw examples by BertTokenizer
    raw_examples_bert = []
    iter_bar = tqdm(raw_examples)
    for raw_example in iter_bar:
        text = raw_example.text_a
        if raw_example.text_b:
            text_pair = (raw_example.text_a, raw_example.text_b)
        else:
            text_pair = None

        text_bert = tokenizer(text=text,
                              text_pair=text_pair,
                              return_tensors="pt",
                              padding="max_length",
                              truncation=True,
                              max_length=input_length,
                              )
        torch_example = TorchExample(guid=raw_example.guid,
                                     input_ids=text_bert["input_ids"],
                                     token_type_ids=text_bert["token_type_ids"],
                                     attention_mask=text_bert["attention_mask"],
                                     )

        if raw_example.label in processor.get_labels():
            # Change label into torch.tensor
            label = raw_example.label
            labels = processor.get_labels()
            # OneHot label
            torch_label = torch.zeros(len(labels))
            torch_label[labels.index(label)] = 1.
            torch_example.label = torch_label.unsqueeze(dim=0)
        raw_examples_bert.append(torch_example)
    print("******** Raw examples have been transformed. ********")

    # Token augment examples by BertTokenizer
    if "unsup" in sub_set:
        aug_examples_bert = []
        iter_bar = tqdm(aug_examples)
        for aug_example in iter_bar:
            text = aug_example.text_a
            if aug_example.text_b:
                text_pair = (aug_example.text_a, aug_example.text_b)
            else:
                text_pair = None

            text_bert = tokenizer(text=text,
                                  text_pair=text_pair,
                                  return_tensors="pt",
                                  padding="max_length",
                                  truncation=True,
                                  max_length=input_length,
                                  )
            torch_example = TorchExample(guid=aug_example.guid,
                                         input_ids=text_bert["input_ids"],
                                         token_type_ids=text_bert["token_type_ids"],
                                         attention_mask=text_bert["attention_mask"],
                                         )

            if aug_example.label in processor.get_labels():
                # Change label into torch.tensor
                label = aug_example.label
                labels = processor.get_labels()
                # OneHot label
                torch_label = torch.zeros(len(labels))
                torch_label.to(dtype=torch.float32)
                torch_label[labels.index(label)] = 1.
                aug_example.label = torch_label.unsqueeze(dim=0)

            aug_examples_bert.append(torch_example)
        print("******** Augmented examples have been transformed. ********")

    ########################
    ########################
    # Save bert input data #
    ########################
    # "train", "test", "pred", unsup_ext, "unsup_ext" or "unsup_in"
    if sub_set == "train":
        if not os.path.exists(os.path.join(task_path[task_name], "sup")):
            os.mkdir(os.path.join(task_path[task_name], "sup"))
        bert_file = os.path.join(task_path[task_name], "sup", "train_bert_input.pt")
        torch.save(raw_examples_bert, bert_file)
    elif sub_set == "test":
        if not os.path.exists(os.path.join(task_path[task_name], "sup")):
            os.mkdir(os.path.join(task_path[task_name], "sup"))
        bert_file = os.path.join(task_path[task_name], "sup", "test_bert_input.pt")
        torch.save(raw_examples_bert, bert_file)
    elif sub_set == "pred":
        if not os.path.exists(os.path.join(task_path[task_name], "pred")):
            os.mkdir(os.path.join(task_path[task_name], "pred"))
        bert_file = os.path.join(task_path[task_name], "pred", "pred_bert_input.pt")
        torch.save(raw_examples_bert, bert_file)
    elif "unsup" in sub_set:
        if not os.path.exists(os.path.join(task_path[task_name], "unsup", aug_ops)):
            os.mkdir(os.path.join(task_path[task_name], "unsup", aug_ops))
        bert_file = os.path.join(task_path[task_name], "unsup", aug_ops, "unsup_bert_input.pt")
        unsup_pairs = list(zip(raw_examples_bert, aug_examples_bert))
        torch.save(unsup_pairs, bert_file)
    print("******** Transformed examples have been stored. ********")


def proc_data_main(cfg: CFG):
    data_to_torch(task_name=cfg.task_name,
                  sub_set="unsup_in",
                  aug_ops=cfg.aug_ops,
                  bert_type=cfg.bert_type,
                  input_length=cfg.input_length,
                  )

    data_to_torch(task_name=cfg.task_name,
                  sub_set="train",
                  aug_ops=None,
                  bert_type=cfg.bert_type,
                  input_length=cfg.input_length,
                  )

    data_to_torch(task_name=cfg.task_name,
                  sub_set="test",
                  aug_ops=None,
                  bert_type=cfg.bert_type,
                  input_length=cfg.input_length,
                  )

    data_to_torch(task_name=cfg.task_name,
                  sub_set="pred",
                  aug_ops=None,
                  bert_type=cfg.bert_type,
                  input_length=cfg.input_length,
                  )


if __name__ == "__main__":
    # Get configuration
    cfg = CFG()
    proc_data_main(cfg=cfg)
