# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sentence level augmentations: back translation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import numpy as np
from bert_uda.utils import raw_data_utils
from bert_uda.augmentation import word_level_augment


def replace_with_length_check(ori_text, new_text, use_min_length, use_max_length_diff_ratio):
    """
    Purpose:
        Back translation;
        Use new_text if the text length satisfies several constraints;
    """
    if len(ori_text) < use_min_length or len(new_text) < use_min_length:
        if random.random() < 0.001:
            print(
                "not replacing due to short text: \n\tori: {:s}\n\tnew: {:s}\n".format(
                    word_level_augment.filter_unicode(ori_text),
                    word_level_augment.filter_unicode(new_text))
            )
        return ori_text

    length_diff_ratio = 1.0 * (len(new_text) - len(ori_text)) / len(ori_text)
    if math.fabs(length_diff_ratio) > use_max_length_diff_ratio:
        if random.random() < 0.001:
            print(
                "not replacing due to too different text length:\n\tori: {:s}\n\tnew: {:s}\n".format(
                    word_level_augment.filter_unicode(ori_text),
                    word_level_augment.filter_unicode(new_text))
            )
        return ori_text

    return new_text


def back_translation(examples: list, aug_ops: str, back_translation_file: str):
    """
    Purpose:
        Run back translation.
        examples: class with attributes: guid, text_a, text_b, label;
        aug_ops: [1]-[2]-[3] or [1]-[2]
            [1] "bt" or "unif" / "tf_idf";
            [2] float: 0.9;
            [3] float: 1;
        sub_set: "sup/unsup", used to track back-translation file;
        # start: the beginning index of target item;
        # end: the ending index of target item;
        # data_total_size: int, the number of data;
    Return:
        augmented examples just like original examples;
    """
    # Set hyper-parameters
    use_min_length = 10
    use_max_length_diff_ratio = 0.5
    print("Running bt augmentation")
    bt_args = aug_ops.split("-")

    if len(bt_args) > 2:
        assert len(bt_args) == 3
        assert float(bt_args[2]) == 1.

    if examples[0].text_b is not None:
        text_per_example = 2
    else:
        text_per_example = 1

    # Load back-translation file
    print("Using back translation file: {:s}".format(back_translation_file))
    with open(back_translation_file) as inf:
        paraphrases = inf.readlines()
    for i in range(len(paraphrases)):
        paraphrases[i] = paraphrases[i].strip()
    assert len(paraphrases) == text_per_example * len(examples)
    # print("{}******{}".format(len(paraphrases), data_total_size))
    # assert len(paraphrases) == data_total_size

    # Get the bt augmented examples from start to end
    # If there are paired sentences in example, the start is:
    # paraphrases = paraphrases[start * text_per_example: end * text_per_example]

    # Get relevant augmented examples
    aug_examples = []
    aug_cnt = 0
    for i in range(len(examples)):
        ori_example = examples[i]
        text_a = replace_with_length_check(
            ori_example.text_a,
            paraphrases[i * text_per_example],
            use_min_length,
            use_max_length_diff_ratio,
        )
        if text_a == paraphrases[i * text_per_example]:
            aug_cnt += 1
        if ori_example.text_b is not None:
            text_b = replace_with_length_check(
                ori_example.text_b,
                paraphrases[i * text_per_example + 1],
                use_min_length,
                use_max_length_diff_ratio,
            )
        else:
            text_b = None

        example = raw_data_utils.InputExample(
            guid=ori_example.guid,
            text_a=text_a,
            text_b=text_b,
            label=ori_example.label)
        aug_examples += [example]

        # Show state randomly
        if np.random.random() < 0.0001:
            print("\tori:\n\t\t{}\n\t\t{}\n\t\t{}\n".format(
                ori_example.text_a, ori_example.text_b, ori_example.label)
            )
            print("\tnew:\n\t\t{}\n\t\t{}\n\t\t{}\n".format(
                example.text_a, example.text_b, example.label)
            )
        if i % 10000 == 0:
            print("processing example # {:d}".format(i))
    print("applied back translation for {:.1f} percent of data".format(aug_cnt * 1. / len(examples) * 100))
    print("finishing running back translation augmentation")
    return aug_examples


def sent_augment(examples, aug_ops, back_translation_file):
    """ Sentence level augmentations. Used before augmentation. """
    if aug_ops:
        if aug_ops.startswith("bt"):
            examples = back_translation(examples, aug_ops, back_translation_file)
        else:
            pass
    return examples
