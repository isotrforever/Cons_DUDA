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
Word level augmentations including Replace words with uniform random words or TF-IDF based word replacement.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import string
import collections
import numpy as np
from absl import flags
from tqdm import tqdm

FLAGS = flags.FLAGS

# Printable string is chars predefined by python 3
printable = set(string.printable)


def filter_unicode(st):
    """ Filter out the unpredictable chars in a sentence """
    return "".join([c for c in st if c in printable])


class EfficientRandomGen(object):
    """ A base class that generate multiple random numbers at the same time. """
    def __init__(self):
        self.token_list = None
        self.token_ptr = None

    def reset_random_prob(self):
        """ Generate many random numbers at the same time and cache them. """
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """ Get a random number. """
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value

    def get_random_token(self):
        """ Get a random token according to a random number. """
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token

    def reset_token_list(self):
        raise NotImplementedError


class UnifRep(EfficientRandomGen):
    """ Uniformly replace word with random words in the vocab. """
    def __init__(self, token_prob, vocab):
        super(UnifRep, self).__init__()
        """
        token_prob: float. to define the change probability;
        vocab: dict. keys are the vocab list;
        """
        self.vocab = vocab
        self.token_prob = token_prob
        self.vocab_size = len(vocab)

        # Get initial token list, token index and random number sequence;
        self.reset_token_list()
        self.reset_random_prob()

    def __call__(self, example):
        example.word_list_a = self.replace_tokens(example.word_list_a)
        example.text_a = " ".join(example.word_list_a)
        if example.text_b:
            example.word_list_b = self.replace_tokens(example.word_list_b)
            example.text_b = " ".join(example.word_list_b)
        return example

    def replace_tokens(self, tokens):
        """ Replace tokens randomly. """
        # Only augment tokens longer than 3;
        if len(tokens) >= 3:

            # Show example randomly;
            if np.random.random() < 0.001:
                show_example = True
            else:
                show_example = False
            if show_example:
                print("before augment: {:s}".format(" ".join(tokens)))

            # Replace token when the random number smaller than token probability;
            for i in range(len(tokens)):
                if self.get_random_prob() < self.token_prob:
                    tokens[i] = self.get_random_token()
            if show_example:
                print("after augment: {:s}".format(filter_unicode(" ".join(tokens))))

        return tokens

    def reset_token_list(self):
        """ Generate many random tokens at the same time and cache them. """
        self.token_list = self.vocab
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)


def get_data_stats(examples):
    """
    Purpose:
        Compute the IDF score for each word. Then compute the TF-IDF score.
        examples: list(class contains {word_list_a, word_list_b, text_b})
    Return:
        {"idf": dic(idf), "tf_idf": dic(tf_idf)}
        idf: {word: idf of the word}
        tf_idf: {word: global tf_idf of the word}
    """
    word_doc_freq = collections.defaultdict(int)

    # Compute IDF, return idf[word].
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i].word_list_a)
        if examples[i].text_b:
            cur_sent += examples[i].word_list_b
        for word in cur_sent:
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(examples) * 1. / word_doc_freq[word])

    # Compute TF-IDF
    tf_idf = {}
    for i in range(len(examples)):
        cur_sent = copy.deepcopy(examples[i].word_list_a)
        if examples[i].text_b:
            cur_sent += examples[i].word_list_b
        for word in cur_sent:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] += 1. / len(cur_sent) * idf[word]

    return {"idf": idf, "tf_idf": tf_idf}


class TfIdfWordRep(EfficientRandomGen):
    """ TF-IDF Based Word Replacement. """

    def __init__(self, token_prob, data_stats):
        super(TfIdfWordRep, self).__init__()
        """
        Parameters:
            data_stats: {"idf": idf, "tf_idf": tf_idf};
            token_prob: float. The probability to replace token. For example: "tf_idf-0.9" is 0.9 token_prob;
        """
        self.token_prob = token_prob
        self.data_stats = data_stats
        self.idf = data_stats["idf"]
        self.tf_idf = data_stats["tf_idf"]

        # Sort tf_idf_items
        data_stats = copy.deepcopy(data_stats)
        tf_idf_items = data_stats["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.tf_idf_keys = []
        self.tf_idf_values = []
        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]

        # Normalize tf_idf_values
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = (self.normalized_tf_idf.max() - self.normalized_tf_idf)
        self.normalized_tf_idf = (self.normalized_tf_idf / self.normalized_tf_idf.sum())

        # Initialize some parameters
        self.reset_token_list()
        self.reset_random_prob()

    def get_replace_prob(self, all_words):
        """
        Purpose:
            Compute the probability of replacing tokens in a sentence;
            all_words: list(string);
        Return:
            replace_prob: list(np.float);
        """
        cur_tf_idf = collections.defaultdict(int)
        for word in all_words:
            cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = replace_prob / replace_prob.sum() * self.token_prob * len(all_words)
        return replace_prob

    def __call__(self, example):
        """
        Purpose:
            Replace tokens in class example;
            :param example, class contains token_list_a, token_list_b, token_b;
        Return:
            Example after replaced;
        """
        # Show examples randomly when running program;
        if self.get_random_prob() < 0.001:
            show_example = True
        else:
            show_example = False
        all_words = copy.deepcopy(example.word_list_a)
        if example.text_b:
            all_words += example.word_list_b
        if show_example:
            print(("before tf_idf_unif aug: {:s}".format(" ".join(all_words))))

        # Replace token list in example;
        replace_prob = self.get_replace_prob(all_words)

        example.word_list_a = self.replace_tokens(
            example.word_list_a,
            replace_prob[:len(example.word_list_a)]
        )
        example.text_a = " ".join(example.word_list_a)

        if example.text_b:
            example.word_list_b = self.replace_tokens(
                example.word_list_b,
                replace_prob[len(example.word_list_a):]
            )
            example.text_b = " ".join(example.word_list_b)

        # Show tokens after replaced;
        if show_example:
            all_words = copy.deepcopy(example.word_list_a)
            if example.text_b:
                all_words += example.word_list_b
            print("after tf_idf_unif aug: {:s}".format(filter_unicode(" ".join(all_words))))

        return example

    def replace_tokens(self, word_list, replace_prob):
        """
        Purpose:
            Replace tokens in a sentence randomly.
        Return:
            word_list: List;
        """
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_random_token()
        return word_list

    def reset_token_list(self):
        """
        Purpose:
            Sample token according to the probability;
        Return:
            token_list: token with frequency according to probability;
            token_ptr: int;
        """
        cache_len = len(self.tf_idf_keys)
        # Sample with
        token_list_idx = np.random.choice(cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1
        print("sampled token list: {:s}".format(" ".join(self.token_list)))


def word_augment(examples: list, aug_ops: str, vocab: list, data_stats: dict):
    """
    Purpose:
        Word level augmentations.
        Used before augmentation.
    Parameter:
        examples: list[example];
            example: token_list_a, token_list_b, token_b;
        aug_ops: "unif-0.9" / "tf_idf-0.9"
        vocab: vocabulary list;
        data_sates: Get from function get_data_states();
    Return:
         examples;
    """
    if aug_ops:
        if aug_ops.startswith("unif"):
            print("\n>>Using augmentation {}".format(aug_ops))
            token_prob = float(aug_ops.split("-")[1])
            op = UnifRep(token_prob, vocab)
            iter_bar = tqdm(range(len(examples)))
            for i in iter_bar:
                examples[i] = op(examples[i])
        elif aug_ops.startswith("tf_idf"):
            print("\n>>Using augmentation {}".format(aug_ops))
            token_prob = float(aug_ops.split("-")[1])
            op = TfIdfWordRep(token_prob, data_stats)
            iter_bar = tqdm(range(len(examples)))
            for i in iter_bar:
                examples[i] = op(examples[i])
    return examples
