"""Basic NLP Tool for Segmentation, POS Tagging, and Dependency Analysis"""

from utils import *
import numpy as np

import jieba
import thulac
from pyltp import Segmentor, Postagger, Parser


# Initialize Segmentation Models
segmentor = Segmentor()
segmentor.load("/home/isotr/Scripts/PycharmProjects/ltp_data_v3.4.0/cws.model")
thu = thulac.thulac(seg_only=True)
# Load POS Tagging Model
postagger = Postagger()
postagger.load("/home/isotr/Scripts/PycharmProjects/ltp_data_v3.4.0/pos.model")
# Load Dependence Model
parser = Parser()
parser.load("/home/isotr/Scripts/PycharmProjects/ltp_data_v3.4.0/parser.model")


class NLPTool:
    # Segment Sentence by jie_ba, THU_LAC, and LTP
    def seg(self, sentence):
        # Method to Fuse Segmented Results
        def ensemble_seg_results(sentence, *args):
            split_tag = np.zeros(shape=(1, len(sentence)))
            # Vote
            for arg in args:
                init_tag = - np.ones(shape=(1, len(sentence)))
                blank_loc = 0
                for token in arg:
                    blank_loc += len(token)
                    init_tag[0, blank_loc - 1] = 1.
                split_tag = np.add(init_tag, split_tag)
            # Final Result
            result = []
            token = ""
            for i in range(split_tag.shape[1]):
                if split_tag[0, i] < 1:
                    token += sentence[i]
                else:
                    token += sentence[i]
                    result.append(token)
                    token = ""
            return result

        # Segment Sentence by Different Models; Fuse Results
        ltp_seg = " ".join(segmentor.segment(sentence)).split(" ")
        thu_seg = [item[0] for item in thu.cut(sentence)]
        jb_seg = " ".join(jieba.cut(sentence)).split(' ')
        fuse_seg = ensemble_seg_results(sentence, ltp_seg, thu_seg, jb_seg)

        return fuse_seg

    # POS Tagging Using LTP
    def pos(self, words):
        postag = " ".join(postagger.postag(words)).split(" ")
        return postag

    # Get Dependence Tree Using LTP
    def dep(self, words):
        postags = self.pos(words)
        arcs = parser.parse(words, postags)
        arcs_head = [item.head for item in arcs]
        arcs_relation = [item.relation for item in arcs]
        return arcs_head, arcs_relation

    # Get Dependence Tree using Words&Postags
    def new_dep(self, words, postags):
        arcs = parser.parse(words, postags)
        arcs_head = [item.head for item in arcs]
        arcs_relation = [item.relation for item in arcs]
        return arcs_head, arcs_relation

    # Parse Labeled Sentence
    def parse_sentence_label(self, sentence_label):
        # temp_word_label: [word, semantic_type]
        new_sentence_label = []
        for word_label in sentence_label:
            if "B" in word_label[1:]:
                temp_word_label = [word_label[0], 0]
                word_label[0] = ""
                temp_word_label[1] = word_label.index("B")
                new_sentence_label.append(temp_word_label)
            elif "I" in word_label[1:]:
                new_sentence_label[-1][0] += word_label[0]
                word_label[0] = ""
                if not new_sentence_label[-1][1] == word_label.index("I"):
                    raise ImportError
            else:
                temp_word_label = [word_label[0], 0]
                new_sentence_label.append(temp_word_label)

        # POS Tagging
        postags = self.pos(words=[word_label[0] for word_label in new_sentence_label])
        for index, element in enumerate(new_sentence_label):
            if element[1] == 1:
                postags[index] = "n"
            elif element[1] == 2:
                postags[index] = "ni"
            elif element[1] == 3:
                postags[index] = "v"
            elif element[1] == 4:
                postags[index] = "n"
            elif element[1] == 5:
                if postags[index] in ["d", "v"]:
                    pass
                else:
                    postags[index] = "a"

        # Dependence Parsing
        words = [word_label[0] for word_label in new_sentence_label]
        semantic_type = [word_label[1] for word_label in new_sentence_label]
        arcs = parser.parse(words, postags)
        arcs_head = [item.head for item in arcs]
        arcs_relation = [item.relation for item in arcs]

        return words, semantic_type, postags, arcs_head, arcs_relation

