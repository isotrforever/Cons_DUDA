"""
Class Data set for ML
"""
import csv
import random
import numpy as np

from gensim import corpora, models


class DataSet:
    def __init__(self, data_file: str = None):
        if data_file:
            self.text_data = self.load_data(data_file=data_file)
        else:
            self.text_data = None

    def load_data(self, data_file: str, mode: str = "filtered"):
        if mode == "filtered":
            with open(data_file, "r") as rf:
                csv_reader = csv.reader(rf)
                data = [[int(row[0]), row[1]] for row in csv_reader]
                self.text_data = data
        elif mode == "all":
            with open(data_file, "r") as rf:
                csv_reader = csv.reader(rf)
                data = [[int(row[0]), row[1]] for row in csv_reader]
            with open("data/train_data/all_item_seg.csv", "r") as rf:
                csv_reader = csv.reader(rf)
                data.extend([[int(row[0]), row[1]] for row in csv_reader])
            self.text_data = data
        return data

    def trans_data(self,
                   train_size,
                   is_tf_idf=True):

        random.shuffle(self.text_data)

        #######################################
        # Get positive data and negative data #
        #######################################
        pos_items = []
        neg_items = []
        for item in self.text_data:
            if item[0] == 1:
                pos_items.append(item)
            else:
                neg_items.append(item)
        all_items = pos_items + neg_items

        #################
        # Get tf matrix #
        #################
        doc_tokenized = [item[1].split("\t") for item in all_items]
        dictionary = corpora.Dictionary()
        BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
        labels = [item[0] for item in all_items]

        tf_mat = np.zeros(shape=(len(all_items), len(dictionary.token2id)), dtype=int)
        for row, doc in enumerate(BoW_corpus):
            for uid, freq in doc:
                tf_mat[row, uid] = freq

        ######################
        # Get tf-idf if need #
        ######################
        if is_tf_idf:
            tfidf = models.TfidfModel(BoW_corpus, smartirs='ntc')
            tf_mat = np.zeros(shape=(len(all_items), len(dictionary.token2id)), dtype=np.float)
            for row, doc in enumerate(tfidf[BoW_corpus]):
                for uid, freq in doc:
                    tf_mat[row, uid] = freq

        #############################
        # Split train and test data #
        #############################
        pos_len = len(pos_items)
        neg_len = len(neg_items)
        pos_x = tf_mat[:pos_len, :]
        pos_y = labels[:pos_len]
        neg_x = tf_mat[pos_len:, :]
        neg_y = labels[pos_len:]

        if train_size == "proportion":
            train_size = int(0.8 * pos_len)
        elif train_size > pos_len:
            train_size = int(0.9*pos_len)

        train_x = np.concatenate((pos_x[:train_size, :], neg_x[:train_size, :]), axis=0)
        train_y = pos_y[:train_size] + neg_y[:train_size]

        test_x = np.concatenate((pos_x[train_size:, :], neg_x[train_size:pos_len, :]), axis=0)
        test_y = pos_y[train_size:] + neg_y[train_size:pos_len]

        return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    data_set = DataSet(data_file="data/train_data/money_seg.csv")

