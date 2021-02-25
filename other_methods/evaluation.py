"""
SVM model;
NB model;
tf_mat
"""
from other_methods.data_loader import DataSet

# Load model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import numpy as np


def main():
    # Hyper parameters
    modes = ["filtered", "all"]
    train_sizes = [30, 50, 100, "proportion"]
    task_names = ["money", "quality"]

    for task_name in task_names:
        for mode in modes:
            for train_size in train_sizes:
                data_set = DataSet()
                data_set.load_data(data_file="data/train_data/"+task_name+"_seg.csv", mode=mode)
                train_x, train_y, test_x, test_y = data_set.trans_data(train_size=train_size, is_tf_idf=False)

                models = {"bernoulli": BernoulliNB(),
                          "gaussian": GaussianNB(),
                          "Multinomial": MultinomialNB(),
                          "svm": SVC(gamma="auto")}

                for model_name in models:
                    model = models[model_name]
                    model.fit(train_x, train_y)
                    predicted = model.predict(test_x)
                    predicted = predicted.tolist()

                    assert len(predicted) == len(test_y)
                    count = 0
                    for index, pred in enumerate(predicted):
                        if pred == test_y[index]:
                            count += 1

                    print("{}-{}-{}-{}:".format(task_name, mode, str(train_size), model_name))
                    print(predicted)
                    print(test_y)
                    print(float(count) / len(test_y))
                    print("\n#######################\n")


if __name__ == '__main__':
    main()
