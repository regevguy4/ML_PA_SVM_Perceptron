import utils
import norm
from SVM import SVM
from perceptron import Perceptron
from PA import PA

import sys
import numpy as np


def train(train_set, per, svm, pa, epochs):
    for i in range(epochs):

        # shuffle the training set .
        np.random.seed(4)
        np.random.shuffle(train_set)

        # splitting the training set to examples and labels.
        x_train, y_train = train_set[:, :-1], train_set[:, -1]

        for x, y in zip(x_train, y_train):
            # training each algorithm.
            per.train(x, y)
            svm.train(x, y)
            pa.train(x, y)

        # updating the learning rate for the perceptron and SVM.
        per.eta = per.eta ** 2
        svm.eta = svm.eta ** 2


def predict(test_set, per, svm, pa):

    for x in test_set:
        per.test(x)
        svm.test(x)
        pa.test(x)


def main(argv):

    # creating the algorithms. the optimal hyper-parameters are hardcoded.
    per = Perceptron()
    svm = SVM()
    pa = PA()

    # loading the train set, including the labels on the rightmost column.
    train_set = utils.load_dset(argv[0], argv[1])

    # loading the test set. add a right column of ones for the bias
    test_x = utils.load_samples(argv[2])

    # normalizing the train and the test sets with min-max norm. min = 4, max = 30.
    mins, denoms = norm.minmax_params(train_set)
    train_set = norm.minmax(train_set, 4, 30)
    test_x = norm.minmax(test_x, 4, 30, mins, denoms)

    # training the algorithms .
    train(train_set, per, svm, pa, 100)

    # predicting and printing the results on the test set.
    predict(test_x, per, svm, pa)


if __name__ == "__main__":
    main(sys.argv[1:])
