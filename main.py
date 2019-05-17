import norm
import utils
from percpetron import Perceptron

import numpy as np
import matplotlib.pyplot as plt


def cross_validate(n_epochs, _min, _max, groups):
    """
    :param n_epochs: number of epochs.
    :param _min: min value for normalization.
    :param _max: max value for normalization.
    :param groups: groups: array of equal sized numpy matrices.
    :return: run groups.shape[0] times, and each time takes a different group
    to be the validation set, and all the rest are the training set. trains new
    perceptron alg' and print its accuracy. returns the average accuracy.
    """

    k = groups.shape[0]
    _sum = 0

    for i in range(k):

        train_set = None
        valid_set = np.copy(groups[i])  # the validation set for th i'th iteration.
        p = Perceptron()  # a new (!) perceptron object.

        for j in range(k):

            if j != i:
                # arange the train set for the i'th iteration.
                if train_set is None:
                    train_set = np.copy(groups[j])
                else:
                    train_set = np.concatenate((train_set, groups[j]), axis=0)

        # normalizing the train set and the valid set.
        # old_mins, denoms = norm.minmax_params(train_set)
        # train_set = norm.minmax(train_set, _min, _max)
        # valid_set = norm.minmax(valid_set, _min, _max, old_mins, denoms)

        # training the model with the i'th training set.
        utils.train(p, train_set, n_epochs)

        # printing the result of the i'th test on the validation set,
        # and saving the sum.
        print("iteration number " + str(i + 1) + " : ", end='')
        _sum += utils.test(p, valid_set)

    accuracy = float(_sum) / k

    # prints the average accuracy of the cross validation.
    print("the average accuracy of this session: " + str(accuracy) + " %\n")
    print("---------\n")

    return accuracy


def plot_epochs(max_ep, _min, _max, groups):
    """
    :param max_ep: the maximum num of epochs to be checked.
    :param _min: the min value for the normalization
    :param _max: the max value for the normalization
    :param groups: array of equal sized numpy matrices.
    :return: none, but print to the screen a graph of the average
    accuracy as a function of the numbers of epochs.
    """
    arr = np.zeros((max_ep, 2))

    # creating arr for showing accuracy as number of iterations
    for ep in range(max_ep):
        print("epoch number: " + str(ep))

        arr[ep, 0] = ep

        # cross_validate returning the average accuracy of the session,
        # and print some information on the session .
        arr[ep, 1] = cross_validate(ep, _min, _max, groups)

    # plotting the array.
    plt.plot(arr[:, 0], arr[:, 1])
    plt.show()

    # printing the max accuracy.
    print("the max accuracy is:" + str(max(arr[:, 1]))
          + " at epoch = " + str(np.argmax(arr[:, 1])))


"""

Main method

"""

n_groups = 5
max_epochs = 70

n_min = 5
n_max = 10

data = utils.load_dset("train_x.txt", "train_y.txt")
_groups = np.copy(np.array_split(data, n_groups))

plot_epochs(max_epochs, n_min, n_max, _groups)
