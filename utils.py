import numpy as np


def load_samples(path):

    """
    for loading the samples, and returning it as an np Array.
    """
    with open(path, "r") as f:
        arr = [[str(num) for num in line.split(',')] for line in f]

    np_arr = np.copy(arr)
    n_samples = np_arr.shape[0]  # the number of samples.

    gender = np_arr[:, 0]  # gets array of ganders (the 0'th col).
    np_arr = np_arr[:, 1:].astype(float)  # gets array of all of the numeric values.
    np_arr = np.c_[np_arr, np.ones((n_samples, 1))]   # add right col of ones for bias.

    one_hot = np.zeros((n_samples, 3))  # creating the one hot encoded array.

    dic = {'M': 0, 'F': 1, 'I': 2}

    # converting genders to their encoded value.
    for i in range(n_samples):
        g = gender[i]
        one_hot[i, dic[g]] = 1

    # concatenate the sample's right features .
    return np.c_[one_hot, np_arr]


def load_labels(path):

    """
    for loading the labels, and returning them as an np Array.
    """

    return np.loadtxt(path)


def load_dset(examples_path, labels_path):

    """
    loading the samples and the labels from their files,
    and returning them as one data set.
    """

    examples = load_samples(examples_path)

    labels = load_labels(labels_path)

    return np.c_[examples, labels]


def split_dset(d_set, line):

    arr = np.copy(d_set[: line, :])
    arr2 = np.copy(d_set[line:, :])

    return arr, arr2


def train(alg, train_set, epochs):

    """
    training the perceptron.
    """

    for i in range(epochs):

        np.random.seed(4)
        np.random.shuffle(train_set)

        x_train, y_train = train_set[:, :-1], train_set[:, -1]

        for x, y in zip(x_train, y_train):

            alg.train(x, y)

        alg.eta = alg.eta ** 2


def test(alg, dset):

    """
    testing the perceptron on the validation set,
    and prints the results.
    """

    errors = 0
    x_val, y_val = dset[:, :-1], dset[:, -1]

    for x, y in zip(x_val, y_val):

        if not alg.test(x, y):
            errors = errors + 1

    loss = float(errors) / dset.shape[0]
    print(str((1 - loss) * 100) + " % of accuracy")

    return (1 - loss) * 100
