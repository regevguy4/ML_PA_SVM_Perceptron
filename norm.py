import numpy as np


# returns the params of the normalization on the data_set .
def minmax_params(dset):
    features = dset.shape[1] - 1

    old_mins = np.zeros(features)
    denoms = np.zeros(features)

    for i in range(features):
        old_mins[i] = np.amin(dset[:, i])
        denoms[i] = np.amax(dset[:, i]) - old_mins[i]

    return old_mins, denoms


def minmax(dset, new_min, new_max, old_mins=None, denoms=None):
    _dset = np.copy(dset)
    features = _dset.shape[1] - 1

    if old_mins is None and denoms is None:

        for i in range(features):
            old_min = np.amin(_dset[:, i])
            denom = np.amax(_dset[:, i]) - old_min

            value = (_dset[:, i] - old_min) / denom

            _dset[:, i] = value * (new_max - new_min) + new_min

        return _dset

    for i in range(features):
        value = (_dset[:, i] - old_mins[i]) / denoms[i]

        _dset[:, i] = value * (new_max - new_min) + new_min

    return _dset


def zscore_params(dset):

    """
    :param dset: the data set which working on.
    :return: the params of the normalization on 'dset'.
    """

    features = dset.shape[1] - 1

    old_means = np.zeros(features)
    old_stds = np.zeros(features)

    for i in range(features):
        old_means[i] = np.mean(dset[:, i])
        old_stds[i] = np.std(dset[:, i])

    return old_means, old_stds


def zscore(dset, train_means=None, train_stds=None):

    """
    :param dset: the data set which we like to normalize.
    :param train_means: (optional) the means of the features of the train set.
    :param train_stds: (optional) the stand' dev of the features of the train set.
    :return: a copy of 'dest', only z-scored normalized.
    """

    _dset = np.copy(dset)
    features = dset.shape[1] - 1

    if train_means is None and train_stds is None:

        for i in range(features):
            mean = np.mean(_dset[:, i])
            stand_dev = np.std(_dset[:, i])

            if stand_dev != 0:
                _dset[:, i] = (_dset[:, i] - mean) / stand_dev

        return _dset

    for i in range(features):
        if train_stds[i] != 0:
            _dset[:, i] = (_dset[:, i] - train_means[i]) / train_stds[i]

    return _dset
