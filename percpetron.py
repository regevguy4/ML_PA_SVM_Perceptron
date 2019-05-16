import numpy as np
import utils


class Perceptron:

    def __init__(self, num_features=10, num_classes=3):

        self.weights = np.zeros((num_classes, num_features))
        self.eta = 0.000001

    def train(self, x, y):

        # predict
        y_hat = np.argmax(np.dot(self.weights, x))

        # update
        if y != y_hat:
            self.weights[int(y), :] = self.weights[int(y), :] + self.eta * x
            self.weights[y_hat, :] = self.weights[y_hat, :] - self.eta * x

    def test(self, x, y):

        y_hat = np.argmax(np.dot(self.weights, x))

        if y != y_hat:
            return False
        else:
            return True


p = Perceptron()

epochs = 10
errors = 0
_min = -1
_max = 1


# loading the data from files and shuffles it.
data = utils.load_dset("train_x.txt", "train_y.txt")
# shuffle(data)

# splitting the data into 3 sets: training, validation and test.
training, validation = utils.split_dset(data, int(0.8 * data.shape[0]))


# get the params for normalizing the valid set, than normalize it and the training set.
# oldmins, denoms = norm.minmax_params(training)
# training = norm.minmax(training, _min, _max)
# validation = norm.minmax(validation, _min, _max, oldmins, denoms)

print("training size = " + str(training.shape))
print("validation size = " + str(validation.shape))
# print("test size = " + str(test.shape))
print('\n')

# training the perceptron, and than check it on the validation set.
utils.train(p, training, epochs)

utils.test(p, training)
print('\n')
utils.test(p, validation)
