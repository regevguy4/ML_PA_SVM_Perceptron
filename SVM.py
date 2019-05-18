import numpy as np


class SVM:

    def __init__(self, eta=0.1, c=1, n_features=11, n_classes=3):

        self.weights = np.zeros((n_classes, n_features))

        self.eta = eta
        self.c = c

    def train(self, x, y):

        # predict
        y_hat = np.argmax(np.dot(self.weights, x))

        # update
        if y != y_hat:

            update = 1 - self.eta * self.c

            self.weights[int(y), :] = update * self.weights[int(y), :] + self.eta * x
            self.weights[y_hat, :] = update * self.weights[y_hat, :] - self.eta * x

            # updating the other weight vector.
            other = 3 - int(y) - y_hat
            self.weights[other, :] = update * self.weights[other, :]

    def test(self, x, y):

        # predict
        y_hat = np.argmax(np.dot(self.weights, x))

        if y != y_hat:
            return False
        else:
            return True
