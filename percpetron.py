import numpy as np


class Perceptron:

    def __init__(self, num_features=10, num_classes=3):

        self.weights = np.zeros((num_classes, num_features))
        self.eta = 1

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

