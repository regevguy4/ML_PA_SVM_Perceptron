import numpy as np
from numpy import linalg as la


class PA:

    def __init__(self, n_features=11, n_classes=3):

            self.w = np.zeros((n_classes, n_features))
            self.tao = 0

    def update_tao(self, x, y, y_hat):

        loss = max(0, 1 - np.dot(self.w[int(y), :], x) + np.dot(self.w[y_hat, :], x))
        self.tao = loss / (2 * (la.norm(x) ** 2))

    def train(self, x, y):

        # predict
        y_hat = np.argmax(np.dot(self.w, x))

        self.update_tao(x, y, y_hat)

        # update
        if y != y_hat:

            self.w[int(y), :] = self.w[int(y), :] + self.tao * x
            self.w[y_hat, :] = self.w[y_hat, :] - self.tao * x

    def test(self, x):

        # predict
        y_hat = np.argmax(np.dot(self.w, x))

        print("pa: " + str(y_hat))

