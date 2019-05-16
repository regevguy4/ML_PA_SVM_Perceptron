import norm
import utils
from percpetron import Perceptron


p = Perceptron()

epochs = 10
errors = 0
_min = 5
_max = 10


# loading the data from files and shuffles it.
data = utils.load_dset("train_x.txt", "train_y.txt")

# splitting the data into 3 sets: training, validation and test.
training, validation = utils.split_dset(data, int(0.8 * data.shape[0]))


# get the params for normalizing the valid set, than normalize it and the training set.
oldmins, denoms = norm.minmax_params(training)
training = norm.minmax(training, _min, _max)
validation = norm.minmax(validation, _min, _max, oldmins, denoms)

print("training size = " + str(training.shape))
print("validation size = " + str(validation.shape))
print('\n')

# training the perceptron, and than check it on the validation set.
utils.train(p, training, epochs)

utils.test(p, training)
print('\n')
utils.test(p, validation)
