import numpy as np
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score


def sigmoid(z):
    """ computes the sigmoid function of z
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    """ computes the gradient of the sigmoid function at z
    """
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNet(object):
    """ class that contains a Neural Network
        and functions for intialising and training it
    """

    def __init__(self, n_hidden, l=1, epsilon=0.12):
        """ creates a neural network with one hidden layer
                n_hidden: number of hidden units
                l: the regularisation when training and calculating cost
                epsilon: the std dev when generating initial random weights
        """
        self._n_hidden = n_hidden
        self._l = l
        self._epsilon = epsilon


    def _cost(self, X, y):
        # computes forward propagation

        # number of training examples
        m = X.shape[0]

        # perform forward propagation. hX is the hypothesised output
        # X already has the bias units added
        hX = X
        hX = sigmoid(np.dot(np.hstack((np.ones((m, 1)), hX)), self._theta1.T))
        hX = sigmoid(np.dot(np.hstack((np.ones((X.shape[0], 1)), hX)), self._theta2.T))

        # calculate the cost
        J = (1.0 / m) * (-y * np.log(hX) - (1.0 - y) * np.log(1.0 - hX)).sum()
        # calculate regularisation
        reg = 0
        reg += (self._theta1[:, 1:] ** 2).sum()
        reg += (self._theta2[:, 1:] ** 2).sum()
        reg *= self._l / (2.0 * m)
        return J + reg


    def _costGrad(self, X, y):
        # number of training examples
        m = X.shape[0]

        # set gradient vectors
        Delta1 = np.zeros(self._theta1.shape, float)
        Delta2 = np.zeros(self._theta2.shape, float)

        # store both thetaGrad1 and thetaGrad2 in a continuous block of memory,
        # so that the whole set of theta parameters can easily be flattened for
        # use by optimisation routines

        thetaGrad = np.zeros(self._theta.shape)

        thetaGrad1 = thetaGrad[:self._theta1.shape[0], :self._theta1.shape[1]]
        thetaGrad2 = thetaGrad[:, self._theta1.shape[1]:].T

        for i in range(m):
            # propagate forward
            a1 = np.vstack((1, X[i, :, np.newaxis]))

            z2 = np.dot(self._theta1, a1)
            a2 = np.vstack((1, sigmoid(z2)))

            z3 = np.dot(self._theta2, a2)
            a3 = sigmoid(z3)

            # calculate error between calculated and actual
            delta3 = a3 - y[i, :, np.newaxis]
            # propagate error backward
            delta2 = np.dot(self._theta2.T, delta3)[1:, :] * sigmoid_grad(z2)

            # accumulate error
            Delta1 += np.dot(delta2, a1.T)
            Delta2 += np.dot(delta3, a2.T)

        # calculate the gradients, with regularisation.
        # Note that the first column of theta is removed, as it corresponds
        # to the bias units
        thetaGrad1[:] = (1.0 / m) * (Delta1 + self._l * np.hstack((
            np.zeros((self._theta1.shape[0], 1)), self._theta1[:, 1:])))
        thetaGrad2[:] = (1.0 / m) * (Delta2 + self._l * np.hstack((
            np.zeros((self._theta2.shape[0], 1)), self._theta2[:, 1:])))
        return thetaGrad.ravel()


    def fit(self, X, y, batch_size=20, n_epochs=1000, learning_rate=0.001):
        """ returns the cost
        """

        # make y a vertical array if it's not
        if y.ndim == 1:
            y = y[:, np.newaxis]

        input_size = X.shape[1]
        output_size = y.shape[1]
        n_samples = X.shape[0]

        # store both theta1 and theta2 in a continuous block of memory, so that
        # the whole set of theta parameters can easily be flattened for use
        # by optimisation routines
        self._theta = np.zeros((self._n_hidden + 1, input_size + output_size + 1))

        # define theta1 and theta2 as views into the theta block
        self._theta1 = self._theta[:self._n_hidden, :input_size + 1]
        self._theta2 = self._theta[:, input_size + 1:].T

        # randomly initialise the weights
        self._theta1[:] = np.random.rand(*self._theta1.shape) * 2 * self._epsilon - self._epsilon
        self._theta2[:] = np.random.rand(*self._theta2.shape) * 2 * self._epsilon - self._epsilon

        # generate randomised indices
        epoch_split = ShuffleSplit(n_samples, n_iter=n_epochs, test_size=batch_size)

        for epoch, (train_index, valid_index) in enumerate(epoch_split):
            # split the training index into minibatches
            for ind in [train_index[i:i+batch_size] for i in range(0, len(train_index), batch_size)]:
                theta_grad = self._costGrad(X[ind], y[ind])
                # update parameters
                self._theta.ravel()[:] = self._theta.ravel()[:] - learning_rate * theta_grad

            # check the model on the last batch in the epoch
            y_pred = self.predict(X[valid_index])
            print("epoch {} accuracy: {}%".format(
                epoch,
                accuracy_score(y[valid_index].argmax(1), y_pred.argmax(1)) * 100))


    def predict(self, X):
        # add the bias units to the input matrix
        m = X.shape[0]

        hX = X
        hX = sigmoid(np.dot(np.hstack((np.ones((m, 1)), hX)), self._theta1.T))
        hX = sigmoid(np.dot(np.hstack((np.ones((X.shape[0], 1)), hX)), self._theta2.T))

        return hX