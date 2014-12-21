import numpy as np
import scipy.optimize as op
import scipy.sparse as sp


def sigmoid(z):
    """ computes the sigmoid function of z
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoidGrad(z):
    """ computes the gradient of the sigmoid function at z
    """
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNet(object):
    """ class that contains a Neural Network
        and functions for intialising and training it
    """

    def __init__(self, n_hidden, maxiter=100, l=1, epsilon=0.12):
        """ creates a neural network with one hidden layer
                n_hidden: number of hidden units
                maxiter: the number of iterations when training
                l: the regularisation when training and calculating cost
                epsilon: the std dev when generating initial random weights
        """
        self._n_hidden = n_hidden
        self._l = l
        self._maxiter = maxiter
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
            delta2 = np.dot(self._theta2.T, delta3)[1:, :]*sigmoidGrad(z2)

            # accumulate error
            Delta1 += np.dot(delta2, a1.T)
            Delta2 += np.dot(delta3, a2.T)

        # calculate the gradients, with regularisation.
        # Note that the first column of theta is removed, as it corresponds
        # to the bias units
        thetaGrad1[:] = (1.0 / m) * (Delta1 + self._l*np.hstack((
                np.zeros((self._theta1.shape[0], 1)), self._theta1[:, 1:])))
        thetaGrad2[:] = (1.0 / m) * (Delta2 + self._l*np.hstack((
                np.zeros((self._theta2.shape[0], 1)), self._theta2[:, 1:])))
        return thetaGrad.ravel()


    def fit(self, X, y):
        """ returns the cost
        """
        #define functions to be minimised
        def f(thetaParams):
            # set theta
            self._theta.ravel()[:] = thetaParams
            # calculate the cost
            return self._cost(X, y)

        def fGrad(thetaParams):
            # set theta
            self._theta.ravel()[:] = thetaParams
            # calculate the cost gradient
            return self._costGrad(X, y)

        # make y a vertical array if it's not
        if y.ndim == 1:
            y = y[:, np.newaxis]

        inputSize = X.shape[1]
        outputSize = y.shape[1]

        # store both theta1 and theta2 in a continuous block of memory, so that
        # the whole set of theta parameters can easily be flattened for use
        # by optimisation routines
        self._theta = np.zeros((self._n_hidden + 1, inputSize+outputSize+1))

        # define theta1 and theta2 as views into the theta block
        self._theta1 = self._theta[:self._n_hidden, :inputSize+1]
        self._theta2 = self._theta[:, inputSize+1:].T

        # randomly initialise the weights
        self._theta1[:] = np.random.rand(*self._theta1.shape) * 2*self._epsilon - self._epsilon
        self._theta2[:] = np.random.rand(*self._theta2.shape) * 2*self._epsilon - self._epsilon

        # minimise
        thetaParams = op.fmin_bfgs(f, self._theta.ravel(), fGrad,
                maxiter=self._maxiter, full_output=0)
        # set parameters
        self._theta.ravel()[:] = thetaParams


    def predict(self, X):
        # add the bias units to the input matrix
        m = X.shape[0]

        hX = X
        hX = sigmoid(np.dot(np.hstack((np.ones((m, 1)), hX)), self._theta1.T))
        hX = sigmoid(np.dot(np.hstack((np.ones((X.shape[0], 1)), hX)), self._theta2.T))

        return hX