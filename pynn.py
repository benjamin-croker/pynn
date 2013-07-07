import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.optimize as op


def sigmoid(z):
    """ computes the sigmoid function of z
    params:
        z - a numpy array
    returns:
        g(z)"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoidGrad(z):
    """ computes the gradient of the sigmoid function at z
    params:
        z - a numpy array
    returns:
        g'(z)"""
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNet(object):
    """ class that contains a Neural Network
    and functions for intialising and training it"""

    def __init__(self, layerSizes):
        """ creates a NeuralNet object
        params:
            layerSizes - a list of the number of units in each layer.
                Must be of length at least 3 (input, hidden and output).
                The sizes do NOT included the bias units, which are included
                on all layers except for the output layer
        """
        # internal variables set:
        #	self._theta - list of theta matrices
        self._theta = []

        # randomly initialise the weights
        # TODO: move EPSILON to a parameter
        EPSILON = 0.12
        for i in range(len(layerSizes) - 1):
            self._theta += [np.random.rand(layerSizes[i + 1], layerSizes[i] + 1)
                            * 2 * EPSILON - EPSILON]

    def _cost(self, X, y, l):
        # first compute forward propogation.
        # number of training examples
        m = X.shape[0]

        # perform forward propagation. hX is the hypothesised output
        hX = X
        for t in self._theta:
            hX = sigmoid(np.dot(np.hstack([np.ones((m, 1)), hX]), t.transpose()))

        # calculate the cost
        J = (1.0 / m) * (-y * np.log(hX) - (1.0 - y) * np.log(1.0 - hX)).sum()
        # calculate regularisation
        reg = 0
        for t in self._theta:
            reg += (t[:, 1:] ** 2).sum()
        reg *= l / (2.0 * m)

        return J + reg

    def _costGrad(self, X, y, l):
        # number of training examples
        m = X.shape[0]
        # set gradient vectors
        Delta = [np.zeros(t.shape, float) for t in self._theta]
        thetaGrad = [np.zeros(t.shape, float) for t in self._theta]

        for i in range(m):
            # first layer just the inputs with the bias unit
            a = [np.vstack([1, X[i, :, np.newaxis]])]
            # there is no input z, so use a dummy value
            z = [0]

            # propagate forward
            for j in range(len(self._theta)):
                z.append(np.dot(self._theta[j], a[-1]))
                a.append(np.vstack([1, sigmoid(z[-1])]))
                # the last layer does not have a bias unit
            a[-1] = a[-1][1:, :]

            # calculate error between calculated and actual
            delta = [a[-1] - y[i, :, np.newaxis]]

            # propagate error backward
            for j in range(len(self._theta))[::-1]:
                # there is no error for the input layer
                if j == 0:
                    delta.insert(0, 0)
                else:
                    delta.insert(0,
                                 np.dot(self._theta[j].transpose(), delta[0])[1:, :]
                                 * sigmoidGrad(z[j])
                    )
                Delta[j] += np.dot(delta[1], a[j].transpose())

        # calculate the gradients, with regularisation.
        # Note that the first column of theta is removed, as it corresponds
        # to the bias units
        for j in range(len(self._theta)):
            s = self._theta[j].shape

            thetaGrad[j] = (1.0 / m) * (Delta[j] + l * np.hstack([
                np.zeros((s[0], 1)), self._theta[j][:, 1:]
            ]))

        return self._unroll(thetaGrad)

    # TODO: Check backprop values & set functions to be stuck into fmin

    def train(self, X, y, l):
        """ returns the cost
        """
        #define functions to be minimised
        def f(thetaParams):
            # set theta
            self._theta = self._roll(thetaParams)
            # calculate the cost
            return self._cost(X, y, l)

        def fGrad(thetaParams):
            # set theta
            self._theta = self._roll(thetaParams)
            # calculate the cost
            return self._costGrad(X, y, l)

        def callback(*args):
            print 'iter'

        # minimise
        thetaParams = op.fmin_cg(f, self._unroll(self._theta), fGrad, maxiter=100,
                                 full_output=1)
        # set parameters
        self._theta = self._roll(thetaParams[0])
        return thetaParams[1]

    def predict(self, X):
        # number of training examples
        m = X.shape[0]

        hX = X
        for t in self._theta:
            hX = sigmoid(np.dot(np.hstack([np.ones((m, 1)), hX]), t.transpose()))

        return hX

    def _unroll(self, arrList):
        """ rolls up all theta parameters into a single vector
            params:
                listVs - a list of arrays
            returns:
                the rolled vector
        """
        return np.hstack([v.ravel() for v in arrList])

    def _roll(self, V):
        """ takes an unrolled up theta vector and rolls it into
        the individual theta vectors
            params:
                V - the rolled up vector
            returns:
                a list of the theta vectors
        """
        # m is an index indicating the number of elements have been unrolled
        m = 0
        theta = []
        # print V.shape
        # for t in self._theta:
        # 	print t.shape

        for i in range(len(self._theta)):
            theta.append(V[m:m + self._theta[i].size].reshape(self._theta[i].shape))
            m = m + self._theta[i].size
        return theta