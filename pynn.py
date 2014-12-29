import numpy as np
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score
import theano
import theano.tensor as T


def sigmoid(z):
    """ computes the sigmoid function of z
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(z):
    """ computes the gradient of the sigmoid function at z
    """
    return sigmoid(z) * (1 - sigmoid(z))


class LogRegressionLayer(object):
    """ class to perform logistic regression
    """

    def __init__(self, input, input_size, output_size):
        # initialise weights and bias
        self._W = theano.shared(
            value=np.zeros(
                (input_size, output_size),
                dtype=theano.config.floatX),
            name='W',
            borrow=True)

        self._b = theano.shared(
            value=np.zeros(
                (output_size,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self._params = [self._W, self._b]

        # input and output points
        self.input = input
        self.output = T.nnet.softmax(T.dot(input, self._W) + self._b)

    def updates(self, cost, learning_rate):
        """ rule for updating the weights
        """
        grad_W = T.grad(cost=cost, wrt=self._W)
        grad_b = T.grad(cost=cost, wrt=self._b)

        return [(self._W, self._W - learning_rate * grad_W),
                (self._b, self._b - learning_rate * grad_b)]

    def neg_log_likelihood(self, y_true):
        """ y_true is a matrix with a 1 in the column indicating the true class
        """
        # TODO: try ':' type indexing
        return -T.mean(T.log(self.output)[T.arange(y_true.shape[0]), y_true.argmax(1)])

    def accuracy_score(self, y_true):
        return T.mean(T.neq(self.output.argmax(1), y_true.argmax(1)))


class HiddenLayer(object):
    """ class to perform model a single neural network layer
    """

    def __init__(self, input, input_size, output_size, activation=T.tanh, random_state=None):
        # initialise weights and bias

        # for a tanh activation function the weights should be random uniform in
        # sqrt(-6./(n_in+n_hidden)) to sqrt(6./(n_in+n_hidden))
        # for a sigmoid activation function, the weights should be 4 times this
        # for any other activation functions, use the tanh initialisation

        if random_state is None:
            random_state = np.random.RandomState()

        if activation == T.nnet.sigmoid:
            self._W = theano.shared(
                value=np.asarray(
                    random_state.uniform(
                        low=-4.0 * np.sqrt(6.0 / (input_size + output_size)),
                        high=4.0 * np.sqrt(6.0 / (input_size + output_size)),
                        size=(input_size, output_size)),
                    dtype=theano.config.floatX),
                name='W',
                borrow=True)
        else:
            self._W = theano.shared(
                value=np.asarray(
                    random_state.uniform(
                        low=-np.sqrt(6.0 / (input_size + output_size)),
                        high=np.sqrt(6.0 / (input_size + output_size)),
                        size=(input_size, output_size)),
                    dtype=theano.config.floatX),
                name='W',
                borrow=True)

        self._b = theano.shared(
            value=np.zeros(
                (output_size,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self._params = [self._W, self._b]

        self.input = input
        self.output = activation(T.dot(X, self._W) + self._b)


    def updates(self, cost, learning_rate):
        """ rule for updating the weights
        """
        grad_W = T.grad(cost=cost, wrt=self._W)
        grad_b = T.grad(cost=cost, wrt=self._b)

        return [(self._W, self._W - learning_rate * grad_W),
                (self._b, self._b - learning_rate * grad_b)]


class LogRegression(object):
    def __init__(self, input_size, output_size):
        # initialise connection points for data
        self._input = T.matrix()
        self._desired_output = T.matrix()

        # initialise the log regression layer
        self._log_layer = LogRegressionLayer(self._input, input_size, output_size)
        self._cost = self._log_layer.neg_log_likelihood(self._desired_output)

        self._output = self._log_layer.output


    def fit(self, X, y, batch_size=20, n_epochs=1000, learning_rate=0.001):
        """ returns the cost
        """

        # make y a vertical array if it's not
        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_samples = X.shape[0]

        # convert the input numpy arrays to theano shared variables
        X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
        y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
        # theano vector for indexing batches
        batch_index = T.lvector()

        validate_batch = theano.function(
            inputs=[batch_index],
            outputs=self._log_layer.accuracy_score(self._desired_output),
            givens={
                self._input:          X[batch_index],
                self._desired_output: y[batch_index]
            }
        )

        train_batch = theano.function(
            inputs=[batch_index],
            outputs=self._cost,
            updates=self._log_layer.updates(self._cost, learning_rate),
            givens={
                self._input:          X[batch_index],
                self._desired_output: y[batch_index]
            }
        )

        epoch_split = ShuffleSplit(n_samples, n_iter=n_epochs, test_size=batch_size)

        for epoch, (train_index, valid_index) in enumerate(epoch_split):
            # split the training index into minibatches
            for ind in [train_index[i:i + batch_size] for i in range(0, len(train_index), batch_size)]:
                # don't train on incomplete batches
                if len(ind) >= batch_size:
                    train_batch(ind)

            # check the model on the last batch in the epoch
            print("epoch {} error: {}%".format(epoch, 100 * validate_batch(valid_index)))

    def predict(self, X):

        # convert the input numpy arrays to theano shared variables
        X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)

        predict_all = theano.function(
            inputs=[],
            outputs=self._output,
            givens={
                self._input: X
            }
        )

        return predict_all()


class NeuralNet(object):
    def __init__(self):
        self._log_layer = None

    def fit(self, X, y, batch_size=20, n_epochs=1000, learning_rate=0.001):
        """ returns the cost
        """

        # make y a vertical array if it's not
        if y.ndim == 1:
            y = y[:, np.newaxis]

        input_size = X.shape[1]
        output_size = y.shape[1]
        n_samples = X.shape[0]

        # convert the input numpy arrays to theano shared variables
        X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
        y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
        # define theano symbolic variables
        X_theano = T.matrix('X')
        y_theano = T.matrix('y')
        batch_index = T.lvector()

        # define the logistic regression layer
        self._log_layer = LogRegressionLayer(X_theano, input_size, output_size)
        # the cost function
        cost = self._log_layer.neg_log_likelihood(y_theano)

        validate_batch = theano.function(
            inputs=[batch_index],
            outputs=self._log_layer.accuracy_score(y_theano),
            givens={
                X_theano: X[batch_index],
                y_theano: y[batch_index]
            }
        )

        train_batch = theano.function(
            inputs=[batch_index],
            outputs=cost,
            updates=self._log_layer.updates(cost, learning_rate),
            givens={
                X_theano: X[batch_index],
                y_theano: y[batch_index]
            }
        )

        epoch_split = ShuffleSplit(n_samples, n_iter=n_epochs, test_size=batch_size)

        for epoch, (train_index, valid_index) in enumerate(epoch_split):
            # split the training index into minibatches
            for ind in [train_index[i:i + batch_size] for i in range(0, len(train_index), batch_size)]:
                # don't train on incomplete batches
                if len(ind) >= batch_size:
                    train_batch(ind)

            # check the model on the last batch in the epoch
            print("epoch {} error: {}%".format(epoch, 100 * validate_batch(valid_index)))

    def predict(self, X):

        # convert the input numpy arrays to theano shared variables
        X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)

        predict_all = theano.function(
            inputs=[],
            outputs=self._log_layer.output,
            givens={
                self._log_layer.input: X
            }
        )

        return predict_all()


class PyNeuralNet(object):
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
            for ind in [train_index[i:i + batch_size] for i in range(0, len(train_index), batch_size)]:
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