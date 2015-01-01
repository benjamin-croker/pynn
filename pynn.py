import numpy as np
from sklearn.cross_validation import ShuffleSplit
import theano
import theano.tensor as T

import logging


class LogRegressionLayer(object):
    """ class to perform logistic regression
    """

    def __init__(self, input_matrix, input_size, output_size):
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
        self.input = input_matrix
        self.output = T.nnet.softmax(T.dot(self.input, self._W) + self._b)

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
    """ class to model a single neural network layer
    """

    def __init__(self, input_matrix, input_size, output_size, activation=T.tanh, random_state=None):
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

        self.input = input_matrix
        self.output = activation(T.dot(self.input, self._W) + self._b)

    def updates(self, cost, learning_rate):
        """ rule for updating the weights
        """
        grad_W = T.grad(cost=cost, wrt=self._W)
        grad_b = T.grad(cost=cost, wrt=self._b)

        return [(self._W, self._W - learning_rate * grad_W),
                (self._b, self._b - learning_rate * grad_b)]


class LogRegression(object):
    def __init__(self, input_size, output_size, learning_rate):
        """ Creates a Logistic Regression classifier
        """

        # initialise connection points for data
        self._input = T.matrix()
        self._desired_output = T.matrix()

        # add the logistic regression classifier
        self._log_layer = LogRegressionLayer(self._input, input_size, output_size)

        # add cost and output connections
        self._cost = self._log_layer.neg_log_likelihood(self._desired_output)
        self._output = self._log_layer.output

        # build the training, validation and testing functions
        # inputs for the theano functions
        X_theano = T.matrix('X_theano')
        y_theano = T.matrix('y_theano')

        # update rules each layer.updates() will return a list of all the params
        # these need to be flattened into a single list
        updates = self._log_layer.updates(self._cost, learning_rate)

        self.partial_fit = theano.function(
            inputs=[X_theano, y_theano],
            outputs=self._cost,
            updates=updates,
            givens={
                self._input: X_theano,
                self._desired_output: y_theano
            }
        )

        self.validate_batch = theano.function(
            inputs=[X_theano, y_theano],
            outputs=self._log_layer.accuracy_score(self._desired_output),
            givens={
                self._input: X_theano,
                self._desired_output: y_theano
            }
        )

        self.predict = theano.function(
            inputs=[X_theano],
            outputs=self._output,
            givens={
                self._input: X_theano
            }
        )

    def fit(self, X, y, batch_size, n_epochs, validation_size=0.25, random_state=None):
        """ Performs minibatch training on the data. X and y must both be
            two dimensional numpy arrays

            validation_size is the percentage of training examples to use for
            tracking performance each epoch.

            One epoch involves randomly splitting the data into a training and validation set.
            All training data is used for training, and the error on the validation
            set is reported each epoch.
        """

        epoch_split = ShuffleSplit(X.shape[0], n_iter=n_epochs, test_size=validation_size,
                                   random_state=random_state)
        for epoch, (train_index, valid_index) in enumerate(epoch_split):
            # split the training index into minibatches
            for batch_index in [train_index[i:i + batch_size]
                                for i in range(0, len(train_index), batch_size)]:
                # don't train on incomplete batches
                if len(batch_index) >= batch_size:
                    self.partial_fit(X[batch_index], y[batch_index])

            logging.debug("epoch {} error: {}%".format(
                epoch, 100 * self.validate_batch(X[valid_index], y[valid_index])))


class NeuralNet(object):
    def __init__(self, input_size, output_size, hidden_sizes, learning_rate, random_state=None):
        """ Creates a fully connected artificial neural network

            hidden_sizes is a list of integers which specifies the number of units
            in each hidden layer. Must have at least 1 element
        """
        # check there is at least one hidden layer
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must be a list with at least one element")

        if random_state is None:
            random_state = np.random.RandomState()

        # initialise connection points for data
        self._input = T.matrix()
        self._desired_output = T.matrix()

        # add the first layer, connected to the input
        self._hidden_layers = [HiddenLayer(self._input, input_size, hidden_sizes[0],
                                           random_state=random_state)]

        # add any other hidden layers
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._hidden_layers += [HiddenLayer(self._hidden_layers[-1].output, in_size, out_size,
                                                random_state=random_state)]

        # add the last layer, a logistic regression classifier
        self._log_layer = LogRegressionLayer(self._hidden_layers[-1].output, hidden_sizes[-1], output_size)

        # add cost and output connections
        self._cost = self._log_layer.neg_log_likelihood(self._desired_output)
        self._output = self._log_layer.output

        # build the training, validation and testing functions
        # inputs for the theano functions
        X_theano = T.matrix('X_theano')
        y_theano = T.matrix('y_theano')

        # update rules each layer.updates() will return a list of all the params
        # these need to be flattened into a single list
        updates = [update
                   for layer in self._hidden_layers + [self._log_layer]
                   for update in layer.updates(self._cost, learning_rate)]

        self.partial_fit = theano.function(
            inputs=[X_theano, y_theano],
            outputs=self._cost,
            updates=updates,
            givens={
                self._input: X_theano,
                self._desired_output: y_theano
            }
        )

        self.validate_batch = theano.function(
            inputs=[X_theano, y_theano],
            outputs=self._log_layer.accuracy_score(self._desired_output),
            givens={
                self._input: X_theano,
                self._desired_output: y_theano
            }
        )

        self.predict = theano.function(
            inputs=[X_theano],
            outputs=self._output,
            givens={
                self._input: X_theano
            }
        )

    def fit(self, X, y, batch_size, n_epochs, validation_size=0.25, random_state=None):
        """ Performs minibatch training on the data. X and y must both be
            two dimensional numpy arrays

            validation_size is the percentage of training examples to use for
            tracking performance each epoch.

            One epoch involves randomly splitting the data into a training and validation set.
            All training data is used for training, and the error on the validation
            set is reported each epoch.
        """

        epoch_split = ShuffleSplit(X.shape[0], n_iter=n_epochs, test_size=validation_size,
                                   random_state=random_state)
        for epoch, (train_index, valid_index) in enumerate(epoch_split):
            # split the training index into minibatches
            for batch_index in [train_index[i:i + batch_size]
                                for i in range(0, len(train_index), batch_size)]:
                # don't train on incomplete batches
                if len(batch_index) >= batch_size:
                    self.partial_fit(X[batch_index], y[batch_index])

            logging.debug("epoch {} error: {}%".format(
                epoch, 100 * self.validate_batch(X[valid_index], y[valid_index])))
