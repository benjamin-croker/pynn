import numpy as np
from sklearn.cross_validation import ShuffleSplit
import theano
import theano.tensor as T

import logging

from layers import LogRegressionLayer, HiddenLayer


class ArtificialNeuralNet(object):
    def __init__(self, layer_sizes, random_state=None):
        """ Creates a fully connected artificial neural network

            layer_sizes is a list of integers which specifies the number of units
            in each hidden layer. There must be at least 3 elements

            the first element is the X input size
            at least one element must specify hidden layer size
            the last element is the y output size

        """
        # check there is at least one hidden layer
        if len(layer_sizes) < 3:
            raise ValueError("layer_sizes must be a list with at least input, hidden and output sizes")

        if random_state is None:
            random_state = np.random.RandomState()

        # initialise connection points for data
        self._input = T.matrix()
        self._desired_output = T.matrix()
        self._lr = T.scalar()

        # add the first layer, connected to the input
        self._layers = [HiddenLayer(self._input,
                                    layer_sizes[0], layer_sizes[1],
                                    random_state=random_state)]

        # add the hidden layers
        for i in range(1, len(layer_sizes) - 2):
            self._layers += [HiddenLayer(self._layers[-1].output,
                                         layer_sizes[i], layer_sizes[i+1])]

        # add the last layer, a logistic regression classifier
        self._layers += [LogRegressionLayer(self._layers[-1].output,
                                            layer_sizes[-2], layer_sizes[-1])]

        # add cost and output connections
        self._cost = self._layers[-1].neg_log_likelihood(self._desired_output)
        self._output = self._layers[-1].output

        # build the training, validation and testing functions
        # inputs for the theano functions
        X_theano = T.matrix('X_theano')
        y_theano = T.matrix('y_theano')
        lr_theano = T.scalar('lr_theano')

        # update rules each layer.updates() will return a list of all the params
        # these need to be flattened into a single list
        updates = [update
                   for layer in self._layers
                   for update in layer.updates(self._cost, self._lr)]

        self.partial_fit = theano.function(
            inputs=[X_theano, y_theano, lr_theano],
            outputs=self._cost,
            updates=updates,
            givens={
                self._input: X_theano,
                self._desired_output: y_theano,
                self._lr: lr_theano
            }
        )

        self.validate_batch = theano.function(
            inputs=[X_theano, y_theano],
            outputs=self._layers[-1].accuracy_score(self._desired_output),
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

    def fit(self, X, y, batch_size, n_epochs, learning_rate, validation_size=0.25, random_state=None):
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
                    self.partial_fit(X[batch_index], y[batch_index], learning_rate)

            logging.debug("epoch {} error: {}%".format(
                epoch, 100 * self.validate_batch(X[valid_index], y[valid_index])))

