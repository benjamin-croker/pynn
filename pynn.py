import numpy as np
from sklearn.cross_validation import ShuffleSplit
import theano
import theano.tensor as T

import logging

from layers import LogRegressionLayer, HiddenLayer, ConvolutionLayer


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
            self._random_state = np.random.RandomState()
        else:
            self._random_state = random_state

        # initialise connection points for data
        self._input = T.matrix()
        self._desired_output = T.matrix()
        self._lr = T.scalar()

        # add the first layer, connected to the input
        self._layers = [HiddenLayer(self._input,
                                    layer_sizes[0], layer_sizes[1],
                                    random_state=self._random_state)]

        # add the hidden layers
        for i in range(1, len(layer_sizes) - 2):
            self._layers += [HiddenLayer(self._layers[-1].output,
                                         layer_sizes[i], layer_sizes[i + 1])]

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

    def fit(self, X, y, X_valid, y_valid,
            batch_size, n_epochs, learning_rate,
            random_state=None):
        """ Performs minibatch training on the data. Fits X and y, using X_valid, y_valid
            as a validation set.
        """

        for epoch in range(n_epochs):
            # create a shuffled index to randomise the order of training data
            index = np.arange(len(X))
            rand_index = self._random_state.shuffle(index)

            # split the training index into minibatches
            for batch_index in [rand_index[i:i + batch_size]
                                for i in range(0, len(rand_index), batch_size)]:
                # don't train on incomplete batches
                if len(batch_index) >= batch_size:
                    self.partial_fit(X[batch_index], y[batch_index], learning_rate)

            logging.debug("epoch {} error: {}%".format(
                epoch, 100 * self.validate_batch(X_valid, y_valid)))


class ConvNet(object):
    def __init__(self, image_shape, image_channels, output_size, random_state=None):
        """ Creates a fully connected artificial neural network

            the first element is the X input size
            at least one element must specify hidden layer size
            the last element is the y output size

        """
        # TODO allow variable filter and max pooling sizes

        if random_state is None:
            self._random_state = np.random.RandomState()
        else:
            self._random_state = random_state


        # initialise connection points for data
        self._input = T.tensor4()
        self._desired_output = T.matrix()
        self._lr = T.scalar()

        self._image_shape = image_shape
        self._image_channels = image_channels

        self._layers = [ConvolutionLayer(self._input,
                                         image_shape=image_shape,
                                         n_input_maps=image_channels,
                                         n_filters=20,
                                         filter_shape=(3, 3),
                                         pool_shape=(2, 2),
                                         random_state=self._random_state)]

        self._layers += [HiddenLayer(self._layers[-1].output.flatten(2),
                                     input_size=20 * np.prod(self._layers[-1].output_image_shape),
                                     output_size=10,
                                     random_state=self._random_state)]

        # add the last layer, a logistic regression classifier
        self._layers += [LogRegressionLayer(self._layers[-1].output,
                                            input_size=10,
                                            output_size=output_size)]

        # add cost and output connections
        self._cost = self._layers[-1].neg_log_likelihood(self._desired_output)
        self._output = self._layers[-1].output

        # build the training, validation and testing functions
        # inputs for the theano functions
        X_theano = T.tensor4('X_theano')
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

        self._validate_batch = theano.function(
            inputs=[X_theano, y_theano],
            outputs=self._layers[-1].accuracy_score(self._desired_output),
            givens={
                self._input: X_theano,
                self._desired_output: y_theano
            }
        )

        self._predict = theano.function(
            inputs=[X_theano],
            outputs=self._output,
            givens={
                self._input: X_theano
            }
        )

    def _reshape_image(self, X):
        """ converts a n_images by n_pixels matrix into a 4-D tensor of size
            (n_images, n_channels, height, width)
        """
        return X.reshape(X.shape[0], self._image_channels, self._image_shape[0], self._image_shape[1])

    def fit(self, X, y, X_valid, y_valid,
            batch_size, n_epochs, learning_rate):
        """ Performs minibatch training on the data. Fits X and y, using X_valid, y_valid
            as a validation set.
        """

        for epoch in range(n_epochs):
            # create a shuffled index to randomise the order of training data
            rand_index = np.arange(len(X))
            self._random_state.shuffle(np.arange(len(X)))

            # split the training index into minibatches
            for batch_index in [rand_index[i:i + batch_size]
                                for i in range(0, len(rand_index), batch_size)]:
                # don't train on incomplete batches
                if len(batch_index) >= batch_size:
                    self.partial_fit(self._reshape_image(X[batch_index]), y[batch_index], learning_rate)

            logging.debug("epoch {} error: {}%".format(
                epoch, 100 * self.validate_batch(X_valid, y_valid)))

    def validate_batch(self, X, y):
        return self._validate_batch(self._reshape_image(X), y)

    def predict(self, X):
        return self._predict(self._reshape_image(X))
