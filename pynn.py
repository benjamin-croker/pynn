import numpy as np
from sklearn.cross_validation import ShuffleSplit
import theano
import theano.tensor as T


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
    """ class to perform model a single neural network layer
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

        updates = self._log_layer.updates(self._cost, learning_rate)

        validate_batch = theano.function(
            inputs=[batch_index],
            outputs=self._log_layer.accuracy_score(self._desired_output),
            givens={
                self._input: X[batch_index],
                self._desired_output: y[batch_index]
            }
        )

        train_batch = theano.function(
            inputs=[batch_index],
            outputs=self._cost,
            updates=updates,
            givens={
                self._input: X[batch_index],
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
    def __init__(self, input_size, output_size, hidden_sizes, learning_rate):
        """ Takes the input and output dimensions, and a list of ints,
            representing the number of neurons in each hidden layer
        """
        # check there is at least one hidden layer
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must be a list with at least one element")

        # initialise connection points for data
        self._input = T.matrix()
        self._desired_output = T.matrix()

        # add the first layer, connected to the input
        self._layers = [HiddenLayer(self._input, input_size, hidden_sizes[0])]

        # add any other hidden layers
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self._layers += [HiddenLayer(self._layers[-1].output, in_size, out_size)]

        # add the last layer, a logistic regression classifier
        self._layers += [LogRegressionLayer(self._layers[-1].output, hidden_sizes[-1], output_size)]

        # add cost and output connections
        self._cost = self._layers[-1].neg_log_likelihood(self._desired_output)
        self._output = self._layers[-1].output

        # build the training, validation and testing functions
        # inputs for the theano functions
        X_theano = T.matrix('X_theano')
        y_theano = T.matrix('y_theano')

        # update rules
        updates = [update
                   for layer in self._layers
                   for update in layer.updates(self._cost, learning_rate)]


        self.train_batch = theano.function(
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


    def fit(self, X, y, batch_size=20, n_epochs=1000):
        """ returns the cost
        """

        # make y a vertical array if it's not
        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_samples = X.shape[0]

        epoch_split = ShuffleSplit(n_samples, n_iter=n_epochs, test_size=batch_size)

        for epoch, (train_index, valid_index) in enumerate(epoch_split):
            # split the training index into minibatches
            for batch_index in [train_index[i:i + batch_size] for i in range(0, len(train_index), batch_size)]:
                # don't train on incomplete batches
                if len(batch_index) >= batch_size:
                    self.train_batch(X[batch_index], y[batch_index])

            # check the model on the last batch in the epoch
            print("epoch {} error: {}%".format(
                        epoch, 100 * self.validate_batch(X[valid_index], y[valid_index])))