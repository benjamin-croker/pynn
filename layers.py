import numpy as np
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

        # create a dummy input and define the output function
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