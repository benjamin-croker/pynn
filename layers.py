import numpy as np
import theano
import theano.printing
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


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


class ConvolutionLayer(object):
    """ class to model a single neural network layer
    """

    def __init__(self,
                 input_matrix, image_shape,
                 n_input_maps, n_filters, filter_shape, pool_shape,
                 activation=T.tanh, random_state=None):
        # initialise weights and bias

        # for a tanh activation function the weights should be random uniform in
        # sqrt(-6./(n_in+n_hidden)) to sqrt(6./(n_in+n_hidden))
        # for a sigmoid activation function, the weights should be 4 times this
        # for any other activation functions, use the tanh initialisation

        # TODO check filter shapes

        if random_state is None:
            random_state = np.random.RandomState()

        # total number of input and output connections
        input_size = n_input_maps * np.prod(filter_shape)
        output_size = n_filters * np.prod(filter_shape) / np.prod(pool_shape)

        if activation == T.nnet.sigmoid:
            self._W = theano.shared(
                value=np.asarray(
                    random_state.uniform(
                        low=-4.0 * np.sqrt(6.0 / (input_size + output_size)),
                        high=4.0 * np.sqrt(6.0 / (input_size + output_size)),
                        size=(n_filters, n_input_maps, filter_shape[0], filter_shape[1])),
                    dtype=theano.config.floatX),
                name='W',
                borrow=True)
        else:
            self._W = theano.shared(
                value=np.asarray(
                    random_state.uniform(
                        low=-np.sqrt(6.0 / (input_size + output_size)),
                        high=np.sqrt(6.0 / (input_size + output_size)),
                        size=(n_filters, n_input_maps, filter_shape[0], filter_shape[1])),
                    dtype=theano.config.floatX),
                name='W',
                borrow=True)

        # b is 1D, there is one bias unit for each output filter
        self._b = theano.shared(
            value=np.zeros(
                (n_filters,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self._params = [self._W, self._b]
        self.input = input_matrix

        # convolve input then downsample with max pooling
        conv_out = conv.conv2d(
            input=self.input,
            filters=self._W,
            filter_shape=(n_filters, n_input_maps, filter_shape[0], filter_shape[1])
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=pool_shape,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(pooled_out + self._b.dimshuffle('x', 0, 'x', 'x'))
        self.output_image_shape = (int((image_shape[0]-filter_shape[0]+1)/pool_shape[0]),
                                   int((image_shape[1]-filter_shape[1]+1)/pool_shape[1]))

    def updates(self, cost, learning_rate):
        """ rule for updating the weights
        """
        grad_W = T.grad(cost=cost, wrt=self._W)
        grad_b = T.grad(cost=cost, wrt=self._b)

        return [(self._W, self._W - learning_rate * grad_W),
                (self._b, self._b - learning_rate * grad_b)]