from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import logging

import pynn

RANDOM_STATE = np.random.RandomState(42)


def iris_demo():
    # load the iris dataset
    iris = load_iris()
    X = iris['data']
    y_labels = iris['target']

    lb = LabelBinarizer()
    y = lb.fit_transform(y_labels)

    # split into training, validation and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=RANDOM_STATE)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.25,
                                                          random_state=RANDOM_STATE)

    # train the neural net
    print("Building logistic regression classifier to classify iris data")
    nn = pynn.ArtificialNeuralNet([X_train.shape[1], 20, y_train.shape[1]])
    print("Training")
    nn.fit(X_train, y_train, X_valid, y_valid,
           batch_size=20, n_epochs=20, learning_rate=0.05,
           random_state=RANDOM_STATE)

    y_pred = nn.predict(X_test)

    print("iris accuracy: {}%".format(
        accuracy_score(y_test.argmax(1), y_pred.argmax(1)) * 100))


def digits_demo():
    # load the digits dataset
    digits = load_digits()
    X = digits['data']
    y_labels = digits['target']

    lb = LabelBinarizer()
    y = lb.fit_transform(y_labels)

    # split into training, validation and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=RANDOM_STATE)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.25,
                                                          random_state=RANDOM_STATE)

    # train the neural net
    print("Building neural net to classify digits")
    nn = pynn.ArtificialNeuralNet([X_train.shape[1], 20, y_train.shape[1]],
                                  random_state=RANDOM_STATE)
    print("Training")
    nn.fit(X_train, y_train, X_valid, y_valid,
           batch_size=20, n_epochs=20, learning_rate=0.05)

    y_pred = nn.predict(X_test)

    print("digits accuracy: {}%".format(
        accuracy_score(y_test.argmax(1), y_pred.argmax(1)) * 100))


def conv_demo():
    # load the digits dataset
    digits = load_digits()
    X = digits['data']
    y_labels = digits['target']

    lb = LabelBinarizer()
    y = lb.fit_transform(y_labels)

    # split into training, validation and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=RANDOM_STATE)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.25,
                                                          random_state=RANDOM_STATE)

    # train the neural net
    print("Building neural net to classify digits")
    conv_net = pynn.ConvNet(digits['images'][0].shape, 1, y.shape[1],
                            random_state=RANDOM_STATE)
    print("Training")
    conv_net.fit(X_train, y_train, X_valid, y_valid,
                 batch_size=20, n_epochs=20, learning_rate=0.05)

    y_pred = conv_net.predict(X_test)

    print("digits accuracy: {}%".format(
        accuracy_score(y_test.argmax(1), y_pred.argmax(1)) * 100))


logging.basicConfig(level=logging.DEBUG)
# iris_demo()
# digits_demo()

conv_demo()