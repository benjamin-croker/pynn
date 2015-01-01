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

    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=RANDOM_STATE)

    # train the neural net
    print("Building logistic regression classifier to classify iris data")
    lr = pynn.LogRegression(X_train.shape[1], y_train.shape[1], learning_rate=0.005)
    print("Training")
    lr.fit(X_train, y_train, batch_size=20, n_epochs=60, random_state=RANDOM_STATE)

    y_pred = lr.predict(X_test)

    print("iris accuracy: {}%".format(
        accuracy_score(y_test.argmax(1), y_pred.argmax(1)) * 100))


def digits_demo():
    # load the digits dataset
    digits = load_digits()
    X = digits['data']
    y_labels = digits['target']

    lb = LabelBinarizer()
    y = lb.fit_transform(y_labels)

    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=RANDOM_STATE)

    # train the neural net
    print("Building neural net to classify digits")
    nn = pynn.NeuralNet(X_train.shape[1], y_train.shape[1],
                        hidden_sizes=[20], learning_rate=0.05, random_state=RANDOM_STATE)
    print("Training")
    nn.fit(X_train, y_train, batch_size=20, n_epochs=20, random_state=RANDOM_STATE)

    y_pred = nn.predict(X_test)

    print("digits accuracy: {}%".format(
        accuracy_score(y_test.argmax(1), y_pred.argmax(1)) * 100))


logging.basicConfig(level=logging.DEBUG)
iris_demo()
digits_demo()