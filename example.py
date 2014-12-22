from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

import pynn

SEED = 42

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
                                                        random_state=SEED)

    # train the neural net
    nn = pynn.NeuralNet(n_hidden=10)
    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)

    print("iris accuracy: {}%".format(
                accuracy_score(y_test.argmax(1), y_pred.argmax(1))*100))


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
                                                        random_state=SEED)

    # train the neural net
    nn = pynn.NeuralNet(n_hidden=100)
    nn.fit(X_train, y_train)

    y_pred = nn.predict(X_test)

    print("digits accuracy: {}%".format(
                accuracy_score(y_test.argmax(1), y_pred.argmax(1))*100))


iris_demo()
# digits_demo()
