from sklearn import datasets
import numpy as np
import scipy.sparse as sp
import pynn

# load the iris dataset
iris = datasets.load_iris()
X = iris['data']
y = iris['target']


# the iris dataset has 4 variables, and three possible outputs (species)
# 10 is arbitrarily chosen as the number of units in the hidden layer
nn = pynn.NeuralNet(hiddenLayers=10)

# adjust the y values to an array, with a 1 in relevant column
yv = np.zeros((y.shape[0], 3))
for i in range(len(y)):
    yv[i, y[i]] = 1

print "Training"
nn.fit(X, yv)

print "Predicting"
yPred = nn.predict(X)

# convert to a column of labels
yPredLabels = yPred.argmax(1)

print 'accuracy:', np.mean(yPredLabels == y) * 100, '%'