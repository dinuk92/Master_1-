# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:42:13 2017

@author: 3200183
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import clone
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six.moves import xrange
from arftools import *

# Parameters
n_classes = 2
n_estimators = 30
plot_colors = "ry"
cmap = plt.cm.RdGy
plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
RANDOM_SEED = 13  # fix the seed on each iteration

# Load data
#iris = load_iris()

plot_idx = 1

model = RandomForestClassifier(n_estimators=n_estimators)
#(x, y) = gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=0, epsilon=0.02)
#(x1, y1) = gen_arti(centerx=1, centery=1, sigma=0.5, nbex=100, data_type=0, epsilon=0.02)

# We only take the two corresponding features
#X = iris.data[:, pair]
#y = iris.target
(X,y) =   gen_arti(centerx=1, centery=1, sigma=0.1, nbex=1000, data_type=1, epsilon=0.02)


# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(RANDOM_SEED)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# Standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Train
clf = clone(model)
clf = model.fit(X, y)

scores = clf.score(X, y)
# Create a title for each column and the console by using str() and
# slicing away useless parts of the string
model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
model_details = model_title

plt.subplot(1, 1, plot_idx)
#if plot_idx <= len(models):
    # Add a title at the top of each column
    #plt.title(model_title)

# Now plot the decision boundary using a fine mesh as input to a
# filled contour plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

   
# Choose alpha blend level with respect to the number of estimators
# that are in use (noting that AdaBoost can use fewer estimators
# than its maximum if it achieves a good enough fit early on)
estimator_alpha = 1.0 / len(model.estimators_)
for tree in model.estimators_:
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z= tree.predict_proba([xx[0][0],yy[0][0]] )
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy ,Z, alpha=estimator_alpha, cmap=cmap)

# Build a coarser grid to plot a set of ensemble classifications
# to show how these are different to what we see in the decision
# surfaces. These points are regularly space and do not have a black outline
xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
                                     np.arange(y_min, y_max, plot_step_coarser))
Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")

# Plot the training points, these are clustered together and have a
# black outline
for i, c in zip(xrange(n_classes), plot_colors):

    ybis= np.where(y<0,0,1)
    print ybis
    idx = np.where(ybis == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=cmap)
    plt.show()
plot_idx += 1  # move on to the next plot in sequence

plt.axis("tight")

plt.show()