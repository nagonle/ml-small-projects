#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2017 nicolas <nicolas@laptop>
#
# Distributed under terms of the MIT license.

"""
In this model it seek to predict the flower specie given measures of the petal and sepal.
It uses the iris dataset, which one it is build a linear model to predict the species of a plant given some measures of the leaf plant.
"""

from sklearn import datasets

verbose = False

print "[+] Working with the Iris Dataset"
# Getting the dataset
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# Some info about the dataset
print "[+] Info about the dataset"
print "\tX shape:", X_iris.shape
print "\tY shape:", y_iris.shape

print "Vector features and target vector"
print "One example:", X_iris[0], y_iris[0]

if (verbose == True):
    print y_iris
    print iris.target_names

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

# Get only the first two attributes
X, y = X_iris[:, :2], y_iris

# "Split data into train and test sets"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

if (verbose == True):
    print "X_train.shape, X_test.shape, y_train.shape, y_test.shape"
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Standardize the features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


print "\n[+] To see how the features are distributed in the two-dimensional space."
import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in xrange(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()


# Creation of model SGD Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)

if (verbose == True):
    print "model coefficients: ", clf.coef_
    print "model intercept: ", clf.intercept_

print "[+] Plot boundaries...",
import numpy as np

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5

xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class ' + str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(xs, ys, hold=True)

plt.show()
print "[OK]"

# Predict a new flower (after normalizing!)
new_example = np.array([[4.7, 3.1]])
print "Predicting a new example:", new_example, "Prediction:", clf.predict(scaler.transform(new_example))
if (verbose):
    print "decision function: ", clf.decision_function(scaler.transform([[4.7, 3.1]]))

# Evaluating results
print "[+] Evaluating results"
from sklearn import metrics
y_train_pred = clf.predict(X_train)
print "Train Accuracy: {0:.2f}".format(metrics.accuracy_score(y_train, y_train_pred))

y_pred = clf.predict(X_test)
print "Test Accuracy: {0:.2f}".format(metrics.accuracy_score(y_test, y_pred))

# Precision recall f1-score support (how many instances of each class we had in the testing set)
print "[+] Classification Report"
if verbose:
    print metrics.classification_report(y_test, y_pred, target_names=iris.target_names)
print "f1-score: {0:.2f}".format(metrics.f1_score(y_test, y_pred, average='weighted'))

if (verbose):
    print metrics.confusion_matrix(y_test, y_pred)

print "[+] Cross-validation"
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
# create a composite estimator made by a pipeline of the standarization and the linear model.
clf = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('linear_model', SGDClassifier())
    ])

# Create a k-fold cross validation iterator of k=5 folds
cv = KFold(X.shape[0], 5, shuffle=True, random_state=42)
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(clf, X, y, cv=cv)
if (verbose):
    print scores

from scipy.stats import sem
def mean_score(scores):
    return ("Mean Accuracy Score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))

print mean_score(scores)
