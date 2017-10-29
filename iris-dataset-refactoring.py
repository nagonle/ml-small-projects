#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2017 nicolas <nicolas@laptop>
#
# Distributed under terms of the MIT license.

"""
In this model it seek to predict the flower specie given measures of the petal and sepal.
It uses the iris dataset, which one it is build a linear model to predict the species of a plant given some measures of the leaf plant.
"""
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.stats import sem
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
VERBOSE = False
PLOTS = False

def mean_score(scores):
    return ("\tMean Accuracy Score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))

#######################
# Getting the dataset #
#######################
print "======================================"
print "    Working with the Iris Dataset "
print "======================================"
iris = pd.read_csv("iris_dataset.csv")
iris = iris.drop("Unnamed: 0", axis=1)
iris = shuffle(iris, random_state=0)
X_iris, y_iris = np.array(iris.iloc[:, 0:4]), np.array(iris.iloc[:,4])

feature_names = list(set(y_iris))
y_iris = np.array([feature_names.index(y) for y in y_iris])

# Some info about the dataset
if (VERBOSE == True):
    print "[+] Info about the dataset"
    print "\tX shape:", X_iris.shape
    print "\tY shape:", y_iris.shape

    print "Vector features and target vector"
    print "One example:", X_iris[0], y_iris[0]
    print y_iris
    print iris.target_names

# Get only the first two attributes
X, y = X_iris[:, :2], y_iris
#X, y = X_iris, y_iris

##########################################
# "Split data into train and test sets"  #
##########################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

if (VERBOSE == True):
    print "X_train.shape, X_test.shape, y_train.shape, y_test.shape"
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

############################
# Standardize the features #
############################ 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##############################################
# Plot features in the two-dimensional space #
##############################################
if PLOTS:
    print "\n[+] To see how the features are distributed in the two-dimensional space."
    colors = ['red', 'greenyellow', 'blue']
    for i in xrange(len(colors)):
            xs = X_train[:, 0][y_train == i]
            ys = X_train[:, 1][y_train == i]
            plt.scatter(xs, ys, c=colors[i])
    plt.legend(iris.target_names)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()

xs = X_train[:, 0][y_train == 0]

#####################################################
# Creation of model SGD Stochastic Gradient Descent #
#####################################################
clf = SGDClassifier(random_state=0)
clf.fit(X_train, y_train)

if (VERBOSE == True):
    print "model coefficients: ", clf.coef_
    print "model intercept: ", clf.intercept_

###################
# Plot boundaries #
###################
if PLOTS:
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

#############################################
# Predict a new flower (after normalizing!) #
#############################################
unseen_flower = np.array([[4.7, 3.1]])
if (VERBOSE):
    print "[+] Predicting an unseen flower", unseen_flower
    print "\tPrediction:", clf.predict(scaler.transform(unseen_flower))
    print "\tDecision function: ", clf.decision_function(scaler.transform([[4.7, 3.1]]))

############################
# Evaluating model results #
############################
print "[+] Evaluating model results"
y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)
print "\tTrain Accuracy: {0:.2f}".format(metrics.accuracy_score(y_train, y_train_pred))
print "\tTest Accuracy: {0:.2f}".format(metrics.accuracy_score(y_test, y_pred))

# Precision recall f1-score support (how many instances of each class we had in the testing set)
print "[+] Classification Report"
if VERBOSE:
    print "\t",metrics.classification_report(y_test, y_pred, target_names=iris.target_names)
    print metrics.confusion_matrix(y_test, y_pred)
print "\tf1-score: {0:.2f}".format(metrics.f1_score(y_test, y_pred, average='weighted'))


###############################################################
# Using cross-validation to improve the accuracy of the model #
###############################################################
print "[+] Cross-validation using f1 measure"
# create a composite estimator made by a pipeline of the standarization and the linear model.
clf = Pipeline([('scaler', preprocessing.StandardScaler()), ('linear_model', SGDClassifier(random_state=0))])

# Create a k-fold cross validation iterator of k=5 folds
cv = KFold(X.shape[0], 5, shuffle=True, random_state=42)
#scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
scores = cross_val_score(clf, X, y, cv=cv)
if (VERBOSE):
    print scores

print mean_score(scores)
