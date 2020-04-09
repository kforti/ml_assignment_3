#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 30 2020
WPI CS539 spring 2020
Team Assignment 3 problem 2
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
import time
import pyflann


# load mnist data from sklearn (Geron pg85-86)
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist['target']

ndataset = 60000
X_train_set = X[:ndataset]
y_train_set = y[:ndataset]
X_test_set = X[ndataset:]
y_test_set = y[ndataset:]

ntrain = 10000
ntest = 1000
x_train = X_train_set[:ntrain]
y_train = y_train_set[:ntrain]
x_test = X_test_set[:ntest]
y_test = y_test_set[:ntest]

# part 1 Run mnistknndemo
# KNN classifier rewrote mnistknndemo.m from pmtk3

# use the knn model to predict
k = 5
knnstart = time.time()
knn = KNN(n_neighbors=k)
knn.fit(x_train, y_train)
knnmid = time.time()
knn_prediction = knn.predict(x_test)
knnstop = time.time()
train_errs = knn.score(x_train, y_train)
test_errs = knn.score(x_test, y_test)
knn_accuracy = accuracy_score(y_test, knn_prediction)
knn_train_time = knnmid - knnstart
knn_test_time = knnstop -knnmid


print('Accuracy for KNN: ' + str(knn_accuracy * 100))
print('Total training time for KNN: ' + str(knn_train_time))
print('Total test time for KNN: ' + str(knn_test_time))

# part 2 FLANN

pyflann.set_distance_type('euclidean', order=2)
flann = pyflann.FLANN()
flannstart = time.time()
index_params = flann.build_index(x_train, log_level='info', algorithm='kmeans', branching=32, iterations=7)
neighbor, dist = flann.nn_index(x_test, num_neigbors=5, checks=index_params['checks'])
flannmid = time.time()
flann_prediction = y_train[neighbor]
flannstop = time.time()
flann_accuracy = accuracy_score(y_test, flann_prediction)
flann_train_time = flannmid - flannstart
flann_test_time = flannstop - flannmid

print('Accuracy for Approximate Nearest Neighbors: ' + str(flann_accuracy * 100))
print('Total training time for Approximate Nearest Neighbors: ' + str(flann_train_time))
print('Total testing time for Approximate Nearest Neighbors: ' + str(flann_test_time))

# part 3 multi-class logistic regression

logistart = time.time()
logi_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=14000)
logi_reg.fit(x_train, y_train)
logimid = time.time()
logi_prediction = logi_reg.predict(x_test)
logistop = time.time()
logi_accuracy = accuracy_score(y_test, logi_prediction)
logi_train_time = logimid - logistart
logi_test_time = logistop - logimid

print('Accuracy for Logistic Regression: ' + str(logi_accuracy * 100))
print('Total training time for Logistic Regression: ' + str(logi_train_time))
print('Total testing time for Logistic Regression: ' + str(logi_test_time))

# support Vector machine using OVA
# parameter

# param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
# svm_ova = SVC(gamma="scale", decision_function_shape='ovr')
# rnd_search_cv = RandomizedSearchCV(svm_ova, param_distributions, n_iter=10, verbose=2, cv=3)
# rnd_search_cv.fit(x_train, y_train)
# print(rnd_search_cv.best_params_)

svmstart = time.time()
svm_ova = SVC(C=3.4, decision_function_shape='ovr', gamma="scale", kernel='rbf')
svm_ova.fit(x_train, y_train)
svmmid = time.time()
svm_ova_pred = svm_ova.predict(x_test)
svmstop = time.time()
svm_accuracy = accuracy_score(y_test, svm_ova_pred)
svm_train_time = svmmid - svmstart
svm_test_time = svmstop - svmmid

print('Accuracy for OVA SVM: ' + str(svm_accuracy * 100))
print('Total training time for OVA SVM: ' + str(svm_train_time))
print('Total testing time for OVA SVM: ' + str(svm_test_time))

