
import time

from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, lasso_path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import l1_min_c

import matplotlib.pyplot as plt


def fit_and_predict(clf, max_iter, x_train, y_train, x_test, y_test, name):
    t1 = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time() - t1

    y_pred = clf.predict(x_test)
    accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
    density = np.mean(clf.coef_ != 0, axis=1) * 100

    print(name, " Number of Iterations: ", max_iter)
    print(name, " Training Time:", train_time)
    print(name, " Accuracy: ", accuracy)
    print(name, " Density: ", density)

    return accuracy, train_time, density


# data and params
TEST_REG_STRATS = True
# Turn down for faster run time
n_samples = 10000
max_iter = 20

X, y = fetch_20newsgroups_vectorized('all', return_X_y=True)
X = X[:n_samples]
y = y[:n_samples]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.1)

if TEST_REG_STRATS:
    models = ['l1', 'l2', 'elasticnet']
    classifiers = {}

    for m in models:
        l1_ratio = None
        if m == 'elasticnet':
            l1_ratio = .5
        clf = LogisticRegression(solver='saga',
                                 multi_class='multinomial',
                                 penalty=m,
                                 max_iter=max_iter,
                                 random_state=42,
                                 warm_start=True,
                                 C=11.05,
                                 l1_ratio=l1_ratio
                                 )
        fit_and_predict(clf,
                        max_iter,
                        x_train=X_train,
                        x_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        name="Multinomial Logistic Regression Classifier {} Regularization".format(m))
        classifiers[m] = clf

#####
# L2 Regularization was determined to be the best performing model
X, y = fetch_20newsgroups_vectorized('all', return_X_y=True)
X = X
y = y
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.1)

clf = LogisticRegression(solver='saga',
                         multi_class='multinomial',
                         penalty='l2',
                         max_iter=max_iter,
                         random_state=42,
                         warm_start=True,
                         C=11.05
                         )
fit_and_predict(clf,
                max_iter,
                x_train=X_train,
                x_test=X_test,
                y_train=y_train,
                y_test=y_test,
                name="Multinomial Logistic Regression Classifier {} Regularization".format('l2'))