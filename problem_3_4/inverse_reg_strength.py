import time

from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

# Turn down for faster run time
n_samples = 10000

X, y = fetch_20newsgroups_vectorized('all', return_X_y=True)
X = X[:n_samples]
y = y[:n_samples]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.1)

cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 5, 10)

max_iter = 20
clf = LogisticRegression(solver='saga',
                         multi_class='multinomial',
                         penalty='l1',
                         max_iter=max_iter,
                         random_state=42,
                         warm_start=True
                         )

times = []
accuracies = []
best_acc = 0
best_C = None
for c in cs:
    clf.set_params(C=c)
    a, t, d = fit_and_predict(clf,
                    max_iter,
                    x_train=X_train,
                    x_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    name="Multinomial Logistic Regression Classifier l1 Regularization")
    if a > best_acc:
        best_acc = a
        best_C = c
    print(c)
    accuracies.append(a)
    times.append(t)

print(best_C)
print()
print(times)
print(accuracies)

# Data for plotting
fig, ax = plt.subplots()
ax.plot(cs, times)

ax.set(xlabel='Inverse Regularization Strength', ylabel='Time (s)',
       title='Inverse Regularization Strength vs Time')
ax.grid()

fig.savefig("C_vs_Time.png")
plt.show()

# Data for plotting
fig, ax = plt.subplots()
ax.plot(cs, accuracies)

ax.set(xlabel='Inverse Regularization Strength', ylabel='accuracy',
       title='Inverse Regularization Strength vs Accuracy')
ax.grid()

fig.savefig("C_vs_Accuracy.png")
plt.show()