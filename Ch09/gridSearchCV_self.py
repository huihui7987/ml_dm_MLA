#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import tree

from sklearn.datasets import load_iris

from matplotlib import pyplot
import scipy as sp
import numpy as np
from matplotlib import pylab

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

print(__doc__)

# Loading the Digits dataset
iris = load_iris()

X = iris.data
y=iris.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }


clf = tree.DecisionTreeClassifier()

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(clf, tuned_parameters, cv=10)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

print("use the best estimator to predict...")

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

'''
###########
'''

#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import tree

from sklearn.datasets import load_iris

from matplotlib import pyplot
import scipy as sp
import numpy as np
from matplotlib import pylab

from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from time import time

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report


# Loading the Digits dataset
iris = load_iris()

X = iris.data
y=iris.target

print (X.shape)
print (y.shape)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.75, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = {"criterion": ["gini", "entropy"],
              "min_samples_split": sp_randint(1, 20),
              "max_depth": sp_randint(1, 20),
              "min_samples_leaf": sp_randint(1, 20),
              "max_leaf_nodes": sp_randint(2,20),
              }

clf = tree.DecisionTreeClassifier()

print("# Tuning hyper-parameters")
print()

n_iter_search = 288
clf = RandomizedSearchCV(clf, \
        param_distributions=tuned_parameters, \
        n_iter=n_iter_search, \
        cv=10)

start = time()
clf.fit(X_train, y_train)


print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), n_iter_search))

print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()

print("use the best...")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()


###

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import scipy as sp
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from time import time
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

tuned_parameters = {#"criterion": ["gini", "entropy"],
              "min_samples_split": sp_randint(1, 20),
              "max_depth": sp_randint(1, 20),
              "min_samples_leaf": sp_randint(1, 20),
              "max_leaf_nodes": sp_randint(2,20),
              }


# Fit regression model
regr_1 = DecisionTreeRegressor()
n_iter_search = 600
regr_1 = RandomizedSearchCV(regr_1,
                         param_distributions=tuned_parameters,
                         n_iter=n_iter_search, cv=10)
start = time()
regr_1.fit(X, y)

# Predict
X_test = np.arange(0.0, 5, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)


print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), n_iter_search))
print(regr_1.best_params_)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="data")
plt.plot(X_test, y_1, c="g", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()