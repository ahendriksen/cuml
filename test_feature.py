#!/usr/bin/env python3

import cupy as cp
import numpy as np
import cuml
from cuml.linear_model import LinearRegression
from cuml.common.array import CumlArray

n_features = 3
n_samples = 1_000
n_targets = 10
fit_intercept = False

sample_weight = cp.linspace(0, 1, n_samples) + 2.

X = cp.random.normal(size=(n_samples, n_features))
alpha = cp.linspace(1, n_targets, n_targets)
beta = cp.linspace(0, 1_000, n_features * n_targets).reshape((n_features, n_targets))
y = X @ beta + alpha

# Single target
lr = LinearRegression(fit_intercept=fit_intercept, output_type="cupy")
lr.fit(X, y[:, 0], sample_weight=sample_weight)

# Multi target
lr_multi = LinearRegression(fit_intercept=fit_intercept, output_type="cupy")
lr_multi.fit(X, y, sample_weight=sample_weight)

print("single coef: ", lr.coef_)
print("multi  coef: ", lr_multi.coef_[:, 0])

print("single intercept: ", lr.intercept_)
print("multi  intercept: ", lr_multi.intercept_[0])

# Predict
y_hat_single = lr.predict(X[:10])
y_hat_multi = lr_multi.predict(X[:10])

print("single predict: ", y_hat_single)
print("multi predict : ", y_hat_multi[:, 0])

# Fail on normalize
lr_multi = LinearRegression(fit_intercept=True, normalize=True, output_type="cupy")
try:
    lr_multi.fit(X, y, sample_weight=sample_weight)
except ValueError:
    print("multi: Succesfully failed to normalize")


# def predict_multi_target(X, coef, intercept=None):
#     assert X.ndim == 2
#     assert coef.ndim == 2

#     if intercept is None:
#         return X @ coef
#     else:
#         return X @ coef + intercept

# def pred(X, lr):
#     return predict_multi_target(X, lr.coef_, lr.intercept_)
