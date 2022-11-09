#!/usr/bin/env python3

import cupy as cp
import numpy as np
import cuml
from cuml.linear_model import LinearRegression
from cuml.common.array import CumlArray

n_features = 3
n_samples = 1_000
n_targets = 10

sample_weight = cp.linspace(0, 1, n_samples) + 2.

X = cp.random.normal(size=(n_samples, n_features))
alpha = cp.linspace(1, n_targets, n_targets)
beta = cp.linspace(0, 1_000, n_features * n_targets).reshape((n_features, n_targets))
y = X @ beta + alpha

# Single target
lr = LinearRegression(fit_intercept=True, output_type="cupy")
lr.fit(X, y[:, 0], sample_weight=sample_weight)

# Multi target
lr_multi = LinearRegression(fit_intercept=True, output_type="cupy")
lr_multi.fit(X, y, sample_weight=sample_weight)

print("single coef: ", lr.coef_)
print("multi  coef: ", lr_multi.coef_[:, 0])

print("single intercept: ", lr.intercept_)
print("multi  intercept: ", lr_multi.intercept_[0])
