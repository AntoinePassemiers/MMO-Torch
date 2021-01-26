# -*- coding: utf-8 -*-
# positive_entries.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch
import warnings


class PositiveEntriesManifold(Manifold):

    def __init__(self, *shape, epsilon=1e-8, **kwargs):
        Manifold.__init__(self, *shape, **kwargs)
        self.epsilon = epsilon

    def _init(self):
        return np.exp(np.random.rand(self.n, self.m))

    def _egrad_to_rgrad(self, X, G):
        return X * G * X

    def _retraction(self, X, G):
        X = np.maximum(X, self.epsilon)
        Y = np.log(X) + (G / X)
        Y = np.exp(Y)
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            warnings.warn('Invalid value encountered during retraction mapping onto PositiveEntriesManifold')
            mask = np.isinf(Y)
            Y[mask] = X[mask]
            Y = np.nan_to_num(Y)
        Y = np.maximum(Y, self.epsilon)

        # Weighted geometric mean for smoothing
        Y = np.exp(0.1 * np.log(Y) + 0.9 * np.log(X))  # TODO
        return Y

    def _inner(self, X, G, H):
        return torch.sum((G * H) / (X * X))

    def _distance(self, X, Y):
        return torch.norm(torch.log(X) - torch.log(Y), p='fro')

    def _norm(self, X, G):
        return torch.sqrt(self.inner(X, G, G))

    def _ndim(self):
        return self.n * self.m
