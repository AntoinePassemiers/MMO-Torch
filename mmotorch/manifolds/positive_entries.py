# -*- coding: utf-8 -*-
# positive_entries.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch
import warnings


class PositiveEntriesManifold(Manifold):

    def __init__(self, *shape, epsilon=1e-8, fast=False, **kwargs):
        Manifold.__init__(self, *shape, **kwargs)
        self.epsilon = epsilon
        self.fast = fast

    def _init(self):
        return np.exp(np.random.rand(self.n, self.m))

    def _egrad_to_rgrad(self, X, G):
        if self.fast:
            return G
        else:
            G_prime = X * G
            G_prime *= X
            return G_prime

    def _retraction(self, X, G):
        Y = np.copy(X)
        if self.fast:
            Y += G
            np.maximum(Y, self.epsilon, out=Y)
        else:
            np.maximum(Y, self.epsilon, out=Y)
            Y[:] = np.log(Y, out=Y) + (G / Y)  # TODO: slow
            np.exp(Y, out=Y)
            if np.any(np.isinf(Y)):
                warnings.warn('Invalid value encountered during retraction mapping onto PositiveEntriesManifold')
                mask = np.isinf(Y)
                Y[mask] = X[mask]
            if np.any(np.isnan(Y)):
                warnings.warn('Invalid value encountered during retraction mapping onto PositiveEntriesManifold')
                np.nan_to_num(Y, copy=False)
            np.maximum(Y, self.epsilon, out=Y)

            # Weighted geometric mean for smoothing
            np.log(Y, out=Y)
            Y *= 0.1
            Y += 0.9 * np.log(X)
            np.exp(Y, out=Y)
        return Y

    def _inner(self, X, G, H):
        return torch.sum((G * H) / (X * X))

    def _distance(self, X, Y):
        return torch.norm(torch.log(X) - torch.log(Y), p='fro')

    def _norm(self, X, G):
        return torch.sqrt(self.inner(X, G, G))

    def _ndim(self):
        return self.n * self.m
