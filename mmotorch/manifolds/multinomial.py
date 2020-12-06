# -*- coding: utf-8 -*-
# multinomial.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np


class MultinomialManifold(Manifold):

    def __init__(self, n, m, le=False, epsilon=1e-7, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)
        self.le = le
        self.epsilon = epsilon

    def _init(self):
        X = np.random.rand(self.n, self.m)
        assert(X.shape == (self.n, self.m))
        X /= np.linalg.norm(X, axis=0)[np.newaxis, :]
        return X ** 2.

    def _step(self, X, G):
        Y = X * np.exp(G / np.maximum(X, self.epsilon))
        Y = np.nan_to_num(Y)
        s = Y.sum(axis=0)
        if self.le:
            mask = (s > 1)
            mask[np.isnan(mask)] = True
        else:
            mask = np.ones(len(s), dtype=np.bool)
        Y[:, mask] /= s[np.newaxis, mask]
        Y = np.maximum(Y, self.epsilon)
        return Y

    def _inner(self, X, G, H):
        return torch.sum((G * H) / X)

    def _distance(self, X, Y):
        return torch.linalg.norm(2. * torch.acos(torch.sum(torch.sqrt(X * Y), 1)), ord='fro')

    def _norm(self, X, G):
        return torch.sqrt(self.inner(X, G, G))

    def _ndim(self):
        return (self.n - 1) * self.m
