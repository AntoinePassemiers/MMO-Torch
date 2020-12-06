# -*- coding: utf-8 -*-
# oblique.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np


class ObliqueManifold(Manifold):

    def __init__(self, n, m, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)

    def _init(self):
        X = np.random.rand(self.n, self.m)
        assert(X.shape == (self.n, self.m))
        X /= np.linalg.norm(X, axis=0)[np.newaxis, :]
        return X

    def _step(self, X, G):
        U = X + G
        U /= np.linalg.norm(U, axis=0)[np.newaxis, :]
        return U

    def _inner(self, X, G, H):
        return torch.sum(G * H)

    def _distance(self, X, Y):
        colsums = torch.sum(X * Y, 0)
        colsums = torch.minimum(colsums, 1)
        return torch.linalg.norm(torch.arccos(colsums), ord='fro')

    def _norm(self, X, G):
        return torch.linalg.norm(G, ord='fro')

    def _ndim(self):
        return (self.m - 1) * self.n
