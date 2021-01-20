# -*- coding: utf-8 -*-
# oblique.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch


class ObliqueManifold(Manifold):

    def __init__(self, n, m, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)

    def _init(self):
        X = np.random.rand(self.n, self.m)
        assert(X.shape == (self.n, self.m))
        X /= np.linalg.norm(X, axis=0)[np.newaxis, :]
        return X

    def _tangent_space_projection(self, X, H):
        return H - X.dot(np.diag(np.diag(X.T.dot(H))))

    def _egrad_to_rgrad(self, X, G):
        return self._tangent_space_projection(X, G)

    def _retraction(self, X, G):
        U = X + G
        U /= np.linalg.norm(U, axis=0)[np.newaxis, :]
        return U

    def _inner(self, X, G, H):
        return torch.sum(G * H)

    def _distance(self, X, Y):
        colsums = torch.sum(X * Y, 0)
        colsums = torch.clamp(colsums, -1, 1)
        return torch.norm(torch.acos(colsums), p='fro')

    def _norm(self, X, G):
        return torch.norm(G, p='fro')

    def _ndim(self):
        return (self.m - 1) * self.n
