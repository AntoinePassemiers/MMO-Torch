# -*- coding: utf-8 -*-
# spd.py
# author: Antoine Passemiers

from mmotorch.functional import matrix_log
from mmotorch.manifolds.base import Manifold

import numpy as np
import scipy.linalg
import torch


class SPDMatrixManifold(Manifold):

    def __init__(self, *shape, **kwargs):
        Manifold.__init__(self, *shape, **kwargs)
        self._shape = shape

    def _ensure_symmetric(self, G):
        return (G + G.T) / 2.

    def _init(self):
        n = self._shape[0]
        eigenvalues = np.random.rand(n) + 1.
        U, _ = np.linalg.qr(np.random.randn(n, n))
        return np.dot(U, eigenvalues[:, np.newaxis] * U.T)

    def _egrad_to_rgrad(self, X, G):
        return X.dot(self._ensure_symmetric(G)).dot(X)

    def _retraction(self, X, G):
        return self._exponential_map(X, G)

    def _exponential_map(self, X, G):
        return np.dot(X, scipy.linalg.expm(np.linalg.solve(X, G)))

    def _inner(self, X, G, H):
        a = torch.solve(G, X)
        b = torch.solve(H, X)
        return torch.sum(a * b)

    def _distance(self, X, Y):
        L_inv = torch.inverse(torch.cholesky(X))
        return matrix_log(L_inv.dot(Y).dot(L_inv.T))

    def _norm(self, X, G):
        L_inv = torch.inverse(torch.cholesky(X))
        return torch.linalg.norm(torch.mm(torch.mm(L_inv, G), L_inv.t()), ord='fro')

    def _ndim(self):
        n = self._shape[0]
        return n * (n + 1) // 2
