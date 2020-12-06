# -*- coding: utf-8 -*-
# stiefel.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch


class StiefelManifold(Manifold):

    def __init__(self, n, m, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)
        self.k = 1

    def _init(self):
        X = np.random.randn(self.n, self.m)
        q, _ = np.linalg.qr(X)
        assert(X.shape == q.shape)
        return torch.nn.Parameter(torch.FloatTensor(q))

    def _step(self, X, G):
        q, r = np.linalg.qr(X + G)
        return np.dot(q, np.diag(np.sign(np.sign(np.diag(r)) + 0.5)))

    def _inner(self, X, G, H):
        return torch.sum(G * H)

    def _distance(self, X, Y):
        raise NotImplementedError

    def _norm(self, X, G):
        return torch.norm(G, p='fro')

    def _ndim(self):
        return self.k * (self.n * self.m - self.m * (self.m + 1) // 2)
