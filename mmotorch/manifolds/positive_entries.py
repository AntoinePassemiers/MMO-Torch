# -*- coding: utf-8 -*-
# positive_entries.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch


class PositiveEntriesManifold(Manifold):

    def __init__(self, n, m, epsilon=1e-7, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)
        self.epsilon = epsilon

    def _init(self):
        return np.random.rand(self.n, self.m)

    def _step(self, X, G):
        Y = X * np.exp(G / np.maximum(X, self.epsilon))
        Y = np.nan_to_num(Y)
        Y = np.maximum(Y, self.epsilon)
        return Y

    def _inner(self, X, G, H):
        return torch.sum((G * H) / (X * X))

    def _distance(self, X, Y):
        return torch.norm(torch.log(X) - torch.log(Y), p='fro')

    def _norm(self, X, G):
        return torch.sqrt(self.inner(X, G, G))

    def _ndim(self):
        return self.n * self.m
