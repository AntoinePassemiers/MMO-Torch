# -*- coding: utf-8 -*-
# euclidean.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch


class EuclideanManifold(Manifold):

    def __init__(self, n, m, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)

    def _init(self):
        return np.random.normal(0, 1, size=(self.n, self.m))

    def _step(self, X, G):
        return X + G

    def _inner(self, X, G, H):
        return torch.sum(G * H)

    def _distance(self, X, Y):
        return torch.norm(X - Y, p='fro')

    def _norm(self, X, G):
        return torch.norm(G, p='fro')

    def _ndim(self):
        return self.n * self.m
