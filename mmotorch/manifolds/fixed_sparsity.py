# -*- coding: utf-8 -*-
# fixed_sparsity.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch


class FixedSparsityPatternManifold(Manifold):

    def __init__(self, S, **kwargs):
        self.S = np.asarray(S, dtype=np.bool)
        Manifold.__init__(self, *self.S.shape, **kwargs)

    def _init(self):
        return self.S * np.random.rand(*self.S.shape)

    def _egrad_to_rgrad(self, X, G):
        return self.S * G

    def _retraction(self, X, G):
        return X + G

    def _inner(self, X, G, H):
        return torch.sum(G * H)

    def _distance(self, X, Y):
        return torch.norm(X - Y, p='fro')

    def _norm(self, X, G):
        return torch.norm(G, p='fro')

    def _ndim(self):
        return np.prod(self.S.shape)
