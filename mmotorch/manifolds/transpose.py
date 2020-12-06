# -*- coding: utf-8 -*-
# transpose.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch


class TransposeManifold(Manifold):

    def __init__(self, manifold, **kwargs):
        n, m = manifold.shape
        Manifold.__init__(self, m, n, **kwargs)
        self.manifold = manifold

    def _init(self):
        X = self.manifold._init()
        return X.T

    def _step(self, X, G):
        X_new = self.manifold.step(X.T, G.T)
        return X_new.T

    def _inner(self, X, G, H):
        return self.manifold.inner(X, G, H)

    def _distance(self, X, Y):
        return self.manifold.distance(X, Y)

    def _norm(self, X, G):
        return self.manifold.norm(X, G)

    def _ndim(self):
        return self.manifold.ndim
