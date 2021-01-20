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
        return self.manifold._init().T

    def _egrad_to_rgrad(self, X, G):
        return self.manifold._egrad_to_rgrad(X.T, G.T).T

    def _retraction(self, X, G):
        return self.manifold._retraction(X.T, G.T).T

    def _inner(self, X, G, H):
        return self.manifold.inner(X, G, H)

    def _distance(self, X, Y):
        return self.manifold.distance(X, Y)

    def _norm(self, X, G):
        return self.manifold.norm(X, G)

    def _ndim(self):
        return self.manifold.ndim
