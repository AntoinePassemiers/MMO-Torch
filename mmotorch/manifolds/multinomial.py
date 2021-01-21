# -*- coding: utf-8 -*-
# multinomial.py
# author: Antoine Passemiers

from mmotorch.manifolds.base import Manifold

import numpy as np
import torch
import scipy.special


class MultinomialManifold(Manifold):

    def __init__(self, n, m, strict_simplex=False, epsilon=1e-8, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)
        self.strict_simplex = strict_simplex
        self.epsilon = epsilon

    def _init(self):
        X = np.random.rand(self.n, self.m)
        assert(X.shape == (self.n, self.m))
        X /= np.sum(X, axis=0)[np.newaxis, :]
        return X

    def _tangent_space_projection(self, X, H):
        return H - np.sum(H, axis=0)[np.newaxis, :] * X

    def _egrad_to_rgrad(self, X, G):
        return self._tangent_space_projection(X, X * G)

    def _retraction(self, X, G):
        X = np.maximum(X, self.epsilon)
        norm = scipy.special.logsumexp(G / X, b=X, axis=0)
        Y = np.log(X) + (G / X)
        if self.strict_simplex:
            Y -= norm[np.newaxis, :]
        else:
            mask = (norm > 0)
            Y[:, mask] -= norm[np.newaxis, mask]
        Y = np.exp(Y)
        Y = np.maximum(Y, self.epsilon)

        assert(not np.any(np.isnan(Y)))
        return Y

    def _inner(self, X, G, H):
        return torch.sum((G * H) / X)

    def _distance(self, X, Y):
        return torch.norm(2. * torch.acos(torch.sum(torch.sqrt(X * Y), 1)), p='fro')

    def _norm(self, X, G):
        return torch.sqrt(self.inner(X, G, G))

    def _ndim(self):
        return (self.n - 1) * self.m
