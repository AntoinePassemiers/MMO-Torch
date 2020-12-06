# -*- coding: utf-8 -*-
# manifold.py
# author: Antoine Passemiers

from mmotorch.nn.utils import *
from mmotorch.optim.parameter_wrapper import ParameterWrapper

from abc import ABCMeta, abstractmethod
import torch


class Manifold(metaclass=ABCMeta):

    def __init__(self, n, m, lb=None, ub=None):
        self.n = n
        self.m = m
        self.lb = lb
        self.ub = ub

    @abstractmethod
    def _init(self):
        pass

    def init(self):
        X = self._clip_values(self._init())
        param = torch.nn.Parameter(torch.Tensor(X))
        ParameterWrapper.wrap(param, self)
        return param

    @abstractmethod
    def _step(self, X, G):
        pass

    def step(self, X, G):
        X = ensure_numpy(X)
        G = ensure_numpy(G)
        X_prime = self._step(X, G)
        return ensure_torch(self._clip_values(X_prime))

    def _clip_values(self, X):
        if self.lb is not None:
            X = np.maximum(X, self.lb)
        if self.ub is not None:
            X = np.minimum(X, self.ub)
        return X

    @property
    def shape(self):
        return (self.n, self.m)


class EuclideanManifold(Manifold):

    def __init__(self, n, m, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)

    def _init(self):
        return np.random.normal(0, 1, size=(self.n, self.m))

    def _step(self, X, G):
        return X + G


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


class MultinomialManifold(Manifold):

    def __init__(self, n, m, le=False, epsilon=1e-7, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)
        self.le = le
        self.epsilon = epsilon

    def _init(self):
        X = np.random.rand(self.n, self.m)
        assert(X.shape == (self.n, self.m))
        X /= np.linalg.norm(X, axis=0)[np.newaxis, :]
        return X ** 2.

    def _step(self, X, G):
        Y = X * np.exp(G / np.maximum(X, self.epsilon))
        Y = np.nan_to_num(Y)
        s = Y.sum(axis=0)
        if self.le:
            mask = (s > 1)
            mask[np.isnan(mask)] = True
        else:
            mask = np.ones(len(s), dtype=np.bool)
        Y[:, mask] /= s[np.newaxis, mask]
        Y = np.maximum(Y, self.epsilon)
        return Y


class StiefelManifold(Manifold):

    def __init__(self, n, m, **kwargs):
        Manifold.__init__(self, n, m, **kwargs)

    def _init(self):
        X = np.random.randn(self.n, self.m)
        q, _ = np.linalg.qr(X)
        assert(X.shape == q.shape)
        return torch.nn.Parameter(torch.FloatTensor(q))

    def _step(self, X, G):
        q, r = np.linalg.qr(X + G)
        return np.dot(q, np.diag(np.sign(np.sign(np.diag(r)) + 0.5)))


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
