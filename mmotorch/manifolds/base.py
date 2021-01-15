# -*- coding: utf-8 -*-
# base.py
# author: Antoine Passemiers

from mmotorch.nn.utils import *
from mmotorch.optim.parameter_wrapper import ParameterWrapper

from abc import ABCMeta, abstractmethod
import torch


class Manifold(metaclass=ABCMeta):

    def __init__(self, *shape, lb=None, ub=None):
        self._shape = tuple(shape)
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

    def inner(self, X, G, H):
        X = ensure_torch(X)
        G = ensure_torch(G)
        H = ensure_torch(H)
        return self._inner(X, G, H)

    @abstractmethod
    def _inner(self, X, G, H):
        pass

    def distance(self, X, Y):
        X = ensure_torch(X)
        Y = ensure_torch(Y)
        return self._distance(X, Y)

    @abstractmethod
    def _distance(self, X, Y):
        pass

    def norm(self, X, G):
        X = ensure_torch(X)
        G = ensure_torch(G)
        return self._norm(X, G)

    @abstractmethod
    def _norm(self, X, G):
        pass

    @abstractmethod
    def _ndim(self):
        pass

    @property
    def ndim(self):
        return self._ndim()
    
    @property
    def shape(self):
        return self._shape

    @property
    def n(self):
        return self.shape[0]

    @property
    def m(self):
        return self.shape[1]
    
