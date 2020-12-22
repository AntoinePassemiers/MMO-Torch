# -*- coding: utf-8 -*-
# product.py
# author: Antoine Passemiers

import numpy as np
import torch


class ProductManifold:

    def __init__(self, manifolds):
        self.manifolds = manifolds

    def init(self):
        return [manifold.init() for manifold in self.manifolds]

    def step(self, X, G):
        return [manifold.step(x, g) for x, g, manifold in zip(X, G, self.manifolds)]

    def inner(self, X, G, H):
        return [manifold.inner(x, g, h) for x, g, h, manifold in zip(X, G, self.manifolds)]

    def distance(self, X, Y):
        return [manifold.distance(x, y) for x, y, manifold in zip(X, Y, self.manifolds)]

    def norm(self, X, G):
        return [manifold.norm(x, g) for x, g, manifold in zip(X, G, self.manifolds)]

    @property
    def ndim(self):
        return [manifold.ndim for manifold in self.manifolds]
    
    @property
    def shape(self):
        return [manifold.shape for manifold in self.manifolds]
