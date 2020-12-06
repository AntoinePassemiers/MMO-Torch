# -*- coding: utf-8 -*-
# distance.py
# author: Antoine Passemiers

from mmotorch.manifolds import ObliqueManifold
from mmotorch.optim import RiemannianSGD

import numpy as np
import torch


# Random oblique matrix
X = np.random.rand(100, 40)
X = X / np.linalg.norm(X, axis=0)[np.newaxis, :]
X = torch.FloatTensor(X)

# Define oblique matrix parameter
manifold = ObliqueManifold(100, 40)
Y = manifold.init()

# Init optimizer
optimizer = RiemannianSGD([Y], lr=1e-5)

# Minimize distance between the 2 matrices
for _ in range(1000):
    loss = manifold.distance(X, Y)
    loss.backward()
    optimizer.step()
    print(loss.item())
