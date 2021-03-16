# -*- coding: utf-8 -*-
# example.py
# author: Antoine Passemiers

from mmotorch.manifolds import StiefelManifold
from mmotorch.optim import RiemannianSGD

import torch


X = torch.rand(100, 40)
Y = torch.rand(100, 40)
manifold = StiefelManifold(40, 40)
W = manifold.init()

optimizer = RiemannianSGD([W], lr=1e-3)

for _ in range(50):
    Y_hat = X.mm(W)
    loss = torch.mean((Y_hat - Y) ** 2.)
    loss.backward()
    optimizer.step()
    print(loss.item())
