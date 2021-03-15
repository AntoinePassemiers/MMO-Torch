# -*- coding: utf-8 -*-
# multivariate_gaussian.py
# author: Antoine Passemiers

from mmotorch.manifolds import SPDMatrixManifold
from mmotorch.optim import RiemannianSGD

import numpy as np
import matplotlib.pyplot as plt
import torch


mean = np.asarray([0, 1])
C = np.asarray([[1, -0.9], [-0.9, 2]])
X = np.random.multivariate_normal(mean, C, size=100)


C = torch.FloatTensor(C)
S = SPDMatrixManifold(2, 2).init()
optimizer = RiemannianSGD([S], lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
        verbose=True, threshold=0.00001, threshold_mode='rel',
        cooldown=0, min_lr=0, eps=1e-25)

for _ in range(10000):
    loss = torch.trace(torch.mm(S, C)) - torch.logdet(S)
    loss.backward()
    optimizer.step()
    print(loss.item(), np.linalg.cond(S.data.numpy()))
    scheduler.step(loss.item())

print(np.linalg.inv(C.data.numpy()))
print(S.data.numpy())
