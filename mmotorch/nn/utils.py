# -*- coding: utf-8 -*-
# utils.py
# author: Antoine Passemiers

import torch
import numpy as np


def ensure_numpy(X):
    if isinstance(X, torch.Tensor):
        X = X.data.numpy()
    return X


def ensure_torch(X):
    if isinstance(X, np.ndarray):
        X = torch.Tensor(X)
    return X
