# -*- coding: utf-8 -*-
# functional.py
# author: Antoine Passemiers

import torch


def matrix_log(X, epsilon=1e-5):
    u, s, v = torch.svd(X)
    s = torch.log(s + epsilon)
    return torch.mm(torch.mm(u, torch.diag(s)), v.t())


def matrix_exp(X):
    u, s, v = torch.svd(X)
    s = torch.exp(s)
    return torch.mm(torch.mm(u, torch.diag(s)), v.t())
