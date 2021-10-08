# -*- coding: utf-8 -*-
# local_convergence.py
# author: Antoine Passemiers

import numpy as np


class LocalConvergence:

    def __init__(self, n_steps_without_improvements, max_n_iter=1000, tau=1e-5):
        self.n_steps_without_improvements = n_steps_without_improvements
        self.max_n_iter = max_n_iter
        self.tau = tau
        self.iterations = 0
        self.nwi = 0
        self.best_loss = np.nan_to_num(np.inf)

    def step(self, loss):
        gain = (self.best_loss - loss) / np.abs(self.best_loss)
        if gain > self.tau:
            self.nwi = 0
        else:
            self.nwi += 1
        if loss < self.best_loss:
            self.best_loss = loss
        self.iterations += 1

    def __call__(self):
        if self.iterations >= self.max_n_iter:
            return True
        else:
            return (self.nwi >= self.n_steps_without_improvements)
