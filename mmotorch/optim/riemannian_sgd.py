# -*- coding: utf-8 -*-
# riemannian_sgd.py
# author: Antoine Passemiers

from mmotorch.optim.parameter_wrapper import ParameterWrapper

import random
import torch


class RiemannianSGD(torch.optim.Optimizer):

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr < 0.:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError('Nesterov momentum requires a momentum and zero dampening')
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        torch.optim.Optimizer.__init__(self, params, defaults)

    def __setstate__(self, state):
        torch.optim.Optimizer.__setstate__(self, state)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:

                manifold = ParameterWrapper.unwrap(p)

                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1.-dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if manifold is not None:
                    new_p = manifold.step(p, -lr * d_p)
                else:
                    new_p = p - lr * d_p
                p.data.copy_(new_p)

        return loss
