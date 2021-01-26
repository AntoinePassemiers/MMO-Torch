# -*- coding: utf-8 -*-
# logger.py
# author: Antoine Passemiers

import numpy as np
import torch
from contextlib import ContextDecorator


class Summary:

    def __init__(self, module):
        self.module = module
        self.alpha = 0.1
        self.scalars = dict()
        self.inputs = dict()
        self.outputs = dict()

    def add_scalar(self, key, value):
        if key not in self.scalars:
            self.scalars[key] = list()
        self.scalars[key].append(value)

    def add_inputs(self, key, tensor):
        if key not in self.inputs:
            self.inputs[key] = list()
        self.inputs[key].append(self._to_array(tensor))

    def add_outputs(self, key, tensor):
        if key not in self.outputs:
            self.outputs[key] = list()
        self.outputs[key].append(self._to_array(tensor))

    def _to_array(self, tensor):
        arr = torch.flatten(tensor).data.numpy()
        mask = (np.random.rand(len(arr)) < self.alpha)
        return arr[mask]

    def get_inputs(self):
        return [np.concatenate(self.inputs[key], 0) for key in self.inputs.keys()]

    def get_outputs(self):
        return [np.concatenate(self.outputs[key], 0) for key in self.outputs.keys()]


class Logger:

    LOGGING = False
    SUMMARY_KEYS = dict()
    SUMMARIES = list()

    def watch(module):
        key = id(module)
        if key not in Logger.SUMMARY_KEYS:
            Logger.SUMMARY_KEYS[key] = len(Logger.SUMMARY_KEYS)
            Logger.SUMMARIES.append(Summary(module))
        summary = Logger.SUMMARIES[Logger.SUMMARY_KEYS[key]]

        f = module.forward
        def wrapper(*args, **kwargs):
            if Logger.LOGGING:
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        summary.add_inputs(i, arg)
            out = f(*args, **kwargs)
            if Logger.LOGGING:
                for i, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        summary.add_outputs(i, arg)
            return out

        module.forward = wrapper

    def summarize():
        summaries = list()
        inputs = list()
        for i in range(len(Logger.SUMMARIES)):
            summaries.append(Logger.SUMMARIES[i])
        return summaries


class logging(ContextDecorator):

    def __enter__(self):
        Logger.LOGGING = True
        return self

    def __exit__(self, *exc):
        Logger.LOGGING = False
        return False
