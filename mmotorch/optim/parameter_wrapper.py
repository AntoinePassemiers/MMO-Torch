# -*- coding: utf-8 -*-
# parameter_wrapper.py
# author: Antoine Passemiers


class ParameterWrapper:

    @staticmethod
    def wrap(parameter, manifold):
        parameter.manifold = manifold
        return parameter

    @staticmethod
    def unwrap(parameter):
        try:
            return parameter.manifold
        except AttributeError:
            return None
