# -*- coding: utf-8 -*-
# setup.py
# author: Antoine Passemiers

from setuptools import setup


packages = [
    'mmotorch',
    'mmotorch.logging',
    'mmotorch.manifolds',
    'mmotorch.nn',
    'mmotorch.optim'
]

setup(
    name='mmmotorch',
    version='1.0.0',
    description='Matrix Manifold Optimization for PyTorch',
    url='https://github.com/AntoinePassemiers/MMO-Torch',
    author='Antoine Passemiers',
    packages=packages,
    include_package_data=False,
    install_requires=['numpy >= 1.13.3', 'torch'])
