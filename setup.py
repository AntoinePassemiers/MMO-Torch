# -*- coding: utf-8 -*-
# setup.py
# author: Antoine Passemiers

import re
from setuptools import setup


packages = [
    'mmotorch',
    'mmotorch.logging',
    'mmotorch.manifolds',
    'mmotorch.nn',
    'mmotorch.optim'
]


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)


setup(
    name='mmmotorch',
    version=get_property('__version__', 'mmotorch'),
    description='Matrix Manifold Optimization for PyTorch',
    url='https://github.com/AntoinePassemiers/MMO-Torch',
    author='Antoine Passemiers',
    packages=packages,
    include_package_data=False,
    install_requires=['numpy >= 1.13.3', 'torch'])
