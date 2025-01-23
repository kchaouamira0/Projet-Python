#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from setuptools import setup

# Get the version from __init__.py
with open('gradient/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

# set the parameter of the setup
setup(name='gradient',  # define the name of the package
      version=version,
      description='Package to compute coefficients with a gradient descent technique',
      author='Antoine Dumas',
      author_email='dumas@phimeca.com',
      packages=['gradient', 'gradient.example'],  # namespace of the package
      # define where the package "gradient" is located, works also with the command find_packages
      # and define subpackage example
      package_dir={'gradient': 'gradient',
                   'gradient.example': 'gradient/example'},
      # some additional data included in the package
      package_data={'gradient': ['data/Sigma_features.pkl',
                                 'data/Sigma_labels.pkl']},

      # List of dependancies
      install_requires=['numpy']
      )
