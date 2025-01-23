# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Gradient Descent'
copyright = '2025, Amira'
author = 'Amira'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Pour générer automatiquement à partir des docstrings
    'sphinx.ext.napoleon',  # Support pour les docstrings Google/NumPy
]

templates_path = ['_templates']
exclude_patterns = []

# Ajouter le chemin vers le projet pour que Sphinx trouve le module
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Ajuster en fonction de ton chemin réel

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Utiliser un thème moderne
html_static_path = ['_static']
