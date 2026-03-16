# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Point Sphinx to your package
sys.path.insert(0, os.path.abspath('../../'))  # root of SciKit/

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SciKit'
copyright = '2026, Shanlong Li'
author = 'Shanlong Li'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

autodoc_mock_imports = ["HyresBuilder"]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True  # Auto-generate stub files

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
