# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# Document the easyspec code of this repository (src/ folder, two levels above docs/source), and NOT an easyspec version installed
# with pip (e.g. the one on PyPI), which may be older than the code on GitHub:
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'easyspec'
copyright = '2025, Raniere de Menezes'
author = 'Raniere de Menezes'

# The full version, including alpha/beta/rc tags
release = '2025'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Packages that are not needed to read the docstrings are "mocked" (replaced by dummy modules), so that the build does not
# fail if they cannot be installed on Read the Docs. Add here any new heavy dependency imported by easyspec:
autodoc_mock_imports = ['ccdproc', 'astroscrappy', 'emcee', 'corner', 'dust_extinction', 'tqdm']

autodoc_member_order = 'bysource'  # functions listed in the same order as in the code
autodoc_default_options = {'members': True, 'undoc-members': True, 'show-inheritance': True}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []