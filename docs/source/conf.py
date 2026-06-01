# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Point Sphinx at the FORGE source so autodoc can import the modules.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "FORGE"
copyright = "2026, Chris Marsden"
author = "Chris Marsden"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # Pull docstrings from source code
    "sphinx.ext.napoleon",      # Support NumPy/Google-style docstrings
    "sphinx.ext.viewcode",      # Add [source] links to API docs
    "sphinx.ext.intersphinx",   # Cross-reference NumPy/SciPy/Python docs
    "sphinx.ext.mathjax",       # Render LaTeX maths in HTML output
]

# Intersphinx mapping — allows :class:`numpy.ndarray` etc. to link out.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}

# Napoleon settings — match the NumPy convention used in FORGE.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = False

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mock imports so docs build without installing every dependency.
# NOTE: numpy is NOT mocked because several modules use np.pi and np.array
# at module level, which breaks the simple mock. Instead, numpy must be
# installed in the docs build environment (it is pulled in as a dependency
# of forge itself).
autodoc_mock_imports = [
    "scipy",
    "matplotlib",
    "shapely",
    "freeqdsk",
    "panel",
    "bokeh",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Sidebar logo
html_logo = "_static/FORGE_logo.svg"
# html_favicon = "_static/favicon.ico"
