# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

sys.path.append(str(Path("../../examples").resolve()))

project = "NeMo-Run"
copyright = "2025, NVIDIA"
author = "NVIDIA"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
python_maximum_signature_line_length = 88

# Autoapi settings
autoapi_generate_api_docs = True
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_type = "python"
autoapi_dirs = ["../../nemo_run"]
autoapi_file_pattern = "*.py"
autoapi_root = "api"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

# Autodoc settings
autodoc_typehints = "signature"

# MyST settings
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
