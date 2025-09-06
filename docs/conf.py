"""Sphinx configuration for sensor_modeling documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
project = "sensor_modeling"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "nbsphinx"]
html_theme = "sphinx_rtd_theme"

master_doc = "index"
nbsphinx_allow_errors = True
nbsphinx_execute = "never"
