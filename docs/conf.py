import polyxios

project = "polyxios"
author = "polyxios contributors"
copyright = "2025, polyxios contributors"
release = polyxios.__version__ if hasattr(polyxios, "__version__") else "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/fury-gl/polyxios",
}

exclude_patterns = ["_build"]
