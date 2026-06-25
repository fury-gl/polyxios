import polyxios

project = "polyxios"
author = "polyxios contributors"
copyright = "2025, polyxios contributors"
release = polyxios.__version__ if hasattr(polyxios, "__version__") else "0.1.0"
html_title = "polyxios"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx_design",
    "sphinx_copybutton",
]

extlinks = {
    "ghpull": ("https://github.com/fury-gl/polyxios/pull/%s", "PR #%s"),
    "ghissue": ("https://github.com/fury-gl/polyxios/issues/%s", "GH#%s"),
}

autosummary_generate = True
numpydoc_show_class_members = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "logo": {
        "text": "polyxios",
    },
    "github_url": "https://github.com/fury-gl/polyxios",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/polyxios",
            "icon": "fa-brands fa-python",
        }
    ],
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "navigation_depth": 2,
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": [],
    "primary_sidebar_end": [],
    "navbar_end": ["navbar-icon-links"],
}

pygments_style = "friendly"

html_sidebars = {
    "index": [],
}

exclude_patterns = ["_build"]
