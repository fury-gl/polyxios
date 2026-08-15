"""Sphinx configuration for the polyxios documentation site.

Retro terminal theme layered on top of pydata-sphinx-theme:
  * light mode  = "paper"  (warm cream, ink text)
  * dark mode   = "crt"    (amber phosphor on near-black)

Both palettes live in _static/css/retro.css as pydata theme variables, so the
theme's own light/dark switcher in the navbar drives them.

The only custom JavaScript is _static/js/homepage.js, which drives the copy
button on the landing page's install command.
"""

import os
from importlib import metadata

import pydata_sphinx_theme
from packaging.version import Version

import polyxios

project = "polyxios"
author = "polyxios contributors"
copyright = "2025, polyxios contributors"
release = getattr(polyxios, "__version__", None) or metadata.version("polyxios")

# "0.3.0.dev0+git..." -> the "0.3" series.
_parsed = Version(release)
version = f"{_parsed.major}.{_parsed.minor}"
_is_dev = _parsed.is_devrelease or _parsed.local is not None

# The switcher entry this build should highlight. A dev build off master is
# "dev"; a tagged build is its own "X.Y" series. SWITCHER_VERSION lets a
# release job pin it explicitly.
switcher_version = os.environ.get("SWITCHER_VERSION") or ("dev" if _is_dev else version)

# The published switcher index. It must be an absolute URL so it resolves the
# same from every page depth; SWITCHER_JSON_URL points a local build at its own
# copy in _static/ instead. The source of truth is docs/_static/switcher.json.
switcher_json_url = os.environ.get(
    "SWITCHER_JSON_URL", "https://fury-gl.github.io/polyxios/_static/switcher.json"
)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
]

extlinks = {
    "ghpull": ("https://github.com/fury-gl/polyxios/pull/%s", "PR #%s"),
    "ghissue": ("https://github.com/fury-gl/polyxios/issues/%s", "GH#%s"),
}

autosummary_generate = True
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

exclude_patterns = ["_build", "_includes", "_templates"]

# -- HTML output ------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "polyxios"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["css/retro.css"]
html_js_files = ["js/homepage.js"]

# Pygments: the retro palettes are tuned against these two. pydata renamed the
# options from "pygment_*" to "pygments_*" in 0.16, and either spelling warns on
# the version that does not own it.
_pygments_prefix = (
    "pygments" if Version(pydata_sphinx_theme.__version__) >= Version("0.16") else "pygment"
)

html_theme_options = {
    f"{_pygments_prefix}_light_style": "friendly",
    f"{_pygments_prefix}_dark_style": "native",
    # Escaped: pydata drops the logo text into the template unescaped, so a raw
    # "<polyxios />" is parsed as an unknown HTML element and renders nothing.
    "logo": {"text": "&lt;polyxios /&gt;"},
    "announcement": (
        "// fast, clean mesh I/O for Python &mdash; one dependency instead of eighteen"
    ),
    "github_url": "https://github.com/fury-gl/polyxios",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/polyxios/",
            "icon": "fa-solid fa-box",
        },
        {
            "name": "FURY",
            "url": "https://fury.gl",
            "icon": "fa-solid fa-cube",
        },
    ],
    # Version switcher. json_url must be absolute so it resolves the same from
    # every page depth; check_switcher stays off so a build without network
    # access (local, or a PR runner) does not fail under -W.
    "switcher": {
        "json_url": switcher_json_url,
        "version_match": switcher_version,
    },
    "check_switcher": False,
    "show_version_warning_banner": True,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "py-version", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_prev_next": True,
    "show_nav_level": 1,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "fury-gl",
    "github_repo": "polyxios",
    "github_version": "master",
    "doc_path": "docs",
    # Start visitors in the CRT palette; the navbar switcher still works.
    "default_mode": "dark",
}

# The landing page is full-bleed: no primary sidebar, no page-toc.
html_sidebars = {"index": []}

html_last_updated_fmt = "%Y-%m-%d"
copybutton_prompt_text = r"\$ |>>> "
copybutton_prompt_is_regex = True
