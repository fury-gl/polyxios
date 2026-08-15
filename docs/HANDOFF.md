# polyxios docs - Sphinx deliverable

Retro terminal theme (the version you approved) built as **pydata-sphinx-theme
overrides**. Nothing here is a fork of the theme: one stylesheet redefines the
theme's CSS variables, and the landing page is a raw HTML include.

## What is in here

```
sphinx-docs/
  conf.py                     replaces docs/conf.py
  requirements.txt            doc build deps
  index.rst                   landing page (raw HTML include + hidden toctree)
  _includes/homepage.html     the landing page markup - edit copy here
  _includes/formats_grid.html the eighteen format cards on formats/index
  _static/css/retro.css       the whole theme: palettes, chrome, landing page
  _static/js/homepage.js      copy button for the hero install command
  _static/switcher.json       version switcher index - edit on every release
  _templates/py-version.html  the "py>=3.11" navbar chip
  _templates/sidebar-nav-bs.html  full-tree section navigation (see below)
  _templates/navbar-nav.html      the four fixed header links
  usage.rst                   quickstart, now pointing at the split-out pages
  lazy_loading.rst            split out of the old usage.rst
  transforms.rst              split out of the old usage.rst
  cli.rst                     split out of the old usage.rst
  plugins.rst                 split out of the old usage.rst
  formats/
    index.rst                 the 18-format table + hidden toctree
    vtk.rst vtr.rst vtp.rst obj.rst ply.rst stl.rst off.rst abaqus.rst
    avs.rst meshb.rst dolfin.rst flac3d.rst gmsh.rst nastran.rst
    tecplot.rst su2.rst tetgen.rst wkt.rst
```

## Install

Copy the tree into the repo's `docs/` directory (it is laid out to drop in
directly), then:

```bash
pip install -r docs/requirements.txt
spin docs --open          # or: cd docs && make html
```

Two new dependencies beyond your current build: `pydata-sphinx-theme` (already
in your `conf.py`) and `sphinx-copybutton` (drives the copy button on code
blocks - drop the extension from `conf.py` if you would rather not add it).

Your existing `installation.rst`, `contributing.rst`, `development.rst`,
`changelog.rst` and `api/` are unchanged and already wired into the toctree in
`index.rst`.

## How the theme works

* **Light mode = "paper"**, dark mode = **"crt"** (amber phosphor). Both are
  defined at the top of `retro.css` as pydata variables, so the theme's own
  navbar switcher drives them - no custom JavaScript. `conf.py` sets
  `default_mode: "dark"` so first-time visitors land in the CRT palette.
* Fonts: JetBrains Mono throughout, imported by `retro.css`. To vendor it
  instead of hitting Google Fonts, drop the woff2 files in `_static/fonts/`
  and swap the `@import` for `@font-face` rules.
* The scanline overlay is `body::after` - delete that block to remove it.
* Section markers (`// ` before every `h2`), the file-tree glyphs in the
  sidebar and the `[!]` quirk bullets are all CSS `::before` content, so the
  reStructuredText stays clean.
* The landing page has no sidebars: `html_sidebars = {"index": []}` plus the
  `:html_theme.sidebar_secondary.remove:` field at the top of `index.rst`. It is
  also full-bleed: `retro.css` drops the theme's 60em article column via
  `.bd-article-container:has(.px-home)`.
* The logo text in `conf.py` is written `&lt;polyxios /&gt;`. pydata drops the
  string into the template unescaped, so a literal `<polyxios />` is parsed as
  an unknown HTML element and renders nothing at all.

## Header links

`_templates/navbar-nav.html` replaces pydata's auto-generated header nav, which
listed every top-level toctree entry and spilled the rest into a "More"
dropdown. The four links - docs, formats, cli, api - are hard-coded in that
template; add one there, not in the toctree.

## Section navigation

pydata's stock `sidebar-nav-bs.html` renders the toctree from `startdepth=1` -
only the pages nested under the current top-level entry. This toctree is flat at
the top level, so that left the sidebar empty on every page. The override in
`_templates/sidebar-nav-bs.html` passes `startdepth=0`, so each page carries the
whole documentation tree.

The landing page is the exception and drops the sidebar entirely, via
`html_sidebars = {"index": []}`.

Sidebar entries read as filenames (`usage.rst`, `formats/`) because the toctree
in `index.rst` gives each page an explicit label: `usage.rst <usage>`. The page's
own title, its breadcrumb and its prev/next label all still come from the `rst`
heading, so only the sidebar changes.

`show_nav_level: 1` keeps sections with children (formats, api) collapsible -
`retro.css` swaps pydata's chevron for a terminal-style `[+]` / `[-]`. Raising
it to `2` expands everything but removes the toggles, which is a poor trade with
eighteen format pages. `collapse_navigation: False` puts every section's children
in the DOM so the toggles work without a page load.

## Version switcher

`conf.py` derives the switcher entry from the installed version: a dev build
(`0.3.0.dev0`) matches `dev`, a tagged build matches its own `X.Y` series. Both
can be overridden by environment variable:

| Variable | Purpose |
| --- | --- |
| `SWITCHER_VERSION` | pin the entry to highlight, e.g. `0.2` in a release job |
| `SWITCHER_JSON_URL` | point at a different index; use `_static/switcher.json` to test locally |

The published index lives at `https://fury-gl.github.io/polyxios/_static/switcher.json`,
built from `docs/_static/switcher.json`. **That URL is not live yet** - there is
no Pages deploy in `.github/workflows/build_docs.yml`, so until one exists the
switcher button renders but its dropdown stays empty. `check_switcher` is off so
this never fails a `-W` build.

On each release: add the new series to `docs/_static/switcher.json`, move
`"preferred": true` onto it, and drop any series you stop publishing (a stale
entry is a dead link).

## Format pages - please review

Each of the eighteen pages follows one shape:

1. badges - extension, read/write, lazy support
2. **Summary of the specification** - prose, 4-6 sentences
3. **Specification at a glance** - a field table of the structural facts
4. a button linking to the full specification
5. **Reading** / **Writing** with the code and any format-specific options
6. **Quirks worth knowing** - what polyxios does that the spec does not say

The summaries were written from the codec sources and the public specifications;
the quirks come from your README's format table. Two things to check:

* the option tables - only `.vtk`, `.ply`, `.stl` and `.bdf` document options
  here; if other codecs take kwargs, they need adding
* the specification URLs, listed below, especially the vendor ones

| Format | Spec link used |
| --- | --- |
| VTK (.vtk/.vtr/.vtp) | docs.vtk.org VTKFileFormats |
| OBJ | paulbourke.net/dataformats/obj |
| PLY | paulbourke.net/dataformats/ply |
| STL | fabbers.com/tech/STL_Format |
| OFF | segeval.cs.princeton.edu OFF format |
| Abaqus | help.3ds.com keywords reference |
| AVS-UCD | lanl.github.io/LaGriT read_avs |
| Medit | people.sc.fsu.edu medit |
| DOLFIN | fenicsproject.org olddocs XMLFile |
| FLAC3D | docs.itascacg.com grid import/export |
| Gmsh | gmsh.info MSH file format |
| Nastran | pynastran-git.readthedocs.io BDF overview |
| Tecplot | tecplot.com data format guide (PDF) |
| SU2 | su2code.github.io Mesh-File |
| TetGen | wias-berlin.de tetgen fformats |
| WKT | libgeos.org specifications/wkt |

## Adding a format later

Copy any file in `formats/`, keep the six-section shape, and add its slug to
the hidden toctree and the table in `formats/index.rst`. The landing page table
in `_includes/homepage.html` is hand-written HTML, so add a row there too.
