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
  _includes/formats_grid.html the twenty-seven format cards on formats/index
  _static/css/retro.css       the whole theme: palettes, chrome, landing page
  _static/js/homepage.js      copy button for the hero install command
  _ext/contributors.py        credits page contributor list, from git
  _ext/seo.py                 canonical/noindex tags + llms.txt generation
  credits.rst                 core team + generated contributor list
  _templates/py-version.html  the "py>=3.11" navbar chip
  _templates/sidebar-nav-bs.html  full-tree section navigation (see below)
  _templates/navbar-nav.html      the four fixed header links
  usage.rst                   quickstart, now pointing at the split-out pages
  lazy_loading.rst            split out of the old usage.rst
  transforms.rst              split out of the old usage.rst
  cli.rst                     split out of the old usage.rst
  plugins.rst                 split out of the old usage.rst
  formats/
    index.rst                 the format card grid + hidden toctree
    vtk.rst vtr.rst vtp.rst vtu.rst vts.rst vti.rst obj.rst ply.rst
    stl.rst off.rst abaqus.rst avs.rst meshb.rst mfem.rst dolfin.rst
    flac3d.rst gmsh.rst nastran.rst tecplot.rst su2.rst tetgen.rst
    wkt.rst netgen.rst ugrid.rst splat.rst
```

Outside `docs/`:

```
.mailmap                      collapses duplicate committer identities
tools/gen_switcher.py         writes switcher.json + the root redirect on gh-pages
.github/workflows/build_docs.yml   build on PR, publish on merge and release
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
twenty-seven format pages. `collapse_navigation: False` puts every section's children
in the DOM so the toggles work without a page load.

## Theme version

`pydata-sphinx-theme` is pinned to `>=0.20,<0.21` in both `pyproject.toml` and
`docs/requirements.txt`. The pin is not cosmetic: 0.16 replaced the sidebar
toggle markup, from `input.toctree-checkbox` + `label.toctree-toggle` to
`<details><summary><span class="toctree-toggle">`. `retro.css` styles that
toggle, so an unpinned theme means CI and a local checkout can render the
sidebar differently.

`retro.css` matches the bare `.toctree-toggle` class and handles both open-state
markups, so it survives either version - but bump the pin deliberately and
re-check the overrides when you do.

One related trap: the glyph goes on `.toctree-toggle::before` with every child
hidden, rather than on the inner `<i>`. Font Awesome's JS build swaps that `<i>`
for an `<svg>` at runtime, so an `i` selector matches nothing by the time the
icon is on screen.

Everything is served from **polyxios.org**. The `fury-gl.github.io/polyxios`
address only 301s there, so `SITE_URL` in `conf.py`, the `--base-url` in the
deploy workflow and `tools/gen_switcher.py` all name the custom domain - a
canonical link must be the final URL, not a redirect source.

## Credits page

`credits.rst` names the core developers by hand - the list lives in `conf.py` as
`polyxios_core_developers`, so the page and the extension agree on who is
already credited.

Everyone else in the git history is listed under "Contributors" by
`docs/_ext/contributors.py`, which shells out to `git shortlog -sne` on
`builder-inited` and writes `_includes/contributors.rst` (generated, gitignored).
A merged pull request is therefore all it takes to appear - there is no list to
edit. Bots are filtered by the `BOT_MARKERS` tuple.

Duplicate identities collapse through `.mailmap` at the repository root, which
`git shortlog` honours on its own. Add a line there rather than special-casing a
name in the extension.

A build outside a git checkout (an sdist, say) degrades to a placeholder
paragraph and a warning rather than failing.

## Version switcher and deployment

### Site layout

The `gh-pages` branch holds one directory per version:

```
/                 index.html (redirect), switcher.json, .nojekyll
/dev/             built from master on every merge
/stable/          a copy of the newest release
/0.3/  /0.2/      built from tags v0.3.0, v0.2.0
```

`switcher.json` and the root redirect are **generated**, not tracked:
`tools/gen_switcher.py` reads the directories actually present on `gh-pages` and
writes both. That is deliberate - a hand-maintained index drifts and starts
offering versions that were never deployed. There is no `switcher.json` in the
repository for the same reason.

The newest release is published twice, under `X.Y/` and under `stable/`, and the
switcher points at `stable/` for it. That URL survives the next release, so it
is the one worth linking to from elsewhere.

Until the first release ships, `dev` is the only entry and the root redirects
there.

### Which entry a build highlights

`conf.py` derives it from the installed version: a dev build (`0.3.0.dev0`)
matches `dev`, a tagged build matches its own `X.Y` series. Two environment
variables override that:

| Variable | Purpose |
| --- | --- |
| `SWITCHER_VERSION` | pin the entry to highlight. The workflow sets this per build rather than trusting the installed version |
| `SWITCHER_JSON_URL` | point at a different index, e.g. a locally generated one |

`check_switcher` is off, so a build with no network access never fails a `-W`
build on an unreachable index.

### The workflow

`.github/workflows/build_docs.yml`:

| Trigger | Builds | Publishes to |
| --- | --- | --- |
| pull request | yes, `-W --keep-going` | nothing |
| push to master | yes | `dev/` |
| release published | yes | `X.Y/` and `stable/` |
| workflow_dispatch | yes | `dev/` |

Pull requests never publish - they can run from forks without write access, and
a preview would overwrite the live dev docs. The deploy job creates the
`gh-pages` branch as an orphan on its first run, so nothing has to be seeded by
hand.

The release trigger parses `X.Y` out of the tag and fails loudly if the tag does
not look like `vX.Y.Z`, rather than publishing to a directory named after a
typo.

### One-time setup

GitHub Pages has to be pointed at the branch once, which no workflow can do for
you:

```bash
gh api -X POST repos/fury-gl/polyxios/pages \
  -f 'source[branch]=gh-pages' -f 'source[path]=/'
```

Or: Settings -> Pages -> Source -> Deploy from a branch -> `gh-pages` / `/`.

Do this **after** the first deploy has created the branch, or the API call has
nothing to point at.

### Testing the switcher locally

```bash
mkdir -p /tmp/site/dev && python tools/gen_switcher.py /tmp/site
cd docs && SWITCHER_JSON_URL=file:///tmp/site/switcher.json make html
```

## SEO and answer engines

### The duplicate-content problem this site has

Publishing `/dev/`, `/stable/` and `/0.3/` means three near-identical copies of
every page. Left alone, search engines treat them as duplicates and split the
ranking between them. Three things keep that under control:

* `html_baseurl` in `conf.py` is `<site>/stable/`, so **every** build emits a
  canonical link pointing at the stable copy.
* `docs/_ext/seo.py` adds `<meta name="robots" content="noindex, follow">` to
  every build that is not stable. `DOCS_IS_STABLE=true` is set by the workflow
  on release builds only.
* `robots.txt` deliberately **allows** everything. A disallowed page is never
  fetched, so the crawler would never see the noindex or the canonical tag -
  blocking there strands the duplicates in whatever state they were last
  indexed in.

### Per-page descriptions

Sphinx emits no `<meta name="description">` on its own. The main pages carry a
hand-written one:

```rst
.. meta::
   :description: One or two sentences, roughly 150-160 characters.
```

The twenty-seven format pages deliberately have none - `sphinxext-opengraph`
derives theirs from the "Summary of the specification" paragraph, which is
already the right text. Write one by hand only when the derived text is poor.

Note that `.. meta::` writes straight into the page's `metatags` string and
never reaches the page context's `meta` dict, which is why `seo.py` reads the
description back out of the rendered tag to mirror it into `og:description`.
Without that the landing page gets none, since it is a raw HTML include with no
prose for opengraph to find.

There is no `<meta name="keywords">` anywhere and none should be added. Google
has ignored it since 2009 and Bing treats it as a spam signal. The keyword
surface that does count is the `<title>`, the description, the headings, and the
JSON-LD `keywords` array on the landing page.

### Social cards

`sphinxext-opengraph` renders a preview PNG per page at build time, which is why
the doc extra is `sphinxext-opengraph[social-cards]` - it pulls matplotlib. Drop
the extra and set `ogp_social_cards = {"enable": False}` together, or the build
warns on every page.

### llms.txt

`seo.py` writes two files into the build output, following the llmstxt.org
convention:

* `llms.txt` - one line per page, title plus a trimmed lead paragraph, in
  toctree order. Autosummary stubs under `api/generated/` are left out.
* `llms-full.txt` - the full prose of every page in one file.

`tools/gen_switcher.py` lifts both, plus `sitemap.xml`, from the canonical
version directory up to the site root, because that is where crawlers and answer
engines look for them.

The summaries live on `app.env` and are purged per document rather than reset
per build. An incremental build only re-reads what changed, so a wholesale reset
would write an `llms.txt` containing only those pages.

## Format pages - please review

Each of the twenty-seven pages follows one shape:

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
| VTK (.vtk/.vtr/.vtp/.vtu/.vts/.vti) | examples.vtk.org VTKFileFormats |
| OBJ | paulbourke.net/dataformats/obj |
| PLY | paulbourke.net/dataformats/ply |
| STL | fabbers.com/tech/STL_Format |
| OFF | segeval.cs.princeton.edu OFF format |
| Abaqus | help.3ds.com keywords reference |
| AVS-UCD | lanl.github.io/LaGriT read_avs |
| Medit | people.sc.fsu.edu medit |
| DOLFIN | people.sc.fsu.edu dolfin_xml |
| FLAC3D | docs.itascacg.com Itasca Grid Format |
| Gmsh | gmsh.info MSH file format |
| Kratos MDPA | github.com/KratosMultiphysics/Kratos wiki Input-Data |
| Nastran | pynastran-git.readthedocs.io BDF overview |
| Tecplot | tecplot.azureedge.net 360-data-format (PDF) |
| SU2 | su2code.github.io Mesh-File |
| TetGen | wias-berlin.de tetgen fformats |
| WKT | libgeos.org specifications/wkt |
| MFEM | mfem.org mesh-format-v1.0 |
| Netgen | github.com/NGSolve/netgen (project, no spec page found) |
| UGRID | simcenter.msstate.edu ugrid |
| Gaussian splat | github.com/antimatter15/splat (reference impl, no formal spec) |

Every link above was checked with `curl -o /dev/null -w '%{http_code}' -L`.
The Abaqus one answers 403 to curl but resolves in a browser; the rest answer 200.
Four were dead and have been replaced: the VTK, DOLFIN, FLAC3D and Tecplot rows.

## Adding a format later

Copy any file in `formats/`, keep the six-section shape, and add its slug to
the hidden toctree and the table in `formats/index.rst`. The landing page table
in `_includes/homepage.html` is hand-written HTML, so add a row there too.
