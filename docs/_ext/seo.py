"""Search-engine and answer-engine metadata for the versioned documentation.

Two jobs, both of which exist because the site publishes several versions of
every page side by side:

*Indexing.* ``/dev/``, ``/stable/`` and ``/0.3/`` serve near-identical HTML. Left
alone, search engines treat them as duplicates and split the ranking between
them. ``conf.py`` sets ``html_baseurl`` to the stable path so every build emits a
canonical link pointing there, and this extension marks non-stable builds
``noindex`` so only one copy competes.

*Answer engines.* ``llms.txt`` and ``llms-full.txt`` are written into the build
output following the llmstxt.org convention: a short index of every page for a
model deciding what to fetch, and the full prose for one that wants to read
everything in a single request.
"""

from pathlib import Path
import re
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

#: Pages that carry no prose worth handing to an answer engine.
SKIP_DOCS = frozenset({"genindex", "modindex", "search"})

#: Autosummary writes one stub per object under here. They repeat what the API
#: index already links to, so they stay out of the llms.txt index.
SKIP_PREFIXES = ("api/generated/",)

MAX_SUMMARY_CHARS = 220

#: The ``.. meta::`` directive writes straight into the page's ``metatags``
#: string and never reaches the page context's ``meta`` dict, so the hand-written
#: description has to be read back out of the rendered tag. Attribute order is
#: not fixed, hence the two alternatives.
_DESCRIPTION_RE = re.compile(
    r'<meta[^>]*?(?:name="description"[^>]*?content="([^"]*)"'
    r'|content="([^"]*)"[^>]*?name="description")',
    re.IGNORECASE,
)


def _first_paragraph(doctree: nodes.document) -> str:
    """Return the first real paragraph of a page, flattened to plain text.

    Parameters
    ----------
    doctree
        The resolved doctree for one page.

    Returns
    -------
    str
        Empty when the page has no paragraph, e.g. a pure toctree stub.
    """
    for node in doctree.findall(nodes.paragraph):
        # docutils reports its own problems as paragraphs inside a
        # system_message; those are not page content.
        if any(isinstance(parent, nodes.system_message) for parent in node.parent.traverse(ascend=True, descend=False)):
            continue
        text = " ".join(node.astext().split())
        # Skip the one-word field-list leftovers and badge rows.
        if len(text) > 40:
            return text
    return ""


def _toctree_order(app: Sphinx) -> list[str]:
    """Return every document in reading order, root first.

    Alphabetical order would open the index with autosummary stubs. Walking the
    toctree gives an answer engine the same path a reader takes.

    Parameters
    ----------
    app
        The Sphinx application.

    Returns
    -------
    list of str
        Document names, depth-first through the toctree, then anything the
        toctree does not reach.
    """
    root = app.config.root_doc
    includes = app.env.toctree_includes
    ordered: list[str] = []
    seen: set[str] = set()

    def walk(docname: str) -> None:
        if docname in seen:
            return
        seen.add(docname)
        ordered.append(docname)
        for child in includes.get(docname, []):
            walk(child)

    walk(root)
    ordered += sorted(d for d in app.env.found_docs if d not in seen)
    return ordered


def _truncate(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """Cut ``text`` to ``limit`` characters on a word boundary.

    Parameters
    ----------
    text
        Text to shorten.
    limit
        Maximum length of the result, before the ellipsis is added.

    Returns
    -------
    str
        The text unchanged when it already fits.
    """
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


def _page_meta(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: nodes.document | None,
) -> None:
    """Add the per-page tags Sphinx and sphinxext-opengraph leave out.

    Two of them:

    ``noindex`` on every build that is not the stable one, so the duplicate
    copies at ``/dev/`` and ``/0.3/`` do not compete with ``/stable/``.
    ``follow`` is kept so the crawler still walks the links and reaches the
    canonical pages.

    ``og:description`` mirrored from a hand-written ``.. meta:: :description:``.
    sphinxext-opengraph derives its description from page prose, so a page built
    out of a raw HTML include - the landing page - would otherwise get none.

    Parameters
    ----------
    app
        The Sphinx application.
    pagename
        Document name being rendered.
    templatename
        Template about to be used; unused.
    context
        The page context, whose ``metatags`` entry is appended to.
    doctree
        The page doctree, or None for generated pages.
    """
    extra = []

    metatags = context.get("metatags", "")
    match = _DESCRIPTION_RE.search(metatags)
    if match and "og:description" not in metatags:
        text = " ".join((match.group(1) or match.group(2)).split())
        extra.append(f'<meta property="og:description" content="{text}" />')

    if not app.config.polyxios_is_stable:
        extra.append('<meta name="robots" content="noindex, follow" />')

    if extra:
        context["metatags"] = context.get("metatags", "") + "\n    " + "\n    ".join(extra)


def _purge(app: Sphinx, env: Any, docname: str) -> None:
    """Drop a document's stored summary when it is about to be re-read.

    The store has to survive between builds: an incremental build only re-reads
    the documents that changed, so resetting it wholesale would write an
    llms.txt containing just those.

    Parameters
    ----------
    app
        The Sphinx application.
    env
        The build environment holding the store.
    docname
        Document being purged.
    """
    getattr(env, "polyxios_summaries", {}).pop(docname, None)


def _merge(app: Sphinx, env: Any, docnames: list[str], other: Any) -> None:
    """Merge a parallel read worker's summaries back into the main environment.

    Parameters
    ----------
    app
        The Sphinx application.
    env
        The main build environment.
    docnames
        Documents the worker read; unused.
    other
        The worker's environment.
    """
    if not hasattr(env, "polyxios_summaries"):
        env.polyxios_summaries = {}
    env.polyxios_summaries.update(getattr(other, "polyxios_summaries", {}))


def _record(app: Sphinx, doctree: nodes.document) -> None:
    """Store one page's title and lead paragraph for the llms.txt index.

    Parameters
    ----------
    app
        The Sphinx application.
    doctree
        The doctree just read.
    """
    docname = app.env.docname
    if docname in SKIP_DOCS:
        return
    title = app.env.titles.get(docname)
    store = getattr(app.env, "polyxios_summaries", None)
    if store is None:
        store = app.env.polyxios_summaries = {}
    store[docname] = (
        title.astext() if title is not None else docname,
        _first_paragraph(doctree),
        doctree.astext(),
    )


def _write_llms_txt(app: Sphinx, exception: Exception | None) -> None:
    """Write ``llms.txt`` and ``llms-full.txt`` into the build output.

    Parameters
    ----------
    app
        The Sphinx application.
    exception
        The build error, if the build failed; nothing is written in that case.
    """
    if exception is not None or app.builder.name != "html":
        return

    summaries = getattr(app.env, "polyxios_summaries", {})
    if not summaries:
        return

    base = app.config.html_baseurl.rstrip("/")
    project = app.config.project
    tagline = app.config.polyxios_tagline
    out = Path(app.outdir)

    # Index: one line per page, so a model can pick what to fetch.
    index = [f"# {project}", "", f"> {tagline}", ""]
    index += [
        "polyxios reads and writes 25 3D mesh and geometry file formats through one "
        "API, with a single runtime dependency (NumPy). It raises on malformed input "
        "rather than silently truncating, and can memory-map large binary files.",
        "",
        "## Documentation",
        "",
    ]
    order = [d for d in _toctree_order(app) if d in summaries]
    for docname in order:
        if docname.startswith(SKIP_PREFIXES):
            continue
        title, summary, _full = summaries[docname]
        url = f"{base}/{docname}.html"
        line = f"- [{title}]({url})"
        if summary:
            line += f": {_truncate(summary, 160)}"
        index.append(line)
    index.append("")
    (out / "llms.txt").write_text("\n".join(index), encoding="utf-8")

    # Full text: the whole documentation in one request.
    full = [f"# {project}", "", f"> {tagline}", ""]
    for docname in order:
        title, _summary, text = summaries[docname]
        full += [
            f"## {title}",
            "",
            f"Source: {base}/{docname}.html",
            "",
            text.strip(),
            "",
        ]
    (out / "llms-full.txt").write_text("\n".join(full), encoding="utf-8")

    logger.info("wrote llms.txt and llms-full.txt (%d pages)", len(summaries))


def setup(app: Sphinx) -> dict[str, object]:
    """Register the extension.

    Parameters
    ----------
    app
        The Sphinx application.

    Returns
    -------
    dict
        Standard Sphinx extension metadata.
    """
    app.add_config_value("polyxios_is_stable", False, "html", bool)
    app.add_config_value("polyxios_tagline", "", "html", str)
    app.connect("env-purge-doc", _purge)
    app.connect("env-merge-info", _merge)
    app.connect("doctree-read", _record)
    app.connect("html-page-context", _page_meta)
    app.connect("build-finished", _write_llms_txt)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
