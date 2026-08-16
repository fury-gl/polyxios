"""Generate the version switcher index and the root redirect for GitHub Pages.

The published site keeps one directory per documentation version::

    /                 index.html (redirect), switcher.json, .nojekyll
    /dev/             built from master on every merge
    /stable/          a copy of the newest release
    /0.3/             built from tag v0.3.0
    /0.2/             built from tag v0.2.0

This script reads the directories that are actually present and writes
``switcher.json`` from them, so the switcher can never offer a version that was
never deployed. It runs in the deploy workflow, against a checkout of the
``gh-pages`` branch.

Usage
-----
::

    python tools/gen_switcher.py <site-root> --base-url https://fury-gl.github.io/polyxios
"""

import argparse
import json
from pathlib import Path
import re
import sys

#: Directory names that hold docs but are not a release series.
DEV_DIR = "dev"
STABLE_DIR = "stable"

_SERIES = re.compile(r"^\d+\.\d+$")


def _series_key(name: str) -> tuple[int, int]:
    """Return a sortable key for an ``X.Y`` directory name.

    Parameters
    ----------
    name
        Directory name, already known to match ``X.Y``.

    Returns
    -------
    tuple of int
        Major and minor, as integers.
    """
    major, minor = name.split(".")
    return int(major), int(minor)


def discover(root: Path) -> tuple[list[str], bool]:
    """Find the release series and the dev build present under ``root``.

    Parameters
    ----------
    root
        The site root, i.e. a checkout of the ``gh-pages`` branch.

    Returns
    -------
    tuple
        The release series newest first, and whether a ``dev`` build exists.
    """
    series = sorted(
        (p.name for p in root.iterdir() if p.is_dir() and _SERIES.match(p.name)),
        key=_series_key,
        reverse=True,
    )
    return series, (root / DEV_DIR).is_dir()


def build_entries(series: list[str], has_dev: bool, base_url: str) -> list[dict]:
    """Build the switcher entries, newest release marked preferred.

    Parameters
    ----------
    series
        Release series names, newest first.
    has_dev
        Whether a dev build is published.
    base_url
        Site root URL, without a trailing slash.

    Returns
    -------
    list of dict
        Entries in the order the switcher should show them.
    """
    entries: list[dict] = []

    if has_dev:
        entries.append({"name": "dev", "version": "dev", "url": f"{base_url}/dev/"})

    for index, name in enumerate(series):
        entry = {
            "name": f"{name} (stable)" if index == 0 else name,
            "version": name,
            # The newest release is also served from /stable/, which is the URL
            # worth handing out - it keeps working across releases.
            "url": f"{base_url}/{STABLE_DIR}/" if index == 0 else f"{base_url}/{name}/",
        }
        if index == 0:
            entry["preferred"] = True
        entries.append(entry)

    # With no release published yet, dev is all there is, so it is what the
    # root redirect and the switcher should prefer.
    if not series and entries:
        entries[0]["preferred"] = True

    return entries


def redirect_target(series: list[str], has_dev: bool) -> str:
    """Pick what the site root should redirect to.

    Parameters
    ----------
    series
        Release series names, newest first.
    has_dev
        Whether a dev build is published.

    Returns
    -------
    str
        A relative directory name with a trailing slash.
    """
    if series:
        return f"{STABLE_DIR}/"
    if has_dev:
        return f"{DEV_DIR}/"
    return f"{DEV_DIR}/"


def render_redirect(target: str) -> str:
    """Return the root ``index.html`` that forwards to ``target``.

    Parameters
    ----------
    target
        Relative path to redirect to.

    Returns
    -------
    str
        A complete HTML document.
    """
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>polyxios documentation</title>
    <meta http-equiv="refresh" content="0; url=./{target}">
    <link rel="canonical" href="./{target}">
  </head>
  <body>
    <p>Redirecting to <a href="./{target}">the polyxios documentation</a>.</p>
  </body>
</html>
"""


def render_robots(base_url: str) -> str:
    """Return a robots.txt that allows everything and points at the sitemap.

    Deliberately no ``Disallow`` for ``/dev/`` or the older series. Those copies
    are kept out of the index by a ``noindex`` tag and a canonical link pointing
    at ``/stable/`` - and a crawler that is disallowed never fetches the page,
    so it never sees either one. Blocking here would strand the duplicates in
    whatever state they were last indexed in.

    Parameters
    ----------
    base_url
        Site root URL, without a trailing slash.

    Returns
    -------
    str
        A complete robots.txt.
    """
    return "\n".join(
        [
            "User-agent: *",
            "Allow: /",
            "",
            "# /dev/ and the older X.Y directories are duplicates of /stable/.",
            "# They are excluded with per-page noindex and canonical tags, not",
            "# here, so crawlers can still fetch them and read those tags.",
            "",
            f"Sitemap: {base_url}/sitemap.xml",
            "",
        ]
    )


def hoist_site_files(root: Path, series: list[str], has_dev: bool) -> list[str]:
    """Copy the version-level sitemap and llms files up to the site root.

    Sphinx writes sitemap.xml, llms.txt and llms-full.txt inside whichever
    version directory it built. Crawlers and answer engines look for them at the
    root, so the canonical version's copies are lifted there.

    Parameters
    ----------
    root
        The site root.
    series
        Release series names, newest first.
    has_dev
        Whether a dev build is published.

    Returns
    -------
    list of str
        Names of the files copied.
    """
    source = root / STABLE_DIR if series else (root / DEV_DIR if has_dev else None)
    if source is None or not source.is_dir():
        return []

    copied = []
    for name in ("sitemap.xml", "llms.txt", "llms-full.txt"):
        candidate = source / name
        if candidate.is_file():
            (root / name).write_bytes(candidate.read_bytes())
            copied.append(name)
    return copied


def main(argv: list[str] | None = None) -> int:
    """Write ``switcher.json``, ``index.html`` and ``.nojekyll`` into the site root.

    Parameters
    ----------
    argv
        Command line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="site root (gh-pages checkout)")
    parser.add_argument(
        "--base-url",
        default="https://fury-gl.github.io/polyxios",
        help="site root URL, without a trailing slash",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    base_url = args.base_url.rstrip("/")
    series, has_dev = discover(root)
    entries = build_entries(series, has_dev, base_url)

    (root / "switcher.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )
    (root / "index.html").write_text(
        render_redirect(redirect_target(series, has_dev)), encoding="utf-8"
    )
    (root / "robots.txt").write_text(render_robots(base_url), encoding="utf-8")
    # Without this, Pages runs Jekyll and drops every _static/ directory.
    (root / ".nojekyll").touch()

    copied = hoist_site_files(root, series, has_dev)

    print(f"versions: dev={has_dev} releases={series or 'none'}")
    print(f"root redirects to {redirect_target(series, has_dev)}")
    print(f"hoisted to root: {', '.join(copied) or 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
