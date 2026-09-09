"""Generate the version switcher index and the root redirect for GitHub Pages.

The published site keeps one directory per documentation version::

    /                 index.html (redirect), switcher.json, .nojekyll
    /dev/             built from master on every merge
    /stable/          a copy of the newest release
    /0.4.1/           built from tag v0.4.1
    /0.4.0/           built from tag v0.4.0
    /0.2.0/           built from tag v0.2.0

One directory per release, not per ``X.Y`` series: two releases of one series
are two sets of documentation, and the switcher entry has to carry the exact
version its pages report or the theme reads them as something other than the
stable docs and banners them as a development build.

This script reads the directories that are actually present and writes
``switcher.json`` from them, so the switcher can never offer a version that was
never deployed. It runs in the deploy workflow, against a checkout of the
``gh-pages`` branch.

Usage
-----
::

    python tools/gen_switcher.py <site-root> --base-url https://polyxios.org
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

#: Directory names that hold docs but are not a release.
DEV_DIR = "dev"
STABLE_DIR = "stable"

#: A release directory. The two-component spelling is what the site used
#: before it published one directory per release; it is still recognised so a
#: directory left over from then is listed rather than dropped.
_RELEASE = re.compile(r"^\d+\.\d+(?:\.\d+)?$")

#: How a built directory records the exact version it was built from.
_BUILT_VERSION = re.compile(r"VERSION:\s*['\"]([^'\"]+)['\"]")


def _release_key(name: str) -> tuple[int, int, int]:
    """Return a sortable key for a release directory name.

    Parameters
    ----------
    name
        Directory name, already known to match ``X.Y`` or ``X.Y.Z``.

    Returns
    -------
    tuple of int
        Major, minor and patch, the patch defaulting to zero.
    """
    parts = [int(part) for part in name.split(".")]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def built_version(directory: Path) -> str | None:
    """Return the exact version a built directory reports, if it says.

    Sphinx writes the release string into ``_static/documentation_options.js``,
    which is how a directory named for a series can still say which release it
    actually holds.

    Parameters
    ----------
    directory
        A version directory under the site root.

    Returns
    -------
    str or None
        The version string, or None when the file is missing or says nothing.
    """
    options = directory / "_static" / "documentation_options.js"
    if not options.is_file():
        return None
    match = _BUILT_VERSION.search(options.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else None


def migrate_series_dirs(root: Path) -> list[tuple[str, str]]:
    """Rename a leftover ``X.Y`` directory to the release it actually holds.

    The site published one directory per series before it published one per
    release. Such a directory holds exactly one release - the newest of its
    series - so it is renamed to that version and the old name is left behind
    as a redirect, since links to it are already out in the world.

    Parameters
    ----------
    root
        The site root.

    Returns
    -------
    list of tuple of str
        The renames performed, as ``(old_name, new_name)`` pairs.
    """
    renamed: list[tuple[str, str]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not re.fullmatch(r"\d+\.\d+", path.name):
            continue
        version = built_version(path)
        if version is None or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            continue
        target = root / version
        if target.exists():
            # The release already has its own directory - published straight
            # there, or renamed on an earlier run. The old name is then a
            # second copy of the same pages, so it becomes a redirect too.
            if built_version(target) != version:
                continue
            shutil.rmtree(path)
            path.mkdir()
        else:
            path.rename(target)
            path.mkdir()
        (path / "index.html").write_text(
            render_redirect(f"../{version}/"), encoding="utf-8"
        )
        renamed.append((path.name, version))
    return renamed


def discover(root: Path) -> tuple[list[str], bool]:
    """Find the releases and the dev build present under ``root``.

    Parameters
    ----------
    root
        The site root, i.e. a checkout of the ``gh-pages`` branch.

    Returns
    -------
    tuple
        The releases newest first, and whether a ``dev`` build exists.
    """
    # A directory counts as a release when it holds a build that says which
    # version it is. The redirect left behind by a rename matches the name
    # pattern too, and listing it would offer the same docs twice.
    series = sorted(
        (
            p.name
            for p in root.iterdir()
            if p.is_dir() and _RELEASE.match(p.name) and built_version(p) is not None
        ),
        key=_release_key,
        reverse=True,
    )
    return series, (root / DEV_DIR).is_dir()


def build_entries(series: list[str], has_dev: bool, base_url: str) -> list[dict]:
    """Build the switcher entries, newest release marked preferred.

    Parameters
    ----------
    series
        Release versions, newest first.
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
        Release versions, newest first.
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
        Release versions, newest first.
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
        default="https://polyxios.org",
        help="site root URL, without a trailing slash",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    base_url = args.base_url.rstrip("/")
    renamed = migrate_series_dirs(root)
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

    for old, new in renamed:
        print(f"{old}/ now redirects to {new}/")
    print(f"versions: dev={has_dev} releases={series or 'none'}")
    print(f"root redirects to {redirect_target(series, has_dev)}")
    print(f"hoisted to root: {', '.join(copied) or 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
