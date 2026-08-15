"""Generate the contributor list on the credits page from the git history.

The core team is named by hand in ``conf.py``; everyone else who has landed a
commit is listed automatically, so a new contributor appears on the next build
with no page to edit. Identities are collapsed through the repository's
``.mailmap``, which ``git shortlog`` honours on its own.

The generated file is written next to the page that includes it and is not
tracked; a build outside a git checkout falls back to a short note instead of
failing.
"""

import subprocess
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

#: Substring match, case-insensitive, against "Name <email>".
BOT_MARKERS = ("[bot]", "dependabot", "pre-commit-ci", "github-actions")

OUTPUT = "_includes/contributors.rst"


def _shortlog(repo_root: Path) -> list[tuple[int, str]]:
    """Return (commit count, name) per contributor, most commits first.

    Parameters
    ----------
    repo_root
        Directory to run ``git`` in.

    Returns
    -------
    list of (int, str)
        Empty when ``git`` is unavailable or the directory is not a checkout.
    """
    try:
        out = subprocess.run(
            # Merges are kept: a maintainer whose commits are all merge
            # commits is still a contributor, and dropping them loses people
            # silently.
            ["git", "shortlog", "-sne", "--all"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    people = []
    for line in out.splitlines():
        count, _, who = line.strip().partition("\t")
        if not who:
            continue
        if any(marker in who.lower() for marker in BOT_MARKERS):
            continue
        name = who.rsplit("<", 1)[0].strip()
        if name:
            people.append((int(count), name))
    return people


def _render(people: list[tuple[int, str]], core: set[str]) -> str:
    """Build the reStructuredText fragment listing the non-core contributors.

    Parameters
    ----------
    people
        (commit count, name) pairs as returned by ``_shortlog``.
    core
        Names already credited as core developers, which are left out here.

    Returns
    -------
    str
        A bullet list, or a placeholder line when nobody qualifies yet.
    """
    if not people:
        return (
            "This list is generated from the git history at build time, and this\n"
            "build had no repository to read.\n"
        )

    others = [name for _count, name in people if name not in core]
    if not others:
        return (
            "Everyone who has contributed so far is listed above. This section\n"
            "fills itself in from the git history as new people land commits.\n"
        )

    lines = ["Listed by commit count, generated from the git history:", ""]
    lines += [f"* {name}" for name in others]
    return "\n".join(lines) + "\n"


def _generate(app: Sphinx) -> None:
    """Write the contributor fragment before the build reads any source.

    Parameters
    ----------
    app
        The Sphinx application, used for its ``srcdir`` and config.
    """
    srcdir = Path(app.srcdir)
    core = set(app.config.polyxios_core_developers)
    people = _shortlog(srcdir.parent)

    if not people:
        logger.warning(
            "credits: no git history available, contributor list will be a "
            "placeholder",
            type="polyxios",
        )

    target = srcdir / OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render(people, core), encoding="utf-8")


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
    app.add_config_value("polyxios_core_developers", [], "env", list)
    app.connect("builder-inited", _generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
