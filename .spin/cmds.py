"""Custom spin commands for polyxios development."""

import datetime
import os
import platform
import re
import shutil
import subprocess
import sys

import click

UPSTREAM_URL = "https://github.com/fury-gl/polyxios.git"
UPSTREAM_NAME = "upstream"


def _run(cmd, *, check=True, capture=True):
    result = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if check and result.returncode != 0:
        if capture:
            click.echo(result.stderr, err=True)
        return None
    return result.stdout.strip() if capture else result.returncode


def _ensure_installed(packages):
    """Ensure specific pip packages are installed before running a command."""
    if isinstance(packages, str):
        packages = [packages]

    click.echo(f"Ensuring dependencies are met: {', '.join(packages)}...")
    _run([sys.executable, "-m", "pip", "install", *packages], capture=False)


def _get_remotes():
    output = _run(["git", "remote", "-v"])
    if not output:
        return {}
    remotes = {}
    for line in output.split("\n"):
        if "(fetch)" in line:
            parts = line.split()
            remotes[parts[0]] = parts[1]
    return remotes


@click.command()
def setup():
    """Set up the development environment.

    Run once after cloning your fork. Adds the upstream remote and prints
    the next steps for building and testing.
    """
    click.echo("Setting up polyxios development environment...\n")

    if _run(["git", "rev-parse", "--git-dir"], check=False) is None:
        click.echo("Error: not in a git repository", err=True)
        sys.exit(1)

    remotes = _get_remotes()
    upstream_remote = next(
        (name for name, url in remotes.items() if "fury-gl/polyxios" in url), None
    )
    if upstream_remote is None:
        click.echo(f"Adding upstream remote: {UPSTREAM_URL}")
        _run(["git", "remote", "add", UPSTREAM_NAME, UPSTREAM_URL])
    else:
        click.echo(f"Upstream remote already present: {upstream_remote}")

    click.echo("\nInstalling dev dependencies...")
    _ensure_installed(
        [
            "meson-python>=0.15",
            "Cython>=3.0",
            "numpy>=1.24",
            "meson",
            "ninja",
            "mypy",
            "pre-commit",
        ]
    )

    if platform.system() == "Darwin":
        click.echo("\nDetected macOS - installing libomp for OpenMP support...")
        _run(["brew", "install", "libomp"], capture=False)

    click.echo("\nSetup complete! Next steps:")
    click.echo("  spin install    # build Cython extensions and install")
    click.echo("  spin test       # run the test suite")
    click.echo("  spin lint       # run ruff + codespell")
    click.echo("  spin docs       # build Sphinx documentation")


@click.command()
@click.option(
    "--editable",
    "-e",
    is_flag=True,
    default=False,
    help="Install in editable mode (development)",
)
def install(editable):
    """Install polyxios with Cython extensions compiled.

    Default: regular install (built wheel, no source link).
    Pass --editable / -e for a development install that reflects source changes.

    Requires meson, ninja, Cython and numpy (run ``spin setup`` first).

    Parameters
    ----------
    editable : bool
        If True, install in editable mode (``pip install -e .``).
    """
    cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation"]
    if editable:
        click.echo("Installing polyxios in editable mode...")
        cmd.append("-e")
    else:
        click.echo("Installing polyxios...")
    cmd.append(".")
    sys.exit(_run(cmd, capture=False, check=False))


@click.command()
@click.option(
    "-k",
    "--match",
    "pattern",
    default=None,
    help="Only run tests matching this pattern (pytest -k)",
)
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.argument("pytest_args", nargs=-1)
def test(pattern, verbose, pytest_args):
    """Run the test suite with pytest.

    Parameters
    ----------
    pattern : str, optional
        Filter tests by name pattern (passed to pytest -k).
    verbose : bool
        Enable verbose pytest output.
    pytest_args : tuple
        Extra arguments forwarded directly to pytest.
    """
    _ensure_installed(["pytest>=7"])

    cmd = ["pytest", "tests/"]

    if pattern:
        cmd.extend(["-k", pattern])
    if verbose:
        cmd.append("-v")
    if pytest_args:
        cmd.extend(pytest_args)

    click.echo(f"Running: {' '.join(cmd)}\n")
    sys.exit(_run(cmd, capture=False, check=False))


@click.command()
@click.option(
    "--fix", is_flag=True, default=False, help="Auto-fix issues where possible"
)
def lint(fix):
    """Run ruff linter, formatter check, and codespell.

    Parameters
    ----------
    fix : bool
        If True, apply automatic fixes with ruff.
    """
    _ensure_installed(["ruff", "codespell"])
    failed = False

    click.echo("Running ruff linter...")
    ruff_cmd = ["ruff", "check", "."]
    if fix:
        ruff_cmd.append("--fix")
    if _run(ruff_cmd, capture=False, check=False) != 0:
        click.echo("Linting issues found.", err=True)
        failed = True

    click.echo("\nRunning ruff formatter...")
    fmt_cmd = ["ruff", "format", "."] if fix else ["ruff", "format", "--check", "."]
    if _run(fmt_cmd, capture=False, check=False) != 0:
        click.echo("Formatting issues found.", err=True)
        failed = True

    click.echo("\nRunning codespell...")
    spell_cmd = [
        "codespell",
        "--skip",
        "*.pyc,.git,_build,*.egg-info,./build",
        "polyxios",
        "tests",
        "docs",
        ".spin",
    ]
    if _run(spell_cmd, capture=False, check=False) != 0:
        click.echo("Spelling issues found.", err=True)
        failed = True

    if failed:
        sys.exit(1)
    click.echo("\nAll checks passed!")


@click.command()
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Remove build directory before building",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    default=False,
    help="Open docs in browser after building",
)
def docs(clean, open_browser):
    """Build Sphinx documentation.

    Parameters
    ----------
    clean : bool
        Remove the _build directory first.
    open_browser : bool
        Open the built docs in the default browser.
    """
    _ensure_installed(
        [
            "sphinx",
            "numpydoc",
            "sphinx-gallery",
            "pydata-sphinx-theme",
        ]
    )
    docs_dir = "docs"
    build_dir = os.path.join(docs_dir, "_build")

    if clean and os.path.exists(build_dir):
        click.echo("Cleaning build directory...")
        shutil.rmtree(build_dir)

    click.echo("Building documentation...")
    result = _run(["make", "-C", docs_dir, "html"], capture=False, check=False)

    if result == 0:
        index = os.path.abspath(os.path.join(build_dir, "html", "index.html"))
        click.echo(f"\nDocs built: {index}")
        if open_browser:
            import webbrowser

            webbrowser.open(f"file://{index}")

    sys.exit(result)


@click.command()
def clean():
    """Remove build artifacts and cache directories."""
    click.echo("Cleaning up...")
    targets = [
        "build",
        "dist",
        "_build",
        "**/__pycache__",
        "**/.pytest_cache",
        "**/*.egg-info",
    ]
    import glob

    for pattern in targets:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isdir(path):
                click.echo(f"  rm -r {path}")
                shutil.rmtree(path)
            elif os.path.isfile(path):
                click.echo(f"  rm {path}")
                os.remove(path)
    click.echo("Done.")


def _next_dev_version(version):
    """Increment minor component and append .dev0."""
    parts = version.split(".")
    parts[1] = str(int(parts[1]) + 1)
    if len(parts) > 2:
        parts[2] = "0"
    return ".".join(parts) + ".dev0"


def _bump_pyproject(pyproject_path, *, new_version):
    with open(pyproject_path) as f:
        content = f.read()
    updated = re.sub(
        r'^(version\s*=\s*")[^"]*(")',
        rf"\g<1>{new_version}\g<2>",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if updated == content:
        click.echo(f"ERROR: version line not found in {pyproject_path}", err=True)
        sys.exit(1)
    with open(pyproject_path, "w") as f:
        f.write(updated)


def _bump_changelog(changes_path, *, release_version, next_version, release_date):
    with open(changes_path) as f:
        content = f.read()

    dated_heading = f"{release_version} ({release_date})"
    updated = re.sub(
        rf"{re.escape(release_version)}\s*\(upcoming\)",
        dated_heading,
        content,
        count=1,
    )
    if updated == content:
        click.echo(
            f"WARNING: '{release_version} (upcoming)' not found in {changes_path}.",
            err=True,
        )
    with open(changes_path, "w") as f:
        f.write(updated)


def _prepend_upcoming_section(changes_path, *, next_version):
    anchor = f".. _changes_{next_version.replace('.dev0', '')}:"
    heading = f"{next_version.replace('.dev0', '')} (upcoming)"
    underline = "-" * len(heading)
    new_section = f"{anchor}\n\n{heading}\n{underline}\n\n(No entries yet.)\n\n"
    with open(changes_path) as f:
        content = f.read()
    marker = ".. _changes_"
    insert_at = content.find(marker, content.find(marker) + 1)
    if insert_at == -1:
        insert_at = content.find(marker)
    with open(changes_path, "w") as f:
        f.write(content[:insert_at] + new_section + content[insert_at:])


def _append_stats_to_changelog(changes_path, *, release_version, prev_tag):
    import importlib.util

    root = _run(["git", "rev-parse", "--show-toplevel"])
    stats_script = os.path.join(root, "tools", "github_stats.py")
    spec = importlib.util.spec_from_file_location("github_stats", stats_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    click.echo("  Fetching GitHub stats (set GITHUB_TOKEN to avoid rate limits)...")
    try:
        stats = mod.generate_stats(since_tag=prev_tag)
    except Exception as exc:
        click.echo(f"  WARNING: could not fetch stats: {exc}", err=True)
        return

    with open(changes_path) as f:
        content = f.read()

    anchor = f".. _changes_{release_version}:"
    next_anchor_pat = re.compile(
        r"\.\. _changes_(?!" + re.escape(release_version) + r")"
    )
    section_start = content.find(anchor)
    if section_start == -1:
        click.echo(f"  WARNING: anchor '{anchor}' not found in CHANGES.rst.", err=True)
        return

    match = next_anchor_pat.search(content, section_start + len(anchor))
    insert_at = match.start() if match else len(content)

    block = f"\n{stats}\n\n"
    with open(changes_path, "w") as f:
        f.write(content[:insert_at] + block + content[insert_at:])


@click.command()
@click.argument("version")
@click.option(
    "--next",
    "next_version",
    default=None,
    help="Next dev version (default: auto-increment minor, e.g. 0.3.0.dev0)",
)
@click.option(
    "--remote",
    default="upstream",
    show_default=True,
    help="Git remote to push tag and commits to.",
)
@click.option(
    "--no-stats",
    is_flag=True,
    default=False,
    help="Skip GitHub stats generation in the changelog.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print all steps without executing them.",
)
def release(version, next_version, remote, no_stats, dry_run):
    """Cut a release: bump version, tag, push, then start the next dev cycle.

    Parameters
    ----------
    version : str
        Release version string, e.g. ``0.2.0``.
    next_version : str, optional
        Next development version string (with .dev0 suffix).
        Defaults to auto-incrementing the minor component.
    remote : str
        Git remote to push to.
    no_stats : bool
        Skip GitHub stats (PR / issue / contributor lists) in changelog.
    dry_run : bool
        If True, print commands without running them.
    """
    if next_version is None:
        next_version = _next_dev_version(version)

    tag = f"v{version}"
    today = datetime.date.today().isoformat()
    root = _run(["git", "rev-parse", "--show-toplevel"])
    pyproject = os.path.join(root, "pyproject.toml")
    changes = os.path.join(root, "CHANGES.rst")

    prev_tag = _run(["git", "describe", "--abbrev=0", "--tags"])

    def step(msg, cmd=None):
        click.echo(f"  {'[DRY-RUN] ' if dry_run else ''}{msg}")
        if cmd and not dry_run:
            result = subprocess.run(cmd, cwd=root)
            if result.returncode != 0:
                click.echo(f"ERROR: command failed: {' '.join(cmd)}", err=True)
                sys.exit(1)

    dirty = _run(["git", "status", "--porcelain"])
    if dirty:
        click.echo("ERROR: working tree has uncommitted changes.", err=True)
        sys.exit(1)

    click.echo(f"\nReleasing polyxios {version} (next dev: {next_version})\n")

    step(f"Bump pyproject.toml to {version}")
    if not dry_run:
        _bump_pyproject(pyproject, new_version=version)

    step(f"Update CHANGES.rst: mark {version} with date {today}")
    if not dry_run:
        _bump_changelog(
            changes,
            release_version=version,
            next_version=next_version,
            release_date=today,
        )

    if not no_stats:
        step(f"Append GitHub stats to CHANGES.rst (since {prev_tag})")
        if not dry_run:
            _append_stats_to_changelog(
                changes, release_version=version, prev_tag=prev_tag
            )

    step(
        f"Commit release: 'MNT: release {version}'",
        ["git", "commit", "-am", f"MNT: release {version}"],
    )

    step(f"Tag {tag}", ["git", "tag", tag])

    step(f"Push master + {tag} to {remote}", ["git", "push", remote, "master", tag])

    step(f"Bump pyproject.toml to {next_version}")
    if not dry_run:
        _bump_pyproject(pyproject, new_version=next_version)

    step(f"Prepend upcoming section for {next_version} in CHANGES.rst")
    if not dry_run:
        _prepend_upcoming_section(changes, next_version=next_version)

    step(
        f"Commit dev bump: 'MNT: back to dev, start {next_version}'",
        ["git", "commit", "-am", f"MNT: back to dev, start {next_version}"],
    )

    step(f"Push master to {remote}", ["git", "push", remote, "master"])

    click.echo(f"\nDone! {tag} is live on {remote}. Next cycle: {next_version}.")
