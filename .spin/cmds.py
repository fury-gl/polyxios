"""Custom spin commands for polyxios development."""

import os
import platform
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

    click.echo("\nInstalling build dependencies...")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "meson-python>=0.15",
            "Cython>=3.0",
            "numpy>=1.24",
            "meson",
            "ninja",
        ],
        capture=False,
    )

    if platform.system() == "Darwin":
        click.echo("\nDetected macOS — installing libomp for OpenMP support...")
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
        "*.pyc,.git,./docs/_build,*.egg-info,./build",
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
