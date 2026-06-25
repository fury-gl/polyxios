Development Guide
=================

polyxios uses `spin <https://github.com/scientific-python/spin>`_ to manage
the development workflow. All common tasks - building, testing, linting, and
documentation - are available as ``spin`` sub-commands.

Getting started
---------------

**1. Fork and clone the repository**, then run the one-time setup::

    pip install spin
    spin setup

``spin setup`` does three things:

- Adds the ``upstream`` remote (``https://github.com/fury-gl/polyxios.git``)
  if it is not already present.
- Installs the dev dependencies (``meson-python``, ``Cython``, ``numpy``,
  ``meson``, ``ninja``, ``mypy``, ``pre-commit``).
- On macOS, installs ``libomp`` via Homebrew so the OpenMP hot-paths in
  ``_core.pyx`` compile correctly.

**2. Install polyxios** with Cython extensions compiled::

    spin install       # regular install
    spin install -e    # editable install - source changes are reflected immediately

Building
--------

To invoke Meson/ninja directly (useful when iterating on ``.pyx`` files)::

    spin build

Testing
-------

Run the full test suite::

    spin test

Run only tests that match a name pattern (passed to ``pytest -k``)::

    spin test -k vtk
    spin test -k "roundtrip and binary"

Pass any extra argument directly to pytest::

    spin test -- --tb=short -x

Linting
-------

Check code style, imports, and spelling::

    spin lint

Auto-fix issues where possible::

    spin lint --fix

This runs three tools in sequence:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Tool
     - What it checks
   * - ``ruff check``
     - PEP 8, unused imports, common bug patterns
   * - ``ruff format``
     - Code formatting (replaces Black)
   * - ``codespell``
     - Spelling mistakes in source and docs

Documentation
-------------

Build the HTML docs::

    spin docs

Remove the previous build first (useful after restructuring)::

    spin docs --clean

Build and immediately open the result in the browser::

    spin docs --open

The built docs land in ``docs/_build/html/``.

Cleaning up
-----------

Remove build artifacts, ``__pycache__``, ``.pytest_cache``, and ``*.egg-info``::

    spin clean

Commit message convention
--------------------------

See :doc:`contributing` for the full commit prefix table and rules enforced
by the pre-commit hook.

Pre-commit hooks
----------------

Install the hooks once (they run automatically on every commit)::

    pip install pre-commit
    pre-commit install
    pre-commit install --hook-type commit-msg

To run all hooks manually against the whole codebase::

    pre-commit run --all-files

Making a release
----------------

Releases are published automatically when a version tag is pushed.
The GitHub Actions ``release`` workflow builds platform wheels, creates
a GitHub Release, and uploads everything to PyPI via Trusted Publishing.

**One-time setup (do once per repository):**

1. On `PyPI <https://pypi.org>`_, configure Trusted Publishing for
   ``polyxios``:

   - Publisher: GitHub Actions
   - Repository: ``fury-gl/polyxios``
   - Workflow: ``release.yml``
   - Environment: ``pypi``

2. In the GitHub repository settings, create a deployment environment
   named ``pypi`` (optional but recommended for approval gates).

**Steps to cut a release:**

1. Make sure all tests pass on ``master``::

       spin test

2. Update :doc:`changelog` — rename the ``upcoming`` heading to the
   release version and date, e.g.::

       0.1.0 (2026-07-01)
       -------------------

3. Remove the ``.dev0`` suffix from ``version`` in ``pyproject.toml``::

       version = "0.1.0"

4. Commit the version bump and changelog::

       git add pyproject.toml CHANGES.rst
       git commit -m "DOC: release 0.1.0"

5. Tag and push::

       git tag v0.1.0
       git push origin master v0.1.0

   GitHub Actions picks up the tag, builds wheels for Linux / macOS /
   Windows, creates a GitHub Release with auto-generated notes, and
   publishes to PyPI.

6. After the release, restore the dev version for the next cycle::

       # pyproject.toml
       version = "0.2.0.dev0"

       # CHANGES.rst — add a new upcoming section at the top
       0.2.0 (upcoming)
       -----------------

   Commit::

       git commit -am "MNT: back to dev, start 0.2.0"
       git push origin master
