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

Releases are published automatically when a version tag is pushed to
``upstream``.  The GitHub Actions ``release`` workflow builds platform wheels,
creates a GitHub Release, and publishes to PyPI via Trusted Publishing.

**One-time setup (do once per repository):**

1. On `PyPI <https://pypi.org>`_, configure Trusted Publishing for
   ``polyxios``:

   - Publisher: GitHub Actions
   - Repository: ``fury-gl/polyxios``
   - Workflow: ``release.yml``
   - Environment: ``pypi``

2. In the GitHub repository settings, create a deployment environment
   named ``pypi`` (optional but recommended for approval gates).

**Cutting a release:**

1. Make sure all tests pass on ``master``::

       spin test

2. Update :doc:`changelog` — fill in the ``upcoming`` section with the
   changes for this release (the release date is stamped automatically).

3. Run the release command::

       spin release 0.2.0

   This single command:

   - Sets ``version = "0.2.0"`` in ``pyproject.toml``
   - Stamps today's date in ``CHANGES.rst`` (replaces ``upcoming``)
   - Commits ``MNT: release 0.2.0``
   - Creates and pushes tag ``v0.2.0`` to ``upstream``
   - Bumps ``pyproject.toml`` to ``0.3.0.dev0``
   - Prepends a new ``0.3.0 (upcoming)`` section to ``CHANGES.rst``
   - Commits ``MNT: back to dev, start 0.3.0.dev0`` and pushes

   The tag push triggers CI, which builds wheels for Linux / macOS /
   Windows and publishes to PyPI.

   The GitHub stats step queries the public API, which is rate-limited to
   60 requests/hour unauthenticated.  Set the ``GITHUB_TOKEN`` environment
   variable to raise the limit::

       export GITHUB_TOKEN=ghp_...
       spin release 0.2.0

**Options:**

.. code-block:: text

    spin release 0.2.0                        # default: next dev = 0.3.0.dev0
    spin release 0.2.0 --next 0.2.1.dev0      # override next dev version
    spin release 0.2.0 --remote origin        # push to a different remote
    spin release 0.2.0 --no-stats             # skip GitHub stats in changelog
    spin release 0.2.0 --dry-run              # print all steps without executing

**Test the release workflow without publishing to PyPI:**

Push a tag whose name contains ``test`` — the CI will build wheels and
create a GitHub Release but skip the PyPI upload::

    git tag v0.2.0-test
    git push upstream v0.2.0-test

Delete the test tag afterwards::

    git push upstream :v0.2.0-test
    git tag -d v0.2.0-test
