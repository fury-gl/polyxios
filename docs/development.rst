Development Guide
=================

polyxios uses `spin <https://github.com/scientific-python/spin>`_ to manage
the development workflow. All common tasks — building, testing, linting, and
documentation — are available as ``spin`` sub-commands.

Getting started
---------------

**1. Fork and clone the repository**, then run the one-time setup::

    pip install spin
    spin setup

``spin setup`` does three things:

- Adds the ``upstream`` remote (``https://github.com/fury-gl/polyxios.git``)
  if it is not already present.
- Installs the build dependencies (``meson-python``, ``Cython``, ``numpy``,
  ``meson``, ``ninja``).
- On macOS, installs ``libomp`` via Homebrew so the OpenMP hot-paths in
  ``_core.pyx`` compile correctly.

**2. Install polyxios** with Cython extensions compiled::

    spin install       # regular install
    spin install -e    # editable install — source changes are reflected immediately

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
