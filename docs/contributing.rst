Contributing
============

Commit message convention
--------------------------

Every commit subject line must start with one of these prefixes followed by
a space:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Prefix
     - Meaning
   * - ``BF:``
     - Bug fix
   * - ``RF:``
     - Refactoring
   * - ``NF:``
     - New feature
   * - ``BW:``
     - Addresses backward-compatibility
   * - ``OPT:``
     - Optimization
   * - ``CI:``
     - Continuous integration
   * - ``MNT:``
     - Maintenance (release prep, dependency bumps, etc.)
   * - ``DOC:``
     - Documentation
   * - ``TEST:``
     - Adding or changing tests
   * - ``STYLE:``
     - Whitespace / formatting — no logic change
   * - ``WIP:``
     - Work in progress, not ready to merge

Additional rules:

- Subject line: minimum 15 characters, maximum 78 characters.
- No trailing period on the subject line.
- Second line, if present, must be blank.
- To reference an issue add ``Issue #XXXX`` to the PR description.
- To close an issue add ``Closes #XXXX`` to the PR description.

Examples::

    NF: add VTK v5.1 reader support
    BF: fix int32 overflow in PLY binary writer
    TEST: add roundtrip tests for OBJ multi-group tags
    DOC: document lazy loading behaviour for binary VTK

The commit message hook runs automatically via ``pre-commit``. Install it
once with::

    pip install pre-commit
    pre-commit install --hook-type commit-msg
