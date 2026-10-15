Contributing
============

.. meta::
   :description: How to contribute to polyxios: set up a development checkout, run the test suite, and open a pull request.


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
     - Whitespace / formatting - no logic change
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

Bot-authored commits
--------------------

The convention applies to every commit on a pull request, whoever wrote it:
the ``commit-messages`` job checks each one against ``origin/master``.
Dependabot (``.github/dependabot.yml``) and the pre-commit update workflow
are configured to write ``MNT:`` subjects and need no attention.

Copilot Autofix is not configurable. A branch it opens for a code scanning
alert carries a subject such as ``Potential fix for code scanning alert
no. 3: ...`` with no prefix and more than 78 characters, so the check fails
until the commit is reworded. Before merging::

    git fetch upstream alert-autofix-N
    git checkout alert-autofix-N
    git commit --amend
    git push --force-with-lease upstream alert-autofix-N

Write the subject as if the change were yours: ``CI:`` for a workflow
hardening, ``BF:`` for a fix to library code, ``MNT:`` for a dependency
bump. Put the alert link in the PR description, not in the subject, and
drop the bot's ``Co-authored-by`` trailer so the message stands on its own.
