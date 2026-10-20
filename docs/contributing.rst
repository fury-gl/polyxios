Contributing
============

.. meta::
   :description: How to contribute to polyxios: set up a development checkout, run the test suite, and open a pull request.


Issue first
-----------

Every change starts as an `issue <https://github.com/fury-gl/polyxios/issues>`_
assigned to the person who will write it, so the problem and the proposed
fix are agreed on before code is written:

1. Find or open an issue describing the change. Ideas can start as a
   `discussion <https://github.com/fury-gl/polyxios/discussions>`_; once
   agreed, a maintainer opens an issue from it.
2. Comment on the issue to ask to work on it. A maintainer assigns it to you.
3. Reference it in the pull request description (``Closes #123``).

The ``PR Policy`` workflow enforces this for every author who is neither a
member of the ``fury-gl`` organisation nor has write access to the
repository, outside collaborators with read or triage access included. A
pull request that references no open issue assigned to
its author is labelled ``needs-issue``, closed, and gets a failing
``pr-policy`` status. It reopens on its own once it references such an issue,
whether the description is edited or the author is assigned, unless someone
else closed it since. GitHub starts no CI for a reopen made by a workflow, so
if you pushed while the pull request was closed, push again to run the checks.
Unassigning the author closes the pull request again. A maintainer can waive
the check for one pull request by reopening it after the policy closed it.
First-time contributors also get a welcome comment that summarises these
rules.

The rule holds for returning contributors too. Regular contributors can
instead join the polyxios community: open a discussion in the
`Community membership <https://github.com/fury-gl/polyxios/discussions/categories/community-membership>`_
category with links to past contributions. Once a maintainer adds the login
to ``.github/community-members.txt``, that author's pull requests skip the
issue-first check. Membership grants no write access to the repository.

AI-assisted contributions are welcome under the same terms as any other:
the author is responsible for every line, has run the code, and answers
review questions personally. Pull requests that read as unreviewed
generated output are closed.


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
