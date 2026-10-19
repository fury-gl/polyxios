# Contributing to polyxios

The full guide lives at <https://polyxios.org/dev/contributing.html>.
The short version:

1. **Issue first, assigned to you.** Before writing code, find or open an
   issue and comment on it to ask to work on it; a maintainer assigns it to
   you. Ideas can start as a discussion, which becomes an issue once agreed.
   Every pull request must reference an open issue assigned to its author
   (`Closes #123`), or it is closed automatically. Regular contributors can ask
   to join the polyxios community in the "Community membership" discussion
   category to skip this check; membership grants no write access.
2. **Commit messages** start with a prefix (`BF:`, `NF:`, `RF:`, `DOC:`,
   `TEST:`, `MNT:`, `CI:`, `OPT:`, `STYLE:`, `BW:`, `WIP:`) and the subject
   is 15 to 78 characters with no trailing period.
3. **CI gates.** `pre-commit run --all-files`, the test suite, and
   `spin docs` (when docs or docstrings change) must all pass.
4. **AI-assisted work** is allowed, but you are responsible for every line:
   you must understand it, have run it, and answer review questions
   yourself. Unreviewed generated pull requests are closed.
