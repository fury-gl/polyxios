#!/usr/bin/env python3
"""Commit message validator for polyxios.

Adapted from the equivalent hook in DIPY, which is itself borrowed from ITK.

Every commit subject line must start with one of these prefixes:

  BF:     bug fix
  RF:     refactoring
  NF:     new feature
  BW:     backward-compatibility concern
  OPT:    optimization
  CI:     continuous integration
  MNT:    maintenance (release prep, dep bumps, etc.)
  DOC:    documentation
  TEST:   adding or changing tests
  STYLE:  whitespace, formatting — no logic change
  WIP:    work in progress, not ready to merge

Subject line rules:
  - Minimum 15 characters
  - Maximum 78 characters (Merge/Revert exempt)
  - No leading or trailing whitespace
  - No trailing period
  - Second line, if present, must be blank

Examples:
  NF: add VTK v5.1 reader support
  BF: fix int32 overflow in PLY binary writer
  TEST: add roundtrip tests for OBJ multi-group tags
"""

import os
from pathlib import Path
import re
import sys

DEFAULT_LINE_LENGTH: int = 78
MIN_SUBJ_LINE_LENGTH: int = 15

PREFIXES = (
    "Merge",
    "Revert",
    "BF:",
    "RF:",
    "NF:",
    "BW:",
    "OPT:",
    "CI:",
    "MNT:",
    "DOC:",
    "TEST:",
    "STYLE:",
    "WIP:",
)

PREFIX_HELP = """\
Start polyxios commit messages with a standard prefix (and a space):
  BF:     - bug fix
  RF:     - refactoring
  NF:     - new feature
  BW:     - addresses backward-compatibility
  OPT:    - optimization
  CI:     - continuous integration
  MNT:    - maintenance tasks (release prep, dep bumps, etc.)
  DOC:    - documentation
  TEST:   - adding or changing tests
  STYLE:  - whitespace/formatting, no logic change
  WIP:    - work in progress, not ready to merge

To reference a GitHub issue: add "Issue #XXXX" to the PR description.
To close an issue: add "Closes #XXXX" to the PR description."""


def _die(message: str, commit_msg_path: Path) -> None:
    print("commit-msg hook failure", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    print(message, file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    print(
        f'\nTo resume editing:\n  git commit -e -F "{commit_msg_path}"',
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    git_dir = Path(os.environ.get("GIT_DIR", ".git")).resolve()
    commit_msg_path = git_dir / "COMMIT_MSG"

    if len(sys.argv) < 2:
        _die(f"Usage: {sys.argv[0]} <commit_message_file>", commit_msg_path)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        _die(f"Missing input file: {sys.argv[1]}", commit_msg_path)

    raw_lines = input_file.read_text().splitlines(keepends=True)

    # Strip comments and leading blank lines
    lines: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("#") or (not lines and not stripped):
            continue
        lines.append(f"{stripped}\n")

    commit_msg_path.write_text("".join(lines))

    if not lines:
        _die("Commit message is empty after stripping comments.", commit_msg_path)

    subject = lines[0]

    if len(subject) < MIN_SUBJ_LINE_LENGTH:
        _die(
            f"Subject line must be at least {MIN_SUBJ_LINE_LENGTH} characters:\n{subject}",
            commit_msg_path,
        )

    if (
        len(subject) > DEFAULT_LINE_LENGTH
        and not subject.startswith("Merge ")
        and not subject.startswith("Revert ")
    ):
        _die(
            f"Subject line may be at most {DEFAULT_LINE_LENGTH} characters:\n"
            + "-" * DEFAULT_LINE_LENGTH
            + f"\n{subject}"
            + "-" * DEFAULT_LINE_LENGTH,
            commit_msg_path,
        )

    if re.match(r"^[ \t]|[ \t]$", subject):
        _die(
            f"Subject line may not have leading or trailing space:\n[{subject}]",
            commit_msg_path,
        )

    if re.search(r"\.$", subject):
        _die(f"Subject line may not end with a period:\n[{subject}]", commit_msg_path)

    pattern = r"^(" + "|".join(re.escape(p) for p in PREFIXES) + r")\s"
    if not re.match(pattern, subject):
        _die(PREFIX_HELP, commit_msg_path)

    if len(lines) > 1 and lines[1].strip():
        _die(
            f"Second line must be blank, got:\n{lines[1]!r}",
            commit_msg_path,
        )


if __name__ == "__main__":
    main()
