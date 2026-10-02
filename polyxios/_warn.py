"""Warnings attributed to the caller's line, whatever depth they come from."""

from __future__ import annotations

import os
import warnings

# The separator keeps a caller's own package that merely starts with the
# same name, ``polyxios_scripts/`` say, from being skipped as ours.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def warn_caller(message: str) -> None:
    """Warn at the first frame outside polyxios.

    Parameters
    ----------
    message
        The warning's text.
    """
    warnings.warn(message, skip_file_prefixes=(_PACKAGE_DIR,))
