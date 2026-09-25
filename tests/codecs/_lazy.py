"""Helpers shared by the lazy-read tests of the mapping codecs."""

from __future__ import annotations

import mmap

import numpy as np


def mapped(arr: np.ndarray) -> bool:
    """Whether the array, through however many views, sits on a mapping.

    Parameters
    ----------
    arr
        Array handed back by a lazy read.

    Returns
    -------
    bool
        True when the innermost base of the array is an ``mmap.mmap``.
    """
    base = arr
    while isinstance(base, np.ndarray):
        base = base.base
    if isinstance(base, memoryview):
        base = base.obj
    return isinstance(base, mmap.mmap)
