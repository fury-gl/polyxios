"""Time one read in the current process and print it as one JSON line.

Each ``read_*.py`` script imports only its own library and calls
:func:`measure` once, so the peak memory it reports belongs to that library
alone and no earlier read has warmed a cache for it.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import resource
import sys
import time

import numpy as np


def peak_rss_mb() -> float:
    """Return the peak resident set size of this process so far, in MiB.

    Returns
    -------
    float
        ``ru_maxrss``, which Linux reports in KiB and macOS in bytes.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def measure(*, load: Callable[[str], tuple[object, int]], path: str) -> None:
    """Read ``path`` with ``load`` and print time, memory and counts as JSON.

    The clock covers the read and one pass over the x coordinates, which is
    what forces a lazy reader to fetch them; the import of the library is
    done before and reported apart as ``import_rss_mb``.

    Parameters
    ----------
    load
        Reads a path and returns its points (anything ``numpy.asarray``
        takes, shape ``(n, 3)``) and its cell count.
    path
        The file to read.
    """
    import_rss = peak_rss_mb()
    start = time.perf_counter()
    points, n_cells = load(path)
    points = np.asarray(points)
    checksum = float(points[:, 0].sum())
    seconds = time.perf_counter() - start
    print(
        json.dumps(
            {
                "seconds": seconds,
                "peak_rss_mb": peak_rss_mb(),
                "import_rss_mb": import_rss,
                "n_points": int(points.shape[0]),
                "n_cells": int(n_cells),
                "checksum": checksum,
            }
        )
    )
