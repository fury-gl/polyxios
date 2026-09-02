from typing import Any

from polyxios._io import Source, source_name
from polyxios._types import PolyData
from polyxios.exceptions import UnsupportedFormatError

EXTENSION: str = ".pvtu"


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Raise UnsupportedFormatError - .pvtu is a parallel/multi-block meta-file.

    Parameters
    ----------
    path
        Path to the .pvtu file.
    lazy
        Ignored; error is raised immediately.

    Raises
    ------
    UnsupportedFormatError
        Always. ``polyxios.helper.read_multiblock`` reads one of these.
    """
    raise UnsupportedFormatError(
        f"'{source_name(path)}' is a parallel/multi-block meta-file (.pvtu): it "
        "contains no geometry, only references to sub-files. "
        "polyxios.helper.read_multiblock() reads it and merges the pieces, "
        "and helper.read_blocks() hands them back one mesh per sub-file. "
        "See examples/read_parallel_vtk.py for a step-by-step tutorial."
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Raise NotImplementedError - writing .pvtu files is not supported.

    Parameters
    ----------
    poly
        Ignored.
    path
        Ignored.
    """
    raise NotImplementedError(
        "Writing .pvtu parallel/multi-block files is not supported."
    )
