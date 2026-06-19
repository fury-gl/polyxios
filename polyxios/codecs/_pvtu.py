from pathlib import Path
from typing import Any

from polyxios._types import PolyData
from polyxios.exceptions import UnsupportedFormatError

EXTENSION: str = ".pvtu"


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Raise UnsupportedFormatError — .pvtu is a parallel/multi-block meta-file.

    Parameters
    ----------
    path
        Path to the .pvtu file.
    lazy
        Ignored; error is raised immediately.

    Raises
    ------
    UnsupportedFormatError
        Always. See examples/read_parallel_vtk.py for a loading tutorial.
    """
    raise UnsupportedFormatError(
        f"'{Path(path).name}' is a parallel/multi-block meta-file (.pvtu): it "
        "contains no geometry, only references to sub-files. "
        "See examples/read_parallel_vtk.py for a step-by-step loading tutorial."
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Raise NotImplementedError — writing .pvtu files is not supported.

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
