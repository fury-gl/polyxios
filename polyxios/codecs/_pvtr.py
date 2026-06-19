from pathlib import Path
from typing import Any

from polyxios._types import PolyData
from polyxios.exceptions import UnsupportedFormatError

EXTENSION: str = ".pvtr"


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Raise UnsupportedFormatError — .pvtr is a parallel/multi-block meta-file.

    Parameters
    ----------
    path
        Path to the .pvtr file.
    lazy
        Ignored; error is raised immediately.

    Raises
    ------
    UnsupportedFormatError
        Always. See examples/read_parallel_vtk.py for a loading tutorial.
    """
    raise UnsupportedFormatError(
        f"'{Path(path).name}' is a parallel/multi-block meta-file (.pvtr): it "
        "contains no geometry, only references to sub-files. "
        "See examples/read_parallel_vtk.py for a step-by-step loading tutorial."
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Raise NotImplementedError — writing .pvtr files is not supported.

    Parameters
    ----------
    poly
        Ignored.
    path
        Ignored.
    """
    raise NotImplementedError(
        "Writing .pvtr parallel/multi-block files is not supported."
    )
