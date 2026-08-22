"""What a face-list format can say about an element's shape, and what it cannot.

A PLY face, an OBJ ``f`` record and their kin are a flat ring of vertex
indices and nothing else: the file names no element type, so a reader has only
the ring's width to go on and calls it a triangle at three, a quad at four and
a polygon otherwise. Anything else a writer puts in one - a solid, a
higher-order element - keeps its vertices and loses the type it was. The rule
and the report live here so the codecs that share the convention cannot drift
apart on it.
"""

import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV

__all__ = ["report_flattened_faces"]

# An element type is a uint8, so a pair of them packs into one integer and the
# pairs a mesh holds are counted in one numpy pass.
_TYPE_CODES: int = 256


def _reads_back_as(widths: np.ndarray) -> np.ndarray:
    """Return the type each ring width names when the file is read again.

    Parameters
    ----------
    widths
        How many vertices each record carries.

    Returns
    -------
    numpy.ndarray
        One element type code per record.

    Notes
    -----
    This is the readers' own rule. Keeping it here rather than as a list of
    types beside them is what stops the report from claiming a round trip the
    readers do not make.
    """
    return np.where(
        widths == 3,
        ELEMENT_TYPES["triangle"],
        np.where(widths == 4, ELEMENT_TYPES["quad"], ELEMENT_TYPES["polygon"]),
    )


def report_flattened_faces(
    *,
    offsets: np.ndarray,
    element_types: np.ndarray,
    face_indices: np.ndarray | list[int],
    fmt: str,
) -> None:
    """Name the elements a face record cannot hold the shape of.

    Parameters
    ----------
    offsets
        The mesh's element offsets.
    element_types
        One type code per element.
    face_indices
        Which of them are written as face records.
    fmt
        The extension, spelling the warning in the codec's own voice.

    Notes
    -----
    A face is a flat ring of vertices and the format spells no other shape, so
    a solid or a higher-order element is written as a ring of its nodes in
    mesh order. Nothing in the file says what it was: a reader names a record
    by how many vertices it holds, so a ``tetra`` comes back a ``quad`` and a
    ``quadratic_triangle`` a ``polygon``. The elements are still written -
    their vertices are the mesh, and refusing them would cost a caller the
    file they asked for - but the type they lose is worth saying.

    A ``polygon`` of three or four vertices is named too: it is a ring the
    format holds exactly, and it still comes back under another type.

    Counting is left to numpy rather than a Python loop: the pairs a mesh
    spreads over are a handful however many elements it holds, and a loop
    would run once per element to say the same few things.
    """
    picked = np.asarray(face_indices, dtype=np.int64)
    if not picked.size:
        return
    bounds = np.asarray(offsets)
    widths = bounds[picked + 1] - bounds[picked]
    reads_as = _reads_back_as(widths)
    held = np.asarray(element_types)[picked]
    changed = np.flatnonzero(reads_as != held)
    if not changed.size:
        return

    pairs = held[changed].astype(np.int64) * _TYPE_CODES + reads_as[changed]
    kinds, counts = np.unique(pairs, return_counts=True)
    named = ", ".join(
        f"{ELEMENT_TYPES_INV.get(was, was)} ({n}) ->"
        f" {ELEMENT_TYPES_INV.get(becomes, becomes)}"
        for was, becomes, n in (
            (kind // _TYPE_CODES, kind % _TYPE_CODES, count)
            for kind, count in zip(kinds.tolist(), counts.tolist())
        )
    )
    warnings.warn(
        f"{fmt}: a face record is a flat ring of vertices and the format"
        " spells no other shape, so these elements keep their vertices and"
        f" lose the type they were: {named}.",
        stacklevel=4,
    )
