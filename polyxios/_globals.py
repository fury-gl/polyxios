"""One rule for what a mesh's own metadata may hold and what survives a write.

``PolyData.global_attrs`` is the whole-mesh slot: the numbers that belong to
the mesh rather than to any one vertex or element - a time value, a material
constant, a solver's convergence tolerance, the ``was_2d`` flag the readers
set. Its values are typed ``Any``, because the formats disagree about what
belongs there: a Kratos ``ModelPartData`` entry is a scalar or a small vector,
a VTK ``<FieldData>`` array is any numeric array, and an OBJ ``mtllib`` line
is a file name.

``Any`` is wide, and a writer has to say what it does with the parts of it a
file has no room for. This module is that answer for every format whose
metadata slot holds arrays - the VTK family's ``<FieldData>`` and the legacy
``FIELD`` block:

1. **Numbers travel.** A value numpy reads as a numeric array - a Python int
   or float, a bool, a list of them, an ``ndarray`` of any shape - is written
   as one array of that name.
2. **Everything else is dropped, and the writer says so.** A string, a dict, a
   ragged list, None: one warning naming the keys, never a silent loss.
3. **A reader hands back arrays.** ``<FieldData>`` holds arrays and nothing
   else, so a scalar written from one comes back as a one-element array. The
   number is intact; the Python type it was handed in is not.

Rule 3 is why a codec's own metadata does not travel this way. A reader that
records the grid it expanded - ``vti_extent``, ``vtk_dimensions`` - writes
that key again from the grid on the way out, so spelling it a second time as
field data would round-trip two copies of it, one of them the wrong shape.
Those keys are the writer's ``reserved`` set and are skipped here.
"""

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from polyxios._types import PolyData

# Kinds a data array holds: bool, signed, unsigned, float. Complex is a
# number numpy names and no VTK array does, so it falls out with the strings.
_NUMERIC_KINDS: frozenset[str] = frozenset("biuf")


def spellable(value: Any) -> np.ndarray | None:
    """Turn one ``global_attrs`` value into the array a file can hold.

    Parameters
    ----------
    value
        Whatever the caller put in ``global_attrs``.

    Returns
    -------
    numpy.ndarray or None
        The value as a numeric array of at least one dimension, or None when
        it is of a kind no numeric array holds.

    Examples
    --------
    >>> spellable(42).tolist()
    [42]
    >>> spellable("run 3") is None
    True
    >>> spellable(np.zeros((2, 0))) is None
    True
    """
    if isinstance(value, (str, bytes, bytearray)):
        # A string is a sequence numpy takes without complaint, as a U dtype
        # that holds no numbers; asking first is what keeps it from arriving
        # at the kind check as a zero-dimensional array of text.
        return None
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        # A ragged list, a dict, an object with no array protocol: numpy
        # refuses these rather than inventing a shape for them.
        return None
    if arr.dtype.kind not in _NUMERIC_KINDS:
        return None
    arr = np.atleast_1d(arr)
    if 0 in arr.shape[1:]:
        # Every axis past the first is a component, and an array of no
        # components has no tuple count either - the two numbers a field
        # header carries are its shape, and neither format spells this one.
        return None
    return arr


def globals_for_write(
    poly: "PolyData",
    *,
    reserved: frozenset[str] | set[str] | tuple[str, ...] = (),
    fmt: str,
) -> dict[str, np.ndarray]:
    """The ``global_attrs`` entries a writer can spell as numeric arrays.

    Parameters
    ----------
    poly
        The mesh being written.
    reserved
        Keys the codec writes itself from the mesh rather than from this
        mapping - the grid a structured reader recorded, and anything else
        it will spell again on its own. Skipped without a warning.
    fmt
        The format's extension, for the warning naming what was dropped.

    Returns
    -------
    dict of str to numpy.ndarray
        One entry per spellable key, in the order the mesh holds them.

    Warns
    -----
    UserWarning
        Once, naming every key whose value no numeric array holds. A key that
        is not a name - the empty string, or something that is not text at
        all - is dropped with them: every format here writes the name as the
        array's own handle, and one that is blank leaves the array unfindable
        where it does not make the file unreadable outright.
    """
    spelled: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for name, value in (poly.global_attrs or {}).items():
        if name in reserved:
            continue
        arr = spellable(value) if isinstance(name, str) and name else None
        if arr is None:
            dropped.append(name)
        else:
            spelled[name] = arr
    if dropped:
        warnings.warn(
            f"{fmt} write: global_attrs {sorted(dropped, key=repr)} hold values no"
            " numeric array can spell; dropped.",
            UserWarning,
            stacklevel=3,
        )
    return spelled
