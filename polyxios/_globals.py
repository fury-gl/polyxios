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
2. **Text travels where the format spells it.** The VTK XML family's
   ``<FieldData>`` holds a text array beside its numeric ones, so a string -
   or a list of them - is written as one there and comes back the same. The
   legacy ``FIELD`` block spells no such thing, and a string handed to it
   falls under the rule below.
3. **Everything else is dropped, and the writer says so.** A dict, a ragged
   list, None: one warning naming the keys, never a silent loss.
4. **A reader hands back arrays.** ``<FieldData>`` holds arrays and nothing
   else, so a scalar written from one comes back as a one-element array. The
   number is intact; the Python type it was handed in is not. Text is the
   exception: a text array names its own strings, so one comes back a ``str``
   and several a list of them.
5. **Shape past two dimensions is not spelled.** A field header carries a
   tuple count and a component count and no third number, so every axis past
   the first is written as components: an ``(n, 3)`` array comes back as it
   went out, an ``(n, 3, 3)`` one comes back ``(n, 9)``. The values are
   intact and in order; only the shape is the file's to lose.

Rule 4 is why a codec's own metadata does not travel this way. A reader that
records the grid it expanded - ``vti_extent``, ``vtr_extents`` - writes that
key again from the grid on the way out, so spelling it a second time as field
data would round-trip two copies of it, one of them the wrong shape. Those
keys are that writer's ``reserved`` set and are skipped here.

Reserved is per writer, not per key. A codec that never spells a key again
reserves nothing: skipping it there would drop it without a word, where the
rule above promises that nothing goes quietly. The legacy ``.vtk`` writer
spells an ``UNSTRUCTURED_GRID`` whatever it was handed, so it carries every
grid key another reader recorded rather than reserving any of them.

The other side of that is what a mesh gathers on a long conversion: a ``.vti``
read and written as ``.vtr`` carries ``vti_extent`` through the field data,
because the ``.vtr`` writer reserves only its own. The keys are the grid each
reader rebuilt, so a mesh picks up a set per format it passes through and
nothing is ever wrong: a reader's own grid is written over anything the field
data names, and a writer rebuilds the grid from the geometry it was handed.
Dropping the other codec's keys instead would be a silent loss, which is the
one thing this module does not do.
"""

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from polyxios._types import PolyData

# Kinds a data array holds: bool, signed, unsigned, float. Complex is a
# number numpy names and no VTK array does, so it falls out with the strings.
_NUMERIC_KINDS: frozenset[str] = frozenset("biuf")


def as_text(value: Any) -> tuple[str, ...] | None:
    """Turn one ``global_attrs`` value into the strings a text array holds.

    Parameters
    ----------
    value
        Whatever the caller put in ``global_attrs``.

    Returns
    -------
    tuple of str or None
        One entry per string, or None when the value is not text at all. A
        bare string is one entry; a list or tuple of them is one apiece, in
        order, which is what a text array of several tuples spells.

    Notes
    -----
    Bytes are not text here. They are a sequence of numbers as readily as a
    run of characters, and a format that spells one encoding cannot say which
    the caller meant; :func:`spellable` refuses them for the same reason.

    Examples
    --------
    >>> as_text("run 3")
    ('run 3',)
    >>> as_text(["a", "b"])
    ('a', 'b')
    >>> as_text(3) is None
    True
    """
    if isinstance(value, str):
        return (value,)
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, str) for item in value)
    ):
        return tuple(value)
    return None


def text_for_write(
    poly: "PolyData",
    *,
    reserved: frozenset[str] | set[str] | tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    """The ``global_attrs`` entries a writer can spell as text arrays.

    Parameters
    ----------
    poly
        The mesh being written.
    reserved
        Keys the codec writes itself from the mesh, skipped here as they are
        by :func:`globals_for_write`.

    Returns
    -------
    dict of str to tuple of str
        One entry per text key, in the order the mesh holds them.

    Notes
    -----
    The other half of :func:`globals_for_write`, for the formats whose
    metadata block holds a text array as well as a numeric one - the VTK XML
    family's ``<FieldData>``. Nothing is warned about here: a key this
    refuses is one that function reports, and a key it takes is one that
    function was told to leave alone, so the two together still name
    everything that does not travel.
    """
    kept: dict[str, tuple[str, ...]] = {}
    for name, value in (poly.global_attrs or {}).items():
        if name in reserved or not isinstance(name, str) or not name:
            continue
        text = as_text(value)
        if text is not None:
            kept[name] = text
    return kept


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
        it is of a kind no numeric array holds. Shape is not checked past the
        first axis: a field header spells a tuple count and a component count
        and nothing else, so an array of three dimensions is written as the
        two-dimensional one its components flatten to.

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
    stacklevel: int = 3,
    text: bool = False,
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
    stacklevel
        How far above this frame the warning should point. The default
        answers a codec's ``write``, which is where this is called from; a
        writer that reaches it through a helper of its own adds a frame to it.
    text
        Whether the caller writes text values itself, through
        :func:`text_for_write`. They are then left out here rather than
        reported as dropped, since they are not.

    Returns
    -------
    dict of str to numpy.ndarray
        One entry per spellable key, in the order the mesh holds them.

    Warns
    -----
    UserWarning
        Once for the keys whose value no numeric array holds, and once for
        the keys that are not names - the empty string, or something that is
        not text at all. Two warnings rather than one, because they say
        different things: the first is about what the mesh holds, the second
        about what it holds it under. Every format here writes the name as
        the array's own handle, and one that is blank leaves the array
        unfindable where it does not make the file unreadable outright.
    """
    spelled: dict[str, np.ndarray] = {}
    unspellable: list[str] = []
    unnamed: list[str] = []
    for name, value in (poly.global_attrs or {}).items():
        if name in reserved:
            continue
        if not isinstance(name, str) or not name:
            unnamed.append(name)
            continue
        if text and as_text(value) is not None:
            continue
        arr = spellable(value)
        if arr is None:
            unspellable.append(name)
        else:
            spelled[name] = arr
    if unspellable:
        warnings.warn(
            f"{fmt} write: global_attrs {sorted(unspellable)} hold values no"
            " numeric array can spell; dropped.",
            UserWarning,
            stacklevel=stacklevel,
        )
    if unnamed:
        warnings.warn(
            f"{fmt} write: global_attrs {sorted(unnamed, key=repr)} have no name"
            " a data array can carry; dropped.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return spelled
