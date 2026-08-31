"""One rule for the numbers a file gave its nodes and elements.

A ``PolyData`` indexes its vertices and elements densely from zero, always.
Most formats agree, but a hand-edited Abaqus deck, a Nastran bulk data file,
a Gmsh ``.msh`` and a FLAC3D grid all let their author number entities freely:
the ids are neither dense nor zero-based, and they may run to any 64-bit value
in any order. Reading such a file has to renumber - nothing downstream can
index ``vertices[7000001]`` - and renumbering on its own is lossy in one
direction. The ids are how the author's other files talk about this mesh: a
load case naming ``GRID 7000001``, a report keyed on element ``4001``. A round
trip that renumbers 1..n leaves those references pointing at the wrong thing.

The reader therefore records what the file said in
``vertex_attrs["original_ids"]`` / ``element_attrs["original_ids"]`` and the
writer puts it back. The key is deliberately not format-prefixed: an id says
something about the entity, not about the file it came from, so a mesh read
from a ``.bdf`` and written as ``.msh`` keeps its numbering.

Three rules, and every id-carrying codec follows them:

1. **Read renumbers.** Indices are dense and zero-based whatever the file
   spelled. Connectivity, tags and attributes all speak indices.
2. **Read remembers, but only what the index does not already say.** A file
   numbering its nodes ``1..n`` in order is recorded as nothing at all: the
   writer's own renumbering reproduces it exactly, so storing it would be a
   column of redundancy on every mesh ever read. The key's absence means "no
   numbering worth keeping", not "numbered from one".
3. **Write asks.** :func:`ids_for_write` returns the ids to spell, one per
   entity. Stored ids win when they can still be written; otherwise the
   writer renumbers from one as it always did.

"Can still be written" is checked at the point of writing rather than trusted,
because a transform moves a mesh out from under its ids without either one
being wrong. ``merge`` concatenates two meshes that each numbered from one, so
the ids collide; ``triangulate`` splits a quad into two triangles that both
carry the quad's id. Either would spell a file with two entities answering to
one number, which no solver loads. So the ids have to be positive, unique, and
one per entity to survive, and a mesh that lost that gets dense numbering and
a warning rather than a broken file.

A format that numbers densely by construction - Medit, OFF, STL, the VTK
family - has no id to record and ignores the key on write. It travels all the
same, so a mesh read from a ``.bdf`` and welded, then written back, keeps the
ids of the vertices that survived. Some of those carriers have no integer
column to travel in: a PLY vertex property and a legacy VTK data array are
float, so the ids come back whole float64s rather than int64s. That is
polyxios's own doing rather than anything the mesh did, so a whole float
column is read as the numbering it is - see :func:`_as_ids` for the bound
that makes it safe.
"""

from typing import TYPE_CHECKING, Final
import warnings

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    from polyxios._types import PolyData

__all__ = [
    "IDS_KEY",
    "ids_for_write",
    "original_ids",
    "record_ids",
    "unwritable",
]

#: Where a reader records the numbers a file gave its nodes or elements.
IDS_KEY: Final[str] = "original_ids"

# What an id is written as. Wide enough for every id-carrying format: a Gmsh
# node tag is a 64-bit integer by spec, and a hand-numbered Abaqus deck reaches
# eight digits routinely.
IDS_DTYPE: Final[np.dtype] = np.dtype(np.int64)

# The magnitude past which float64 stops spelling every whole number, so the
# bound under which a float column is an id column in another dtype. It is
# polyxios that puts the ids there: a PLY vertex property and a legacy VTK
# data array have no integer form, so a mesh carried through one of those
# comes back holding whole float64s where it held int64s.
_EXACT_IN_FLOAT: Final[int] = 2**53


def _as_ids(ids: np.ndarray) -> np.ndarray | None:
    """Return ``ids`` as int64 when every value is a whole number, else None.

    Parameters
    ----------
    ids
        The candidate column, of any dtype.

    Returns
    -------
    numpy.ndarray or None
        An int64 view or copy holding the same numbers, or None when the
        column is not integral - a fraction, a NaN, an infinity, a magnitude
        float64 no longer spells exactly, or a dtype that is not a number.

    Notes
    -----
    The float64 case costs one int64 array, which the caller needs anyway,
    and one pass to check the cast lost nothing. ``min``/``max`` rather than
    ``abs(...).max()``: no temporary the length of the mesh, and a NaN
    poisons both comparisons the same way an out-of-range value does.
    """
    kind = ids.dtype.kind
    if kind in "iu":
        return ids.astype(IDS_DTYPE, copy=False)
    if kind != "f":
        return None
    if ids.size and not (-_EXACT_IN_FLOAT < ids.min() and ids.max() < _EXACT_IN_FLOAT):
        return None
    whole = ids.astype(IDS_DTYPE)
    # int64 -> float64 is exact inside the bound above, so an unequal element
    # is one the cast truncated: a fraction, and no id.
    return whole if np.array_equal(whole, ids) else None


def _dense_from_one(ids: np.ndarray) -> bool:
    """Report whether ``ids`` is exactly ``1, 2, ... n`` in that order."""
    if ids.size == 0:
        return True
    # The two scalar comparisons stand in front of the ``diff`` rather than
    # beside it: ``and`` short-circuits, so a numbering that is not dense at
    # all - the case this asks about on every read - answers in constant time
    # and allocates nothing the length of the mesh.
    return bool(ids[0] == 1 and ids[-1] == ids.size and np.all(np.diff(ids) == 1))


def unwritable(
    ids: np.ndarray, count: int, what: str, *, limit: int | None = None
) -> str | None:
    """Say why ``ids`` cannot be written, or None when they can.

    Parameters
    ----------
    ids
        The candidate ids.
    count
        How many entities the mesh holds.
    what
        ``"vertex"`` or ``"element"``, named in the reason so a warning
        reads as a sentence.
    limit
        The widest id the format can spell, for one whose own reader holds
        ids narrower than int64. None for a format that spells any of them.

    Returns
    -------
    str or None
        A phrase completing "the ids ...", or None when nothing is wrong.

    Notes
    -----
    One test, used on the way in and again on the way out. A reader stores
    only ids a writer would honour, so the key's presence is a promise rather
    than a hint; the writer asks again all the same, because a transform moves
    a mesh out from under its ids without either one being wrong.
    """
    return _judge(np.asarray(ids), count, what, limit=limit)[0]


def _judge(
    ids: np.ndarray, count: int, what: str, *, limit: int | None = None
) -> tuple[str | None, np.ndarray | None]:
    """Say why ``ids`` cannot be written, and hand back the int64 column.

    Parameters
    ----------
    ids
        The candidate ids.
    count
        How many entities the mesh holds.
    what
        ``"vertex"`` or ``"element"``, named in the reason.
    limit
        The widest id the format can spell, or None when it spells any.

    Returns
    -------
    tuple
        ``(reason, None)`` when the ids cannot be written, ``(None, column)``
        when they can. The column is what :func:`unwritable`'s callers would
        otherwise cast for themselves, handed over so the walk over the ids
        happens once however many of them there are.
    """
    if ids.ndim != 1 or ids.size != count:
        return f"are not one per {what}", None
    whole = _as_ids(ids)
    if whole is None:
        return "are not integers", None
    if whole.size == 0:
        return None, whole
    if whole.min() <= 0:
        return "are not all positive", None
    # Ahead of the sort below, being the cheaper of the two passes: a Gmsh
    # node tag is a 64-bit integer by spec and a Nastran field holds 31 bits,
    # so a mesh can arrive carrying an id its next format has no room for.
    if limit is not None and whole.max() > limit:
        return f"run past {limit}, the widest id this format spells", None
    # ``unique`` rather than a set: one sort over int64 instead of a Python
    # object per entity, which is the difference on a grid of any size.
    if np.unique(whole).size != whole.size:
        return "are not unique", None
    return None, whole


def record_ids(ids, *, count: int) -> dict[str, np.ndarray]:
    """Return the ``*_attrs`` entry a reader that saw file ids should carry.

    Parameters
    ----------
    ids
        The id each entity carried in the file, in mesh index order. Anything
        array-like of integers, so a reader holding a ``dict`` mapping id to
        index can hand over ``list(node_map)`` directly.
    count
        How many entities the mesh holds, which the ids must match one for
        one. A reader that dropped an entity has to drop its id with it.

    Returns
    -------
    dict
        ``{"original_ids": array}`` when the numbering says something the
        index does not and a writer could spell it back, an empty dict
        otherwise, so a reader can splat it into the mapping it is building
        without a branch of its own.

    Notes
    -----
    A file numbering ``1..n`` in order records nothing: a writer renumbering
    from one reproduces it exactly, so the entry would be redundant on the
    great majority of real files. Nor is a numbering recorded that no writer
    could honour - a count that does not match the mesh, a duplicate, a
    non-positive id - since the key's presence is meant to be a promise that
    the numbering survives a round trip.

    Examples
    --------
    >>> record_ids([1, 2, 3], count=3)
    {}
    >>> sorted(record_ids([10, 20, 30], count=3))
    ['original_ids']
    """
    why, whole = _judge(np.asarray(ids), count, "entity")
    if why is not None or _dense_from_one(whole):
        return {}
    return {IDS_KEY: whole}


def original_ids(poly: "PolyData", *, kind: str) -> np.ndarray | None:
    """Return the ids a mesh remembers for one kind of entity.

    Parameters
    ----------
    poly
        The mesh to ask.
    kind
        ``"vertex"`` or ``"element"``, naming which attribute mapping holds
        the entry.

    Returns
    -------
    numpy.ndarray or None
        The stored ids, or None when the mesh carries none. No check that
        they can still be written - :func:`ids_for_write` is what asks that.

    Raises
    ------
    ValueError
        If ``kind`` is neither ``"vertex"`` nor ``"element"``.
    """
    if kind == "vertex":
        attrs = poly.vertex_attrs
    elif kind == "element":
        attrs = poly.element_attrs
    else:
        raise ValueError(f"kind must be 'vertex' or 'element', got {kind!r}.")
    found = (attrs or {}).get(IDS_KEY)
    return None if found is None else np.asarray(found)


def ids_for_write(
    poly: "PolyData",
    *,
    kind: str,
    count: int,
    fmt: str,
    default: np.ndarray | None = None,
    limit: int | None = None,
    stacklevel: int = 3,
) -> np.ndarray:
    """Return the id to write for each entity, one per entity, from one up.

    Parameters
    ----------
    poly
        The mesh being written.
    kind
        ``"vertex"`` or ``"element"``.
    count
        How many ids the writer needs, which is how many entities of that
        kind the mesh holds.
    fmt
        The format's own name, ``".bdf"`` and the like, so the warning a mesh
        with unusable ids raises names the file it is about to land in.
    default
        What to number with when the mesh remembers nothing usable, one
        entry per entity. None means ``1..count``. A codec that skips
        elements it has no spelling for passes its own, so an unremembered
        mesh keeps landing numbered the way it did before the policy existed.
    limit
        The widest id this format can spell, for one whose own reader holds
        ids narrower than int64. None for a format that spells any of them.
        A mesh carrying a wider id is numbered densely rather than written
        into a file nothing can read back.
    stacklevel
        Passed to :func:`warnings.warn`. The default points at the caller of
        the codec's own ``write``, which is two frames above this helper.

    Returns
    -------
    numpy.ndarray
        int64, length ``count``. The ids the mesh remembers when they are
        positive, unique, one per entity and inside ``limit``; ``1..count``
        otherwise.

    Warns
    -----
    UserWarning
        When the mesh carries ids that cannot be written - duplicated by a
        merge or a triangulation, non-positive, fractional, wider than this
        format spells, or no longer one per entity. A mesh read from a Gmsh
        file reaches the last of those honestly: a node tag is 64-bit by
        spec, and a Nastran field holds 31 bits. The file is written with
        dense numbering rather than refused, since a mesh is still a mesh
        without the numbering its author gave it.

    Raises
    ------
    ValueError
        If ``default`` is given and is not one id per entity. That is a bug
        in the calling codec rather than anything about the mesh, so it is
        refused outright instead of quietly numbering the file wrong.

    Examples
    --------
    >>> import numpy as np
    >>> from polyxios import make_polydata
    >>> poly = make_polydata(
    ...     np.zeros((3, 3)),
    ...     [("triangle", np.array([[0, 1, 2]]))],
    ...     vertex_attrs={"original_ids": np.array([10, 20, 30])},
    ... )
    >>> ids_for_write(poly, kind="vertex", count=3, fmt=".bdf")
    array([10, 20, 30])
    """
    found = original_ids(poly, kind=kind)
    if found is not None:
        # Asked before the fallback is built: a mesh that remembers a usable
        # numbering never pays for an arange the length of itself.
        why, whole = _judge(found, count, kind, limit=limit)
        if why is None:
            return whole
        warnings.warn(
            f"{fmt}: the mesh carries {kind} ids from the file it was read"
            f" from, but they {why}; it was written numbered from one"
            " instead.",
            stacklevel=stacklevel,
        )
    if default is None:
        return np.arange(1, count + 1, dtype=IDS_DTYPE)
    dense = np.asarray(default, dtype=IDS_DTYPE)
    if dense.shape != (count,):
        raise ValueError(
            f"{fmt}: default numbering has shape {dense.shape},"
            f" not one id per {kind} ({count})."
        )
    return dense
