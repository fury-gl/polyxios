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
ids of the vertices that survived.
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
]

#: Where a reader records the numbers a file gave its nodes or elements.
IDS_KEY: Final[str] = "original_ids"

# What an id is written as. Wide enough for every id-carrying format: a Gmsh
# node tag is a 64-bit integer by spec, and a hand-numbered Abaqus deck reaches
# eight digits routinely.
IDS_DTYPE: Final[np.dtype] = np.dtype(np.int64)


def _dense_from_one(ids: np.ndarray) -> bool:
    """Report whether ``ids`` is exactly ``1, 2, ... n`` in that order."""
    if ids.size == 0:
        return True
    # Checked without building arange: the comparison below is one pass and no
    # allocation the length of the mesh, which matters on a large grid.
    return bool(ids[0] == 1 and ids[-1] == ids.size and np.all(np.diff(ids) == 1))


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
        index does not, an empty dict otherwise, so a reader can splat it into
        the mapping it is building without a branch of its own.

    Notes
    -----
    A file numbering ``1..n`` in order records nothing: a writer renumbering
    from one reproduces it exactly, so the entry would be redundant on the
    great majority of real files. The same is true of a mismatched count -
    the ids no longer describe this mesh, and a wrong id is worse than none.

    Examples
    --------
    >>> record_ids([1, 2, 3], count=3)
    {}
    >>> sorted(record_ids([10, 20, 30], count=3))
    ['original_ids']
    """
    arr = np.asarray(ids)
    if arr.ndim != 1 or arr.size != count:
        return {}
    if arr.size == 0:
        return {}
    if arr.dtype.kind not in "iu":
        return {}
    arr = arr.astype(IDS_DTYPE, copy=False)
    if _dense_from_one(arr):
        return {}
    return {IDS_KEY: arr}


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
    stacklevel
        Passed to :func:`warnings.warn`. The default points at the caller of
        the codec's own ``write``, which is two frames above this helper.

    Returns
    -------
    numpy.ndarray
        int64, length ``count``. The ids the mesh remembers when they are
        positive, unique and one per entity; ``1..count`` otherwise.

    Warns
    -----
    UserWarning
        When the mesh carries ids that cannot be written - duplicated by a
        merge or a triangulation, non-positive, or no longer one per entity.
        The file is written with dense numbering rather than refused, since a
        mesh is still a mesh without the numbering its author gave it.

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
    dense = np.arange(1, count + 1, dtype=IDS_DTYPE)
    found = original_ids(poly, kind=kind)
    if found is None:
        return dense

    what = f"{kind} ids"
    if found.ndim != 1 or found.size != count:
        _warn_dropped(fmt, what, "are not one per " + kind, stacklevel)
        return dense
    if found.dtype.kind not in "iu":
        _warn_dropped(fmt, what, "are not integers", stacklevel)
        return dense
    found = found.astype(IDS_DTYPE, copy=False)
    if count == 0:
        return dense
    if found.min() <= 0:
        _warn_dropped(fmt, what, "are not all positive", stacklevel)
        return dense
    # ``unique`` rather than a set: one sort over int64 instead of a Python
    # object per entity, which is the difference on a grid of any size.
    if np.unique(found).size != found.size:
        _warn_dropped(fmt, what, "are not unique", stacklevel)
        return dense
    return found


def _warn_dropped(fmt: str, what: str, why: str, stacklevel: int) -> None:
    """Report that a mesh's stored ids gave way to dense numbering."""
    warnings.warn(
        f"{fmt}: the mesh carries {what} from the file it was read from, but"
        f" they {why}; it was written numbered from one instead.",
        stacklevel=stacklevel,
    )
