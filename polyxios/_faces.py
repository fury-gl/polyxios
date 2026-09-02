"""What a face-list format can say about an element's shape, and what it cannot.

A PLY face, an OBJ ``f`` record and their kin are a flat ring of vertex
indices and nothing else: the file names no element type, so a reader has only
the ring's width to go on and calls it a triangle at three, a quad at four and
a polygon otherwise. Anything else a writer puts in one - a solid, a
higher-order element - keeps its vertices and loses the type it was. The rule
and the report live here so the codecs that share the convention cannot drift
apart on it.

The other kind of face lives here too: the one a solver names as a *side of an
element* rather than as an element of its own. An Abaqus ``*Surface`` says
"face S3 of every element in this set", and there is no element in the deck to
hang the name on. polyxios has no separate face-set container - a mesh is
vertices and elements - so such a face is read as the element it describes: a
triangle or a quad of the parent's face nodes, tagged with the surface's name,
the way the UGRID, SU2 and Netgen readers already hand back their boundary
faces. A quadratic parent gives a linear face: the table below numbers a
volume element's faces by their corner nodes, so a side of a C3D10 or a C3D20
is the triangle or quadrilateral its corners describe and the mid-side nodes
along its edges stay with the parent. The face still answers
:func:`is_parent_face`, which compares the same corners, so a deck written
back carries the ``*Surface`` it came in as; a mesh handed to another format
carries the linear face.

What the arrangement costs is the link back to the parent, and the two columns
below are that link:

| key | holds |
|---|---|
| ``face_parent`` | the element this one is a face of, -1 when it is not a face |
| ``face_index`` | which of that element's faces it is, in the numbering of ``ELEMENT_FACES``, -1 when it is not a face |

They are ``element_attrs`` like any other, so they travel through every format
with an attribute channel and survive the transforms that keep attributes
aligned. ``face_parent`` is an *index*, though, so a transform that reorders
or drops elements leaves it pointing at the wrong one - the same hazard
``original_ids`` has. ``merge`` is the one transform that knows where each
mesh's elements landed, and it shifts the column the way it shifts the tag
groups; the rest is why a writer asks :func:`parent_face_mask` rather than
trusting the column: a face whose vertices are no longer its parent's is
written as the ordinary element it has become.

Being a face is a claim about the element's type as well as its vertices. A
face this module builds is the triangle, quad or polygon its ring spells, so
an element of any other type is not one however its vertices fall - a
tetrahedron whose four nodes are exactly a hexahedron's face is a solid that
happens to sit there, and writing it as that face would cost the mesh a
volume element.
"""

from typing import TYPE_CHECKING
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_FACES, ELEMENT_TYPES, ELEMENT_TYPES_INV

if TYPE_CHECKING:  # pragma: no cover - typing only
    from polyxios._types import PolyData

__all__ = [
    "FACE_INDEX_KEY",
    "FACE_KEY_STRIDE",
    "FACE_PARENT_KEY",
    "face_keys",
    "is_parent_face",
    "parent_face_columns",
    "parent_face_mask",
    "parent_faces",
    "report_flattened_faces",
]

# Where a face records the element it is a side of. Deliberately not
# format-prefixed: it says something about the element, not about the file it
# came out of, so a surface read from an Abaqus deck and written as a .vtu
# comes back the same pair of faces.
FACE_PARENT_KEY: str = "face_parent"
FACE_INDEX_KEY: str = "face_index"

# An element type is a uint8, so a pair of them packs into one integer and the
# pairs a mesh holds are counted in one numpy pass.
_TYPE_CODES: int = 256

# What a type code is multiplied by to pack a face number in beside it. Wider
# than the widest entry of ``ELEMENT_FACES`` by enough that a corrupt column
# naming a face no type has is refused rather than folded onto another type's.
FACE_KEY_STRIDE: int = 32

# Past this an int64 and a float64 no longer name the same numbers, so a
# column read back as a double is only an index below it.
_EXACT_IN_FLOAT: int = 2**53


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
    stacklevel: int = 4,
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
    stacklevel
        How far above this frame the warning should point. The default
        answers a codec's ``write`` through the helper that gathers its face
        records, which is where both callers reach this from.

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
        stacklevel=stacklevel,
    )


def parent_face_columns(
    parents: dict[int, tuple[int, int]], count: int
) -> dict[str, np.ndarray]:
    """Return the two columns that link faces back to their elements.

    Parameters
    ----------
    parents
        Element index to ``(parent element, local face)``, for the elements
        that are faces of another.
    count
        How many elements the mesh holds.

    Returns
    -------
    dict of str to numpy.ndarray
        The ``face_parent`` and ``face_index`` columns, or an empty mapping
        when no element is a face of another - a mesh that carries no surface
        is not worth two columns of -1.

    Examples
    --------
    >>> parent_face_columns({2: (0, 3)}, 3)["face_parent"].tolist()
    [-1, -1, 0]
    """
    if not parents:
        return {}
    parent_col = np.full(count, -1, dtype=np.int32)
    index_col = np.full(count, -1, dtype=np.int32)
    for element, (parent, local) in parents.items():
        parent_col[element] = parent
        index_col[element] = local
    return {FACE_PARENT_KEY: parent_col, FACE_INDEX_KEY: index_col}


def _whole_column(stored: object, n_elems: int) -> np.ndarray | None:
    """Return one stored column as int64 indices, or None when it is not that.

    Parameters
    ----------
    stored
        The column as the mesh carries it.
    n_elems
        How many elements the mesh holds.

    Returns
    -------
    numpy.ndarray or None
        The column as int64, or None when it is not one whole number per
        element.

    Notes
    -----
    A float column that holds whole numbers counts. It has to: a legacy
    ``.vtk`` cell array is a double whatever it was handed, so a mesh that
    went out through one comes back with these columns as doubles, and
    refusing them would lose every surface a deck round-tripped through that
    format. Nothing is rounded into place - a column holding a fraction, or
    a magnitude float64 no longer spells exactly, is refused whole - and
    :func:`is_parent_face` still checks the claim against the vertices, so a
    number that survives here has not moved a face onto another element.

    A value that is not a number at all reads as the ``-1`` that says "not a
    face" rather than refusing the column. That is what a merge writes: it
    fills a mesh carrying no such column with the blank its dtype spells,
    which for a float one is NaN, so refusing would drop every surface the
    *other* mesh did carry - the whole reason ``merge`` renumbers the column
    in the first place.
    """
    values = np.asarray(stored)
    if values.ndim != 1 or values.shape[0] != n_elems:
        return None
    if values.dtype.kind in "iub":
        return values.astype(np.int64, copy=False)
    if values.dtype.kind != "f":
        return None
    # One pass to find the blanks, and a second array only when there are
    # some: a column straight off a reader holds none, and that is the path
    # every write takes.
    known = np.isfinite(values)
    held = values if known.all() else np.where(known, values, -1.0)
    if held.size and not (
        -_EXACT_IN_FLOAT < held.min() and held.max() < _EXACT_IN_FLOAT
    ):
        return None
    whole = held.astype(np.int64)
    return whole if np.array_equal(whole, held) else None


def parent_faces(poly: "PolyData") -> tuple[np.ndarray, np.ndarray] | None:
    """Return the parent and local-face columns a mesh carries, if usable.

    Parameters
    ----------
    poly
        The mesh.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray] or None
        The two columns as int64, or None when the mesh carries neither, or
        carries something that is not one whole number per element - the
        columns index elements, and a rounded index names another element.
    """
    n_elems = len(poly.element_types)
    columns = []
    for key in (FACE_PARENT_KEY, FACE_INDEX_KEY):
        stored = (poly.element_attrs or {}).get(key)
        if stored is None:
            return None
        values = _whole_column(stored, n_elems)
        if values is None:
            return None
        columns.append(values)
    return columns[0], columns[1]


def face_keys(type_codes: np.ndarray, locals_: np.ndarray) -> np.ndarray:
    """Pack a parent's type and one of its face numbers into a single integer.

    Parameters
    ----------
    type_codes
        One element type code per entry.
    locals_
        The face of that type each names, in the numbering of
        ``ELEMENT_FACES``.

    Returns
    -------
    numpy.ndarray
        One int64 key per entry.

    Notes
    -----
    Everything a face of a known type needs - its width, its corner indices,
    the label a format calls it - depends on the pair and nothing else, so a
    walk over the faces a mesh holds is a walk over the handful of distinct
    pairs it spreads them over. Packing the pair is what lets numpy group
    them in one pass, the way :func:`~polyxios._tags.group_by_value` groups a
    label column.
    """
    return type_codes.astype(np.int64) * FACE_KEY_STRIDE + locals_


def parent_face_mask(
    poly: "PolyData",
    elements: np.ndarray,
    parents: np.ndarray,
    locals_: np.ndarray,
) -> np.ndarray:
    """Say, for each claim, whether the element still is the face it names.

    Parameters
    ----------
    poly
        The mesh.
    elements
        The elements said to be faces. An index this mesh has no element for
        answers False, the way a parent's does: the columns are indices, and
        a transform that dropped elements leaves both ends of the link stale.
    parents
        The element each is said to be a face of.
    locals_
        Which of that parent's faces each is said to be.

    Returns
    -------
    numpy.ndarray
        One bool per claim: True where the parent holds a face of that number
        and the element's vertices are exactly its vertices.

    Notes
    -----
    Asked at the point of writing rather than trusted, because a transform
    moves a mesh out from under these columns without either being wrong:
    ``filter_element_type`` drops the solids a surface belonged to, and a
    caller may have built the columns by hand. Comparing the vertices is what
    tells a face that still is one from an index that now names a different
    element.

    Whole columns rather than one claim at a time, because a writer asks this
    of every member of every surface it writes and a mesh carrying a skin
    asks it as often as it has faces. The claims are grouped by
    :func:`face_keys`, so the corner table is read once per distinct pair
    rather than once per face and the comparison itself is one sort per
    group.

    The offsets are clipped to the connectivity rather than trusted, which is
    what slicing a row out of it did when this was asked one claim at a time:
    a mesh whose offsets run past its connectivity holds a shorter row than
    it declares, and a row that is short of the face's own corners is no
    match for it.
    """
    elements = np.asarray(elements, dtype=np.int64)
    parents = np.asarray(parents, dtype=np.int64)
    locals_ = np.asarray(locals_, dtype=np.int64)
    n_elems = len(poly.element_types)
    out = np.zeros(elements.shape, dtype=bool)

    live = np.flatnonzero(
        (elements >= 0)
        & (elements < n_elems)
        & (parents >= 0)
        & (parents < n_elems)
        & (parents != elements)
        & (locals_ >= 0)
        & (locals_ < FACE_KEY_STRIDE)
    )
    if not live.size:
        return out

    ele, par, loc = elements[live], parents[live], locals_[live]
    conn = poly.connectivity
    n_conn = conn.size
    offsets = poly.offsets
    p_start = np.clip(offsets[par], 0, n_conn)
    p_width = np.clip(offsets[par + 1], 0, n_conn) - p_start
    e_start = np.clip(offsets[ele], 0, n_conn)
    e_width = np.clip(offsets[ele + 1], 0, n_conn) - e_start

    keys = face_keys(poly.element_types[par], loc)
    order = np.argsort(keys, kind="stable")
    ranked = keys[order]
    starts = np.flatnonzero(np.concatenate(([True], ranked[1:] != ranked[:-1])))
    for key, group in zip(ranked[starts], np.split(order, starts[1:])):
        faces = ELEMENT_FACES.get(
            ELEMENT_TYPES_INV.get(int(key) // FACE_KEY_STRIDE, "")
        )
        local = int(key) % FACE_KEY_STRIDE
        if faces is None or local >= len(faces):
            continue
        corners = np.asarray(faces[local], dtype=np.int64)
        # The width, and the type that width names. A face this library built
        # is the triangle, quad or polygon its ring spells, so an element of
        # any other type is not one however its vertices fall - a tetra whose
        # four nodes are exactly a hexahedron's face is still a solid, and
        # writing it as that face would cost the mesh a volume element.
        is_face = int(_reads_back_as(np.array([corners.size]))[0])
        group = group[
            (p_width[group] > int(corners.max()))
            & (e_width[group] == corners.size)
            & (poly.element_types[ele[group]] == is_face)
        ]
        if not group.size:
            continue
        wanted = conn[p_start[group][:, None] + corners]
        held = conn[e_start[group][:, None] + np.arange(corners.size)]
        # Sorted rather than as two sets: a set forgets how often a vertex
        # appears, so a collapsed face repeating one of its corners would
        # answer for the face it collapsed from, and be written back as a
        # side the element no longer has.
        wanted.sort(axis=1)
        held.sort(axis=1)
        out[live[group[(held == wanted).all(axis=1)]]] = True
    return out


def is_parent_face(poly: "PolyData", element: int, parent: int, local: int) -> bool:
    """Say whether an element still is the face its columns claim.

    Parameters
    ----------
    poly
        The mesh.
    element
        The element said to be a face. An index this mesh has no element for
        answers False, the way a parent's does: the columns are indices, and
        a transform that dropped elements leaves both ends of the link stale.
    parent
        The element it is said to be a face of.
    local
        Which of the parent's faces it is said to be.

    Returns
    -------
    bool
        True when the parent holds a face of that number and the element's
        vertices are exactly its vertices.

    Notes
    -----
    One claim asked of :func:`parent_face_mask`, which is where the rule
    lives: a writer asks it of whole columns, and two spellings of the same
    comparison would be two chances to disagree about what a face is.
    """
    return bool(
        parent_face_mask(poly, np.array([element]), np.array([parent]), [local])[0]
    )
