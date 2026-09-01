"""Kratos Multiphysics MDPA ``.mdpa`` codec (ASCII) - read + write.

An MDPA file is a flat list of ``Begin <Section>`` / ``End <Section>`` blocks.
Everything the format holds lives in one of them: ``Nodes`` carries the
coordinates, an ``Elements`` block carries the cells of one Kratos element
class, ``NodalData`` and ``ElementalData`` carry fields, ``ModelPartData``
carries whole-mesh values, and a ``SubModelPart`` names the nodes and elements
of a group. Nothing is counted in a header, so a block is read until its
``End``.

Two things about the format need a decision rather than a translation.

The first is the element class name. Kratos spells a cell as
``Element<space dim>D<node count>N``, which is a registered element class
rather than a geometry, and the pair does not identify one: ``Element3D4N`` is
a tetrahedron, and a quadrilateral sitting in space would want the same name.
polyxios writes the class name with the *topological* dimension of the element
- a quadrilateral is ``Element2D4N`` whatever plane it lies in, a tetrahedron
is ``Element3D4N`` - which makes the pair unique again and keeps a round trip
exact. Reading is wider than writing. A Kratos application registers its own
element under the same suffix - ``SmallDisplacementElement3D4N``,
``SurfaceLoadCondition3D3N``, ``VMS3D4N`` - and the suffix is the whole of
what the name says about the shape, so it is the suffix that is read rather
than the name entire. The explicit geometry names Kratos uses in its own
tables (``Tetrahedra3D4``, ``Quadrilateral3D4``, ``Prism3D6``) are recognised
too, and they say outright what a generic name only implies.

The second is ``Conditions``. Kratos keeps boundary conditions in their own
list with their own numbering, but they are cells of the same mesh, so they
are read as elements after the elements. That merges two id spaces, and two
entities answering to one number is a numbering no writer can spell back, so
such a file comes back without ``original_ids`` (see :mod:`polyxios._ids`).
Conditions are written back under ``Elements``: which cells a solver should
treat as boundary is a modelling choice this codec has no way to recover.

Node ordering follows Kratos, which numbers its higher-order nodes the way GiD
does. That agrees with polyxios for every type but two: a 20-node hexahedron
and a 15-node wedge list their vertical mid-edge nodes before the ones on the
top face, and both are permuted on the way in and back on the way out.

A data block spells a vector two ways and both are read. A ``Variable<Vector>``
declares its length - ``[3] (1.0,2.0,3.0)`` - and an ``array_1d<double,3>``,
which is what ``DISPLACEMENT``, ``VELOCITY`` and the rest of the
three-component nodal variables are, does not: ``(1.0,2.0,3.0)`` on its own.
The bare form is the commoner of the two in files Kratos itself writes. A
matrix - ``[3,3] ((…),(…),(…))`` - is more than one value per entity, which no
attribute column holds, so its block is passed over with a warning rather than
read as the vector its first row resembles.
"""

from collections.abc import Iterable
import re
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    MAX_SAFE_CONN,
    MAX_SAFE_ELEMENTS,
    MAX_SAFE_VERTICES,
)
from polyxios._ids import IDS_KEY, ids_for_write, record_ids
from polyxios._io import Source, read_text, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".mdpa"

#: Where a reader records the Properties block each element pointed at.
PROPERTY_KEY: str = "mdpa_property_id"

# ``Element2D3N``: the generic class name, spelled as (space dimension, node
# count). Anchored at the end only: an application's element carries the same
# suffix behind its own name - ``SmallDisplacementElement3D4N`` - and a
# pattern anchored at ``Element`` would read a real Kratos file as a mesh with
# no cells in it. The pair is ambiguous in the format at large - a
# quadrilateral in space and a tetrahedron are both "3D4N" - so this table is
# the reading half of the convention the module docstring states, and
# _WRITE_CLASS is the writing half. The two agree, which is what makes the
# round trip exact.
_GENERIC_RE = re.compile(r"(\d+)D(\d+)N$", re.IGNORECASE)

_GENERIC_TO_TYPE: dict[tuple[int, int], str] = {
    (2, 1): "vertex",
    (3, 1): "vertex",
    (2, 2): "line",
    (3, 2): "line",
    (2, 3): "triangle",
    (3, 3): "triangle",
    (2, 4): "quad",
    (3, 4): "tetra",
    (2, 6): "quadratic_triangle",
    (3, 6): "wedge",
    (2, 8): "quadratic_quad",
    (3, 8): "hexahedron",
    (2, 9): "biquadratic_quad",
    (3, 5): "pyramid",
    (3, 10): "quadratic_tetra",
    (3, 13): "quadratic_pyramid",
    (3, 15): "quadratic_wedge",
    (3, 20): "quadratic_hexahedron",
}

# The geometry names Kratos uses where it means a shape rather than a class.
# They are unambiguous, so a file that spells one is read by it in preference
# to what the generic pattern would have guessed.
_GEOMETRY_TO_TYPE: dict[str, str] = {
    "point2d": "vertex",
    "point3d": "vertex",
    "line2d2": "line",
    "line3d2": "line",
    "line2d3": "quadratic_edge",
    "line3d3": "quadratic_edge",
    "triangle2d3": "triangle",
    "triangle3d3": "triangle",
    "triangle2d6": "quadratic_triangle",
    "triangle3d6": "quadratic_triangle",
    "quadrilateral2d4": "quad",
    "quadrilateral3d4": "quad",
    "quadrilateral2d8": "quadratic_quad",
    "quadrilateral3d8": "quadratic_quad",
    "quadrilateral2d9": "biquadratic_quad",
    "quadrilateral3d9": "biquadratic_quad",
    "tetrahedra3d4": "tetra",
    "tetrahedra3d10": "quadratic_tetra",
    "hexahedra3d8": "hexahedron",
    "hexahedra3d20": "quadratic_hexahedron",
    "prism3d6": "wedge",
    "prism3d15": "quadratic_wedge",
    "pyramid3d5": "pyramid",
    "pyramid3d13": "quadratic_pyramid",
}

# What each type is written as, and how many nodes that spells. A type absent
# here has no unambiguous Kratos class name - a 3-node line reads as
# ``Element2D3N``, which is how a triangle is spelled - and is dropped with a
# warning rather than written as something that reads back as another shape.
_WRITE_CLASS: dict[str, str] = {
    "vertex": "Element3D1N",
    "line": "Element2D2N",
    "triangle": "Element2D3N",
    "quad": "Element2D4N",
    "tetra": "Element3D4N",
    "pyramid": "Element3D5N",
    "wedge": "Element3D6N",
    "hexahedron": "Element3D8N",
    "quadratic_triangle": "Element2D6N",
    "quadratic_quad": "Element2D8N",
    "biquadratic_quad": "Element2D9N",
    "quadratic_tetra": "Element3D10N",
    "quadratic_pyramid": "Element3D13N",
    "quadratic_wedge": "Element3D15N",
    "quadratic_hexahedron": "Element3D20N",
}

_NODES_PER_TYPE: dict[str, int] = {
    name: n for (_dim, n), name in _GENERIC_TO_TYPE.items()
}
_NODES_PER_TYPE.update({"quadratic_edge": 3})

# Kratos node order -> polyxios node order, for the two types where they
# differ: Kratos lists the mid-edge nodes of the vertical edges before those
# of the top face, and polyxios (like VTK) lists the top face first. Both
# permutations swap two blocks of the same length, so each is its own inverse
# and the same table serves reading and writing.
_NODE_ORDER: dict[str, tuple[int, ...]] = {
    "quadratic_hexahedron": (
        *range(12),
        16,
        17,
        18,
        19,
        12,
        13,
        14,
        15,
    ),
    "quadratic_wedge": (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11),
}

# A section name travels alone on its line and a variable name opens a data
# block, so whitespace inside either would split it in two on the way back in
# and ``//`` would comment out its own tail.
_UNSAFE_NAME_RE = re.compile(r"[\s/]+")

# ``[3] (1.0,2.0,3.0)`` and the bare ``(1.0,2.0,3.0)``: the two ways Kratos
# spells a vector value in a data block. A ``Variable<Vector>`` carries the
# ``[n]``; an ``array_1d<double,3>`` - which is what DISPLACEMENT, VELOCITY
# and every other three-component nodal variable is - goes without one, and a
# reader that insisted on the prefix would refuse the commoner of the two
# spellings. So the prefix is optional, and its absence means the width is
# whatever the parentheses hold. The inner character class excludes
# parentheses as well, so one row of a matrix cannot be read as the whole of
# a vector.
_VECTOR_RE = re.compile(r"(?:\[\s*(\d+)\s*\]\s*)?\(([^()]*)\)")

# ``[3,3] ((1,2,3),(4,5,6),(7,8,9))``: a matrix value. Nothing in a PolyData
# holds one of those per entity, so a block spelling one is passed over whole
# and warned about, rather than read as the vector its first row looks like
# or raised over as if the file were malformed - it is not, it is only
# carrying more than a mesh can.
_MATRIX_RE = re.compile(r"\[\s*\d+\s*,|\(\s*\(")

# A ModelPartData value runs to the end of its line, so a line break inside
# one would spell a second entry and ``//`` would comment the value out. An
# empty one is unspellable for the other reason: the entry would be a line of
# one word, which is not an entry at all on the way back in.
_UNSPELLABLE_VALUE_RE = re.compile(r"[\r\n]|//")

# How deep a SubModelPart may nest. A Kratos hierarchy is a model tree a few
# levels tall, and the reader walks one level per frame, so a file nesting
# past this is refused rather than left to exhaust the interpreter's stack
# and come back as a RecursionError from somewhere inside numpy.
_MAX_PART_DEPTH: int = 64

# What a ModelPartData key cannot be. A line is a section boundary when its
# first word is one of these, so such a key would cut its own block in two
# rather than travel inside it - and ``Begin`` would swallow the rest of the
# file into a block nothing closes.
_SECTIONING_WORDS: frozenset[str] = frozenset({"begin", "end"})


def _scan(text: str) -> list[tuple[int, str]]:
    """Return the file's meaningful lines as ``(line number, text)`` pairs.

    ``//`` opens a comment and blank lines carry nothing, so both are dropped
    here rather than guarded against at every use. The original line number
    rides along so an error can name the line of the file, not of the list.
    """
    records: list[tuple[int, str]] = []
    for no, raw in enumerate(text.splitlines(), start=1):
        # ``partition`` allocates three strings and a tuple for every line of
        # the file. A comment needs a '/' to open, and the great majority of
        # lines hold none, so one scan for the character buys all of them out.
        line = raw.partition("//")[0].strip() if "/" in raw else raw.strip()
        if line:
            records.append((no, line))
    return records


def _opened(line: str) -> tuple[str, str] | None:
    """Return the ``(section, argument)`` a ``Begin`` line opens, or None."""
    parts = line.split()
    if len(parts) < 2 or parts[0].lower() != "begin":
        return None
    return parts[1], " ".join(parts[2:])


def _closed(line: str) -> str | None:
    """Return the section an ``End`` line closes, or None."""
    parts = line.split()
    if len(parts) < 2 or parts[0].lower() != "end":
        return None
    return parts[1]


def _block(
    records: list[tuple[int, str]], start: int, section: str
) -> tuple[list[tuple[int, str]], int]:
    """Return the lines inside the block opened at ``start`` and what follows.

    Nested blocks - the ``SubModelPartNodes`` inside a ``SubModelPart`` - are
    returned with their own ``Begin`` and ``End`` lines intact, so a caller
    that cares can scan them again and one that does not can ignore them.

    Raises
    ------
    CodecError
        If the block is never closed, or is closed by a name other than the
        one it opened: a file whose ``End`` lines do not match its ``Begin``
        lines cannot be split into sections at all.
    """
    body: list[tuple[int, str]] = []
    depth = 0
    for i in range(start + 1, len(records)):
        no, line = records[i]
        # Only a line whose first word is Begin or End can be a boundary, and
        # only a line starting with one of four letters can be either. The
        # test stands in front of the two splits because a big Nodes block
        # holds millions of lines that are none of the above, and splitting
        # each of them twice to find that out costs more than the read.
        if line[0] in "bBeE":
            if _opened(line) is not None:
                depth += 1
            else:
                shut = _closed(line)
                if shut is not None:
                    if depth == 0:
                        if shut.lower() != section.lower():
                            raise CodecError(
                                f".mdpa: line {no} closes {shut!r} inside a"
                                f" {section!r} block opened on line"
                                f" {records[start][0]}."
                            )
                        return body, i + 1
                    depth -= 1
        body.append((no, line))
    raise CodecError(
        f".mdpa: {section!r} block opened on line {records[start][0]} is never closed."
    )


def _type_of(klass: str) -> str | None:
    """Return the polyxios element type a Kratos class name names, or None."""
    known = _GEOMETRY_TO_TYPE.get(klass.lower())
    if known is not None:
        return known
    match = _GENERIC_RE.search(klass)
    if match is None:
        return None
    return _GENERIC_TO_TYPE.get((int(match.group(1)), int(match.group(2))))


def _as_number(token: str) -> int | float:
    """Parse a value token as the number it spells, integer where it is one.

    Notes
    -----
    ``int`` first and ``float`` in the handler would raise once per value of
    every float column in the file, and raising is most of what makes a parse
    slow. ``str.isdecimal`` settles which of the two to call without one, and
    every token it says yes to is a token ``int`` reads.
    """
    body = token[1:] if token[:1] in "+-" else token
    if body.isdecimal():
        return int(token)
    return float(token)


def _read_nodes(
    body: list[tuple[int, str]],
    coords: list[float],
    node_map: dict[int, int],
) -> None:
    """Read one ``Nodes`` block into ``coords`` and ``node_map``.

    Raises
    ------
    CodecError
        On a short or malformed node line, an id the file already used, or a
        node count past the safety cap.
    """
    for no, line in body:
        parts = line.split()
        if len(parts) < 4:
            raise CodecError(
                f".mdpa: node line {no} carries {len(parts)} value(s), expected"
                " an id and three coordinates."
            )
        try:
            nid = int(parts[0])
            xyz = [float(tok) for tok in parts[1:4]]
        except ValueError as exc:
            raise CodecError(f".mdpa: malformed node line {no}: {line!r}.") from exc
        if nid in node_map:
            raise CodecError(f".mdpa: node id {nid} is declared twice, line {no}.")
        if len(node_map) >= MAX_SAFE_VERTICES:
            raise CodecError(
                f".mdpa: node count exceeds the safety cap {MAX_SAFE_VERTICES}."
            )
        node_map[nid] = len(node_map)
        coords.extend(xyz)


class _Cells:
    """The cells read so far, and the id space each block numbered them in.

    Elements and conditions are one flat list here and two lists in the file,
    so which one an id belongs to has to be kept apart: a ``SubModelPart``
    naming element 7 and condition 7 means two different cells.
    """

    def __init__(self) -> None:
        self.conn: list[int] = []
        self.offsets: list[int] = [0]
        self.types: list[int] = []
        self.ids: list[int] = []
        self.properties: list[int] = []
        self.index_of: dict[str, dict[int, int]] = {"element": {}, "condition": {}}


def _read_cells(
    body: list[tuple[int, str]],
    klass: str,
    kind: str,
    cells: _Cells,
    node_map: dict[int, int],
    unknown: dict[str, int],
) -> None:
    """Read one ``Elements`` or ``Conditions`` block into ``cells``.

    A block whose class name this codec does not know is counted in
    ``unknown`` and skipped whole: reading its lines without knowing how many
    nodes a cell of that class has would pull one cell's nodes into the next.

    Raises
    ------
    CodecError
        On a short or malformed cell line, a node reference the file never
        declared, an id the block's own id space already used, or a cell or
        connectivity count past the safety cap.
    """
    name = _type_of(klass)
    if name is None:
        # A block that names no class at all is unreadable for the same
        # reason and is counted under a name of this codec's own, so the
        # warning says something rather than trailing off.
        named = klass or "(unnamed)"
        unknown[named] = unknown.get(named, 0) + len(body)
        return
    n_nodes = _NODES_PER_TYPE[name]
    order = _NODE_ORDER.get(name)
    code = ELEMENT_TYPES[name]
    index_of = cells.index_of[kind]

    for no, line in body:
        parts = line.split()
        if len(parts) < 2 + n_nodes:
            raise CodecError(
                f".mdpa: {kind} line {no} carries {max(len(parts) - 2, 0)}"
                f" node(s), expected {n_nodes} for {klass}."
            )
        try:
            cid = int(parts[0])
            prop = int(parts[1])
            raw = [int(tok) for tok in parts[2 : 2 + n_nodes]]
        except ValueError as exc:
            raise CodecError(f".mdpa: malformed {kind} line {no}: {line!r}.") from exc
        if cid in index_of:
            raise CodecError(f".mdpa: {kind} id {cid} is declared twice, line {no}.")
        if len(cells.types) >= MAX_SAFE_ELEMENTS:
            raise CodecError(
                f".mdpa: element count exceeds the safety cap {MAX_SAFE_ELEMENTS}."
            )
        if len(cells.conn) + n_nodes > MAX_SAFE_CONN:
            raise CodecError(
                f".mdpa: connectivity exceeds the safety cap {MAX_SAFE_CONN}."
            )
        if order is not None:
            raw = [raw[k] for k in order]
        try:
            nodes = [node_map[nid] for nid in raw]
        except KeyError as exc:
            raise CodecError(
                f".mdpa: {kind} line {no} references node {exc.args[0]}, which"
                " no Nodes block declares."
            ) from exc
        index_of[cid] = len(cells.types)
        cells.conn.extend(nodes)
        cells.offsets.append(cells.offsets[-1] + n_nodes)
        cells.types.append(code)
        cells.ids.append(cid)
        cells.properties.append(prop)


def _read_values(
    body: list[tuple[int, str]], what: str
) -> tuple[dict[int, Any], int] | None:
    """Return the ``{entity id: value}`` a data block holds and its width.

    Parameters
    ----------
    body
        The lines inside the block.
    what
        ``"nodal"`` or ``"elemental"``, named in an error so it reads as a
        sentence about the file.

    Returns
    -------
    tuple or None
        The values by entity id, and the width of the column they fill - or
        None when the block spells a matrix per entity, which a PolyData has
        no column for and which the caller passes over with a warning. A
        width of 0 means the block spelled bare scalars; anything larger is
        the length of the vectors it held, a declared ``[1]`` included - a
        one-component vector is a column of one, and reading it back as a
        scalar would reshape a mesh on its way through. A block mixing the
        two is read as the widest it saw, since one row of a column cannot
        be a different shape than the rest.

    Raises
    ------
    CodecError
        On a line with no value, a malformed id or value, or a vector whose
        declared length does not match the values in its parentheses.

    Notes
    -----
    Both of the format's vector spellings are read: the ``[n] (…)`` of a
    ``Variable<Vector>`` and the bare ``(…)`` of an ``array_1d<double,3>``.
    A row spelled without the prefix declares nothing, so its width is what
    its parentheses hold.
    """
    values: dict[int, Any] = {}
    width = 0
    for no, line in body:
        parts = line.split()
        if len(parts) < 2:
            raise CodecError(
                f".mdpa: {what} data line {no} carries no value: {line!r}."
            )
        try:
            eid = int(parts[0])
        except ValueError as exc:
            raise CodecError(
                f".mdpa: malformed id on {what} data line {no}: {line!r}."
            ) from exc
        # The parenthesis test stands in front of the searches because most
        # data blocks are scalar and most lines of a big one hold none at all.
        vector = None
        if "(" in line:
            # A matrix nests its parentheses and declares two dimensions, so
            # the substring test settles the common bare ``(x,y,z)`` without
            # a second search and only a bracketed row pays for one.
            if "((" in line or ("[" in line and _MATRIX_RE.search(line)):
                return None
            vector = _VECTOR_RE.search(line)
        try:
            if vector is not None:
                items = [
                    _as_number(tok) for tok in vector.group(2).replace(",", " ").split()
                ]
                spelled = vector.group(1)
                if spelled is not None and int(spelled) != len(items):
                    raise CodecError(
                        f".mdpa: {what} data line {no} declares a"
                        f" vector of {int(spelled)} but carries {len(items)}."
                    )
                values[eid] = items
                width = max(width, len(items))
            else:
                # ``id value`` and ``id is_fixed value`` are both legal, and
                # the value is the last token in either.
                values[eid] = _as_number(parts[-1])
        except ValueError as exc:
            raise CodecError(
                f".mdpa: malformed value on {what} data line {no}: {line!r}."
            ) from exc
    return values, width


def _as_column(
    values: dict[int, Any], index_of: dict[int, int], count: int, width: int
) -> tuple[np.ndarray | None, int]:
    """Return the array a data block fills, and what it could not place.

    Parameters
    ----------
    values
        The values the block held, by entity id.
    index_of
        Where the mesh keeps each of those ids.
    count
        How many entities of this kind the mesh holds.
    width
        0 for a column of scalars, otherwise how many components a row has.

    Returns
    -------
    tuple
        The column - or None when the block named no entity this mesh holds -
        and how many of its rows named an entity the mesh does not hold.
        Those rows are dropped, and the count is warned about: a data block
        naming an id nothing in the mesh answers to says the file disagrees
        with itself.

    Notes
    -----
    An entity the block does not mention keeps zero: the format lists only
    the entities a variable was set on, so a missing row is an unset value
    rather than a gap in the file.
    """
    rows: list[tuple[int, Any]] = []
    for eid, val in values.items():
        index = index_of.get(eid)
        if index is not None:
            rows.append((index, val))
    unplaced = len(values) - len(rows)
    if not rows:
        return None, unplaced
    # A generator rather than a flattened list: the first float settles the
    # dtype, and a column of a million integers is never materialised twice.
    integral = all(
        isinstance(v, int)
        for _, val in rows
        for v in (val if isinstance(val, list) else (val,))
    )
    dtype = np.int64 if integral else np.float64
    column = np.zeros((count, width) if width else (count,), dtype=dtype)
    # One assignment for the whole block where every row is the shape the
    # column is - which is every block a writer produces and nearly every one
    # a solver does. A scalar-by-scalar loop over a million rows costs more
    # than the rest of the read together, and the scan that decides between
    # the two only measures lengths.
    if width:
        uniform = all(type(val) is list and len(val) == width for _, val in rows)
    else:
        uniform = not any(type(val) is list for _, val in rows)
    if uniform:
        where = np.fromiter(
            (index for index, _ in rows), dtype=np.intp, count=len(rows)
        )
        column[where] = np.array([val for _, val in rows], dtype=dtype)
    elif width:
        for index, val in rows:
            row = val if isinstance(val, list) else (val,)
            column[index, : len(row)] = row
    else:
        for index, val in rows:
            # ``[0] ()`` is a vector of nothing, which leaves the row unset
            # rather than raising on an empty list.
            if not isinstance(val, list):
                column[index] = val
            elif val:
                column[index] = val[0]
    return column, unplaced


def _unique(name: str, taken: set[str]) -> str:
    """Return ``name``, or the first ``name_2``, ``name_3``, ... free of it.

    Parameters
    ----------
    name
        The name to claim.
    taken
        The names already claimed, added to in place.

    Returns
    -------
    str
        The claimed name.
    """
    candidate = name
    suffix = 2
    while candidate in taken:
        candidate = f"{name}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _place(
    blocks: dict[str, tuple[dict[int, Any], int]],
    index_of: dict[int, int],
    count: int,
    section: str,
    reserved_keys: tuple[str, ...],
    unplaced: dict[str, int],
    reserved: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Return the attribute mapping a set of data blocks fills.

    Parameters
    ----------
    blocks
        The values and width of each variable the section held, by name.
    index_of
        Where the mesh keeps each entity id the blocks name.
    count
        How many entities of this kind the mesh holds.
    section
        ``"NodalData"`` or ``"ElementalData"``, named in the warnings.
    reserved_keys
        The keys polyxios keeps for itself in this mapping. A file variable
        spelling one is read under a suffixed name instead of being buried
        under - or, worse, left posing as - a key the library gives a
        meaning of its own.
    unplaced
        Counts of rows naming an entity the mesh does not hold, added to in
        place and warned about by the caller.
    reserved
        The names moved off a reserved key, by section, added to in place.

    Returns
    -------
    dict
        The columns, by the name each is stored under.
    """
    columns: dict[str, np.ndarray] = {}
    taken: set[str] = set(reserved_keys)
    for name, (values, width) in blocks.items():
        column, missed = _as_column(values, index_of, count, width)
        if missed:
            unplaced[section] = unplaced.get(section, 0) + missed
        if column is None:
            continue
        if name in taken:
            reserved.setdefault(section, []).append(name)
            name = _unique(name, taken)
        else:
            taken.add(name)
        columns[name] = column
    return columns


def _read_sub_model_part(
    body: list[tuple[int, str]],
    name: str,
    cells: _Cells,
    node_map: dict[int, int],
    vertex_tags: dict[str, list[int]],
    element_tags: dict[str, list[int]],
    missing: dict[str, int],
    taken: set[str],
    renamed: list[str],
    depth: int = 0,
) -> None:
    """Read one ``SubModelPart`` block into the tag mappings.

    A nested ``SubModelPart`` is read under its own name rather than folded
    into its parent: Kratos nests them to say a group is part of a larger one,
    which polyxios has no way to spell, and folding the members upward would
    put a node in a group the file never named it in. Kratos only asks that
    siblings differ, so two parts under different parents can carry one name;
    the second to be read gets a ``_2`` rather than its members poured into
    the first. A part that contributes no member claims nothing, so an empty
    one cannot push a namesake that does carry members off its own name.

    Raises
    ------
    CodecError
        If the parts nest past ``_MAX_PART_DEPTH``.
    """
    if depth > _MAX_PART_DEPTH:
        raise CodecError(
            f".mdpa: SubModelPart {name!r} nests more than {_MAX_PART_DEPTH} deep."
        )
    # The name is claimed at the first member found rather than on the way in.
    # A part that contributes none - one carrying only SubModelPartData, only
    # the children it groups, or a membership list that turned out empty -
    # becomes no tag at all, and a part that became no tag has no business
    # pushing a later namesake to a suffix.
    held: list[str] = []

    def claim() -> str:
        if not held:
            got = _unique(name, taken)
            if got != name:
                renamed.append(name)
            held.append(got)
        return held[0]

    i = 0
    while i < len(body):
        head = _opened(body[i][1])
        if head is None:
            i += 1
            continue
        section, argument = head
        inner, i = _block(body, i, section)
        lower = section.lower()
        if lower == "submodelpart":
            _read_sub_model_part(
                inner,
                argument.strip() or f"{name}_sub",
                cells,
                node_map,
                vertex_tags,
                element_tags,
                missing,
                taken,
                renamed,
                depth + 1,
            )
        elif lower == "submodelpartnodes":
            found: list[int] = []
            _collect(inner, node_map, found, "node", missing)
            if found:
                vertex_tags.setdefault(claim(), []).extend(found)
        elif lower in ("submodelpartelements", "submodelpartconditions"):
            kind = "element" if lower.endswith("elements") else "condition"
            found = []
            _collect(inner, cells.index_of[kind], found, kind, missing)
            if found:
                element_tags.setdefault(claim(), []).extend(found)
        # Any other nested section - SubModelPartData, SubModelPartTables -
        # says nothing about membership and is skipped.


def _collect(
    body: list[tuple[int, str]],
    index_of: dict[int, int],
    members: list[int],
    kind: str,
    missing: dict[str, int],
) -> None:
    """Resolve one membership list, counting the ids the file never declared."""
    for _, line in body:
        for token in line.split():
            index = index_of.get(_int_or_none(token))
            if index is None:
                missing[kind] = missing.get(kind, 0) + 1
            else:
                members.append(index)


def _int_or_none(token: str) -> int | None:
    """Return the integer a membership token spells, or None when it is not one."""
    try:
        return int(token)
    except ValueError:
        return None


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a Kratos MDPA file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.mdpa`` file, or an open binary handle.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData
        Elements in file order, followed by the cells of every ``Conditions``
        block. ``NodalData`` becomes ``vertex_attrs``, ``ElementalData``
        becomes ``element_attrs``, ``ModelPartData`` becomes ``global_attrs``
        and each ``SubModelPart`` becomes an ``element_tags`` and/or
        ``vertex_tags`` entry named after it.

    Raises
    ------
    CodecError
        On a block that is never closed or closed under another name, a
        ``Nodes`` section that is missing or declares nothing, a malformed
        or short node, cell or data line, a duplicate id, a node reference
        the file never declares, or a count past the safety caps.

    Notes
    -----
    The property id each cell points at is kept in
    ``element_attrs["mdpa_property_id"]``, but only when some cell points at
    something other than property 0 - a file that uses one property says
    nothing a writer could not reproduce. ``Conditions`` are read as elements,
    which merges two id spaces: when a condition id repeats an element id the
    mesh comes back without ``original_ids``, since a numbering with a
    duplicate in it cannot be written back. A ``NodalData`` or
    ``ElementalData`` row naming an entity the mesh does not hold is dropped
    with a warning, as is a whole block spelling a matrix per entity. A
    variable named after a key polyxios keeps for itself - ``original_ids``,
    ``mdpa_property_id`` - is read under a suffixed name and warned about,
    rather than buried under the key or left posing as it.
    A ``ModelPartData`` value carries no type of its own, so
    one spelling a number or ``true``/``false`` comes back as one: a mesh
    that wrote the string ``"42"`` reads it back as the integer 42. A
    ``Geometries`` block, a ``ConditionalData`` block and ``Tables`` are not
    read, and each is warned about. A ``Properties`` block is passed over
    without one: every MDPA file carries at least one, and the id it
    declares travels in
    ``element_attrs["mdpa_property_id"]`` even though its contents do not.
    Two names the reader may have to invent are warned about as well: a data
    block that names no variable is read as ``unnamed``, and a
    ``SubModelPart`` whose name another one already took gets a ``_2`` rather
    than its members poured into the first - Kratos asks only that siblings
    differ, so two parts under different parents may carry one name. A part
    contributing no member of its own claims no name and becomes no tag: one
    that only groups its children, or whose membership list turned out empty,
    has nothing to push a later namesake aside for.
    """
    if lazy:
        warnings.warn(
            ".mdpa: lazy=True is not supported; loading eagerly.", stacklevel=2
        )

    # 'utf-8-sig' so a byte-order mark cannot glue itself to the first Begin;
    # errors="replace" keeps a file written in some other 8-bit encoding
    # inside a CodecError, since only a section name can be hurt.
    records = _scan(read_text(path, encoding="utf-8-sig", errors="replace"))

    coords: list[float] = []
    node_map: dict[int, int] = {}
    cells = _Cells()
    global_attrs: dict[str, Any] = {}
    nodal: dict[str, tuple[dict[int, Any], int]] = {}
    elemental: dict[str, tuple[dict[int, Any], int]] = {}
    vertex_members: dict[str, list[int]] = {}
    element_members: dict[str, list[int]] = {}
    part_names: set[str] = set()
    renamed_parts: list[str] = []
    unknown_classes: dict[str, int] = {}
    missing_members: dict[str, int] = {}
    skipped: dict[str, int] = {}
    matrix_data: dict[str, list[str]] = {}
    unnamed_data = 0

    i = 0
    while i < len(records):
        head = _opened(records[i][1])
        if head is None:
            i += 1
            continue
        section, argument = head
        body, i = _block(records, i, section)
        lower = section.lower()
        argument = argument.strip()

        if lower == "nodes":
            _read_nodes(body, coords, node_map)
        elif lower in ("elements", "conditions"):
            kind = "element" if lower == "elements" else "condition"
            _read_cells(body, argument, kind, cells, node_map, unknown_classes)
        elif lower == "modelpartdata":
            for _, line in body:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    global_attrs[parts[0]] = _as_value(parts[1].strip())
        elif lower in ("nodaldata", "elementaldata"):
            on_nodes = lower == "nodaldata"
            store = nodal if on_nodes else elemental
            # A data block with no variable name is malformed, but its values
            # are still one per entity, so they travel under a name of this
            # codec's own rather than under an empty one.
            if not argument:
                unnamed_data += 1
            key = argument or "unnamed"
            block = _read_values(body, "nodal" if on_nodes else "elemental")
            if block is None:
                matrix_data.setdefault(section, []).append(key)
            else:
                store[key] = _merged(store.get(key), block)
        elif lower == "submodelpart":
            _read_sub_model_part(
                body,
                argument or "submodelpart",
                cells,
                node_map,
                vertex_members,
                element_members,
                missing_members,
                part_names,
                renamed_parts,
            )
        elif lower != "properties":
            skipped[section] = skipped.get(section, 0) + 1

    if not node_map:
        raise CodecError(".mdpa: no 'Begin Nodes' section declares a node.")

    n_verts = len(node_map)
    n_elems = len(cells.types)
    # The columns are built before the warnings go out rather than after, so
    # the rows a data block spent on an entity the file never declared are
    # counted in time to be reported with everything else the reader dropped.
    unplaced: dict[str, int] = {}
    reserved: dict[str, list[str]] = {}
    vertex_attrs = _place(
        nodal, node_map, n_verts, "NodalData", (IDS_KEY,), unplaced, reserved
    )
    element_attrs = _place(
        elemental,
        cells.index_of["element"],
        n_elems,
        "ElementalData",
        (IDS_KEY, PROPERTY_KEY),
        unplaced,
        reserved,
    )

    _warn_read(
        unknown_classes,
        missing_members,
        skipped,
        unnamed_data,
        renamed_parts,
        unplaced,
        matrix_data,
        reserved,
    )

    vertex_attrs.update(record_ids(list(node_map), count=n_verts))
    # One id space for two of the file's own, so a condition numbered like an
    # element leaves the mesh with no numbering rather than a broken one.
    element_attrs.update(record_ids(cells.ids, count=n_elems))
    if any(cells.properties):
        element_attrs[PROPERTY_KEY] = np.array(cells.properties, dtype=np.int64)

    return PolyData(
        vertices=np.array(coords, dtype=np.float64).reshape(n_verts, 3),
        connectivity=np.array(cells.conn, dtype=np.int32),
        offsets=np.array(cells.offsets, dtype=np.int32),
        element_types=np.array(cells.types, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        # np.unique rather than sorted(set(...)): a tag on a large mesh holds
        # as many members as the mesh holds entities, and the sort runs in C.
        vertex_tags={
            name: np.unique(np.asarray(members, dtype=np.int32))
            for name, members in vertex_members.items()
            if members
        },
        element_tags={
            name: np.unique(np.asarray(members, dtype=np.int32))
            for name, members in element_members.items()
            if members
        },
        global_attrs=global_attrs,
    )


def _merged(
    seen: tuple[dict[int, Any], int] | None, block: tuple[dict[int, Any], int]
) -> tuple[dict[int, Any], int]:
    """Fold a second data block for one variable into the first."""
    if seen is None:
        return block
    values, width = seen
    more, more_width = block
    values.update(more)
    return values, max(width, more_width)


def _as_value(text: str) -> Any:
    """Return the Python value a ``ModelPartData`` entry spells."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return _as_number(text)
    except ValueError:
        return text


def _warn_read(
    unknown: dict[str, int],
    missing: dict[str, int],
    skipped: dict[str, int],
    unnamed: int,
    renamed: list[str],
    unplaced: dict[str, int],
    matrix: dict[str, list[str]],
    reserved: dict[str, list[str]],
) -> None:
    """Report what the reader passed over, one warning per kind."""
    if unknown:
        named = ", ".join(sorted(unknown))
        warnings.warn(
            f".mdpa: {sum(unknown.values())} cell(s) in block(s) named {named}"
            " use an element class this codec does not know; skipped.",
            stacklevel=3,
        )
    for kind, count in sorted(missing.items()):
        warnings.warn(
            f".mdpa: {count} SubModelPart {kind} member(s) name an id the file"
            " never declares; dropped.",
            stacklevel=3,
        )
    for section, count in sorted(unplaced.items()):
        warnings.warn(
            f".mdpa: {count} {section} row(s) name an entity this mesh does"
            " not hold; dropped.",
            stacklevel=3,
        )
    if skipped:
        named = ", ".join(sorted(skipped))
        warnings.warn(
            f".mdpa: section(s) {named} are not read by this codec; ignored.",
            stacklevel=3,
        )
    if unnamed:
        warnings.warn(
            f".mdpa: {unnamed} data block(s) name no variable; read as 'unnamed'.",
            stacklevel=3,
        )
    if renamed:
        warnings.warn(
            f".mdpa: SubModelPart name(s) {sorted(set(renamed))} are used by"
            " more than one part; the later ones were read under a suffixed"
            " name.",
            stacklevel=3,
        )
    for section, names in sorted(matrix.items()):
        warnings.warn(
            f".mdpa: {section} block(s) {sorted(set(names))} spell a matrix"
            " per entity, which no attribute column holds; skipped.",
            stacklevel=3,
        )
    for section, names in sorted(reserved.items()):
        warnings.warn(
            f".mdpa: {section} variable(s) {sorted(set(names))} name a key"
            " that is already spoken for - one polyxios keeps for itself, or"
            " one another block took; read under a suffixed name.",
            stacklevel=3,
        )


def _safe_name(name: str, seen: set[str], unsafe: list[str]) -> str:
    """Return a name safe to write on a line of its own, uniquely."""
    clean = _UNSAFE_NAME_RE.sub("_", name.strip()) or "unnamed"
    if clean != name:
        unsafe.append(name)
    return _unique(clean, seen)


def _number(value: Any) -> str:
    """Spell one value the way Kratos reads it back.

    ``repr`` of a Python float is the shortest text that round-trips exactly;
    a fixed width would truncate a float64 coordinate silently.
    """
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _data_rows(
    values: np.ndarray, ids: Iterable[int], *, fixed: bool, indent: str = ""
) -> list[str]:
    """Return the ``NodalData`` / ``ElementalData`` lines a column spells.

    The indent is written into the line rather than glued on by the caller:
    a big mesh spells one of these per node, and prefixing them afterwards
    would build every one of them twice.

    Notes
    -----
    A column has one dtype, so which of the two spellings its values want is
    settled once here rather than asked of :func:`_number` a million times
    over. ``repr`` of a float is the shortest text that reads back exactly,
    and ``str`` of an int is the int; a boolean column reached this function
    as 0/1 already, since ``true`` is a word ``ModelPartData`` knows and a
    data block does not.
    """
    flag = " 0" if fixed else ""
    spell = repr if values.dtype.kind == "f" else str
    if values.ndim == 1:
        return [
            f"{indent}{eid}{flag} {spell(val)}"
            for eid, val in zip(ids, values.tolist(), strict=True)
        ]
    return [
        f"{indent}{eid}{flag} [{len(row)}] ({','.join(map(spell, row))})"
        for eid, row in zip(ids, values.tolist(), strict=True)
    ]


def _writable_columns(
    attrs: dict[str, np.ndarray] | None, count: int, reserved: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Split an attribute mapping into what a data block can hold and what not.

    A data block spells one number or one bracketed vector per entity, so a
    numeric column of one or two dimensions travels and anything else - a
    tensor, a string column, a column that is not one value per entity - does
    not. A column of no components at all is turned away with them: its
    ``[0] ()`` rows carry nothing and read back as a column of scalars, which
    is a shape the mesh did not have. A boolean column travels as 0/1: Kratos
    reads ``true`` in a ``ModelPartData`` entry and nowhere else, so the word
    would spell a file this codec's own reader refuses.
    """
    keep: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for name, raw in (attrs or {}).items():
        if name in reserved:
            continue
        values = np.asarray(raw)
        if (
            values.ndim in (1, 2)
            and values.shape[0] == count
            and values.dtype.kind in "iufb"
            and (values.ndim == 1 or values.shape[1] > 0)
        ):
            keep[name] = values.astype(np.int64) if values.dtype.kind == "b" else values
        else:
            dropped.append(name)
    return keep, dropped


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to a Kratos MDPA file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output ``.mdpa`` path, or an open binary handle.
    **opts
        None are recognized; any that are passed are warned about and ignored.

    Notes
    -----
    Elements are grouped by Kratos element class, in the order the mesh first
    reaches each, so a mesh whose types are interleaved comes back from
    :func:`read` grouped. Every cell is written under ``Elements``, conditions
    included: which cells a solver treats as boundary is a modelling choice
    the mesh does not carry.

    An element of a type Kratos has no unambiguous class name for - a 3-node
    line, whose ``Element2D3N`` is how a triangle is spelled - is dropped with
    a warning, as is one whose node count does not match its type. A tag naming
    a dropped element leaves it out of its ``SubModelPart``, warned about
    apart from a member that indexes nothing at all: Kratos refuses to load a
    file whose part names an element the file never declared.
    ``vertex_attrs`` and ``element_attrs`` become ``NodalData`` and
    ``ElementalData`` blocks; a column that is neither a scalar nor a vector
    per entity has nowhere to go and is dropped with a warning, and a boolean
    one travels as 0/1. ``global_attrs`` becomes ``ModelPartData``, with a
    non-scalar entry dropped the same way and a value that is empty or nothing
    but whitespace, or a string holding a line break or a ``//``, dropped as
    well, since none survives the line it is written on - a blank one would
    spell a line of one word, which is no entry at all on the way back in. An
    entry named ``Begin`` or ``End`` is dropped too: a line whose first word
    is either is a section boundary, so the entry would cut its own block in
    half. A string value keeps no
    surrounding whitespace, which the line it travels on cannot hold apart
    from the space that separates it from its name.

    Each ``vertex_tags`` and ``element_tags`` entry becomes a
    ``SubModelPart``; the two mappings share a name space here, so a name in
    both writes one part holding nodes and elements, while a member indexing
    an entity the mesh does not hold is dropped with a warning. A name is
    made unique only against the others of its own kind: Kratos looks a
    ``ModelPartData`` key, a ``NodalData`` variable, an ``ElementalData``
    variable and a ``SubModelPart`` name up in four tables, so one mesh may
    spell the same name in all four.
    """
    if opts:
        warnings.warn(
            f".mdpa write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)
    offsets = np.asarray(poly.offsets, dtype=np.int64).tolist()
    conn = np.asarray(poly.connectivity, dtype=np.int64).tolist()
    names = [ELEMENT_TYPES_INV.get(int(code), "") for code in poly.element_types]

    node_ids = ids_for_write(poly, kind="vertex", count=n_verts, fmt=".mdpa").tolist()

    # Which elements get written, settled before any of them is: the ids have
    # to be in hand while the blocks are formed, since a SubModelPart and an
    # ElementalData row both name an element by id.
    groups: dict[str, list[int]] = {}
    unspellable = 0
    malformed = 0
    for i, name in enumerate(names):
        klass = _WRITE_CLASS.get(name)
        if klass is None:
            unspellable += 1
            continue
        if offsets[i + 1] - offsets[i] != _NODES_PER_TYPE[name]:
            malformed += 1
            continue
        groups.setdefault(klass, []).append(i)

    written = [i for members in groups.values() for i in members]
    default_ids = np.zeros(n_elems, dtype=np.int64)
    for number, i in enumerate(written, start=1):
        default_ids[i] = number
    kept = ids_for_write(
        poly, kind="element", count=n_elems, fmt=".mdpa", default=default_ids
    )
    elem_ids = default_ids
    if not np.array_equal(kept, default_ids):
        # A dropped element never got an id, and a SubModelPart naming one
        # Kratos cannot find is a file it refuses to load. Only when the mesh
        # brought its own numbering: the default already holds the zeroes.
        elem_ids = np.where(default_ids > 0, kept, 0)

    properties, dropped_property = _property_ids(poly, n_elems)
    lines: list[str] = []
    unsafe: list[str] = []
    # Four name spaces rather than one: Kratos looks a ModelPartData key, a
    # NodalData variable, an ElementalData variable and a SubModelPart name up
    # in tables of its own, so a mesh calling all four TEMPERATURE writes the
    # name four times instead of renaming three of them behind the author.
    global_names: set[str] = set()
    nodal_names: set[str] = set()
    elemental_names: set[str] = set()
    part_names: set[str] = set()

    global_lines = []
    dropped_globals: list[str] = []
    unspellable_globals: list[str] = []
    sectioning_globals: list[str] = []
    for name, value in (poly.global_attrs or {}).items():
        if not isinstance(
            value, (str, bool, int, float, np.bool_, np.integer, np.floating)
        ):
            dropped_globals.append(name)
            continue
        # _safe_name only turns whitespace and '/' into '_', so a name that
        # would sanitise into a section word is already one, and the test can
        # stand in front of the claim rather than take a name back after it.
        if name.strip().lower() in _SECTIONING_WORDS:
            sectioning_globals.append(name)
            continue
        spelled = _number(value)
        # ``strip()`` rather than the text entire: a value of nothing but
        # whitespace writes a line of one word, which is no entry at all on
        # the way back in, so it is as unspellable as an empty one.
        if not spelled.strip() or _UNSPELLABLE_VALUE_RE.search(spelled):
            unspellable_globals.append(name)
            continue
        global_lines.append(f"{_safe_name(name, global_names, unsafe)} {spelled}")
    lines.append("Begin ModelPartData")
    lines.extend(f"    {entry}" for entry in global_lines)
    lines.append("End ModelPartData")
    lines.append("")

    for prop in sorted({int(properties[i]) for i in written} or {0}):
        lines.append(f"Begin Properties {prop}")
        lines.append("End Properties")
        lines.append("")

    lines.append("Begin Nodes")
    lines.extend(
        f"    {nid} {x!r} {y!r} {z!r}"
        for nid, (x, y, z) in zip(
            node_ids, np.asarray(poly.vertices, dtype=np.float64).tolist(), strict=True
        )
    )
    lines.append("End Nodes")
    lines.append("")

    # One list of Python ints for the element loop: formatting a numpy scalar
    # goes the long way round, and a big mesh formats one per element.
    spelled_ids = elem_ids.tolist()
    for klass, members in groups.items():
        lines.append(f"Begin Elements {klass}")
        for i in members:
            nodes = conn[offsets[i] : offsets[i + 1]]
            order = _NODE_ORDER.get(names[i])
            if order is not None:
                nodes = [nodes[k] for k in order]
            spelled = " ".join(str(node_ids[n]) for n in nodes)
            lines.append(f"    {spelled_ids[i]} {properties[i]} {spelled}")
        lines.append("End Elements")
        lines.append("")

    vertex_columns, dropped_vertex = _writable_columns(
        poly.vertex_attrs, n_verts, (IDS_KEY,)
    )
    for name, values in vertex_columns.items():
        lines.append(f"Begin NodalData {_safe_name(name, nodal_names, unsafe)}")
        lines.extend(_data_rows(values, node_ids, fixed=True, indent="    "))
        lines.append("End NodalData")
        lines.append("")

    element_columns, dropped_element = _writable_columns(
        poly.element_attrs, n_elems, (IDS_KEY, PROPERTY_KEY)
    )
    for name, values in element_columns.items():
        lines.append(f"Begin ElementalData {_safe_name(name, elemental_names, unsafe)}")
        lines.extend(
            _data_rows(
                values[written],
                elem_ids[written].tolist(),
                fixed=False,
                indent="    ",
            )
        )
        lines.append("End ElementalData")
        lines.append("")

    part_lines, out_of_range, unwritten = _sub_model_parts(
        poly, node_ids, elem_ids, part_names, unsafe
    )
    lines.extend(part_lines)

    _warn_write(
        unspellable=unspellable,
        malformed=malformed,
        dropped_vertex=dropped_vertex,
        dropped_element=dropped_element,
        dropped_globals=dropped_globals,
        unspellable_globals=unspellable_globals,
        sectioning_globals=sectioning_globals,
        dropped_property=dropped_property,
        out_of_range=out_of_range,
        unwritten=unwritten,
        unsafe=unsafe,
    )
    write_text(path, "\n".join(lines) + "\n")


def _property_ids(poly: PolyData, count: int) -> tuple[list[int], bool]:
    """Return the Properties id to write for each element.

    Returns
    -------
    tuple
        The id per element, and whether the mesh carried a property column
        this writer had to ignore. The key is kept out of ``ElementalData``
        as a matter of course, so a column a transform moved out from under
        the mesh would otherwise go without either a home or a word.
    """
    stored = (poly.element_attrs or {}).get(PROPERTY_KEY)
    if stored is None:
        return [0] * count, False
    values = np.asarray(stored)
    if values.shape != (count,) or values.dtype.kind not in "iu":
        return [0] * count, True
    return values.astype(np.int64).tolist(), False


def _sub_model_parts(
    poly: PolyData,
    node_ids: list[int],
    elem_ids: np.ndarray,
    taken: set[str],
    unsafe: list[str],
) -> tuple[list[str], int]:
    """Return the ``SubModelPart`` blocks the mesh's tags spell.

    A name carried by both mappings writes one part holding both lists, which
    is what Kratos means by a sub model part: a named piece of the mesh, not a
    named piece of one of its two entity lists.

    Returns
    -------
    tuple
        The lines, how many tag members indexed an entity the mesh does not
        hold, and how many named an element this writer had no class name
        for. Both are dropped: Kratos refuses to load a file whose sub model
        part names an entity the file never declared.
    """
    vertex_tags = poly.vertex_tags or {}
    element_tags = poly.element_tags or {}
    # One array for the whole write rather than one scalar lookup per member:
    # a tag on a large mesh holds as many entries as the mesh holds entities.
    all_node_ids = np.asarray(node_ids, dtype=np.int64)
    lines: list[str] = []
    out_of_range = 0
    unwritten = 0
    for name in list(vertex_tags) + [n for n in element_tags if n not in vertex_tags]:
        nodes = np.unique(np.asarray(vertex_tags.get(name, ()), dtype=np.int64))
        elems = np.unique(np.asarray(element_tags.get(name, ()), dtype=np.int64))
        kept_nodes = nodes[(nodes >= 0) & (nodes < all_node_ids.size)]
        kept_elems = elems[(elems >= 0) & (elems < elem_ids.size)]
        out_of_range += (nodes.size - kept_nodes.size) + (elems.size - kept_elems.size)
        spelled_nodes = all_node_ids[kept_nodes]
        # A dropped element never got an id, so a part naming it names nothing.
        # Counted apart from the members that index nothing at all: this one
        # is the mesh's own element, and it is the file that cannot hold it.
        spelled_elems = elem_ids[kept_elems]
        writable = spelled_elems > 0
        unwritten += int(spelled_elems.size - np.count_nonzero(writable))
        spelled_elems = spelled_elems[writable]
        lines.append(f"Begin SubModelPart {_safe_name(name, taken, unsafe)}")
        lines.append("    Begin SubModelPartNodes")
        lines.extend(f"        {nid}" for nid in spelled_nodes.tolist())
        lines.append("    End SubModelPartNodes")
        lines.append("    Begin SubModelPartElements")
        lines.extend(f"        {eid}" for eid in spelled_elems.tolist())
        lines.append("    End SubModelPartElements")
        lines.append("End SubModelPart")
        lines.append("")
    return lines, out_of_range, unwritten


def _warn_write(
    *,
    unspellable: int,
    malformed: int,
    dropped_vertex: list[str],
    dropped_element: list[str],
    dropped_globals: list[str],
    unspellable_globals: list[str],
    sectioning_globals: list[str],
    dropped_property: bool,
    out_of_range: int,
    unwritten: int,
    unsafe: list[str],
) -> None:
    """Report what the writer could not spell, one warning per kind."""
    if unspellable:
        warnings.warn(
            f".mdpa write: {unspellable} element(s) have no unambiguous Kratos"
            " element class; dropped.",
            stacklevel=3,
        )
    if malformed:
        warnings.warn(
            f".mdpa write: {malformed} element(s) carry a node count their type"
            " does not have; dropped.",
            stacklevel=3,
        )
    for what, dropped in (
        ("vertex_attrs", dropped_vertex),
        ("element_attrs", dropped_element),
        ("global_attrs", dropped_globals),
    ):
        if dropped:
            warnings.warn(
                f".mdpa write: {what} {sorted(dropped)} are not one number or"
                " one vector per entity; dropped.",
                stacklevel=3,
            )
    if unspellable_globals:
        warnings.warn(
            f".mdpa write: global_attrs {sorted(unspellable_globals)} carry a"
            " value no ModelPartData entry spells - empty, or holding a"
            " line break or a comment marker; dropped.",
            stacklevel=3,
        )
    if sectioning_globals:
        warnings.warn(
            f".mdpa write: global_attrs {sorted(sectioning_globals)} are named"
            " after a word that opens or closes a section, which no"
            " ModelPartData entry can be; dropped.",
            stacklevel=3,
        )
    if dropped_property:
        warnings.warn(
            f".mdpa write: element_attrs[{PROPERTY_KEY!r}] is not one whole"
            " number per element; every element was written under property 0.",
            stacklevel=3,
        )
    if out_of_range:
        warnings.warn(
            f".mdpa write: {out_of_range} tag member(s) index an entity the"
            " mesh does not hold; dropped.",
            stacklevel=3,
        )
    if unwritten:
        warnings.warn(
            f".mdpa write: {unwritten} tag member(s) name an element this"
            " file has no class name for; left out of their SubModelPart.",
            stacklevel=3,
        )
    if unsafe:
        warnings.warn(
            f".mdpa write: name(s) {sorted(set(unsafe))} carry whitespace or a"
            " comment marker; written with those replaced by '_'.",
            stacklevel=3,
        )
