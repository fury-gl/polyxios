"""Netgen mesh codec (ASCII ``.vol``) - read + write.

A ``.vol`` file opens with the word ``mesh3d`` and is then a sequence of named
sections, each one a keyword line, a count line, and that many records::

    mesh3d
    dimension
    3
    geomtype
    0

    # surfnr    bcnr   domin  domout      np      p1      p2      p3
    surfaceelements
    1
    1 1 0 0 3 1 2 3

    #  matnr      np      p1      p2      p3      p4
    volumeelements
    1
    1 4 1 2 3 4

    points
    4
    0 0 0
    ...
    endmesh

The leading fields differ per section and are what say which element an index
belongs to: a surface element spells ``surfnr bcnr domin domout np`` before its
nodes and a volume element spells ``matnr np``, so the node list starts at
column 5 in one and column 2 in the other. ``np`` is read per record rather
than assumed, since one section holds every element of its dimension whatever
its order - a section of triangles may hold a six-node one next to a three-node
one.

The ``…gi``/``…uv`` spellings of the element sections carry extra geometry or
parameter values *after* the nodes. Records are parsed a line at a time so those
trailing values fall off the end of the record instead of shifting the one
behind them, and each record is cut down to its nodes and its index before
anything is turned into a number - the tail is floating point in both spellings
and reading it as an integer would refuse the whole file.

Netgen numbers nodes from 1 and orders them differently from polyxios (and
VTK): a tetrahedron's second and third nodes are swapped, and the volume types
turn the other way around their axis. Every type whose order differs is
permuted in both directions - a cell read without the permutation is inside
out, which is still four valid nodes and so goes unnoticed until a solver
reports a negative Jacobian.

Each element carries one integer index - ``bcnr`` on a face, ``matnr`` on a
volume cell - which becomes an ``element_tags`` entry. The optional
``materials``/``bcnames``/``cd2names``/``cd3names`` sections name those indices,
and the names are what the tags are called when they are there. One name over
several indices of a codimension is how the format spells a single boundary
carried by several surfaces, so those indices come back as one tag.

Only the plain-text flavour is handled here; ``.vol.gz`` is a gzip stream and is
refused rather than parsed as text.
"""

from collections.abc import Iterable
from itertools import repeat
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
    NODES_PER_ELEMENT,
)
from polyxios._io import Source, read_bytes, source_name, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".vol"

# Matches the other ASCII writers (.obj, .avs, .su2, .ugrid); pass float_fmt to
# widen it.
_DEFAULT_FLOAT_FMT: str = ".10g"

# Element sections: keyword to (topological dimension, index column, first node
# column, node-count column). A None node-count column means the section's
# records are all one width, given by _FIXED_WIDTH.
_CELL_SECTIONS: dict[str, tuple[int, int, int, int | None]] = {
    "pointelements": (0, 1, 0, None),
    "edgesegments": (1, 0, 2, None),
    "edgesegmentsgi": (1, 0, 2, None),
    "edgesegmentsgi2": (1, 0, 2, None),
    "surfaceelements": (2, 1, 5, 4),
    "surfaceelementsgi": (2, 1, 5, 4),
    "surfaceelementsuv": (2, 1, 5, 4),
    "volumeelements": (3, 0, 2, 1),
}
_FIXED_WIDTH: dict[int, int] = {0: 1, 1: 2}

# Netgen names an element by its node count within its dimension, so the same
# number means different things in two sections: six nodes is a quadratic
# triangle on a face and a wedge in a volume.
_TYPE_BY_DIM: dict[int, dict[int, str]] = {
    0: {1: "vertex"},
    1: {2: "line"},
    2: {3: "triangle", 4: "quad", 6: "quadratic_triangle", 8: "quadratic_quad"},
    3: {
        4: "tetra",
        5: "pyramid",
        6: "wedge",
        8: "hexahedron",
        10: "quadratic_tetra",
        13: "quadratic_pyramid",
        15: "quadratic_wedge",
        20: "quadratic_hexahedron",
    },
}

# The section each dimension is written into, in the order a .vol file lists
# them. Reading accepts any order; writing has to fix one.
_SECTION_BY_DIM: dict[int, str] = {
    2: "surfaceelements",
    3: "volumeelements",
    1: "edgesegmentsgi2",
    0: "pointelements",
}

# Node order per type, as gather indices out of Netgen's order into polyxios's:
# ``polyxios[i] = netgen[_TO_POLYXIOS[type][i]]``. A type absent from the table
# already agrees with polyxios and is read as it lies.
_TO_POLYXIOS: dict[str, list[int]] = {
    "quadratic_triangle": [0, 1, 2, 5, 3, 4],
    "quadratic_quad": [0, 1, 2, 3, 4, 7, 5, 6],
    "tetra": [0, 2, 1, 3],
    "quadratic_tetra": [0, 2, 1, 3, 5, 7, 4, 6, 9, 8],
    "pyramid": [0, 3, 2, 1, 4],
    "quadratic_pyramid": [0, 3, 2, 1, 4, 7, 6, 8, 5, 9, 12, 11, 10],
    "wedge": [0, 2, 1, 3, 5, 4],
    "quadratic_wedge": [0, 2, 1, 3, 5, 4, 7, 8, 6, 13, 14, 12, 9, 11, 10],
    "hexahedron": [0, 3, 2, 1, 4, 7, 6, 5],
    "quadratic_hexahedron": [
        0,
        3,
        2,
        1,
        4,
        7,
        6,
        5,
        10,
        9,
        11,
        8,
        16,
        19,
        18,
        17,
        14,
        13,
        15,
        12,
    ],  # fmt: skip
}


def _invert(perm: list[int]) -> list[int]:
    """Return the permutation undoing ``perm``."""
    out = [0] * len(perm)
    for i, src in enumerate(perm):
        out[src] = i
    return out


_FROM_POLYXIOS: dict[str, list[int]] = {
    name: _invert(perm) for name, perm in _TO_POLYXIOS.items()
}

# Sections carrying names for the element indices, by the codimension they
# number. Codimension rather than dimension because that is what the format
# fixes: ``materials`` names the cells of the mesh's own dimension whether that
# is 2 or 3.
_NAME_SECTIONS: dict[str, int] = {
    "materials": 0,
    "bcnames": 1,
    "cd2names": 2,
    "cd3names": 3,
}
_NAME_PREFIX: dict[int, str] = {0: "material", 1: "bc", 2: "cd2", 3: "cd3"}
_NAME_KEYWORD: dict[int, str] = {codim: kw for kw, codim in _NAME_SECTIONS.items()}

# Sections this codec carries no home for. Each is a count and that many
# records, so they can be stepped over without being understood.
_SKIP_SECTIONS: frozenset[str] = frozenset(
    {
        "identifications",
        "face_colours",
        "singular_points",
        "singular_edge_left",
        "singular_edge_right",
        "singular_face_inside",
        "singular_face_outside",
    }
)

# A tag name reading gave an index, which writing reads the number back out of
# so a round trip keeps the index it started with.
_INDEX_RE = re.compile(r"^(?:material|bc|cd2|cd3)_(-?\d+)$")

_DIGITS_RE = re.compile(r"(\d+)")

# How large an index the names sections are willing to spell. They are written
# as a dense list from 1 to the largest index in use, and every entry has to
# carry a name, so one huge index would otherwise turn a small mesh into a file
# of millions of placeholder entries.
_MAX_NAMED_INDEX: int = 100_000

# The largest index the per-element array can hold. ``element_tags`` is the
# caller's to fill, so a name may spell a number past int64; assigning it into
# that array raises OverflowError, which is not something this codec promises.
_MAX_INDEX: int = int(np.iinfo(np.int64).max)


def _natural_key(name: str) -> tuple[tuple[int, Any], ...]:
    """Return a sort key ordering embedded numbers by value, not by spelling.

    Tag names fix the order indices are handed out in, and the names reading
    gives back are numbered - ``bc_1``, ``bc_2``, …. Plain lexicographic order
    puts ``bc_10`` between ``bc_1`` and ``bc_2``, which would hand an unnumbered
    tag an index out of turn.
    """
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in _DIGITS_RE.split(name)
    )


def _read_lines(path: Source) -> list[str]:
    """Return the file's lines, stripped, without blanks or comments.

    Netgen writes a commented column header above most sections and blank lines
    between them, and neither belongs to any section's record count.
    """
    try:
        raw = read_bytes(path)
    except OSError as exc:
        raise CodecError(f".vol: cannot read '{source_name(path)}': {exc}") from exc
    if raw[:2] == b"\x1f\x8b":
        # A .vol.gz opens with the gzip magic. Decoded as text it becomes a
        # first line of mojibake, and the error naming it would be about a
        # missing 'mesh3d' rather than about the compression.
        raise CodecError(
            f".vol: '{source_name(path)}' is gzip-compressed; only plain-text .vol files"
            " are supported. Decompress it first."
        )
    text = raw.decode("utf-8-sig", errors="replace")
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _count(lines: list[str], pos: int, cap: int, what: str) -> tuple[int, int]:
    """Read a section's record count, rejecting negative and absurd values.

    A corrupt count is the cheapest way to make a reader allocate a mesh the
    machine cannot hold, so the declared size is checked before it is trusted.
    """
    if pos >= len(lines):
        raise CodecError(f".vol: file ends where the {what} count was expected.")
    try:
        count = int(lines[pos])
    except ValueError as exc:
        raise CodecError(f".vol: non-integer {what} count {lines[pos]!r}.") from exc
    if count < 0:
        raise CodecError(f".vol: negative {what} count {count}.")
    if count > cap:
        raise CodecError(f".vol: {what} count {count} exceeds the safety cap {cap}.")
    return count, pos + 1


def _is_count(line: str) -> bool:
    """Return whether ``line`` is the bare count a section is followed by.

    What tells a section keyword apart from a stray column header: a section is
    followed by its count, a header by whatever comes next. A count is a plain
    run of digits, so a signed or otherwise decorated number is a header - a
    negative one is not a record count any section could declare.
    """
    return line.isdigit()


def _is_numbers(line: str) -> bool:
    """Return whether every field on ``line`` is a number.

    What tells the tail of a record apart from the keyword line of the section
    behind it: a record is numbers all the way across and a keyword is a word.
    """
    fields = line.split()
    if not fields:
        return False
    for field in fields:
        try:
            float(field)
        except ValueError:
            return False
    return True


def _records(
    lines: list[str], pos: int, count: int, what: str, *, per_record: int = 1
) -> tuple[list[str], int]:
    """Return the lines of the ``count`` records a section declares.

    Parameters
    ----------
    lines
        The file's lines, stripped of blanks and comments.
    pos
        Line the section's records start on.
    count
        How many records the section declares.
    what
        The section's name, for the error message.
    per_record
        How many lines one record takes. The counting is in records rather than
        lines so a truncated two-line section is reported in the units its count
        line is written in.
    """
    stop = pos + count * per_record
    if stop > len(lines):
        raise CodecError(
            f".vol: file truncated - the {what} section declares {count} record(s),"
            f" found {(len(lines) - pos) // per_record}."
        )
    return lines[pos:stop], stop


def _read_points(lines: list[str], pos: int) -> tuple[np.ndarray, int]:
    """Parse the ``points`` section into an (n, 3) float64 array.

    A file of dimension 2 spells two coordinates per node; the third is padded
    with zero rather than refused, since polyxios meshes are always 3D.
    """
    count, pos = _count(lines, pos, MAX_SAFE_VERTICES, "point")
    records, pos = _records(lines, pos, count, "points")
    if not records:
        return np.zeros((0, 3), dtype=np.float64), pos
    rows: list[list[str]] = []
    for record in records:
        xyz = record.split()[:3]
        if len(xyz) < 2:
            raise CodecError(
                f".vol: point {len(rows)} carries {len(xyz)} coordinate(s),"
                " expected 2 or 3."
            )
        rows.append(xyz + ["0"] * (3 - len(xyz)))
    try:
        return np.array(rows, dtype=np.float64), pos
    except ValueError as exc:
        raise CodecError(".vol: non-numeric point coordinate.") from exc


def _read_cells(
    lines: list[str], pos: int, keyword: str, *, two_line_edges: bool
) -> tuple[list[tuple[str, int, np.ndarray, np.ndarray]], int]:
    """Parse one element section into per-type blocks.

    Returns
    -------
    list
        ``(type name, topological dimension, node references, indices)`` per
        block. One section yields one block per distinct element width, since a
        section holds every element of its dimension whatever its order.
    int
        The line after the section.
    """
    dim, i_index, pi0, np_col = _CELL_SECTIONS[keyword]
    count, pos = _count(lines, pos, MAX_SAFE_ELEMENTS, f"{keyword} element")
    # The two-line spelling of edgesegmentsgi2 splits each record over a pair of
    # lines; joined back together they are the record the one-line spelling
    # holds, and every column below still lands where it is expected. The
    # 'surf1 surf2 p1 p2' header says so when the file carries it. Netgen
    # comments that header out in some versions and comments are gone by here,
    # so the layout is read off the records too: the one-line spelling always
    # carries its geometry values behind the nodes, so a first record that stops
    # at the nodes and is followed by a line of a different width is the head and
    # tail of one record rather than two records. The second line has to be
    # numbers for that to hold - a section declaring a single bare record is
    # followed by the keyword line of the next section, which is the same shape
    # as a tail and would take that whole section with it.
    per_record = 1
    if keyword == "edgesegmentsgi2" and count:
        split = two_line_edges
        if not split and pos + 1 < len(lines):
            head = len(lines[pos].split())
            tail = len(lines[pos + 1].split())
            split = (
                head <= pi0 + _FIXED_WIDTH[dim]
                and tail != head
                and _is_numbers(lines[pos + 1])
            )
        per_record = 2 if split else 1
    records, pos = _records(lines, pos, count, keyword, per_record=per_record)
    if per_record == 2:
        records = [f"{records[i]} {records[i + 1]}" for i in range(0, len(records), 2)]

    # Grouped by width before any parsing: every record of one width shares a
    # type, so the whole group converts in one numpy call rather than one per
    # element. Each record is cut down to its nodes and its index as it is
    # grouped - the …gi/…uv spellings carry a tail of geometry and parameter
    # values that is not read, and carrying it into the array would cost the
    # mesh over again in memory.
    groups: dict[int, list[list[str]]] = {}
    for record in records:
        fields = record.split()
        if np_col is None:
            width = _FIXED_WIDTH[dim]
        else:
            if len(fields) <= np_col:
                raise CodecError(
                    f".vol: a {keyword} record carries {len(fields)} field(s),"
                    f" too few to hold its node count at column {np_col}."
                )
            try:
                width = int(fields[np_col])
            except ValueError as exc:
                raise CodecError(
                    f".vol: non-integer node count {fields[np_col]!r} in {keyword}."
                ) from exc
            if width < 0:
                raise CodecError(f".vol: negative node count {width} in {keyword}.")
        if len(fields) < max(pi0 + width, i_index + 1):
            raise CodecError(
                f".vol: a {keyword} record carries {len(fields)} field(s), too few"
                f" for {width} node(s) starting at column {pi0}."
            )
        groups.setdefault(width, []).append(
            [*fields[pi0 : pi0 + width], fields[i_index]]
        )

    blocks: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for width, rows in groups.items():
        name = _TYPE_BY_DIM[dim].get(width)
        if name is None:
            warnings.warn(
                f".vol: {len(rows)} {keyword} record(s) carry {width} node(s), which"
                " is not an element type polyxios spells; skipped.",
                stacklevel=3,
            )
            continue
        # Only the nodes and the index are turned into numbers; the geometry
        # tail behind them is text Netgen writes as floating point, and putting
        # it through the same conversion would refuse a file whose element data
        # is perfectly good.
        text = np.array(rows, dtype=np.str_)
        try:
            refs = text[:, :width].astype(np.int64)
            block_indices = text[:, width].astype(np.int64)
        except (ValueError, OverflowError) as exc:
            # OverflowError rather than ValueError is what a value too large for
            # int64 raises, and it would come out of here uncaught.
            raise CodecError(
                f".vol: non-integer or out-of-range value in {keyword}."
            ) from exc
        perm = _TO_POLYXIOS.get(name)
        if perm is not None:
            refs = refs[:, perm]
        blocks.append((name, dim, refs, block_indices))
    return blocks, pos


def _read_names(lines: list[str], pos: int, keyword: str) -> tuple[dict[int, str], int]:
    """Parse a ``materials``/``bcnames``/``cd?names`` section.

    An entry with no name is left out rather than recorded as an empty string:
    Netgen writes the list densely from 1 to the largest index it knows, so the
    gaps between the indices in use are blank by construction.
    """
    count, pos = _count(lines, pos, MAX_SAFE_ELEMENTS, keyword)
    records, pos = _records(lines, pos, count, keyword)
    names: dict[int, str] = {}
    for record in records:
        fields = record.split(maxsplit=1)
        if len(fields) != 2:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        names[index] = fields[1].strip()
    return names, pos


def _skip_section(lines: list[str], pos: int, keyword: str) -> tuple[int, int]:
    """Step over a section this codec carries no home for.

    Returns
    -------
    int
        The line after the section.
    int
        How many entries were stepped over, so the caller can say what was
        dropped. An empty section costs the mesh nothing and is worth no word.
    """
    if keyword == "identificationtypes":
        # The odd one out: its count is how many values follow on a single line,
        # not how many lines follow. Counting lines here would swallow whatever
        # section comes next. A count of zero is the same trap the other way
        # round - Netgen writes no values at all, only the empty line the blank
        # stripping has already dropped, so the line behind the count is the
        # keyword of the next section and stepping over it would take that whole
        # section with it.
        count, pos = _count(lines, pos, MAX_SAFE_ELEMENTS, keyword)
        return (pos, 0) if count == 0 else (min(pos + 1, len(lines)), count)
    count, pos = _count(lines, pos, MAX_SAFE_ELEMENTS, keyword)
    _, pos = _records(lines, pos, count, keyword)
    return pos, count


def _tag_name(
    index: int, dim: int, *, dimension: int, names: dict[int, dict[int, str]]
) -> str:
    """Return the tag name an element index goes under."""
    codim = dimension - dim
    named = names.get(codim, {}).get(index)
    if named:
        return named
    return f"{_NAME_PREFIX.get(codim, f'netgen{dim}d')}_{index}"


def _element_tags(
    indices: np.ndarray,
    dims: np.ndarray,
    dimension: int,
    names: dict[int, dict[int, str]],
) -> dict[str, np.ndarray]:
    """Fold the per-element indices into named element tags.

    Index 0 is Netgen's word for unset and is dropped. Indices are numbered per
    dimension - a ``bcnr`` of 1 on a face and a ``matnr`` of 1 on a cell are two
    different things - so the grouping is per (dimension, index) pair.

    A names section naming two indices of one codimension alike is Netgen's way
    of spelling one boundary carried by several surfaces, so those merge into a
    single tag. Two *codimensions* spelling one name are two different things and
    are kept apart under a suffixed name instead.
    """
    tags: dict[str, np.ndarray] = {}
    marked = np.flatnonzero(indices)
    if marked.size == 0:
        return tags
    # One stable sort rather than a pass over every element per distinct pair: a
    # mesh carrying many boundary patches would otherwise walk the whole array
    # once for each of them. The sort is stable and the marked positions go in
    # ascending, so each group comes out in element order.
    pairs = np.stack([dims[marked].astype(np.int64), indices[marked]], axis=1)
    ranked = np.unique(pairs, axis=0, return_inverse=True)[1].reshape(-1)
    rank_order = np.argsort(ranked, kind="stable")
    order = marked[rank_order]
    ranked = ranked[rank_order]
    starts = np.flatnonzero(np.concatenate(([True], ranked[1:] != ranked[:-1])))
    stops = np.append(starts[1:], ranked.size)

    # Collected per (name, codimension) first: several indices of one
    # codimension sharing a name are one tag, and appending straight into
    # ``tags`` would let the last of them overwrite the others and take their
    # elements with it.
    grouped: dict[tuple[str, int], list[np.ndarray]] = {}
    for start, stop in zip(starts, stops):
        members = order[start:stop]
        first = int(members[0])
        codim = dimension - int(dims[first])
        name = _tag_name(
            int(indices[first]), int(dims[first]), dimension=dimension, names=names
        )
        grouped.setdefault((name, codim), []).append(members)

    for (name, codim), parts in grouped.items():
        members = parts[0] if len(parts) == 1 else np.sort(np.concatenate(parts))
        label = name
        # Two codimensions whose names sections spell one name would otherwise
        # fuse into a single tag, and nothing downstream could tell the faces
        # from the cells again.
        suffix = 1
        while label in tags:
            label = f"{name}_cd{codim}" if suffix == 1 else f"{name}_cd{codim}_{suffix}"
            suffix += 1
        tags[label] = members.astype(np.int32)
    return tags


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a Netgen ASCII ``.vol`` file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.vol`` file.
    lazy
        Ignored; a true value is warned about (ASCII format, always loads
        eagerly).

    Returns
    -------
    PolyData
        Elements grouped by the section they came from and by node count within
        it, in the order the file lists them.

    Raises
    ------
    CodecError
        On a gzip-compressed file, a file not opening with ``mesh3d``, a missing
        ``points`` section, a negative or absurd count, a truncated section, a
        non-numeric coordinate or node reference, a record too short for the
        nodes it declares, or a node reference outside the point array.

    Notes
    -----
    Node references are 1-based in the file and 0-based in the PolyData, and
    every type whose node order differs from polyxios's - tetrahedra, pyramids,
    wedges, hexahedra and their quadratic forms - is permuted on the way in.

    Each element's index (``bcnr`` on a face, ``matnr`` on a cell) becomes an
    ``element_tags`` entry, named by the ``materials``/``bcnames``/``cd2names``/
    ``cd3names`` sections when the file carries them and ``bc_<n>``/
    ``material_<n>`` otherwise. Index 0 means unset and is dropped. Several
    indices of one codimension named alike merge into a single tag; two
    codimensions named alike stay apart under a ``_cd<n>``-suffixed name.

    Sections holding data polyxios has no home for - ``identifications``,
    ``face_colours``, the ``singular_*`` family - are stepped over, with a
    warning when one is not empty, since what it held is dropped. A section this
    codec does not know is stepped over as a counted block with a warning too,
    and a line no count follows is taken for a column header and ignored.
    """
    if lazy:
        warnings.warn(
            ".vol: lazy=True is not supported; loading eagerly.", stacklevel=2
        )

    lines = _read_lines(path)
    if not lines:
        raise CodecError(
            f".vol: '{source_name(path)}' is an empty file; this is not a Netgen mesh file."
        )
    if lines[0].lower() != "mesh3d":
        raise CodecError(
            f".vol: '{source_name(path)}' opens with {lines[0]!r}, not 'mesh3d'; this is not"
            " a Netgen mesh file."
        )

    pos = 1
    dimension = 3
    vertices: np.ndarray | None = None
    blocks: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    names: dict[int, dict[int, str]] = {}
    two_line_edges = False

    while pos < len(lines):
        keyword = lines[pos].lower()
        pos += 1
        if keyword == "endmesh":
            break
        if keyword == "dimension":
            if pos >= len(lines):
                raise CodecError(".vol: file ends where the dimension was expected.")
            try:
                dimension = int(lines[pos])
            except ValueError as exc:
                raise CodecError(
                    f".vol: non-integer dimension {lines[pos]!r}."
                ) from exc
            pos += 1
        elif keyword == "geomtype":
            if pos >= len(lines):
                raise CodecError(".vol: file ends where the geomtype was expected.")
            pos += 1
        elif keyword == "points":
            if vertices is not None:
                raise CodecError(".vol: more than one 'points' section.")
            vertices, pos = _read_points(lines, pos)
        elif keyword in _CELL_SECTIONS:
            found, pos = _read_cells(lines, pos, keyword, two_line_edges=two_line_edges)
            blocks.extend(found)
        elif keyword in _NAME_SECTIONS:
            found_names, pos = _read_names(lines, pos, keyword)
            names.setdefault(_NAME_SECTIONS[keyword], {}).update(found_names)
        elif keyword in _SKIP_SECTIONS or keyword == "identificationtypes":
            pos, dropped = _skip_section(lines, pos, keyword)
            if dropped:
                # Same reasoning as the unknown section below: the section is
                # understood well enough to step over, but what it held is being
                # dropped, and saying so is the only word the caller gets. An
                # empty one costs the mesh nothing and is passed over in silence.
                warnings.warn(
                    f".vol: the {keyword!r} section holds {dropped} entry/entries"
                    " polyxios has no home for; skipped.",
                    stacklevel=2,
                )
        elif keyword.split()[:4] == ["surf1", "surf2", "p1", "p2"]:
            # Netgen writes this bare column header when it splits each
            # edgesegmentsgi2 record over two lines. Reading on without noticing
            # pairs every record with the tail of the one before.
            two_line_edges = True
        elif pos < len(lines) and _is_count(lines[pos]):
            # Every Netgen section is a count and that many records, so one this
            # codec has never heard of can still be stepped over - quietly would
            # be worse, since whatever it held is being dropped.
            warnings.warn(
                f".vol: unknown section {lines[pos - 1]!r}; skipped.", stacklevel=2
            )
            pos, _dropped = _skip_section(lines, pos, keyword)
        else:
            # No count follows, so this is not a section at all. Netgen writes a
            # column header above most sections and does not always comment it
            # out; stepping over it as a section would swallow the keyword line
            # of the section it belongs to and take the whole section with it.
            warnings.warn(
                f".vol: unrecognized line {lines[pos - 1]!r}; ignored.", stacklevel=2
            )

    if vertices is None:
        raise CodecError(f".vol: '{source_name(path)}' has no 'points' section.")

    n_elems = sum(refs.shape[0] for _n, _d, refs, _i in blocks)
    total_conn = sum(int(refs.size) for _n, _d, refs, _i in blocks)
    if n_elems > MAX_SAFE_ELEMENTS:
        raise CodecError(
            f".vol: element count {n_elems} exceeds the safety cap {MAX_SAFE_ELEMENTS}."
        )
    if total_conn > MAX_SAFE_CONN:
        raise CodecError(
            f".vol: connectivity {total_conn} exceeds the safety cap {MAX_SAFE_CONN}."
        )

    n_points = vertices.shape[0]
    idx_dtype = np.int64 if total_conn > np.iinfo(np.int32).max else np.int32
    conn = (
        np.concatenate([refs.reshape(-1) for _n, _d, refs, _i in blocks])
        if total_conn
        else np.zeros(0, dtype=np.int64)
    )
    if conn.size:
        lo, hi = int(conn.min()), int(conn.max())
        if lo < 1 or hi > n_points:
            # Netgen counts nodes from 1, so 0 is as much out of range as a
            # reference past the end; either would index the wrong vertex, and
            # a negative one would index from the far end in silence.
            where = f"1..{n_points}" if n_points else "an empty point array"
            raise CodecError(
                f".vol: element references node {lo}..{hi}, outside {where}."
            )
    conn = conn - 1

    offsets = np.zeros(n_elems + 1, dtype=idx_dtype)
    element_types = np.zeros(n_elems, dtype=np.uint8)
    indices = np.zeros(n_elems, dtype=np.int64)
    dims = np.zeros(n_elems, dtype=np.int8)
    at = 0
    base = 0
    for name, dim, refs, block_indices in blocks:
        count, width = refs.shape
        # Filled a block at a time rather than by summing a per-element width
        # array: every element in one block is the same width, so its offsets
        # are an arange. The width array and its cumulative sum would each be as
        # long as the mesh and both alive at once.
        offsets[at + 1 : at + 1 + count] = np.arange(
            base + width, base + width * count + 1, width, dtype=np.int64
        )
        element_types[at : at + count] = ELEMENT_TYPES[name]
        indices[at : at + count] = block_indices
        dims[at : at + count] = dim
        at += count
        base += width * count

    return PolyData(
        vertices=vertices,
        connectivity=conn.astype(idx_dtype),
        offsets=offsets,
        element_types=element_types,
        element_tags=_element_tags(indices, dims, dimension, names),
    )


def _group_elements(poly: PolyData) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sort the mesh's elements by type, keeping only what a ``.vol`` spells.

    Returns
    -------
    dict
        Type name to the element indices it takes, in mesh order.
    numpy.ndarray
        The mesh's offsets as int64, handed back so the caller gathering the
        connectivity does not widen the same array a second time.

    Raises
    ------
    CodecError
        If the offsets array does not close over the elements, or an element's
        node count does not match its type - the node count is written into the
        record, so a wrong one puts a file on disk that reads back as some other
        mesh.
    """
    types = np.asarray(poly.element_types)
    offsets = np.asarray(poly.offsets).astype(np.int64)
    if offsets.size != types.shape[0] + 1:
        raise CodecError(
            f".vol: offsets carry {offsets.size} entry/entries for"
            f" {types.shape[0]} element(s), expected {types.shape[0] + 1}."
        )
    widths = np.diff(offsets)
    groups: dict[str, np.ndarray] = {}
    known = np.zeros(types.shape[0], dtype=bool)
    # Which type codes the mesh carries, read off in one pass so the scan below
    # only walks the array for the types that are in it: a mesh of one type
    # would otherwise be swept once per type the format spells. Only when the
    # array is the uint8 the PolyData contract asks for - a wider dtype is
    # scanned type by type as before rather than folded into 256 slots, where a
    # code past the end would land on some other type's.
    present: np.ndarray | None = None
    if types.dtype == np.uint8:
        present = np.zeros(np.iinfo(np.uint8).max + 1, dtype=bool)
        present[types] = True

    for by_width in _TYPE_BY_DIM.values():
        for name in by_width.values():
            code = ELEMENT_TYPES[name]
            if present is not None and not present[code]:
                continue
            members = np.flatnonzero(types == code)
            if members.size == 0:
                continue
            known[members] = True
            width = NODES_PER_ELEMENT[name]
            bad = np.flatnonzero(widths[members] != width)
            if bad.size:
                raise CodecError(
                    f".vol: element {int(members[bad[0]])} is a {name} carrying"
                    f" {int(widths[members[bad[0]]])} node(s), expected {width}."
                )
            groups[name] = members

    if not np.all(known):
        skipped = {
            ELEMENT_TYPES_INV.get(int(code), f"code {int(code)}")
            for code in np.unique(types[~known])
        }
        warnings.warn(
            f".vol: unsupported element type(s) {sorted(skipped)}; skipped. Netgen"
            " knows vertices, lines, triangles, quads, tetrahedra, pyramids,"
            " wedges, hexahedra and their quadratic forms.",
            stacklevel=3,
        )
    return groups, offsets


def _indices_for(
    poly: PolyData, members: np.ndarray, n_elems: int
) -> tuple[np.ndarray, dict[int, str]]:
    """Hand every element of one dimension the index it is written with.

    A Netgen element carries a single index, so an element in two tags keeps the
    first that claims it in name order. ``bc_<n>`` and its siblings recover the
    number reading gave them; any other name is numbered from 1 upward so the
    tag still travels, and its name is written into the matching names section
    so it comes back spelled the way it went in. An element no tag claims takes
    the first index no tag is using - 1 on a mesh with no tags at all, which is
    what a ``.vol`` carrying no boundary information holds. Index 0 is Netgen's
    word for unset, so nothing is written with it.

    Tag members outside the mesh are dropped silently here; the caller warns
    about them once for the whole file rather than once per dimension.

    Returns
    -------
    numpy.ndarray
        One index per element of ``members``, in that order.
    dict
        Index to the tag name that took it, for the names section.
    """
    slot = np.full(n_elems, -1, dtype=np.int64)
    slot[members] = np.arange(members.size)

    ids = np.zeros(members.size, dtype=np.int64)
    claimed = np.zeros(members.size, dtype=bool)
    shared = 0
    collided = 0
    used: set[int] = set()
    writing: list[tuple[int, str, np.ndarray]] = []

    for name in sorted(poly.element_tags or {}, key=lambda n: _natural_key(str(n))):
        match = _INDEX_RE.match(str(name))
        value = int(match.group(1)) if match is not None else 0
        # A number outside 1..int64 cannot be written as it stands: 0 means
        # unset, a negative index is not something the format spells, and one
        # past int64 does not fit the array the indices are gathered in. Such a
        # tag takes a fresh number the same way an unnumbered name does, and its
        # name goes into the names section, so only the number changes.
        if not 0 < value <= _MAX_INDEX:
            value = 0
        idx = np.unique(np.asarray(poly.element_tags[name]).ravel().astype(np.int64))
        # Indices outside the mesh are dropped here and counted once for the
        # whole write by _out_of_range: this runs per dimension, so warning from
        # inside would say the same thing up to four times over.
        inside = idx[(idx >= 0) & (idx < n_elems)]
        slots = slot[inside]
        on_dim = slots[slots >= 0]
        fresh = on_dim[~claimed[on_dim]]
        shared += on_dim.size - fresh.size
        # Only a tag writing at least one element takes an index at all: one
        # whose every member landed in another dimension, or was already
        # claimed, puts nothing in this section, and reserving a number for it
        # would push every tag after it up by one.
        if fresh.size == 0:
            continue
        if value in used:
            # Two names spelling one index - ``bc_3`` and ``bc_03`` - would come
            # back as a single fused patch. The later is renumbered instead.
            collided += 1
            value = 0
        claimed[fresh] = True
        writing.append((value, str(name), fresh))
        if value:
            used.add(value)

    names: dict[int, str] = {}
    nxt = 1
    for value, name, fresh in writing:
        if value == 0:
            while nxt in used:
                nxt += 1
            value = nxt
            used.add(value)
        ids[fresh] = value
        names[value] = name

    if collided:
        warnings.warn(
            f".vol: {collided} element tag(s) name an index another tag already"
            " holds; a shared index would come back as one fused group, so those"
            " tags were renumbered.",
            stacklevel=3,
        )
    if shared:
        warnings.warn(
            f".vol: {shared} element(s) belong to more than one tag; a Netgen"
            " element carries one index, so each kept the first tag that claims"
            " it.",
            stacklevel=3,
        )
    # The index is the tag's own number, so a large one is written as it stands
    # rather than renumbered - that is what lets a numbered tag come back under
    # the name it went in with. It is not written quietly, though: Netgen sizes
    # its per-dimension tables off the largest index it reads, so the file costs
    # it an entry for every number below that one. (The surface number beside it
    # carries nothing, so writing renumbers that one densely instead.)
    oversized = [value for value in names if value > _MAX_NAMED_INDEX]
    if oversized:
        warnings.warn(
            f".vol: {len(oversized)} element index(es) run past"
            f" {_MAX_NAMED_INDEX} (largest {max(oversized)}); Netgen sizes its"
            " per-dimension tables off the largest index it reads, so this file"
            " costs it an entry for every number below that one.",
            stacklevel=3,
        )
    # An element no tag reached still carries an index. It takes a free one out
    # of the same allocator the tags drew from: handing it 1 outright would put
    # it under whichever tag already holds 1, and the two would come back fused.
    if not np.all(claimed):
        while nxt in used:
            nxt += 1
        ids[~claimed] = nxt
    return ids, names


def _warn_out_of_range(poly: PolyData, n_elems: int) -> None:
    """Warn once for every tag member that is not an element of the mesh.

    The indices are collected rather than counted: one stray index two tags both
    name is one index the mesh does not hold, and counting per tag would say two.
    """
    astray: set[int] = set()
    for members in (poly.element_tags or {}).values():
        idx = np.unique(np.asarray(members).ravel().astype(np.int64))
        astray.update(idx[(idx < 0) | (idx >= n_elems)].tolist())
    if astray:
        warnings.warn(
            f".vol: element tag(s) name {len(astray)} index(es) outside the mesh's"
            f" {n_elems} element(s); those members were dropped.",
            stacklevel=3,
        )


def _float_fmt(opts: dict[str, Any]) -> str:
    """Return the coordinate format specifier, refusing an unusable one."""
    fmt = opts.pop("float_fmt", _DEFAULT_FLOAT_FMT)
    if opts:
        warnings.warn(
            f".vol write: unrecognized options {set(opts)}; ignored.", stacklevel=3
        )
    try:
        probe = format(-1234.5, fmt)
    except (TypeError, ValueError) as exc:
        raise CodecError(f".vol: float_fmt {fmt!r} is not usable.") from exc
    try:
        # A grouping or percent format is a legal specifier that writes tokens
        # like ``-1,234.50``; the file would land on disk and only fail on the
        # way back in, so it is refused here rather than by the reader.
        float(probe)
    except ValueError as exc:
        raise CodecError(
            f".vol: float_fmt {fmt!r} writes {probe!r}, which is not a number the"
            " reader can parse back."
        ) from exc
    return fmt


def _gather(
    poly: PolyData, offsets: np.ndarray, members: np.ndarray, width: int, n_verts: int
) -> np.ndarray:
    """Return the 1-based Netgen node references of ``members``."""
    connectivity = np.asarray(poly.connectivity)
    n_conn = connectivity.shape[0] if connectivity.ndim else 0
    starts = offsets[members]
    # The gather below reads ``width`` values from each start. A start past the
    # end of the connectivity would raise a bare IndexError from inside the
    # gather, and a negative one is worse: numpy counts it from the end and
    # writes some other element's nodes without a word.
    astray = np.flatnonzero((starts < 0) | (starts + width > n_conn))
    if astray.size:
        first = int(astray[0])
        raise CodecError(
            f".vol: element {int(members[first])} starts at offset"
            f" {int(starts[first])}, so its {width} node(s) fall outside the"
            f" connectivity's {n_conn} value(s)."
        )
    refs = connectivity[starts[:, None] + np.arange(width)].astype(np.int64)
    lo, hi = int(refs.min()), int(refs.max())
    if lo < 0 or hi >= n_verts:
        where = f"0..{n_verts - 1}" if n_verts else "an empty vertex array"
        raise CodecError(
            f".vol: element references vertex {lo}..{hi}, outside {where}."
        )
    return refs + 1


def _name_lines(keyword: str, names: dict[int, str]) -> list[str]:
    """Return a names section listing ``names`` densely from index 1.

    Netgen reads the section as ``index name`` pairs and splits on whitespace,
    so a name carrying a space would come back as the first word alone; the
    spaces are folded into underscores instead of writing a file that reads back
    differently from the one intended.

    For the same reason no entry is left blank. The list runs from 1 and the
    indices whose name reading works out on its own are not recorded, so there
    are gaps in it; written as a bare number, Netgen's ``index >> name`` reads
    across the newline, takes the *next* index for this one's name, and every
    entry behind it lands one field out. A gap is written with the name reading
    would give it anyway, which is what keeps the round trip honest.
    """
    if not names:
        return []
    prefix = _NAME_PREFIX[_NAME_SECTIONS[keyword]]
    highest = max(names)
    if highest > _MAX_NAMED_INDEX:
        warnings.warn(
            f".vol: index {highest} is too large to write a '{keyword}' section,"
            " which lists every index from 1 upward; the names were dropped and"
            " the indices themselves are unaffected.",
            stacklevel=3,
        )
        return []
    lines = [keyword, str(highest)]
    squashed = 0
    for index in range(1, highest + 1):
        name = names.get(index, "")
        folded = "_".join(name.split())
        squashed += folded != name
        lines.append(f"{index} {folded or f'{prefix}_{index}'}")
    if squashed:
        warnings.warn(
            f".vol: {squashed} tag name(s) carry whitespace, which the format"
            " cannot spell; it was folded into underscores.",
            stacklevel=3,
        )
    return lines


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to a Netgen ASCII ``.vol`` file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output ``.vol`` path.
    **opts
        ``float_fmt`` overrides the ASCII coordinate format specifier. Any other
        option is warned about and ignored.

    Raises
    ------
    CodecError
        If ``float_fmt`` is not a usable format specifier or writes a token that
        is not a number, the vertices are not an ``(n, 3)`` array, the offsets do
        not close over the elements or over the connectivity, an element's node
        count does not match its type, or an element references a vertex that
        does not exist.

    Notes
    -----
    Elements are written section by section in the order the format fixes -
    faces, cells, edges, then point elements - so an element's index changes
    unless the mesh already lay in that order. Node references are written
    1-based, and every type whose node order differs from polyxios's is permuted
    into Netgen's.

    Elements of a type the format cannot spell are skipped with a warning.
    ``element_tags`` fold into one index per element, numbered independently per
    dimension: a ``bc_<n>``-style name keeps its number - unless that number is
    negative or past ``int64``, neither of which the format spells, in which case
    it takes a fresh one and only the number changes - and any other name is
    numbered from 1 upward and written into the matching ``materials``/
    ``bcnames``/``cd2names``/``cd3names`` section so it comes back by name. An
    element in two tags keeps the first that claims it in name order, and one no
    tag claims takes the first index no tag is using - 1 on a mesh carrying no
    tags - and reads back tagged with it. A tag holding elements of more than one
    dimension is written into each of them, and comes back split one tag per
    dimension, since the two sections number their indices apart.

    Coordinates go out under ``float_fmt``, ``.10g`` by default, which is not
    enough digits to name a float64 exactly - a coordinate read back differs in
    the last few. Pass ``float_fmt='.17g'`` for a bit-exact round trip, at the
    cost of a larger file.

    The format carries no fields, so ``vertex_attrs``, ``element_attrs``,
    ``vertex_tags`` and ``global_attrs`` are not written.
    """
    fmt = _float_fmt(dict(opts))

    vertices = np.asarray(poly.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        # The coordinates are written a row at a time, so an array of any other
        # width would put rows of that width on disk under a header counting
        # threes - a file that only fails on the way back in.
        raise CodecError(
            f".vol: vertices have shape {vertices.shape}, expected (n, 3)."
        )

    groups, offsets = _group_elements(poly)
    n_verts = vertices.shape[0]
    n_elems = len(poly.element_types)
    _warn_out_of_range(poly, n_elems)

    lines: list[str] = ["mesh3d", "", "dimension", "3", "", "geomtype", "0", ""]

    section_names: dict[int, dict[int, str]] = {}
    for dim, keyword in _SECTION_BY_DIM.items():
        members = [
            groups[name] for name in _TYPE_BY_DIM[dim].values() if name in groups
        ]
        # Only the surface section spells a surface number, so the other three
        # dimensions never build the array - it is as long as the whole mesh.
        surface_of: np.ndarray | None = None
        # Folded even when the dimension is empty: nothing is written for it,
        # and the indices the other dimensions hand out are unaffected.
        if members:
            positions = np.concatenate(members)
            ids, names = _indices_for(poly, positions, n_elems)
            # A tag already named after the index it holds needs no entry: the
            # names section is written densely from 1, so recording ``bc_9``
            # under 9 would spell out the eight blank entries before it to say
            # what reading works out on its own.
            prefix = _NAME_PREFIX[3 - dim]
            named = {
                index: name
                for index, name in names.items()
                if name != f"{prefix}_{index}"
            }
            if named:
                section_names[dim] = named
            # Scattered into an array over the whole mesh rather than a dict:
            # each block below then gathers its own indices in one step, and a
            # mesh of any size never builds a Python dict as long as itself.
            index_of = np.zeros(n_elems, dtype=np.int64)
            index_of[positions] = ids
            if dim == 2:
                # The surface number is numbered densely from 1 over the indices
                # in use rather than written from the index itself: Netgen holds
                # one face descriptor per surface number and fills the gaps, so a
                # single face numbered 999999 would cost it a million of them.
                surface_of = np.zeros(n_elems, dtype=np.int64)
                surface_of[positions] = (
                    np.unique(ids, return_inverse=True)[1].reshape(-1) + 1
                )
        else:
            index_of = np.zeros(n_elems, dtype=np.int64)

        records: list[str] = []
        for name in _TYPE_BY_DIM[dim].values():
            if name not in groups:
                continue
            width = NODES_PER_ELEMENT[name]
            refs = _gather(poly, offsets, groups[name], width, n_verts)
            perm = _FROM_POLYXIOS.get(name)
            if perm is not None:
                refs = refs[:, perm]
            surfaces: Iterable[int] = (
                repeat(0) if surface_of is None else surface_of[groups[name]].tolist()
            )
            for surface, index, row in zip(
                surfaces, index_of[groups[name]].tolist(), refs.tolist()
            ):
                nodes = " ".join(str(v) for v in row)
                if dim == 2:
                    # surfnr bcnr domin domout np p1…pnp. Two boundary groups get
                    # two surface numbers, so they stay two surfaces rather than
                    # one surface carrying two conditions.
                    records.append(f"{surface} {index} 0 0 {width} {nodes}")
                elif dim == 3:
                    records.append(f"{index} {width} {nodes}")
                elif dim == 1:
                    # surfid 0 p1 p2, then the geometry fields the gi2 spelling
                    # carries; Netgen's own defaults for a mesh with no geometry.
                    records.append(f"{index} 0 {nodes} -1 -1 0 0 1 0 1 0")
                else:
                    records.append(f"{nodes} {index}")

        if dim == 2:
            lines.append(
                "# surfnr    bcnr   domin  domout      np      p1      p2      p3"
            )
        elif dim == 3:
            lines.append("#  matnr      np      p1      p2      p3      p4")
        lines.append(keyword)
        lines.append(str(len(records)))
        # Extended rather than splatted into a tuple: a splat would build a
        # second copy of every record in the section before appending any of it.
        lines.extend(records)
        lines.append("")

    lines.extend(("#          X             Y             Z", "points", str(n_verts)))
    # One crossing into Python per array rather than one per number: indexing a
    # numpy array a scalar at a time is what costs on a mesh of any size.
    lines.extend(" ".join(format(c, fmt) for c in xyz) for xyz in vertices.tolist())
    lines.append("")

    # Written by codimension, the way the format numbers them: the mesh is
    # written as a 3D one, so the cells are codimension 0 and the faces 1.
    for dim, names in sorted(section_names.items(), reverse=True):
        section = _name_lines(_NAME_KEYWORD[3 - dim], names)
        if section:
            lines.extend((*section, ""))

    lines.append("endmesh")
    lines.append("")
    write_text(path, "\n".join(lines), encoding="utf-8")
