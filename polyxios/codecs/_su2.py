"""SU2 mesh codec (ASCII) — read + write.

An SU2 mesh is split in two. ``NELEM`` holds the cells that fill the domain,
and every ``MARKER_TAG`` block holds the cells one dimension below them that
skin it. The two never overlap: a boundary triangle of a tetrahedral mesh is
listed once, inside its marker, and never in ``NELEM``.

polyxios keeps one flat element list, so the volume cells are read first and
each marker's cells are appended after them, with the marker's name becoming
an ``element_tags`` entry over the elements it contributed. Writing undoes
that split by element dimension, which is what keeps a mesh from gaining a
copy of every tagged element on each round trip.

Element types are spelled with the VTK type codes SU2 borrows, and node
references are 0-based in both directions, so neither needs a shift.
"""

from pathlib import Path
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
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".su2"

# SU2 uses VTK element type codes (the same numeric values, not polyxios's).
_SU2_TO_POLYXIOS: dict[int, str] = {
    3: "line",
    5: "triangle",
    9: "quad",
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}
_POLYXIOS_TO_SU2: dict[str, int] = {v: k for k, v in _SU2_TO_POLYXIOS.items()}

# Nodes per element type. A line declaring fewer is truncated, and reading it
# as if it were whole would pull the next element's nodes into this one.
_NODES_PER_TYPE: dict[int, int] = {
    3: 2,
    5: 3,
    9: 4,
    10: 4,
    12: 8,
    13: 6,
    14: 5,
}

# Topological dimension of each supported type. This is what sorts a mesh into
# the volume cells SU2 lists under NELEM and the boundary cells it lists under
# a marker; guessing instead would either duplicate elements or lose them.
_ELEMENT_DIM: dict[str, int] = {
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "tetra": 3,
    "hexahedron": 3,
    "wedge": 3,
    "pyramid": 3,
}

# ``KEY=`` opens a record. The optional space before the ``=`` is tolerated
# because writers emit it, and matching the key is what lets a data reader
# stop at the next record instead of parsing ``NPOIN=`` as an element.
_KEY_RE = re.compile(r"^([A-Za-z_]+)\s*=(.*)$")

# A marker name travels alone on its line, so whitespace inside one would split
# it in two on the way back in and ``%`` would comment out its own tail.
_UNSAFE_NAME_RE = re.compile(r"[\s%]+")

# Name given to boundary elements no tag claims. SU2 has nowhere else to put
# them, and dropping them would lose the skin of the mesh.
_DEFAULT_MARKER: str = "unnamed"


def _scan(text: str) -> list[tuple[int, str]]:
    """Return the file's meaningful lines as ``(line number, text)`` pairs.

    ``%`` opens a comment and blank lines carry nothing, so both are dropped
    here rather than guarded against at every use. The original line number
    rides along so an error can name the line of the file, not of the list.
    """
    records: list[tuple[int, str]] = []
    for no, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("%", 1)[0].strip()
        if line:
            records.append((no, line))
    return records


def _key_of(line: str) -> str | None:
    """Return the upper-cased ``KEY`` of a ``KEY=`` line, or None."""
    match = _KEY_RE.match(line)
    return match.group(1).upper() if match else None


def _value_of(line: str) -> str:
    """Return the text after the ``=`` of a ``KEY=`` line."""
    match = _KEY_RE.match(line)
    return match.group(2).strip() if match else ""


def _find_key(records: list[tuple[int, str]], key: str, *, start: int = 0) -> int:
    """Return the index of the first ``key=`` record at or after ``start``."""
    for i in range(start, len(records)):
        if _key_of(records[i][1]) == key.upper():
            return i
    return -1


def _checked_count(value: str, cap: int, what: str) -> int:
    """Parse a declared count, rejecting negatives and absurd values.

    A corrupt header is the cheapest way to make a reader allocate a mesh the
    machine cannot hold, so the declared size is checked before it is trusted.
    """
    try:
        count = int(value.split()[0])
    except (ValueError, IndexError) as exc:
        raise CodecError(f".su2: non-integer {what} count {value!r}.") from exc
    if count < 0:
        raise CodecError(f".su2: negative {what} count {count}.")
    if count > cap:
        raise CodecError(f".su2: {what} count {count} exceeds the safety cap {cap}.")
    return count


def _read_scalar(
    records: list[tuple[int, str]],
    key: str,
    cap: int,
    what: str,
    *,
    default: int | None = None,
) -> tuple[int, int]:
    """Return the integer a ``KEY=`` record declares and the record's index.

    The index rides along because every caller needs it to find the block the
    record opens, and searching for the same key twice is wasted work.
    """
    idx = _find_key(records, key)
    if idx < 0:
        if default is None:
            raise CodecError(f".su2: missing {key} record.")
        return default, idx
    return _checked_count(_value_of(records[idx][1]), cap, what), idx


def _read_cells(
    records: list[tuple[int, str]],
    start: int,
    count: int,
    what: str,
    *,
    stacklevel: int = 3,
) -> tuple[list[int], list[int], list[int], int]:
    """Read ``count`` element records, returning them and the next index.

    Returns
    -------
    tuple
        ``(connectivity, widths, type codes, next index)``.

    Raises
    ------
    CodecError
        If the block runs out of lines, a new record opens inside it, or an
        element line is malformed, carries too few nodes, or names a type
        this codec does not know.
    """
    conn: list[int] = []
    widths: list[int] = []
    codes: list[int] = []
    # Counted rather than collected: a corrupt file can hold millions of these
    # and only their number and the first one are ever reported.
    overlong = 0
    first_overlong = 0
    i = start
    for k in range(count):
        if i >= len(records):
            raise CodecError(
                f".su2: file truncated — {what} declares {count} element(s), found {k}."
            )
        no, line = records[i]
        opened = _key_of(line)
        if opened is not None:
            raise CodecError(
                f".su2: {what} declares {count} element(s) but only {k} are"
                f" listed; line {no} opens {opened}=."
            )
        parts = line.split()
        try:
            vtk_type = int(parts[0])
        except ValueError as exc:
            raise CodecError(
                f".su2: malformed element type on line {no}: {line!r}."
            ) from exc
        n_nodes = _NODES_PER_TYPE.get(vtk_type)
        if n_nodes is None:
            raise CodecError(f".su2: unsupported element type {vtk_type} on line {no}.")
        if len(parts) - 1 < n_nodes:
            # SU2 closes an element line with an optional element index, so a
            # line may be longer than the type needs — never shorter.
            raise CodecError(
                f".su2: element line {no} carries {len(parts) - 1} node(s),"
                f" expected {n_nodes} for type {vtk_type}."
            )
        if len(parts) - 1 > n_nodes + 1:
            # One trailing token is the element index SU2 allows; more than
            # that means the tokens being dropped are not an index, and the
            # likeliest reason is a type code that understates the element —
            # a quad spelled 5 reads as a triangle with a node quietly gone.
            if not overlong:
                first_overlong = no
            overlong += 1
        try:
            nodes = [int(tok) for tok in parts[1 : 1 + n_nodes]]
        except ValueError as exc:
            raise CodecError(
                f".su2: malformed node reference on line {no}: {line!r}."
            ) from exc
        conn.extend(nodes)
        widths.append(n_nodes)
        codes.append(ELEMENT_TYPES[_SU2_TO_POLYXIOS[vtk_type]])
        i += 1
    if overlong:
        warnings.warn(
            f".su2: {overlong} {what} element line(s) carry more tokens than the"
            f" type needs plus an index, first at line {first_overlong}; the"
            " extra tokens were ignored.",
            stacklevel=stacklevel,
        )
    return conn, widths, codes, i


def _warn_extra(
    records: list[tuple[int, str]], start: int, what: str, *, stacklevel: int = 3
) -> None:
    """Warn about data lines left over between a block and the next record.

    A block holds exactly as many lines as its count declares, so anything
    after it and before the next ``KEY=`` is data this reader will never see.
    Dropping it quietly would report a smaller mesh than the file holds.
    """
    extra = 0
    for i in range(start, len(records)):
        if _key_of(records[i][1]) is not None:
            break
        extra += 1
    if extra:
        warnings.warn(
            f".su2: {extra} line(s) follow the {what} block past the count it"
            f" declares, first at line {records[start][0]}; ignored.",
            stacklevel=stacklevel,
        )


def _read_points(
    records: list[tuple[int, str]],
    start: int,
    count: int,
    ndim: int,
) -> tuple[np.ndarray, int]:
    """Read ``count`` node records, returning them and the next index.

    Raises
    ------
    CodecError
        If the block runs out of lines, a new record opens inside it, or a
        node line is malformed or carries fewer than ``NDIME`` coordinates.
    """
    coords = np.zeros((count, 3), dtype=np.float64)
    # Counted, not collected: a corrupt file can hold millions of these and
    # only their number and the first one are ever reported.
    overlong = 0
    first_overlong = 0
    i = start
    for k in range(count):
        if i >= len(records):
            raise CodecError(
                f".su2: file truncated — NPOIN declares {count} node(s), found {k}."
            )
        no, line = records[i]
        opened = _key_of(line)
        if opened is not None:
            raise CodecError(
                f".su2: NPOIN declares {count} node(s) but only {k} are listed;"
                f" line {no} opens {opened}=."
            )
        parts = line.split()
        if len(parts) < ndim:
            # A node line closes with an optional point index, so extra tokens
            # are ordinary; missing coordinates are not.
            raise CodecError(
                f".su2: node line {no} carries {len(parts)} value(s), expected"
                f" at least NDIME={ndim}."
            )
        if len(parts) > ndim + 1:
            # One trailing token is the point index SU2 allows; more than that
            # means a coordinate is being dropped, which on an NDIME= 2 file is
            # the z of a mesh that was never flat.
            if not overlong:
                first_overlong = no
            overlong += 1
        try:
            coords[k, :ndim] = [float(tok) for tok in parts[:ndim]]
        except ValueError as exc:
            raise CodecError(f".su2: malformed node line {no}: {line!r}.") from exc
        i += 1
    if overlong:
        warnings.warn(
            f".su2: {overlong} node line(s) carry more values than NDIME={ndim}"
            f" plus an index, first at line {first_overlong}; the extra values"
            " were ignored.",
            stacklevel=3,
        )
    return coords, i


def _read_markers(
    records: list[tuple[int, str]],
    n_marks: int,
    start: int,
    first_elem: int,
) -> tuple[list[int], list[int], list[int], dict[str, list[int]]]:
    """Read every marker block, returning its cells and the tags over them.

    Parameters
    ----------
    records
        The file's meaningful lines.
    n_marks
        How many markers ``NMARK`` declares.
    start
        Index to start searching for the first ``MARKER_TAG``.
    first_elem
        Element index the first marker element will take, i.e. how many
        volume elements were read before them.
    """
    conn: list[int] = []
    widths: list[int] = []
    codes: list[int] = []
    tags: dict[str, list[int]] = {}
    unnamed = 0
    merged: list[str] = []
    next_elem = first_elem
    i = start

    for m in range(n_marks):
        tag_idx = _find_key(records, "MARKER_TAG", start=i)
        if tag_idx < 0:
            raise CodecError(
                f".su2: NMARK declares {n_marks} marker(s) but only {m} carry a"
                " MARKER_TAG=."
            )
        name = _value_of(records[tag_idx][1])
        # A quoted tag reads back with its quotes glued on, which is a
        # different name than the one the file meant.
        if len(name) >= 2 and name[0] in "\"'" and name[-1] == name[0]:
            name = name[1:-1].strip()

        # The count has to be this marker's own: searching past the next
        # MARKER_TAG would read the following marker's block under this
        # marker's name and lose that marker whole.
        next_tag = _find_key(records, "MARKER_TAG", start=tag_idx + 1)
        limit = len(records) if next_tag < 0 else next_tag
        elems_idx = _find_key(records, "MARKER_ELEMS", start=tag_idx + 1)
        if not 0 <= elems_idx < limit:
            raise CodecError(f".su2: marker {name or m + 1} declares no MARKER_ELEMS=.")
        n_elems = _checked_count(
            _value_of(records[elems_idx][1]), MAX_SAFE_ELEMENTS, "marker element"
        )
        block_conn, block_widths, block_codes, i = _read_cells(
            records, elems_idx + 1, n_elems, f"marker {name or m + 1}", stacklevel=4
        )
        _warn_extra(records, i, f"marker {name or m + 1}", stacklevel=4)
        conn.extend(block_conn)
        widths.extend(block_widths)
        codes.extend(block_codes)

        members = list(range(next_elem, next_elem + len(block_widths)))
        next_elem += len(block_widths)

        if not name:
            unnamed += 1
            name = f"marker_{m + 1}"
        if name in tags:
            # Two blocks under one name is not legal SU2; keeping only the
            # last would drop the first block's elements from every tag.
            merged.append(name)
        tags.setdefault(name, []).extend(members)

    trailing = _find_key(records, "MARKER_TAG", start=i)
    if trailing >= 0:
        # An under-declared NMARK is the mirror of an over-declared one: the
        # blocks past it are markers the file means, and dropping them in
        # silence would return a mesh missing part of its boundary.
        warnings.warn(
            f".su2: NMARK declares {n_marks} marker(s) but the file holds more,"
            f" first at line {records[trailing][0]}; the rest were not read.",
            stacklevel=3,
        )
    if unnamed:
        warnings.warn(
            f".su2: {unnamed} marker(s) declare an empty MARKER_TAG=; named"
            " them marker_<n>.",
            stacklevel=3,
        )
    if merged:
        warnings.warn(
            f".su2: marker name(s) {sorted(set(merged))} appear more than once;"
            " their elements were merged into one tag.",
            stacklevel=3,
        )
    return conn, widths, codes, tags


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse an SU2 mesh file and return a PolyData.

    Parameters
    ----------
    path
        Path to the .su2 file.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData
        Volume elements first, then the elements of each marker in file order.

    Raises
    ------
    CodecError
        On a missing or out-of-range ``NDIME``, a missing, negative or absurd
        count, a truncated or malformed section, an unsupported element type,
        or a node reference outside the declared node array.

    Notes
    -----
    Each ``MARKER_TAG`` becomes an ``element_tags`` entry over the elements
    that marker contributed. A marker with an empty name is tagged
    ``marker_<n>``, and two markers sharing a name are merged into one tag;
    both are warned about, as are marker blocks past the count ``NMARK``
    declares, which are not read. A file that declares ``NZONE`` above one is
    read down to its first zone, an element line carrying more tokens than its
    type needs plus an index has the rest ignored, and ``NMARK= 0`` beside
    ``MARKER_TAG`` blocks is believed — each with a warning. So is a node line
    carrying more values than ``NDIME`` plus an index.
    """
    if lazy:
        warnings.warn(
            ".su2: lazy=True is not supported; loading eagerly.", stacklevel=2
        )

    # 'utf-8-sig' so a byte-order mark cannot glue itself to NDIME and hide the
    # first record; errors="replace" keeps a file written in some other 8-bit
    # encoding inside a CodecError, since only a marker name can be hurt.
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    records = _scan(text)

    ndim_idx = _find_key(records, "NDIME")
    if ndim_idx < 0:
        raise CodecError(".su2: missing NDIME record.")
    raw_ndim = _value_of(records[ndim_idx][1]).split()
    try:
        ndim = int(raw_ndim[0])
    except (IndexError, ValueError) as exc:
        raise CodecError(
            f".su2: non-integer NDIME {_value_of(records[ndim_idx][1])!r}."
        ) from exc
    if ndim not in (2, 3):
        raise CodecError(f".su2: NDIME={ndim}, expected 2 or 3.")

    # A multi-zone file repeats every section once per zone, and this reader
    # matches the first of each; saying which zone was kept beats returning one
    # zone of several as if it were the whole file.
    n_zones, _ = _read_scalar(records, "NZONE", MAX_SAFE_ELEMENTS, "zone", default=1)
    if n_zones > 1:
        warnings.warn(
            f".su2: the file declares NZONE= {n_zones}; only the first zone was read.",
            stacklevel=2,
        )

    n_points, point_idx = _read_scalar(records, "NPOIN", MAX_SAFE_VERTICES, "node")
    n_elems, elem_idx = _read_scalar(records, "NELEM", MAX_SAFE_ELEMENTS, "element")

    conn, widths, codes, after_elems = _read_cells(
        records, elem_idx + 1, n_elems, "NELEM"
    )
    _warn_extra(records, after_elems, "NELEM")

    vertices, after_points = _read_points(records, point_idx + 1, n_points, ndim)
    _warn_extra(records, after_points, "NPOIN")

    element_tags: dict[str, np.ndarray] = {}
    n_marks, mark_idx = _read_scalar(
        records, "NMARK", MAX_SAFE_ELEMENTS, "marker", default=0
    )
    # A file that reads no marker is worth a word only if it holds one, and
    # both ways of reaching that state count zero of them. The search starts
    # past NMARK — at 0 when there is none, since ``mark_idx`` is -1 there —
    # so a file that does declare its markers never pays for the scan.
    if n_marks == 0 and _find_key(records, "MARKER_TAG", start=mark_idx + 1) >= 0:
        if mark_idx >= 0:
            # NMARK= 0 next to MARKER_TAG blocks is the file contradicting
            # itself. The count is still believed — it is the record SU2 reads
            # — but going quiet here would drop the boundary without a word.
            warnings.warn(
                ".su2: the file declares NMARK= 0 but holds MARKER_TAG blocks;"
                " the declared count was believed and the markers were not read.",
                stacklevel=2,
            )
        else:
            # A file that forgot NMARK still means its markers to be read;
            # saying so beats returning a mesh quietly missing its whole
            # boundary. A file that declares NMARK= 0 has said there are none.
            warnings.warn(
                ".su2: the file holds MARKER_TAG blocks but declares no NMARK=;"
                " the markers were not read.",
                stacklevel=2,
            )
    if n_marks and mark_idx >= 0:
        mark_conn, mark_widths, mark_codes, tags = _read_markers(
            records, n_marks, mark_idx + 1, len(widths)
        )
        conn.extend(mark_conn)
        widths.extend(mark_widths)
        codes.extend(mark_codes)
        # NELEM and every marker are capped one at a time, so only their sum
        # can walk past the cap, which is what keeps an element_tags entry
        # inside the int32 it is spelled in.
        if len(widths) > MAX_SAFE_ELEMENTS:
            raise CodecError(
                f".su2: element count exceeds the safety cap {MAX_SAFE_ELEMENTS}."
            )
        element_tags = {
            name: np.array(members, dtype=np.int32) for name, members in tags.items()
        }

    if len(conn) > MAX_SAFE_CONN:
        raise CodecError(f".su2: connectivity exceeds the safety cap {MAX_SAFE_CONN}.")

    # Node references stay under the vertex cap, but the running offset is a
    # cumulative sum MAX_SAFE_CONN lets past 2**31, where int32 would overflow;
    # widen the pair together so the CSR dtypes stay consistent.
    idx_dtype = np.int64 if len(conn) > np.iinfo(np.int32).max else np.int32
    connectivity = np.array(conn, dtype=idx_dtype)
    if connectivity.size:
        lo, hi = int(connectivity.min()), int(connectivity.max())
        if lo < 0 or hi >= n_points:
            # A reference past the node array would index out of the vertex
            # array on the first transform that touched it.
            where = f"0..{n_points - 1}" if n_points else "an empty node array"
            raise CodecError(
                f".su2: element references node {lo}..{hi}, outside {where}."
            )

    offsets = np.zeros(len(widths) + 1, dtype=idx_dtype)
    if widths:
        offsets[1:] = np.cumsum(widths, dtype=np.int64)

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=np.array(codes, dtype=np.uint8),
        element_tags=element_tags,
    )


def _safe_marker_name(name: str) -> str:
    """Return a marker name that survives its own line in the file."""
    return _UNSAFE_NAME_RE.sub("_", str(name).strip()) or _DEFAULT_MARKER


def _unique_name(name: str, used: set[str]) -> str:
    """Return ``name`` made unique against ``used``, which it is added to."""
    candidate, k = name, 1
    while candidate in used:
        k += 1
        candidate = f"{name}_{k}"
    used.add(candidate)
    return candidate


def _element_nodes(poly: PolyData, index: int, name: str, n_verts: int) -> list[int]:
    """Return one element's node references, checked against its type.

    Raises
    ------
    CodecError
        If the element's node count does not match its type — SU2 records no
        width, so a wrong one silently steals the next element's nodes — or a
        reference falls outside the vertex array.
    """
    start, end = int(poly.offsets[index]), int(poly.offsets[index + 1])
    expected = _NODES_PER_TYPE[_POLYXIOS_TO_SU2[name]]
    if end - start != expected:
        raise CodecError(
            f".su2: element {index} has {end - start} node(s), expected"
            f" {expected} for {name!r}."
        )
    nodes = [int(v) for v in poly.connectivity[start:end]]
    for nid in nodes:
        if not 0 <= nid < n_verts:
            where = f"0..{n_verts - 1}" if n_verts else "an empty vertex array"
            raise CodecError(
                f".su2: element {index} references vertex {nid}, outside {where}."
            )
    return nodes


def _split_by_dimension(
    poly: PolyData,
) -> tuple[list[int], list[int], dict[int, str], int]:
    """Sort the mesh into SU2's volume cells and boundary cells.

    Returns
    -------
    tuple
        ``(volume element indices, boundary element indices, index to type
        name, mesh dimension)``. Elements of a type SU2 cannot spell, and
        elements more than one dimension below the volume cells, are in
        neither list. The mesh dimension is 0 when no element survives.
    """
    names: dict[int, str] = {}
    skipped: set[str] = set()
    for i in range(len(poly.element_types)):
        code = int(poly.element_types[i])
        name = ELEMENT_TYPES_INV.get(code, "")
        if name in _POLYXIOS_TO_SU2:
            names[i] = name
        else:
            skipped.add(name or f"code {code}")
    if skipped:
        warnings.warn(
            f".su2: unsupported element type(s) {sorted(skipped)}; skipped."
            " SU2 knows line, triangle, quad, tetra, hexahedron, wedge and"
            " pyramid.",
            stacklevel=3,
        )

    if not names:
        return [], [], names, 0

    mesh_dim = max(_ELEMENT_DIM[name] for name in names.values())
    volume = [i for i, name in names.items() if _ELEMENT_DIM[name] == mesh_dim]
    boundary = [i for i, name in names.items() if _ELEMENT_DIM[name] == mesh_dim - 1]

    # A marker holds the cells exactly one dimension below the volume ones, so
    # an edge of a tetrahedral mesh has nowhere to go: writing it under a
    # marker anyway spells a line element inside an NDIME= 3 marker block,
    # which SU2 refuses to read back.
    deep = sorted(
        {name for name in names.values() if _ELEMENT_DIM[name] < mesh_dim - 1}
    )
    if deep:
        warnings.warn(
            f".su2: element type(s) {deep} sit more than one dimension below"
            f" the {mesh_dim}-D volume cells; SU2 marks only the dimension"
            " directly below them, so they were dropped.",
            stacklevel=3,
        )
    return volume, boundary, names, mesh_dim


def _marker_groups(poly: PolyData, boundary: list[int]) -> dict[str, list[int]]:
    """Group the boundary elements into the markers that will hold them.

    An element belongs to one marker: SU2 lists a marker's connectivity in
    full, so writing an element under two markers would read back as two
    elements. A tag naming one element twice is the same trap, so a tag is
    read as the set of elements it names. Boundary elements no tag claims
    still have to travel, so they land in a default marker of their own.
    """
    n_elems = len(poly.element_types)
    boundary_set = set(boundary)
    groups: dict[str, list[int]] = {}
    claimed: set[int] = set()
    used: set[str] = set()
    shared = 0
    empty: list[str] = []
    out_of_range: dict[str, int] = {}
    lost: list[str] = []
    dropped: dict[str, int] = {}
    renamed: list[tuple[str, str]] = []

    for name, members in (poly.element_tags or {}).items():
        # dict.fromkeys, not set(), so a tag that names an element twice is
        # written once without its members losing their order.
        idx = list(dict.fromkeys(int(v) for v in np.asarray(members).ravel()))
        if not idx:
            # A tag that names nothing is what reading MARKER_ELEMS= 0 leaves
            # behind, and is a different fault from one whose members SU2
            # cannot mark; saying the latter would send the reader hunting.
            empty.append(str(name))
            continue
        in_mesh = [i for i in idx if 0 <= i < n_elems]
        if len(in_mesh) < len(idx):
            # An index past the element array names no element at all, so
            # reporting it as a volume element would name the wrong fault.
            out_of_range[str(name)] = len(idx) - len(in_mesh)
        on_boundary = [i for i in in_mesh if i in boundary_set]
        if not on_boundary:
            if in_mesh:
                lost.append(str(name))
            continue
        if len(on_boundary) < len(in_mesh):
            # SU2 has no room for the rest: a volume element belongs under
            # NELEM, where no marker reaches it.
            dropped[str(name)] = len(in_mesh) - len(on_boundary)
        fresh = [i for i in on_boundary if i not in claimed]
        shared += len(on_boundary) - len(fresh)
        if not fresh:
            # A tag every one of whose members another marker already claims
            # is reported as sharing, not as naming nothing markable.
            continue
        claimed.update(fresh)
        safe = _unique_name(_safe_marker_name(name), used)
        if safe != str(name):
            renamed.append((str(name), safe))
        groups[safe] = sorted(fresh)

    orphans = sorted(boundary_set - claimed)
    if orphans:
        groups[_unique_name(_DEFAULT_MARKER, used)] = orphans

    if renamed:
        warnings.warn(
            ".su2: a marker name holds a line to itself and cannot carry"
            f" whitespace, '%' or a repeat; wrote {sorted(renamed)}.",
            stacklevel=3,
        )
    if shared:
        warnings.warn(
            f".su2: {shared} element(s) belong to more than one tag; each was"
            " written under the first marker that claims it.",
            stacklevel=3,
        )
    if lost:
        warnings.warn(
            f".su2: tag(s) {sorted(lost)} name no boundary element; SU2 marks"
            " only the elements one dimension below the volume cells, so they"
            " were dropped.",
            stacklevel=3,
        )
    if dropped:
        warnings.warn(
            f".su2: tag(s) {sorted(dropped)} also name {sum(dropped.values())}"
            " element(s) that are not boundary elements; SU2 marks only the"
            " elements one dimension below the volume cells, so those members"
            " were dropped.",
            stacklevel=3,
        )
    if empty:
        warnings.warn(
            f".su2: tag(s) {sorted(empty)} name no element; dropped.",
            stacklevel=3,
        )
    if out_of_range:
        warnings.warn(
            f".su2: tag(s) {sorted(out_of_range)} name"
            f" {sum(out_of_range.values())} index(es) outside the mesh's"
            f" {n_elems} element(s); those members were dropped.",
            stacklevel=3,
        )
    return groups


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Serialise PolyData to an SU2 mesh file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .su2 path.
    **opts
        None are recognized; any that are passed are warned about and ignored.

    Raises
    ------
    CodecError
        If an element's node count does not match its type, or an element
        references a vertex that does not exist.

    Notes
    -----
    Elements are split by dimension the way SU2 expects: the highest-dimension
    elements are written under ``NELEM``, and the ones below them under a
    ``MARKER_TAG`` block, so a mesh does not gain a copy of every tagged
    element per round trip. A marker holds the dimension directly below the
    volume cells and nothing lower, so an edge of a tetrahedral mesh is dropped
    with a warning. An ``element_tags`` entry naming only volume elements has
    nowhere to go and is dropped with a warning, as are the volume members of a
    tag that names both and an element of a type SU2 cannot spell.
    A tag naming one element twice marks it once, a tag naming no element at
    all is dropped, and tag members outside the element array are dropped —
    each with a warning of its own. Boundary elements no tag claims are
    gathered into a marker named ``unnamed``, and whitespace or ``%`` in a
    marker name becomes ``_`` since SU2 gives the name a line of its own.

    ``NDIME`` follows the mesh: 3 for volume cells or a mesh with a z extent,
    2 otherwise. A mesh whose highest elements sit below that — a curved
    surface, a mesh of lines — is written whole and reads back whole here, but
    SU2 fills an ``NDIME``-dimensional domain with cells of that dimension and
    will refuse the file, so it is warned about.

    The format carries no fields, so ``vertex_attrs``, ``element_attrs``,
    ``vertex_tags`` and ``global_attrs`` are not written.
    """
    if opts:
        warnings.warn(
            f".su2 write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )

    path = Path(path)
    n_verts = poly.vertices.shape[0]

    volume, boundary, names, mesh_dim = _split_by_dimension(poly)
    groups = _marker_groups(poly, boundary)

    # A mesh of 3-D cells is 3-D whatever its coordinates say, and a flat one
    # is 2-D: SU2 reads exactly NDIME coordinates per node, so declaring 2 for
    # a mesh with a z extent would flatten it.
    spatial = bool(n_verts) and bool(np.any(poly.vertices[:, 2] != 0))
    ndim = 3 if mesh_dim == 3 or spatial else 2
    if volume and mesh_dim < ndim:
        # SU2 fills an NDIME-dimensional domain with NDIME-dimensional cells,
        # so a surface with a z extent has no home: NDIME= 2 would flatten it
        # and NDIME= 3 puts faces where SU2 looks for volume cells. The file
        # reads back here whole, which is what keeps it worth writing.
        warnings.warn(
            f".su2: the mesh's highest elements are {mesh_dim}-D and were"
            f" written under NELEM of an NDIME= {ndim} file; SU2 fills an"
            f" NDIME= {ndim} domain with {ndim}-D cells and will refuse it."
            " polyxios reads the file back whole.",
            stacklevel=2,
        )

    lines: list[str] = [f"NDIME= {ndim}", f"NELEM= {len(volume)}"]
    for seq, i in enumerate(volume):
        nodes = _element_nodes(poly, i, names[i], n_verts)
        node_str = " ".join(str(nid) for nid in nodes)
        lines.append(f"{_POLYXIOS_TO_SU2[names[i]]} {node_str} {seq}")

    lines.append(f"NPOIN= {n_verts}")
    lines.extend(
        "\t".join(f"{c:.10g}" for c in vertex[:ndim]) for vertex in poly.vertices
    )

    lines.append(f"NMARK= {len(groups)}")
    for name, members in groups.items():
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {len(members)}")
        for i in members:
            nodes = _element_nodes(poly, i, names[i], n_verts)
            node_str = " ".join(str(nid) for nid in nodes)
            lines.append(f"{_POLYXIOS_TO_SU2[names[i]]} {node_str}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
