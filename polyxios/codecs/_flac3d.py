"""FLAC3D .f3grid ASCII codec - read + write."""

from array import array
import io
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._io import Source, read_text, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".f3grid"

# Record keywords. FLAC3D writes the single-letter form; the long form is
# accepted for hand-written files.
_GRIDPOINT_KW: frozenset[str] = frozenset({"G", "GRIDPOINT"})
_ZONE_KW: frozenset[str] = frozenset({"Z", "ZONE"})
_FACE_KW: frozenset[str] = frozenset({"F", "FACE"})
_ZGROUP_KW: frozenset[str] = frozenset({"ZGROUP", "GROUP"})
_FACE_GROUP_KW: frozenset[str] = frozenset({"FGROUP"})

# FLAC3D zone type codes → (polyxios name, n_ids in the record)
# B7 is a hexahedron with a collapsed corner; FLAC3D stores only 7 ids and the
# eighth corner repeats the seventh.
_ZONE_TO_POLYXIOS: dict[str, tuple[str, int]] = {
    "T4": ("tetra", 4),
    "DT4": ("tetra", 4),
    "P5": ("pyramid", 5),
    "W6": ("wedge", 6),
    "B7": ("hexahedron", 7),
    "B8": ("hexahedron", 8),
}
_POLYXIOS_TO_ZONE: dict[str, str] = {
    "tetra": "T4",
    "pyramid": "P5",
    "wedge": "W6",
    "hexahedron": "B8",
}

# FLAC3D face type codes → (polyxios name, n_ids in the record). Faces are the
# 2D records; FLAC3D keeps them in a section of their own, apart from zones.
_FACE_TO_POLYXIOS: dict[str, tuple[str, int]] = {
    "T3": ("triangle", 3),
    "Q4": ("quad", 4),
}
_POLYXIOS_TO_FACE: dict[str, str] = {
    "triangle": "T3",
    "quad": "Q4",
}

# Corner ordering. FLAC3D does not use the VTK corner order for P5/W6/B8,
# so connectivity must be permuted in both directions. T3/Q4 already agree.
_READ_ORDER: dict[str, list[int]] = {
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 4, 2, 3],
    "wedge": [0, 1, 3, 2, 4, 5],
    "hexahedron": [0, 1, 4, 2, 3, 6, 7, 5],
}
_WRITE_ORDER: dict[str, list[int]] = {
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 3, 4, 2],
    "wedge": [0, 1, 3, 2, 4, 5],
    "hexahedron": [0, 1, 3, 4, 2, 7, 5, 6],
}
# Used when the element is inverted: FLAC3D requires zones with a positive
# volume (outward face normals).
_WRITE_ORDER_FLIPPED: dict[str, list[int]] = {
    "tetra": [0, 2, 1, 3],
    "pyramid": [0, 3, 1, 4, 2],
    "wedge": [0, 2, 3, 1, 5, 4],
    "hexahedron": [0, 3, 1, 4, 2, 5, 7, 6],
}

# Element boundary faces in polyxios (VTK) corner order, wound so the normals
# point outward for a positively oriented element. Used to compute the signed
# volume that decides whether a zone must be flipped on write.
_FACES: dict[str, tuple[tuple[int, ...], ...]] = {
    "tetra": ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    "pyramid": ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    # Note: VTK's own wedge face table is wound inward, unlike its tetra,
    # pyramid and hexahedron tables. These faces are the reversed (outward) form.
    "wedge": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
    "hexahedron": (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}

# Record ids per line inside a written ZGROUP/FGROUP block.
_GROUP_IDS_PER_LINE: int = 10

# Text encoding. FLAC3D writes plain ASCII, but group names in files produced by
# other tools are UTF-8; utf-8-sig also drops a byte-order mark, which would
# otherwise hide the first record of the file.
_READ_ENCODING: str = "utf-8-sig"
_WRITE_ENCODING: str = "utf-8"

# A zone counts as degenerate when its volume falls below this fraction of the
# cube of its bounding-box diagonal.
_DEGENERATE_REL_TOL: float = 1e-12


def _extent_cubed(
    px: list[float],
    py: list[float],
    pz: list[float],
    nodes: list[int],
) -> float:
    """Cube of the bounding-box diagonal of an element.

    Gives ``_signed_volume`` a length scale, so the zero-volume test is
    relative to element size instead of an absolute float comparison.

    Parameters
    ----------
    px, py, pz
        Per-axis vertex coordinates.
    nodes
        Vertex indices of the element.

    Returns
    -------
    float
    """
    total = 0.0
    for axis in (px, py, pz):
        lo = hi = axis[nodes[0]]
        for n in nodes:
            v = axis[n]
            if v < lo:
                lo = v
            elif v > hi:
                hi = v
        total += (hi - lo) ** 2
    return total**1.5


def _strip_comment(line: str) -> str:
    """Drop a trailing ``*`` comment, ignoring ``*`` inside quoted names.

    Parameters
    ----------
    line
        Raw file line.

    Returns
    -------
    str
        The line up to the first unquoted ``*``.
    """
    if "*" not in line:
        return line
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == "*" and not quoted:
            return line[:i]
    return line


def _split_quoted(line: str) -> list[str]:
    """Split on whitespace, keeping double-quoted spans together.

    Parameters
    ----------
    line
        Line to split. Quote characters are dropped from the result.

    Returns
    -------
    list of str
        Tokens. A quoted span always yields one token, empty ones included, so
        that ``ZGROUP "" SLOT 1`` does not read ``SLOT`` as the group name.
    """
    tokens: list[str] = []
    buf: list[str] = []
    quoted = False
    for ch in line:
        if ch == '"':
            if quoted:
                tokens.append("".join(buf))
                buf = []
            quoted = not quoted
        elif quoted or not ch.isspace():
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def _group_name(line: str, *, after: str) -> str:
    """Return the token that follows ``after`` on a record line.

    Parameters
    ----------
    line
        Record line, comment already stripped.
    after
        Upper-case keyword the name follows (``ZGROUP``, ``GROUP``, ...).

    Returns
    -------
    str
        Group name, or ``""`` when the keyword carries no name.
    """
    tokens = _split_quoted(line) if '"' in line else line.split()
    for i, tok in enumerate(tokens):
        if tok.upper() == after and i + 1 < len(tokens):
            return tokens[i + 1]
    return ""


def _signed_volume(
    px: list[float],
    py: list[float],
    pz: list[float],
    nodes: list[int],
    faces: tuple[tuple[int, ...], ...],
) -> float:
    """Six times the signed volume of an element, from its boundary faces.

    Positive when the element is oriented as polyxios expects; negative when
    it is inverted (mirrored), which FLAC3D rejects. Computed over the whole
    boundary rather than from one corner frame, so a distorted but valid
    hexahedron or wedge is not mistaken for an inverted one.

    Parameters
    ----------
    px, py, pz
        Per-axis vertex coordinates.
    nodes
        Vertex indices of the element, in polyxios corner order.
    faces
        Boundary faces as tuples of local corner indices, wound outward.

    Returns
    -------
    float
    """
    ox, oy, oz = px[nodes[0]], py[nodes[0]], pz[nodes[0]]
    total = 0.0
    for face in faces:
        a = nodes[face[0]]
        ax, ay, az = px[a] - ox, py[a] - oy, pz[a] - oz
        for k in range(1, len(face) - 1):
            b, c = nodes[face[k]], nodes[face[k + 1]]
            bx, by, bz = px[b] - ox, py[b] - oy, pz[b] - oz
            cx, cy, cz = px[c] - ox, py[c] - oy, pz[c] - oz
            total += (
                ax * (by * cz - bz * cy)
                - ay * (bx * cz - bz * cx)
                + az * (bx * cy - by * cx)
            )
    return total


# Ids are held and matched as int64, which is as wide as the arrays that
# collect them and the numpy sort that resolves them against each other.
_MAX_ID: int = 2**63 - 1


def _record_id(field: str) -> int:
    """Return an id a record names, refusing one no machine integer holds.

    Parameters
    ----------
    field
        The id as the record spells it.

    Returns
    -------
    int
        The id.

    Raises
    ------
    ValueError
        On a field that is not an integer, which the caller reports as a
        non-numeric record.
    CodecError
        On an integer past the width every id here is held at. Python counts
        as high as memory allows and the arrays and the numpy sort behind
        them do not, so an id that overflows would otherwise surface as an
        OverflowError from whichever of them reached it first.
    """
    value = int(field)
    if not -_MAX_ID - 1 <= value <= _MAX_ID:
        raise CodecError(f".f3grid: the id {field} does not fit in a 64-bit integer.")
    return value


def _resolve_gridpoints(
    conn_raw: "array[int]",
    node_map: dict[int, int],
) -> np.ndarray:
    """Map file gridpoint ids to zero-based vertex indices.

    Done after the whole file is read so a ZONE or FACE record may reference
    a GRIDPOINT declared later in the file.

    Parameters
    ----------
    conn_raw
        Flat connectivity holding file gridpoint ids.
    node_map
        Mapping from file gridpoint id to vertex index.

    Returns
    -------
    numpy.ndarray
        Flat connectivity of vertex indices, dtype int32.

    Raises
    ------
    CodecError
        If a gridpoint id is never declared.
    """
    if not len(conn_raw):
        return np.empty(0, dtype=np.int32)

    raw = np.frombuffer(conn_raw, dtype=np.int64)
    keys = np.fromiter(node_map.keys(), dtype=np.int64, count=len(node_map))
    vals = np.fromiter(node_map.values(), dtype=np.int64, count=len(node_map))
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]

    pos = np.searchsorted(keys, raw)
    np.clip(pos, 0, len(keys) - 1, out=pos)
    missing = keys[pos] != raw
    if missing.any():
        raise CodecError(
            f".f3grid: record references undefined GRIDPOINT id"
            f" {int(raw[int(missing.argmax())])}."
        )
    return vals[pos].astype(np.int32)


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a FLAC3D .f3grid ASCII file.

    Both the single-letter record keywords written by FLAC3D (``G``, ``Z``,
    ``F``) and the long forms (``GRIDPOINT``, ``ZONE``, ``FACE``) are
    accepted. Zone records become 3D elements (tetra, pyramid, wedge,
    hexahedron) and face records become 2D elements (triangle, quad), all in
    the order they appear in the file; corners are permuted from FLAC3D order
    to polyxios order.

    ``ZGROUP`` and ``FGROUP`` blocks and the inline ``GROUP "name"`` form of
    FLAC3D 7 become ``element_tags``. Group slots are not preserved: groups
    sharing a name in different slots merge into one tag, as do a zone group
    and a face group of the same name.

    Malformed records are counted and reported in a single warning per kind
    rather than one warning per line.

    Parameters
    ----------
    path
        Path to the .f3grid file.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData

    Raises
    ------
    CodecError
        If no gridpoint entries are found, if a record holds non-numeric
        fields, or if a record references an undefined gridpoint id.
    """
    if lazy:
        warnings.warn(
            ".f3grid: lazy=True ignored; ASCII format always loads eagerly.",
            stacklevel=2,
        )

    # Iterated rather than collected: a large grid runs to millions of lines,
    # and the list of them costs more than the file itself - str.splitlines()
    # would build every one of them up front and hold them all at once.
    # StringIO hands them over one at a time instead, so only the text and the
    # line in hand are held. errors="replace": the format is ASCII-spec, but
    # real files carry extended characters in comments and group names.
    lines = (
        stripped
        for ln in io.StringIO(
            read_text(path, encoding=_READ_ENCODING, errors="replace")
        )
        if (stripped := _strip_comment(ln).strip())
    )

    node_map: dict[int, int] = {}
    # array rather than list: each holds one machine number per entry instead
    # of a pointer to a boxed one, which is four times less for a coordinate
    # and the difference between a large grid fitting and not.
    coords: array[float] = array("d")
    conn_raw: array[int] = array("q")
    offsets_list: array[int] = array("q", [0])
    types_list: array[int] = array("B")
    # Zones and faces are numbered in separate id spaces in FLAC3D, so groups
    # of each kind resolve against their own index.
    record_index: dict[str, dict[int, int]] = {"ZONE": {}, "FACE": {}}
    block_groups: dict[str, dict[str, list[int]]] = {"ZONE": {}, "FACE": {}}
    n_dup: dict[str, int] = {"ZONE": 0, "FACE": 0}
    n_bad: dict[str, int] = {"ZONE": 0, "FACE": 0}
    first_bad: dict[str, str] = {"ZONE": "", "FACE": ""}
    unknown_types: dict[str, set[str]] = {"ZONE": set(), "FACE": set()}
    inline_groups: dict[str, list[int]] = {}
    current_group: str | None = None
    current_kind = "ZONE"
    seen_dt4 = False
    seen_b7 = False
    n_dup_gp = 0
    n_bad_gp = 0
    first_bad_gp = ""

    for ln in lines:
        parts = ln.split()
        kw = parts[0].upper()

        if kw in _GRIDPOINT_KW:
            current_group = None
            if len(parts) < 5:
                n_bad_gp += 1
                first_bad_gp = first_bad_gp or ln
                continue
            try:
                gp_id = _record_id(parts[1])
                xyz = [float(parts[2]), float(parts[3]), float(parts[4])]
            except ValueError as exc:
                raise CodecError(
                    f".f3grid: non-numeric GRIDPOINT record: {ln!r}."
                ) from exc
            slot = node_map.get(gp_id)
            if slot is None:
                node_map[gp_id] = len(coords) // 3
                coords.extend(xyz)
            else:
                # Redeclared id: last definition wins, no orphan vertex.
                coords[3 * slot : 3 * slot + 3] = array("d", xyz)
                n_dup_gp += 1

        elif kw in _ZONE_KW or kw in _FACE_KW:
            kind = "FACE" if kw in _FACE_KW else "ZONE"
            table = _FACE_TO_POLYXIOS if kind == "FACE" else _ZONE_TO_POLYXIOS
            current_group = None
            if len(parts) < 2:
                n_bad[kind] += 1
                first_bad[kind] = first_bad[kind] or ln
                continue
            rec_type = parts[1].upper()
            if rec_type == "DT4":
                seen_dt4 = True
            elif rec_type == "B7":
                seen_b7 = True
            if rec_type not in table:
                unknown_types[kind].add(rec_type)
                continue
            elem_name, n_ids = table[rec_type]
            if len(parts) < 3 + n_ids:
                n_bad[kind] += 1
                first_bad[kind] = first_bad[kind] or ln
                continue
            try:
                rec_id = _record_id(parts[2])
                ids = [_record_id(parts[3 + j]) for j in range(n_ids)]
            except ValueError as exc:
                raise CodecError(
                    f".f3grid: non-numeric {kind} record: {ln!r}."
                ) from exc
            if rec_type == "B7":
                # Collapsed hexahedron: the eighth corner repeats the seventh.
                ids.append(ids[-1])
            elem_idx = len(types_list)
            conn_raw.extend(ids[k] for k in _READ_ORDER[elem_name])
            offsets_list.append(offsets_list[-1] + len(ids))
            types_list.append(ELEMENT_TYPES[elem_name])
            if rec_id in record_index[kind]:
                n_dup[kind] += 1
            record_index[kind][rec_id] = elem_idx
            # FLAC3D 7 may append `GROUP "name"` to the record itself.
            tail = parts[3 + n_ids :]
            if tail and any(tok.upper() == "GROUP" for tok in tail):
                name = _group_name(ln, after="GROUP")
                if name:
                    inline_groups.setdefault(name, []).append(elem_idx)

        elif kw in _ZGROUP_KW or kw in _FACE_GROUP_KW:
            current_kind = "FACE" if kw in _FACE_GROUP_KW else "ZONE"
            current_group = _group_name(ln, after=kw) or None
            if current_group is not None:
                block_groups[current_kind].setdefault(current_group, [])

        elif current_group is not None:
            # Continuation line of a group block: bare record ids.
            try:
                member_ids = [int(tok) for tok in parts]
            except ValueError:
                # Not a member list - the block ended at the previous line.
                current_group = None
            else:
                block_groups[current_kind][current_group].extend(member_ids)

    if n_dup_gp:
        warnings.warn(
            f".f3grid: {n_dup_gp} duplicate GRIDPOINT id(s); last definition used.",
            stacklevel=2,
        )
    if n_bad_gp:
        warnings.warn(
            f".f3grid: {n_bad_gp} malformed GRIDPOINT line(s) skipped"
            f" (first: {first_bad_gp!r}).",
            stacklevel=2,
        )
    for kind in ("ZONE", "FACE"):
        if n_dup[kind]:
            warnings.warn(
                f".f3grid: {n_dup[kind]} duplicate {kind} id(s); groups referencing"
                f" them tag the last {kind.lower()} with that id.",
                stacklevel=2,
            )
        if n_bad[kind]:
            warnings.warn(
                f".f3grid: {n_bad[kind]} short {kind} line(s) skipped"
                f" (first: {first_bad[kind]!r}).",
                stacklevel=2,
            )
        if unknown_types[kind]:
            warnings.warn(
                f".f3grid: unknown {kind.lower()} type(s) skipped:"
                f" {sorted(unknown_types[kind])}.",
                stacklevel=2,
            )
    if seen_dt4:
        warnings.warn(
            ".f3grid: DT4 zone type normalised to T4 on read"
            " (lossy; write will emit T4).",
            stacklevel=2,
        )
    if seen_b7:
        warnings.warn(
            ".f3grid: B7 zone type read as a hexahedron with a repeated corner"
            " (lossy; write will emit B8).",
            stacklevel=2,
        )

    if not len(coords):
        raise CodecError(".f3grid: no GRIDPOINT entries found.")

    if not types_list:
        warnings.warn(
            ".f3grid: no recognised ZONE or FACE entries found;"
            " connectivity will be empty.",
            stacklevel=2,
        )

    members: dict[str, list[int]] = {}
    n_unresolved = 0
    for kind in ("ZONE", "FACE"):
        for name, member_ids in block_groups[kind].items():
            target = members.setdefault(name, [])
            for rid in member_ids:
                idx = record_index[kind].get(rid)
                if idx is None:
                    n_unresolved += 1
                else:
                    target.append(idx)
    for name, elem_ids in inline_groups.items():
        members.setdefault(name, []).extend(elem_ids)

    if n_unresolved:
        warnings.warn(
            f".f3grid: {n_unresolved} group member(s) reference zones or faces"
            " that were not read; dropped from element_tags.",
            stacklevel=2,
        )

    element_tags = {
        name: np.array(sorted(set(idxs)), dtype=np.int32)
        for name, idxs in members.items()
        if idxs
    }

    n_verts = len(coords) // 3
    # A view rather than a copy: the array is the only other reference to
    # these bytes and it dies with the call, so copying them would double the
    # peak for the largest array a grid holds and hand back the same numbers.
    vertices = np.frombuffer(coords, dtype=np.float64).reshape(n_verts, 3)

    return PolyData(
        vertices=vertices,
        connectivity=_resolve_gridpoints(conn_raw, node_map),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        element_tags=element_tags,
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write PolyData to FLAC3D .f3grid ASCII format.

    Records are written with the single-letter keywords FLAC3D itself emits
    (``G``, ``Z``, ``F``). 3D elements (tetra, pyramid, wedge, hexahedron)
    become zone records and 2D elements (triangle, quad) become face
    records; FLAC3D keeps the two in separate sections, so all zones are
    written before all faces and a mesh that interleaves them comes back
    from :func:`read` with the faces moved to the end.

    Corners are permuted from polyxios order to FLAC3D order; zones with a
    negative volume are reordered so the volume is positive, which means
    such elements come back from :func:`read` with a different (but
    geometrically equivalent) corner order.

    ``element_tags`` are written as ``ZGROUP`` blocks for their zone members
    and ``FGROUP`` blocks for their face members. Element types FLAC3D has
    no record for are skipped with a warning.

    Coordinates are written at full float64 precision, and the file is
    encoded UTF-8 to match :func:`read`.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .f3grid path.
    opts
        Ignored; accepted so the codec matches the generic write signature.

    Raises
    ------
    CodecError
        If offsets are inconsistent with the connectivity array, or a
        connectivity entry is out of range for the vertex array.
    """
    n_elems = len(poly.element_types)
    if len(poly.offsets) < n_elems + 1:
        raise CodecError(
            f".f3grid write: offsets length {len(poly.offsets)}"
            f" < n_elems + 1 ({n_elems + 1})."
        )
    if int(poly.offsets[-1]) > len(poly.connectivity):
        raise CodecError(
            f".f3grid write: offsets[-1] ({int(poly.offsets[-1])})"
            f" > connectivity length ({len(poly.connectivity)})."
        )

    verts = np.asarray(poly.vertices, dtype=np.float64)
    used = np.asarray(poly.connectivity[: int(poly.offsets[-1])])
    if used.size and (int(used.min()) < 0 or int(used.max()) >= len(verts)):
        raise CodecError(
            f".f3grid write: connectivity index out of range"
            f" [{int(used.min())}, {int(used.max())}] for {len(verts)} vertices."
        )

    malformed = 0
    first_malformed = ""
    degenerate = 0

    # Plain Python lists: the per-zone volume test below runs once per
    # element, where numpy scalar indexing costs more than float arithmetic.
    px, py, pz = (verts[:, k].tolist() for k in range(3))

    lines: list[str] = ["* FLAC3D grid exported by polyxios", "* GRIDPOINTS"]
    # repr() of a Python float is the shortest representation that round-trips
    # exactly; a fixed %g width would silently truncate float64 coordinates.
    lines.extend(
        f"G {i + 1}  {x!r}  {y!r}  {z!r}"
        for i, (x, y, z) in enumerate(zip(px, py, pz, strict=True))
    )

    conn = np.asarray(poly.connectivity, dtype=np.int64).tolist()
    offsets = np.asarray(poly.offsets, dtype=np.int64).tolist()
    elem_types = np.asarray(poly.element_types).tolist()
    names = [ELEMENT_TYPES_INV.get(code, "") for code in elem_types]
    skipped = sum(
        1
        for name in names
        if name not in _POLYXIOS_TO_ZONE and name not in _POLYXIOS_TO_FACE
    )

    # Global record id shared by zones and faces, so a ZGROUP id can never
    # collide with an FGROUP id.
    record_of_elem: list[int] = [0] * n_elems
    is_face_elem: list[bool] = [False] * n_elems
    record_id = 0

    zone_lines: list[str] = []
    for i, name in enumerate(names):
        if name not in _POLYXIOS_TO_ZONE:
            continue
        nodes = conn[offsets[i] : offsets[i + 1]]
        order = _WRITE_ORDER[name]
        if len(nodes) != len(order):
            malformed += 1
            first_malformed = first_malformed or f"{name} element {i}"
            continue
        vol = _signed_volume(px, py, pz, nodes, _FACES[name])
        # Scale-relative tolerance: an exact `== 0.0` test misses near-flat
        # zones and lets round-off noise flip them.
        tol = _DEGENERATE_REL_TOL * _extent_cubed(px, py, pz, nodes)
        if vol < -tol:
            order = _WRITE_ORDER_FLIPPED[name]
        elif vol <= tol:
            degenerate += 1
        record_id += 1
        record_of_elem[i] = record_id
        node_str = "  ".join(str(nodes[k] + 1) for k in order)
        zone_lines.append(f"Z  {_POLYXIOS_TO_ZONE[name]}  {record_id}  {node_str}")

    # Faces come after every zone: FLAC3D keeps them in a section of their own.
    face_lines: list[str] = []
    for i, name in enumerate(names):
        if name not in _POLYXIOS_TO_FACE:
            continue
        nodes = conn[offsets[i] : offsets[i + 1]]
        order = _WRITE_ORDER[name]
        if len(nodes) != len(order):
            malformed += 1
            first_malformed = first_malformed or f"{name} element {i}"
            continue
        record_id += 1
        record_of_elem[i] = record_id
        is_face_elem[i] = True
        node_str = "  ".join(str(nodes[k] + 1) for k in order)
        face_lines.append(f"F  {_POLYXIOS_TO_FACE[name]}  {record_id}  {node_str}")

    zone_group_lines: list[str] = []
    face_group_lines: list[str] = []
    empty_tags = 0
    dropped_members = 0
    for name, tag in poly.element_tags.items():
        members = [int(v) for v in np.asarray(tag).ravel()]
        written = [e for e in members if 0 <= e < n_elems and record_of_elem[e]]
        if not written:
            empty_tags += 1
            continue
        # Partial drops only; a tag that lost every member is reported above.
        dropped_members += len(members) - len(written)
        safe_name = str(name).replace('"', "'")
        for target, kw, ids in (
            (
                zone_group_lines,
                "ZGROUP",
                sorted({record_of_elem[e] for e in written if not is_face_elem[e]}),
            ),
            (
                face_group_lines,
                "FGROUP",
                sorted({record_of_elem[e] for e in written if is_face_elem[e]}),
            ),
        ):
            if not ids:
                continue
            # SLOT is mandatory in the FLAC3D group syntax; polyxios has no
            # slot concept, so every group is written to slot 1.
            target.append(f'{kw} "{safe_name}" SLOT 1')
            target.extend(
                "     "
                + " ".join(str(z) for z in ids[start : start + _GROUP_IDS_PER_LINE])
                for start in range(0, len(ids), _GROUP_IDS_PER_LINE)
            )

    for header, section in (
        ("* ZONES", zone_lines),
        ("* ZONE GROUPS", zone_group_lines),
        ("* FACES", face_lines),
        ("* FACE GROUPS", face_group_lines),
    ):
        if section:
            lines.append(header)
            lines.extend(section)

    if verts.size and not np.isfinite(verts).all():
        n_nonfinite = int((~np.isfinite(verts)).any(axis=1).sum())
        warnings.warn(
            f".f3grid write: {n_nonfinite} gridpoint(s) have non-finite"
            " coordinates; FLAC3D will not load the file.",
            stacklevel=2,
        )
    if degenerate:
        warnings.warn(
            f".f3grid write: {degenerate} zone(s) have zero volume within"
            " tolerance; FLAC3D will reject them.",
            stacklevel=2,
        )
    if malformed:
        warnings.warn(
            f".f3grid write: {malformed} element(s) with the wrong corner count"
            f" skipped (first: {first_malformed}).",
            stacklevel=2,
        )
    if skipped:
        warnings.warn(
            f".f3grid write: {skipped} element(s) skipped (unsupported type - only"
            " tetra/pyramid/wedge/hexahedron/triangle/quad written).",
            stacklevel=2,
        )
    if empty_tags:
        warnings.warn(
            f".f3grid write: {empty_tags} element tag(s) dropped;"
            " no member element was written.",
            stacklevel=2,
        )
    if dropped_members:
        warnings.warn(
            f".f3grid write: {dropped_members} tag member(s) dropped; they are"
            " out of range or refer to an element that was not written.",
            stacklevel=2,
        )

    lines.append("")
    write_text(path, "\n".join(lines), encoding=_WRITE_ENCODING)
