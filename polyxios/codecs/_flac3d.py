"""FLAC3D .f3grid ASCII codec — read + write."""

from pathlib import Path
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
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

# FLAC3D zone type codes → (polyxios name, n_nodes)
_ZONE_TO_POLYXIOS: dict[str, tuple[str, int]] = {
    "T4": ("tetra", 4),
    "DT4": ("tetra", 4),
    "P5": ("pyramid", 5),
    "W6": ("wedge", 6),
    "B8": ("hexahedron", 8),
}
_POLYXIOS_TO_ZONE: dict[str, str] = {
    "tetra": "T4",
    "pyramid": "P5",
    "wedge": "W6",
    "hexahedron": "B8",
}

# Corner ordering. FLAC3D does not use the VTK corner order for P5/W6/B8,
# so connectivity must be permuted in both directions.
_READ_ORDER: dict[str, list[int]] = {
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 4, 2, 3],
    "wedge": [0, 1, 3, 2, 4, 5],
    "hexahedron": [0, 1, 4, 2, 3, 6, 7, 5],
}
_WRITE_ORDER: dict[str, list[int]] = {
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

# Zone ids per line inside a written ZGROUP block.
_GROUP_IDS_PER_LINE: int = 10

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
        Tokens.
    """
    tokens: list[str] = []
    buf: list[str] = []
    quoted = False
    for ch in line:
        if ch == '"':
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


def _resolve_gridpoints(
    conn_raw: list[int],
    node_map: dict[int, int],
) -> np.ndarray:
    """Map file gridpoint ids to zero-based vertex indices.

    Done after the whole file is read so a ZONE record may reference a
    GRIDPOINT declared later in the file.

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
    if not conn_raw:
        return np.empty(0, dtype=np.int32)

    raw = np.array(conn_raw, dtype=np.int64)
    keys = np.fromiter(node_map.keys(), dtype=np.int64, count=len(node_map))
    vals = np.fromiter(node_map.values(), dtype=np.int64, count=len(node_map))
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]

    pos = np.searchsorted(keys, raw)
    np.clip(pos, 0, len(keys) - 1, out=pos)
    missing = keys[pos] != raw
    if missing.any():
        raise CodecError(
            f".f3grid: ZONE references undefined GRIDPOINT id"
            f" {int(raw[int(missing.argmax())])}."
        )
    return vals[pos].astype(np.int32)


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a FLAC3D .f3grid ASCII file.

    Both the single-letter record keywords written by FLAC3D (``G``, ``Z``)
    and the long forms (``GRIDPOINT``, ``ZONE``) are accepted. Zone corners
    are permuted from FLAC3D order to polyxios order. ``ZGROUP`` blocks and
    the inline ``GROUP "name"`` form of FLAC3D 7 become ``element_tags``.
    Face records (``F``) and ``FGROUP`` blocks are skipped with a warning.
    Group slots are not preserved: groups sharing a name in different slots
    merge into one tag.

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
        fields, or if a zone references an undefined gridpoint id.
    """
    if lazy:
        warnings.warn(
            ".f3grid: lazy=True ignored; ASCII format always loads eagerly.",
            stacklevel=2,
        )

    lines = [
        stripped
        # latin-1: format is ASCII-spec but real files carry extended chars in comments
        for ln in Path(path).read_text(encoding="latin-1").splitlines()
        if (stripped := _strip_comment(ln).strip())
    ]

    node_map: dict[int, int] = {}
    coords: list[float] = []
    conn_raw: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    zone_index: dict[int, int] = {}
    block_groups: dict[str, list[int]] = {}
    inline_groups: dict[str, list[int]] = {}
    unknown_types: set[str] = set()
    current_group: str | None = None
    seen_dt4 = False
    n_dup_gp = 0
    n_dup_zone = 0
    n_bad_gp = 0
    n_bad_zone = 0
    first_bad_gp = ""
    first_bad_zone = ""
    n_faces = 0
    n_face_groups = 0

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
                gp_id = int(parts[1])
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
                coords[3 * slot : 3 * slot + 3] = xyz
                n_dup_gp += 1

        elif kw in _ZONE_KW:
            current_group = None
            if len(parts) < 2:
                n_bad_zone += 1
                first_bad_zone = first_bad_zone or ln
                continue
            zone_type = parts[1].upper()
            if zone_type == "DT4":
                seen_dt4 = True
            if zone_type not in _ZONE_TO_POLYXIOS:
                unknown_types.add(zone_type)
                continue
            elem_name, n_nodes = _ZONE_TO_POLYXIOS[zone_type]
            if len(parts) < 3 + n_nodes:
                n_bad_zone += 1
                first_bad_zone = first_bad_zone or ln
                continue
            try:
                zone_id = int(parts[2])
                ids = [int(parts[3 + j]) for j in range(n_nodes)]
            except ValueError as exc:
                raise CodecError(f".f3grid: non-numeric ZONE record: {ln!r}.") from exc
            elem_idx = len(types_list)
            conn_raw.extend(ids[k] for k in _READ_ORDER[elem_name])
            offsets_list.append(offsets_list[-1] + n_nodes)
            types_list.append(ELEMENT_TYPES[elem_name])
            if zone_id in zone_index:
                n_dup_zone += 1
            zone_index[zone_id] = elem_idx
            # FLAC3D 7 may append `GROUP "name"` to the zone record itself.
            tail = parts[3 + n_nodes :]
            if tail and any(tok.upper() == "GROUP" for tok in tail):
                name = _group_name(ln, after="GROUP")
                if name:
                    inline_groups.setdefault(name, []).append(elem_idx)

        elif kw in _FACE_KW:
            current_group = None
            n_faces += 1

        elif kw in _FACE_GROUP_KW:
            current_group = None
            n_face_groups += 1

        elif kw in _ZGROUP_KW:
            current_group = _group_name(ln, after=kw) or None
            if current_group is not None:
                block_groups.setdefault(current_group, [])

        elif current_group is not None:
            # Continuation line of a ZGROUP block: bare zone ids.
            try:
                zone_ids = [int(tok) for tok in parts]
            except ValueError:
                # Not a member list — the block ended at the previous line.
                current_group = None
            else:
                block_groups[current_group].extend(zone_ids)

    if n_dup_gp:
        warnings.warn(
            f".f3grid: {n_dup_gp} duplicate GRIDPOINT id(s); last definition used.",
            stacklevel=2,
        )
    if n_dup_zone:
        warnings.warn(
            f".f3grid: {n_dup_zone} duplicate ZONE id(s); groups referencing them"
            " tag the last zone with that id.",
            stacklevel=2,
        )
    if n_bad_gp:
        warnings.warn(
            f".f3grid: {n_bad_gp} malformed GRIDPOINT line(s) skipped"
            f" (first: {first_bad_gp!r}).",
            stacklevel=2,
        )
    if n_bad_zone:
        warnings.warn(
            f".f3grid: {n_bad_zone} short ZONE line(s) skipped"
            f" (first: {first_bad_zone!r}).",
            stacklevel=2,
        )
    if seen_dt4:
        warnings.warn(
            ".f3grid: DT4 zone type normalised to T4 on read"
            " (lossy; write will emit T4).",
            stacklevel=2,
        )
    if unknown_types:
        warnings.warn(
            f".f3grid: unknown zone type(s) skipped: {sorted(unknown_types)}.",
            stacklevel=2,
        )
    if n_faces:
        warnings.warn(
            f".f3grid: {n_faces} face record(s) skipped (zones only).",
            stacklevel=2,
        )
    if n_face_groups:
        warnings.warn(
            f".f3grid: {n_face_groups} FGROUP block(s) skipped (faces not read).",
            stacklevel=2,
        )

    if not coords:
        raise CodecError(".f3grid: no GRIDPOINT entries found.")

    if not types_list:
        warnings.warn(
            ".f3grid: no recognised ZONE entries found; connectivity will be empty.",
            stacklevel=2,
        )

    members: dict[str, list[int]] = {}
    n_unresolved = 0
    for name, zone_ids in block_groups.items():
        target = members.setdefault(name, [])
        for zid in zone_ids:
            idx = zone_index.get(zid)
            if idx is None:
                n_unresolved += 1
            else:
                target.append(idx)
    for name, elem_ids in inline_groups.items():
        members.setdefault(name, []).extend(elem_ids)

    if n_unresolved:
        warnings.warn(
            f".f3grid: {n_unresolved} group member(s) reference zones that were"
            " not read; dropped from element_tags.",
            stacklevel=2,
        )

    element_tags = {
        name: np.array(sorted(set(idxs)), dtype=np.int32)
        for name, idxs in members.items()
        if idxs
    }

    n_verts = len(coords) // 3
    vertices = np.array(coords, dtype=np.float64).reshape(n_verts, 3)

    return PolyData(
        vertices=vertices,
        connectivity=_resolve_gridpoints(conn_raw, node_map),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        element_tags=element_tags,
    )


def write(poly: PolyData, path: Path | str) -> None:
    """Write PolyData to FLAC3D .f3grid ASCII format.

    Records are written with the single-letter keywords FLAC3D itself
    emits (``G``, ``Z``). Corners are permuted from polyxios order to
    FLAC3D order; zones with a negative volume are reordered so the volume
    is positive, which means such elements come back from :func:`read` with
    a different (but geometrically equivalent) corner order.

    ``element_tags`` are written as ``ZGROUP`` blocks. Element types not
    supported by FLAC3D (anything other than tetra, pyramid, wedge,
    hexahedron) are skipped with a warning.

    Coordinates are written at full float64 precision. The file is encoded
    latin-1 to match :func:`read`; characters outside that range are
    replaced with ``?`` and reported.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .f3grid path.

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

    skipped = 0
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

    lines.append("* ZONES")
    conn = np.asarray(poly.connectivity, dtype=np.int64).tolist()
    offsets = np.asarray(poly.offsets, dtype=np.int64).tolist()
    elem_types = np.asarray(poly.element_types).tolist()
    zone_of_elem: list[int] = [0] * n_elems
    zone_id = 0
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(elem_types[i], "")
        if name not in _POLYXIOS_TO_ZONE:
            skipped += 1
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
        zone_id += 1
        zone_of_elem[i] = zone_id
        node_str = "  ".join(str(nodes[k] + 1) for k in order)
        lines.append(f"Z  {_POLYXIOS_TO_ZONE[name]}  {zone_id}  {node_str}")

    group_lines: list[str] = []
    empty_tags = 0
    dropped_members = 0
    mangled_names = 0
    for name, tag in poly.element_tags.items():
        members = [int(v) for v in np.asarray(tag).ravel()]
        ids = sorted(
            {zone_of_elem[e] for e in members if 0 <= e < n_elems and zone_of_elem[e]}
        )
        if not ids:
            empty_tags += 1
            continue
        # Partial drops only; a tag that lost every member is reported above.
        dropped_members += sum(
            1 for e in members if not (0 <= e < n_elems and zone_of_elem[e])
        )
        safe_name = str(name).replace('"', "'")
        # The file is latin-1; anything outside that range cannot be written.
        narrowed = safe_name.encode("latin-1", "replace").decode("latin-1")
        if narrowed != safe_name:
            mangled_names += 1
            safe_name = narrowed
        # SLOT is mandatory in the FLAC3D group syntax; polyxios has no slot
        # concept, so every group is written to slot 1.
        group_lines.append(f'ZGROUP "{safe_name}" SLOT 1')
        group_lines.extend(
            "     " + " ".join(str(z) for z in ids[start : start + _GROUP_IDS_PER_LINE])
            for start in range(0, len(ids), _GROUP_IDS_PER_LINE)
        )
    if group_lines:
        lines.append("* GROUPS")
        lines.extend(group_lines)

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
            f".f3grid write: {skipped} element(s) skipped"
            " (unsupported type — only tetra/pyramid/wedge/hexahedron written).",
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
    if mangled_names:
        warnings.warn(
            f".f3grid write: {mangled_names} group name(s) contain characters"
            " outside latin-1; those characters were replaced with '?'.",
            stacklevel=2,
        )

    lines.append("")
    # latin-1 to match read(); group names were already narrowed to that range.
    Path(path).write_text("\n".join(lines), encoding="latin-1")
