"""Gmsh .msh ASCII codec - read (v2.2 and v4.1) + write (v2.2)."""

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
from polyxios._io import Source, read_text, write_text
from polyxios._tags import member_indices
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".msh"

# Gmsh element type code → (polyxios name, n_nodes). Numbering is fixed by the
# MSH specification and is shared by every format version.
#
# Gmsh type 14 (14-node pyramid) is deliberately absent: VTK has no 14-node
# pyramid, so there is nowhere correct to put one, and inventing a place is
# worse than skipping the element with a warning.
_GMSH_TO_POLYXIOS: dict[int, tuple[str, int]] = {
    1: ("line", 2),
    2: ("triangle", 3),
    3: ("quad", 4),
    4: ("tetra", 4),
    5: ("hexahedron", 8),
    6: ("wedge", 6),
    7: ("pyramid", 5),
    8: ("quadratic_edge", 3),
    9: ("quadratic_triangle", 6),
    10: ("biquadratic_quad", 9),
    11: ("quadratic_tetra", 10),
    12: ("triquadratic_hexahedron", 27),
    13: ("biquadratic_quadratic_wedge", 18),
    15: ("vertex", 1),
    16: ("quadratic_quad", 8),
    17: ("quadratic_hexahedron", 20),
    18: ("quadratic_wedge", 15),
    19: ("quadratic_pyramid", 13),
}
_POLYXIOS_TO_GMSH: dict[str, int] = {
    name: code for code, (name, _) in _GMSH_TO_POLYXIOS.items()
}
# Node count Gmsh requires for each writable type; a record of any other length
# would be silently misread by Gmsh as the next element's fields.
_GMSH_NODE_COUNT: dict[str, int] = {
    name: n_nodes for _, (name, n_nodes) in _GMSH_TO_POLYXIOS.items()
}

# Topological dimension per element, used for the ``$PhysicalNames`` records.
# The sections a field travels in, which are collected apart from the mesh
# sections: a file carries one per field and per time step.
_DATA_KEYWORDS: frozenset[str] = frozenset({"$NodeData", "$ElementData"})

# The component counts a data section may declare - a scalar, a vector and a
# tensor. Gmsh refuses a section naming anything else, so a writer has only
# these three to reach for however wide the field it was handed.
_DECLARED_WIDTHS: tuple[int, ...] = (1, 3, 9)
_MAX_DECLARED_WIDTH: int = _DECLARED_WIDTHS[-1]


def _declared_width(held: int) -> int:
    """Return the component count a field of this width is declared at.

    Parameters
    ----------
    held
        How many components the field actually carries.

    Returns
    -------
    int
        The narrowest of 1, 3 and 9 that holds them, or ``held`` itself when
        the field is wider than any of the three and there is nothing legal
        to reach for.
    """
    return next((w for w in _DECLARED_WIDTHS if w >= held), held)


_ELEMENT_DIM: dict[str, int] = {
    "vertex": 0,
    "line": 1,
    "quadratic_edge": 1,
    "triangle": 2,
    "quad": 2,
    "quadratic_triangle": 2,
    "quadratic_quad": 2,
    "biquadratic_quad": 2,
    "tetra": 3,
    "hexahedron": 3,
    "wedge": 3,
    "pyramid": 3,
    "quadratic_tetra": 3,
    "quadratic_hexahedron": 3,
    "triquadratic_hexahedron": 3,
    "quadratic_wedge": 3,
    "biquadratic_quadratic_wedge": 3,
    "quadratic_pyramid": 3,
}

# Node ordering. Corner nodes agree between Gmsh and VTK for every supported
# element, but the mid-edge and face nodes of the higher-order elements do not:
# Gmsh numbers them by its own edge/face tables. Entry ``i`` holds the Gmsh
# position of the node that belongs at VTK position ``i``. Types absent from
# these tables need no permutation.

# Hexahedron mid-edge nodes: VTK lists the four bottom edges, then the four top
# edges, then the four vertical ones; Gmsh interleaves them by vertex.
_HEX_EDGES: tuple[int, ...] = (8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15)
# Hexahedron face centres, VTK's x-min/x-max/y-min/y-max/z-min/z-max order,
# then the body centre; Gmsh starts at z-min and ends at z-max.
_HEX_FACES: tuple[int, ...] = (22, 23, 21, 24, 20, 25, 26)

_READ_ORDER: dict[str, tuple[int, ...]] = {
    # VTK edges (0,1)(1,2)(0,2)(0,3)(1,3)(2,3); Gmsh swaps the last pair.
    "quadratic_tetra": (0, 1, 2, 3, 4, 5, 6, 7, 9, 8),
    "quadratic_hexahedron": (*range(8), *_HEX_EDGES),
    "triquadratic_hexahedron": (*range(8), *_HEX_EDGES, *_HEX_FACES),
    # Bottom triangle edges, top triangle edges, then the vertical edges.
    "quadratic_wedge": (0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11),
    # Base edges, then the four lateral edges meeting at the apex.
    "quadratic_pyramid": (0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12),
    # The 15-node wedge's edges, then the three quadrilateral face centres:
    # VTK orders them (0,1,4,3), (1,2,5,4), (2,0,3,5) while Gmsh's face table
    # runs (0,1,4,3), (0,3,5,2), (1,2,5,4), so the last two swap.
    "biquadratic_quadratic_wedge": (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        9,
        7,
        12,
        14,
        13,
        8,
        10,
        11,
        15,
        17,
        16,
    ),
}
_WRITE_ORDER: dict[str, tuple[int, ...]] = {
    name: tuple(order.index(i) for i in range(len(order)))
    for name, order in _READ_ORDER.items()
}

# Gmsh writes plain ASCII, but physical-group names in files produced by other
# tools are UTF-8; utf-8-sig also drops a byte-order mark, which would otherwise
# hide the ``$MeshFormat`` keyword.
_READ_ENCODING: str = "utf-8-sig"
_WRITE_ENCODING: str = "utf-8"

# Default coordinate format: 17 significant digits round-trip a float64 exactly.
_DEFAULT_FLOAT_FMT: str = ".17g"


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a Gmsh .msh file in ASCII format version 2.2 or 4.1.

    Parameters
    ----------
    path
        Path to the .msh file.
    lazy
        Ignored; the ASCII format always loads eagerly.

    Returns
    -------
    PolyData
        Element physical tags are stored in ``element_attrs["phys_tag"]``.
        Names declared in ``$PhysicalNames`` become ``element_tags`` entries.

    Raises
    ------
    CodecError
        On a missing ``$MeshFormat``, ``$Nodes`` or ``$Elements`` section, on a
        binary or unsupported-version file, on a declared count that exceeds the
        records actually present or the safety caps, or on an element that
        references an undefined node tag.

    Notes
    -----
    Gmsh type 14, the 14-node pyramid, has no VTK equivalent and is skipped
    with a warning, as are any element types outside the MSH spec.

    ``$NodeData`` and ``$ElementData`` become ``vertex_attrs`` and
    ``element_attrs``, at whatever component count the file declares; a field
    is scattered by the tag each row names, so one covering part of the mesh
    lands where it belongs and the rest stays ``NaN``. A row naming an entity
    the mesh does not hold is dropped with a warning, and a field the mesh
    cannot hold at all costs that field alone.
    ``$ElementNodeData``, ``$Periodic`` and every other section are ignored.
    Duplicate node tags are reported with a warning and the last definition of
    each wins.
    """
    if lazy:
        warnings.warn(
            ".msh: lazy=True ignored; ASCII format always loads eagerly.",
            stacklevel=2,
        )

    try:
        text = read_text(path, encoding=_READ_ENCODING)
    except UnicodeDecodeError as exc:
        raise CodecError(
            ".msh: file is not ASCII/UTF-8 text; binary .msh is not supported."
        ) from exc

    sections, data_blocks = _split_sections(text)
    for required in ("$MeshFormat", "$Nodes", "$Elements"):
        if required not in sections:
            raise CodecError(f".msh: missing {required} section.")

    version, is_binary = _parse_mesh_format(sections["$MeshFormat"])
    if is_binary:
        raise CodecError(".msh: binary format is not supported; ASCII only.")

    if version.startswith("2"):
        tags, vertices = _parse_nodes_v2(sections["$Nodes"])
        entity_phys: dict[tuple[int, int], int] = {}
    elif version.startswith("4.1"):
        tags, vertices = _parse_nodes_v41(sections["$Nodes"])
        entity_phys = _parse_entities_v41(sections.get("$Entities", []))
        if "$Entities" not in sections and "$PhysicalNames" in sections:
            # In 4.1 the element blocks name an entity, not a physical group;
            # without $Entities the two cannot be linked and the names are inert.
            warnings.warn(
                ".msh: $PhysicalNames present but $Entities is missing;"
                " physical tags cannot be recovered and default to 0.",
                stacklevel=2,
            )
    else:
        raise CodecError(
            f".msh: unsupported format version {version!r};"
            " only ASCII 2.x and 4.1 are supported."
        )

    node_index = _build_node_index(tags)

    if version.startswith("2"):
        builder = _parse_elements_v2(sections["$Elements"])
    else:
        builder = _parse_elements_v41(sections["$Elements"], entity_phys)
    conn_raw, offsets, types, phys_tags = builder.finish()

    connectivity = _resolve_nodes(conn_raw, node_index, vertices.shape[0])

    element_attrs = {"phys_tag": phys_tags} if len(phys_tags) else {}
    element_tags = _physical_name_tags(
        sections.get("$PhysicalNames", []), phys_tags, types
    )

    vertex_attrs: dict[str, np.ndarray] = {}
    _read_data_sections(
        data_blocks,
        node_index,
        # Gmsh numbers its elements from 1, so a 0 is an element that answers
        # to no tag and must not be reachable from a data row.
        {tag: i for i, tag in enumerate(builder.ids) if tag > 0},
        vertices.shape[0],
        len(types),
        vertex_attrs,
        element_attrs,
    )

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        element_tags=element_tags,
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write PolyData to Gmsh ASCII .msh format version 2.2.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .msh path.
    **opts
        ``float_fmt`` overrides the coordinate format specifier.

    Raises
    ------
    CodecError
        When an element does not carry the node count its Gmsh type requires.

    Notes
    -----
    ``element_attrs["phys_tag"]`` is written back as the first element tag.
    When it is absent, or does not hold one value per element, physical tags
    are derived from ``element_tags``; either way, ``element_tags`` whose members
    share a single physical tag are emitted as a ``$PhysicalNames`` section.
    A tag is scoped to a topological dimension, so two groups may share one as
    long as each stays within a dimension of its own.
    Elements whose type has no Gmsh equivalent are skipped with a warning, as
    are groups that cannot be reduced to one physical tag.
    Numeric ``vertex_attrs`` and ``element_attrs`` are written as ``$NodeData``
    and ``$ElementData`` sections; the rest are skipped with a warning.
    """
    float_fmt = opts.get("float_fmt", _DEFAULT_FLOAT_FMT)
    n_elems = len(poly.element_types)

    writable = [
        i
        for i in range(n_elems)
        if ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "") in _POLYXIOS_TO_GMSH
    ]
    if len(writable) < n_elems:
        warnings.warn(
            f".msh: {n_elems - len(writable)} element(s) have no Gmsh equivalent"
            " and were skipped.",
            stacklevel=2,
        )

    phys_tags, phys_names = _resolve_physical_groups(poly)

    lines: list[str] = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat"]

    if phys_names:
        lines.append("$PhysicalNames")
        lines.append(str(len(phys_names)))
        lines.extend(
            f'{dim} {tag} "{_sanitize_name(name)}"'
            for name, (dim, tag) in phys_names.items()
        )
        lines.append("$EndPhysicalNames")

    lines.append("$Nodes")
    lines.append(str(poly.vertices.shape[0]))
    lines.extend(
        f"{i + 1} {v[0]:{float_fmt}} {v[1]:{float_fmt}} {v[2]:{float_fmt}}"
        for i, v in enumerate(poly.vertices)
    )
    lines.append("$EndNodes")

    lines.append("$Elements")
    lines.append(str(len(writable)))
    connectivity = np.asarray(poly.connectivity)
    for out_idx, ei in enumerate(writable):
        name = ELEMENT_TYPES_INV[int(poly.element_types[ei])]
        nodes = connectivity[int(poly.offsets[ei]) : int(poly.offsets[ei + 1])]
        expected = _GMSH_NODE_COUNT[name]
        if len(nodes) != expected:
            raise CodecError(
                f".msh: element {ei} of type {name!r} has {len(nodes)} node(s),"
                f" expected {expected}; Gmsh type {_POLYXIOS_TO_GMSH[name]} reads"
                " a fixed number of nodes per element."
            )
        order = _WRITE_ORDER.get(name)
        if order is not None:
            nodes = nodes[list(order)]
        node_str = " ".join(str(int(n) + 1) for n in nodes)
        tag = int(phys_tags[ei])
        lines.append(
            f"{out_idx + 1} {_POLYXIOS_TO_GMSH[name]} 2 {tag} {tag} {node_str}"
        )
    lines.append("$EndElements")

    lines.extend(
        _data_section_lines(
            poly.vertex_attrs, "NodeData", poly.vertices.shape[0], None, float_fmt
        )
    )
    lines.extend(
        _data_section_lines(
            {k: v for k, v in (poly.element_attrs or {}).items() if k != "phys_tag"},
            "ElementData",
            n_elems,
            writable,
            float_fmt,
        )
    )
    lines.append("")

    write_text(path, "\n".join(lines), encoding=_WRITE_ENCODING)


def _data_section_lines(
    attrs: dict[str, np.ndarray],
    keyword: str,
    count: int,
    written: list[int] | None,
    float_fmt: str,
) -> list[str]:
    """Return one ``$NodeData`` / ``$ElementData`` section per numeric attribute.

    Parameters
    ----------
    attrs
        Named attribute arrays.
    keyword
        ``NodeData`` or ``ElementData``.
    count
        How many entities the mesh holds, which is how long an attribute has
        to be to describe them.
    written
        For element data, the mesh indices that made it into the file, in the
        order they were written; None for node data, where every node goes.
    float_fmt
        Format specifier for the values.

    Returns
    -------
    list of str
        The section lines, empty when nothing could be written.

    Notes
    -----
    The reader takes a field of any width, but a writer cannot: MSH2 declares
    a component count of 1, 3 or 9, and Gmsh refuses a section naming
    anything else outright. A field of another width is padded out to the next
    of the three with zero columns, so the values it carries all reach the
    file and the file still loads; the padding is reported, since it is a
    column the caller did not hand over. A field wider than 9 has no legal
    width to reach and is written at its own, which polyxios reads back and
    Gmsh does not.

    Non-numeric or wrong-length attributes have no data section to sit in and
    are reported rather than dropped in silence.

    A row that is not finite throughout is left out, and the declared entity
    count drops with it: the format spells no "no value here", so a NaN read
    in from a field covering part of the mesh would go back out as the token
    ``nan``, which is not a number a reader expecting one accepts. A field
    covering part of the mesh is written over that part, the way it came in.
    """
    lines: list[str] = []
    skipped: set[str] = set()
    thinned: dict[str, int] = {}
    padded: dict[str, tuple[int, int]] = {}
    overwide: dict[str, int] = {}
    for name, raw in (attrs or {}).items():
        array = np.asarray(raw)
        if (
            not name.strip()
            or array.dtype.kind not in "fiub"
            or array.ndim not in (1, 2)
            or array.shape[0] != count
        ):
            skipped.add(name)
            continue
        values = array.reshape(count, -1).astype(np.float64, copy=False)
        held = values.shape[1]
        width = _declared_width(held)
        if width > held:
            padded[name] = (held, width)
            values = np.pad(values, ((0, 0), (0, width - held)))
        elif held > _MAX_DECLARED_WIDTH:
            overwide[name] = held
        # Element data is numbered by the file, not by the mesh: an element
        # with no Gmsh equivalent never got an id, so its value has no row.
        picked = (
            np.arange(count, dtype=np.int64)
            if written is None
            else np.asarray(written, dtype=np.int64)
        )
        if not picked.size:
            continue
        rows = values[picked]
        # The tag names the entity as this file numbers it, which is the row's
        # place among the ones written - so dropping a row must not renumber
        # the rest.
        live = np.isfinite(rows).all(axis=1)
        if not live.all():
            thinned[name] = int(np.count_nonzero(~live))
            rows = rows[live]
        tags = np.flatnonzero(live) + 1
        if not tags.size:
            continue
        lines.extend(
            [
                f"${keyword}",
                "1",
                f'"{_sanitize_name(name)}"',
                "1",
                "0.0",
                "3",
                "0",
                str(values.shape[1]),
                str(tags.size),
            ]
        )
        lines.extend(
            f"{tag} " + " ".join(f"{value:{float_fmt}}" for value in row)
            for tag, row in zip(tags.tolist(), rows)
        )
        lines.append(f"$End{keyword}")

    kind = "vertex" if keyword == "NodeData" else "element"
    if skipped:
        warnings.warn(
            f".msh: only named numeric {kind} attributes, one row per"
            f" {kind}, can be written; skipped {sorted(skipped)}.",
            stacklevel=3,
        )
    if thinned:
        named = ", ".join(f"{name} ({n})" for name, n in sorted(thinned.items()))
        warnings.warn(
            f".msh: the format spells no missing value, so the {kind}(s)"
            f" carrying one were left out of their data section: {named}.",
            stacklevel=3,
        )
    if padded:
        named = ", ".join(
            f"{name} ({held} -> {width})"
            for name, (held, width) in sorted(padded.items())
        )
        warnings.warn(
            f".msh: a data section declares 1, 3 or 9 components and no other"
            f" count loads, so the {kind} attribute(s) {named} were padded out"
            " with zero column(s).",
            stacklevel=3,
        )
    if overwide:
        named = ", ".join(f"{name} ({n})" for name, n in sorted(overwide.items()))
        warnings.warn(
            f".msh: the {kind} attribute(s) {named} carry more than"
            f" {_MAX_DECLARED_WIDTH} components, which no data section can"
            " declare; they were written at their own width, which polyxios"
            " reads back and Gmsh refuses.",
            stacklevel=3,
        )
    return lines


def _split_sections(
    text: str,
) -> tuple[dict[str, list[str]], list[tuple[str, list[str]]]]:
    """Split a .msh file into its sections, blanks removed.

    Parameters
    ----------
    text
        The whole file.

    Returns
    -------
    dict of str to list of str
        ``$Keyword`` to its content lines, one entry per keyword.
    list of (str, list of str)
        The ``$NodeData`` / ``$ElementData`` blocks, in file order.

    Notes
    -----
    The data blocks come back alongside rather than through the mapping: a
    file carries one per field and per time step, and a mapping keyed by the
    keyword keeps one and drops the rest. Both are collected in this one walk,
    since the file runs to millions of lines and splitting it twice costs as
    much again.
    """
    sections: dict[str, list[str]] = {}
    data: list[tuple[str, list[str]]] = []
    current: str | None = None
    content: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("$End"):
            if current is not None:
                sections[current] = content
                if current in _DATA_KEYWORDS:
                    data.append((current, content))
            current = None
            content = []
        elif ln.startswith("$"):
            current = ln
            content = []
        elif current is not None:
            content.append(ln)
    # An unterminated final section still carries usable records.
    if current is not None:
        if current not in sections:
            sections[current] = content
        if current in _DATA_KEYWORDS:
            data.append((current, content))
    return sections, data


def _parse_mesh_format(lines: list[str]) -> tuple[str, bool]:
    """Return the declared version string and whether the payload is binary."""
    if not lines:
        raise CodecError(".msh: empty $MeshFormat section.")
    fields = lines[0].split()
    if not fields:
        raise CodecError(".msh: malformed $MeshFormat header.")
    file_type = fields[1] if len(fields) > 1 else "0"
    return fields[0], file_type != "0"


def _checked_count(value: str, cap: int, what: str) -> int:
    """Parse a declared record count, rejecting negatives and absurd values."""
    try:
        count = int(value)
    except ValueError as exc:
        raise CodecError(f".msh: non-integer {what} count {value!r}.") from exc
    if count < 0:
        raise CodecError(f".msh: negative {what} count {count}.")
    if count > cap:
        raise CodecError(f".msh: {what} count {count} exceeds the safety cap {cap}.")
    return count


def _parse_nodes_v2(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Parse a format 2.x ``$Nodes`` section into (node tags, coordinates)."""
    if not lines:
        raise CodecError(".msh: empty $Nodes section.")
    n_nodes = _checked_count(lines[0], MAX_SAFE_VERTICES, "node")
    records = lines[1 : n_nodes + 1]
    if len(records) < n_nodes:
        raise CodecError(
            f".msh: $Nodes declares {n_nodes} nodes but only {len(records)} follow."
        )

    bulk = _bulk_parse_nodes_v2(records, n_nodes)
    if bulk is not None:
        return bulk

    tags = np.empty(n_nodes, dtype=np.int64)
    coords = np.empty((n_nodes, 3), dtype=np.float64)
    for i, ln in enumerate(records):
        parts = ln.split()
        if len(parts) < 4:
            raise CodecError(f".msh: malformed node record {ln!r}.")
        try:
            tags[i] = int(parts[0])
            coords[i] = (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError as exc:
            raise CodecError(f".msh: non-numeric node record {ln!r}.") from exc
        except OverflowError as exc:
            raise CodecError(
                f".msh: node tag in {ln!r} does not fit in a 64-bit integer."
            ) from exc
    return tags, coords


def _first_field_ints(records: list[str]) -> np.ndarray | None:
    """Read the leading integer of every record, or None when one is unusable."""
    try:
        return np.array([ln.split(maxsplit=1)[0] for ln in records], dtype=np.int64)
    except (ValueError, OverflowError, IndexError):
        return None


def _bulk_parse_nodes_v2(
    records: list[str], n_nodes: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse "tag x y z" records in one pass, or None when that is not safe.

    A single bulk parse beats a per-line split, but it only sees a flat stream
    of numbers: a record with a missing or extra field shifts every later value
    into the wrong column, which would corrupt the mesh silently. The result is
    therefore only returned once the alignment is confirmed - first by the
    common case of tags running 1..n, then by re-reading the leading field of
    every record and requiring it to match the column the bulk parse produced.
    """
    flat = np.fromstring(" ".join(records), sep=" ", dtype=np.float64)
    if flat.size != 4 * n_nodes:
        return None

    table = flat.reshape(n_nodes, 4)
    coords = np.ascontiguousarray(table[:, 1:4])
    tags = table[:, 0]
    if not n_nodes:
        return tags.astype(np.int64), coords
    if (
        tags[0] == 1
        and tags[-1] == n_nodes
        and np.array_equal(tags, np.arange(1, n_nodes + 1))
    ):
        return tags.astype(np.int64), coords

    # Tags are not the plain 1..n sequence, so the columns are checked against
    # the records themselves. This also keeps tags beyond 2**53 exact, which a
    # float64 column cannot hold.
    exact = _first_field_ints(records)
    if exact is None or not np.array_equal(exact.astype(np.float64), tags):
        return None
    return exact, coords


def _parse_nodes_v41(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Parse a format 4.1 ``$Nodes`` section into (node tags, coordinates).

    Format 4.1 groups nodes into per-entity blocks and lists all tags of a block
    before all of its coordinates.
    """
    if not lines:
        raise CodecError(".msh: empty $Nodes section.")
    header = lines[0].split()
    if len(header) < 2:
        raise CodecError(f".msh: malformed $Nodes header {lines[0]!r}.")
    n_blocks = _checked_count(header[0], MAX_SAFE_VERTICES, "node block")
    n_nodes = _checked_count(header[1], MAX_SAFE_VERTICES, "node")

    tags = np.empty(n_nodes, dtype=np.int64)
    coords = np.empty((n_nodes, 3), dtype=np.float64)
    cursor = 1
    filled = 0
    for _ in range(n_blocks):
        if cursor >= len(lines):
            raise CodecError(".msh: $Nodes ended before all blocks were read.")
        block = lines[cursor].split()
        if len(block) < 4:
            raise CodecError(f".msh: malformed $Nodes block header {lines[cursor]!r}.")
        cursor += 1
        n_in_block = _checked_count(block[3], MAX_SAFE_VERTICES, "node")
        if filled + n_in_block > n_nodes:
            raise CodecError(".msh: $Nodes blocks hold more nodes than declared.")
        if cursor + 2 * n_in_block > len(lines):
            raise CodecError(".msh: $Nodes block is truncated.")

        try:
            for j in range(n_in_block):
                tags[filled + j] = int(lines[cursor + j])
            cursor += n_in_block
            for j in range(n_in_block):
                parts = lines[cursor + j].split()
                coords[filled + j] = (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                )
        except (ValueError, IndexError) as exc:
            raise CodecError(".msh: malformed record in a $Nodes block.") from exc
        except OverflowError as exc:
            raise CodecError(
                ".msh: a node tag in a $Nodes block does not fit in a 64-bit integer."
            ) from exc
        cursor += n_in_block
        filled += n_in_block

    if filled != n_nodes:
        raise CodecError(
            f".msh: $Nodes declares {n_nodes} nodes but blocks hold {filled}."
        )
    return tags, coords


def _lenient_int(field: str) -> int:
    """Read an integer field, tolerating writers that emit it as a float."""
    try:
        return int(field)
    except ValueError:
        return int(float(field))


def _element_id(field: str) -> int:
    """Read an element tag, or 0 for one no number can be made of.

    A tag names the element a ``$ElementData`` row describes and nothing else,
    so a file that spells one strangely is still a file whose mesh is worth
    reading. Gmsh numbers its elements from 1, which is what leaves 0 free to
    mean "this element answers to no tag".
    """
    try:
        return _lenient_int(field)
    except (ValueError, OverflowError):
        return 0


def _parse_entities_v41(lines: list[str]) -> dict[tuple[int, int], int]:
    """Map (dimension, entity tag) → first physical tag from ``$Entities``.

    A malformed section only costs the physical tags, so it is reported as a
    warning rather than an error.
    """
    if not lines:
        return {}
    header = lines[0].split()
    if len(header) < 4:
        warnings.warn(
            ".msh: malformed $Entities header; physical tags dropped.",
            stacklevel=2,
        )
        return {}

    out: dict[tuple[int, int], int] = {}
    cursor = 1
    try:
        counts = [int(header[i]) for i in range(4)]
        for dim, count in enumerate(counts):
            for _ in range(count):
                fields = lines[cursor].split()
                cursor += 1
                # Point entities carry x y z; the others carry a 6-value box.
                n_phys_at = 4 if dim == 0 else 7
                n_phys = int(fields[n_phys_at])
                if n_phys > 0:
                    entity_tag = _lenient_int(fields[0])
                    out[(dim, entity_tag)] = _lenient_int(fields[n_phys_at + 1])
    except (ValueError, IndexError, OverflowError):
        warnings.warn(
            ".msh: malformed $Entities record; physical tags dropped.",
            stacklevel=2,
        )
        return {}
    return out


def _parse_elements_v2(lines: list[str]) -> "_ElementBuilder":
    """Parse a format 2.x ``$Elements`` section."""
    if not lines:
        raise CodecError(".msh: empty $Elements section.")
    n_elems = _checked_count(lines[0], MAX_SAFE_ELEMENTS, "element")
    records = lines[1 : n_elems + 1]
    if len(records) < n_elems:
        raise CodecError(
            f".msh: $Elements declares {n_elems} elements but only"
            f" {len(records)} follow."
        )

    builder = _ElementBuilder()
    for ln in records:
        parts = ln.split()
        if len(parts) < 3:
            raise CodecError(f".msh: malformed element record {ln!r}.")
        try:
            gmsh_type = int(parts[1])
            n_tags = int(parts[2])
        except ValueError as exc:
            raise CodecError(f".msh: non-integer field in element {ln!r}.") from exc
        if n_tags < 0 or len(parts) < 3 + n_tags:
            raise CodecError(
                f".msh: element declares {n_tags} tag(s) but the record is"
                f" too short: {ln!r}."
            )
        try:
            phys_tag = int(parts[3]) if n_tags > 0 else 0
        except ValueError as exc:
            raise CodecError(f".msh: non-integer physical tag in {ln!r}.") from exc
        builder.add(gmsh_type, parts[3 + n_tags :], phys_tag, ln, _element_id(parts[0]))
    return builder


def _parse_elements_v41(
    lines: list[str], entity_phys: dict[tuple[int, int], int]
) -> "_ElementBuilder":
    """Parse a format 4.1 ``$Elements`` section, one block per entity."""
    if not lines:
        raise CodecError(".msh: empty $Elements section.")
    header = lines[0].split()
    if len(header) < 2:
        raise CodecError(f".msh: malformed $Elements header {lines[0]!r}.")
    n_blocks = _checked_count(header[0], MAX_SAFE_ELEMENTS, "element block")
    n_declared = _checked_count(header[1], MAX_SAFE_ELEMENTS, "element")

    builder = _ElementBuilder()
    seen = 0
    cursor = 1
    for _ in range(n_blocks):
        if cursor >= len(lines):
            raise CodecError(".msh: $Elements ended before all blocks were read.")
        block = lines[cursor].split()
        if len(block) < 4:
            raise CodecError(
                f".msh: malformed $Elements block header {lines[cursor]!r}."
            )
        cursor += 1
        try:
            entity_dim = int(block[0])
            entity_tag = int(block[1])
            gmsh_type = int(block[2])
        except ValueError as exc:
            raise CodecError(".msh: non-integer $Elements block header.") from exc
        n_in_block = _checked_count(block[3], MAX_SAFE_ELEMENTS, "element")
        if cursor + n_in_block > len(lines):
            raise CodecError(".msh: $Elements block is truncated.")

        phys_tag = entity_phys.get((entity_dim, entity_tag), 0)
        for j in range(n_in_block):
            ln = lines[cursor + j]
            # Field 0 is the element tag; the rest are node tags.
            fields = ln.split()
            builder.add(
                gmsh_type,
                fields[1:],
                phys_tag,
                ln,
                _element_id(fields[0]) if fields else 0,
            )
        cursor += n_in_block
        seen += n_in_block

    # The header total is redundant with the per-block counts; a disagreement
    # means the writer is inconsistent, not that records are missing, so the
    # blocks are kept and the mismatch only reported.
    if seen != n_declared:
        warnings.warn(
            f".msh: $Elements declares {n_declared} element(s) but the blocks"
            f" hold {seen}; the blocks were used.",
            stacklevel=2,
        )
    return builder


class _ElementBuilder:
    """Accumulate elements in CSR layout, permuting Gmsh node order to VTK."""

    def __init__(self) -> None:
        self.conn: list[int] = []
        self.offsets: list[int] = [0]
        self.types: list[int] = []
        self.phys: list[int] = []
        self.skipped: set[int] = set()
        # Element tags as the file numbers them, so a $ElementData field can
        # be matched to the elements that survived the read.
        self.ids: list[int] = []

    def add(
        self,
        gmsh_type: int,
        node_fields: list[str],
        phys_tag: int,
        record: str,
        elem_id: int = 0,
    ) -> None:
        """Append one element; unsupported Gmsh types are recorded and dropped."""
        mapped = _GMSH_TO_POLYXIOS.get(gmsh_type)
        if mapped is None:
            self.skipped.add(gmsh_type)
            return
        name, n_nodes = mapped
        # Both callers hand over exactly the node fields of the record, so a
        # different count means the record is misaligned; reading the first
        # ``n_nodes`` of it would quietly build the wrong element.
        if len(node_fields) != n_nodes:
            raise CodecError(
                f".msh: element of type {gmsh_type} needs {n_nodes} nodes,"
                f" got {len(node_fields)} in {record!r}."
            )
        try:
            nodes = [int(node_fields[j]) for j in range(n_nodes)]
        except ValueError as exc:
            raise CodecError(f".msh: non-integer node tag in {record!r}.") from exc

        order = _READ_ORDER.get(name)
        if order is not None:
            nodes = [nodes[k] for k in order]

        self.conn.extend(nodes)
        if len(self.conn) > MAX_SAFE_CONN:
            raise CodecError(
                f".msh: connectivity exceeds the safety cap {MAX_SAFE_CONN}."
            )
        self.offsets.append(self.offsets[-1] + n_nodes)
        self.types.append(ELEMENT_TYPES[name])
        self.phys.append(phys_tag)
        self.ids.append(elem_id)

    def finish(self) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        """Emit (raw node tags, offsets, element types, physical tags)."""
        if self.skipped:
            warnings.warn(
                ".msh: skipped elements of unsupported Gmsh type(s)"
                f" {sorted(self.skipped)}.",
                stacklevel=2,
            )
        try:
            phys = np.array(self.phys, dtype=np.int32)
        except OverflowError as exc:
            raise CodecError(
                ".msh: a physical tag does not fit in a 32-bit integer."
            ) from exc
        return (
            self.conn,
            np.array(self.offsets, dtype=np.int32),
            np.array(self.types, dtype=np.uint8),
            phys,
        )


def _parse_data_block(
    lines: list[str], kind: str
) -> tuple[str, np.ndarray, np.ndarray]:
    """Return a data block's name, the tags it names, and its values.

    Parameters
    ----------
    lines
        The block's content lines, blanks already removed.
    kind
        ``$NodeData`` or ``$ElementData``, named in any error raised.

    Returns
    -------
    str
        The field name, from the first string tag.
    numpy.ndarray
        The node or element tags the block names, one per row.
    numpy.ndarray
        Values, shape ``(n_rows, n_components)``.

    Raises
    ------
    CodecError
        On a header that does not spell three tag counts, a component count
        that is absent or not positive, or a row that is short or not numeric.
    """
    cursor = 0

    def take_count(what: str) -> int:
        nonlocal cursor
        if cursor >= len(lines):
            raise CodecError(f".msh: {kind} ends before its {what} count.")
        try:
            count = int(lines[cursor])
        except ValueError as exc:
            raise CodecError(
                f".msh: {kind} {what} count {lines[cursor]!r} is not an integer."
            ) from exc
        cursor += 1
        if count < 0:
            raise CodecError(f".msh: {kind} declares {count} {what} tag(s).")
        if cursor + count > len(lines):
            raise CodecError(f".msh: {kind} ends inside its {what} tags.")
        return count

    n_strings = take_count("string")
    strings = lines[cursor : cursor + n_strings]
    cursor += n_strings
    n_reals = take_count("real")
    cursor += n_reals
    n_ints = take_count("integer")
    int_tags = lines[cursor : cursor + n_ints]
    cursor += n_ints

    if not strings:
        raise CodecError(f".msh: {kind} carries no name.")
    name = strings[0].strip().strip('"').strip()
    if not name:
        raise CodecError(f".msh: {kind} carries an empty name.")

    # Integer tags are (time step, components, entities, [partition]); the
    # spec puts no ceiling on the component count, so any positive width is
    # read rather than only the 1/3/9 a tensor field happens to use.
    if len(int_tags) < 2:
        raise CodecError(f".msh: {kind} '{name}' declares no component count.")
    try:
        n_components = int(int_tags[1])
        n_rows = int(int_tags[2]) if len(int_tags) > 2 else len(lines) - cursor
    except ValueError as exc:
        raise CodecError(f".msh: {kind} '{name}' has a non-integer tag.") from exc
    if n_components <= 0:
        raise CodecError(f".msh: {kind} '{name}' declares {n_components} component(s).")
    if n_rows < 0 or cursor + n_rows > len(lines):
        raise CodecError(
            f".msh: {kind} '{name}' declares {n_rows} row(s) but only"
            f" {len(lines) - cursor} follow."
        )

    tags = np.empty(n_rows, dtype=np.int64)
    values = np.empty((n_rows, n_components), dtype=np.float64)
    for i, ln in enumerate(lines[cursor : cursor + n_rows]):
        fields = ln.split()
        if len(fields) < n_components + 1:
            raise CodecError(
                f".msh: {kind} '{name}' row {ln!r} holds {len(fields) - 1}"
                f" value(s), expected {n_components}."
            )
        try:
            tags[i] = int(fields[0])
            values[i] = [float(tok) for tok in fields[1 : n_components + 1]]
        except ValueError as exc:
            raise CodecError(
                f".msh: {kind} '{name}' has a malformed row {ln!r}."
            ) from exc
    return name, tags, values


def _unique_key(name: str, taken: dict[str, np.ndarray]) -> str:
    """Return a name free in ``taken``, so a clash does not hide a field."""
    candidate, k = name, 1
    while candidate in taken:
        k += 1
        candidate = f"{name}_{k}"
    return candidate


def _read_data_sections(
    blocks: list[tuple[str, list[str]]],
    node_index: dict[int, int] | None,
    elem_index: dict[int, int],
    n_verts: int,
    n_elems: int,
    vertex_attrs: dict[str, np.ndarray],
    element_attrs: dict[str, np.ndarray],
) -> None:
    """Read every ``$NodeData`` / ``$ElementData`` block into the attribute dicts.

    A field is scattered by the tag each row names rather than by row order,
    so a block covering half the mesh lands on the half it names; the rest
    stays NaN rather than shifting onto the wrong entities. A row naming an
    entity the mesh does not hold - an element of a type this codec skipped,
    say - costs that row and not the whole field, and a block whose every row
    does is dropped. A malformed block costs that field alone: the mesh around
    it is still worth having.
    """
    for kind, lines in blocks:
        try:
            name, tags, values = _parse_data_block(lines, kind)
        except CodecError as exc:
            warnings.warn(f".msh: skipping a data section - {exc}", stacklevel=2)
            continue

        if kind == "$NodeData":
            target, count = vertex_attrs, n_verts
            rows = (
                [node_index.get(int(t), -1) for t in tags]
                if node_index is not None
                else [int(t) - 1 for t in tags]
            )
        else:
            target, count = element_attrs, n_elems
            rows = [elem_index.get(int(t), -1) for t in tags]

        index = np.array(rows, dtype=np.int64)
        held = (index >= 0) & (index < count)
        unknown = int(np.count_nonzero(~held))
        if unknown:
            warnings.warn(
                f".msh: {kind} '{name}' names {unknown} tag(s) the mesh does not"
                f" hold, first {int(tags[np.argmax(~held)])}; those rows are"
                " dropped.",
                stacklevel=2,
            )
        if count == 0 or not held.any():
            continue

        field = np.full((count, values.shape[1]), np.nan, dtype=np.float64)
        field[index[held]] = values[held]
        target[_unique_key(name, target)] = (
            field[:, 0].copy() if values.shape[1] == 1 else field
        )


def _build_node_index(tags: np.ndarray) -> dict[int, int] | None:
    """Return a tag → row mapping, or None when tags are exactly 1..n."""
    n = tags.size
    if (
        n
        and tags[0] == 1
        and tags[-1] == n
        and np.array_equal(tags, np.arange(1, n + 1))
    ):
        return None
    index = {int(tag): i for i, tag in enumerate(tags)}
    if len(index) != n:
        warnings.warn(
            f".msh: $Nodes declares {n - len(index)} duplicate node tag(s);"
            " the last coordinates given for each tag are used.",
            stacklevel=3,
        )
    return index


def _resolve_nodes(
    raw: list[int],
    index: dict[int, int] | None,
    n_vertices: int,
) -> np.ndarray:
    """Translate raw Gmsh node tags into vertex row indices."""
    if not raw:
        return np.empty(0, dtype=np.int32)
    if index is None:
        # Tags are exactly 1..n, so the row is the tag minus one - but a tag
        # outside that span would silently index the wrong vertex.
        try:
            resolved = np.array(raw, dtype=np.int64) - 1
        except OverflowError as exc:
            limit = 1 << 63
            bad_tag = next(tag for tag in raw if not -limit <= tag < limit)
            raise CodecError(
                f".msh: element references undefined node tag {bad_tag}."
            ) from exc
        bad = resolved[(resolved < 0) | (resolved >= n_vertices)]
        if bad.size:
            raise CodecError(
                f".msh: element references undefined node tag {int(bad[0]) + 1}."
            )
        return resolved.astype(np.int32)
    try:
        return np.array([index[tag] for tag in raw], dtype=np.int32)
    except KeyError as exc:
        raise CodecError(f".msh: element references undefined node tag {exc}.") from exc


def _physical_name_tags(
    lines: list[str],
    phys_tags: np.ndarray,
    element_types: np.ndarray,
) -> dict[str, np.ndarray]:
    """Turn ``$PhysicalNames`` into element_tags keyed by group name.

    A physical tag is only unique within a dimension, so two names may share
    one tag. Members are then split by the declared dimension of each name;
    with no collision the tag alone selects them, which keeps groups intact
    when a writer declares a dimension its members do not all match.
    """
    if not lines or not phys_tags.size:
        return {}

    # The leading record count is optional in practice; dropping it blindly
    # would swallow the first name when a writer omits it.
    body = lines[1:] if lines[0].strip().isdigit() else lines

    records: list[tuple[int, int, str]] = []
    for ln in body:
        # "dim tag "name"" - the name is quoted and may contain spaces.
        head, sep, quoted = ln.partition('"')
        if not sep:
            continue
        fields = head.split()
        if len(fields) < 2:
            continue
        try:
            dim = int(fields[0])
            tag = int(fields[1])
        except ValueError:
            continue
        name = quoted.rsplit('"', 1)[0]
        # Gmsh numbers physical groups from 1; tag 0 is the value carried by
        # every element that belongs to no group, so a record claiming it would
        # sweep up the whole untagged remainder of the mesh.
        if name and tag != 0:
            records.append((dim, tag, name))

    shared = {
        tag
        for tag in {rec[1] for rec in records}
        if sum(1 for rec in records if rec[1] == tag) > 1
    }
    element_dims = _element_dims(element_types) if shared else None

    out: dict[str, np.ndarray] = {}
    for dim, tag, name in records:
        selected = phys_tags == tag
        if tag in shared and element_dims is not None:
            selected &= element_dims == dim
        members = np.flatnonzero(selected).astype(np.int32)
        if members.size:
            out[name] = members
    return out


def _element_dims(element_types: np.ndarray) -> np.ndarray:
    """Topological dimension of every element, ``-1`` when unknown."""
    lookup = {code: _ELEMENT_DIM.get(name, -1) for name, code in ELEMENT_TYPES.items()}
    return np.array(
        [lookup.get(int(code), -1) for code in element_types], dtype=np.int8
    )


def _resolve_physical_groups(
    poly: PolyData,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Derive the per-element physical tag and the ``$PhysicalNames`` records.

    Returns the physical tag of every element plus a name → (dimension, tag)
    mapping for the groups that resolve to a single tag.
    """
    n_elems = len(poly.element_types)
    stored = poly.element_attrs.get("phys_tag")

    if stored is not None and len(stored) != n_elems:
        warnings.warn(
            f".msh: element_attrs['phys_tag'] holds {len(stored)} value(s) for"
            f" {n_elems} element(s) and was ignored; physical tags were derived"
            " from element_tags instead.",
            stacklevel=3,
        )
        stored = None

    if stored is not None:
        phys_tags = np.asarray(stored, dtype=np.int32)
    else:
        # No stored tags: hand each named group the next free id, first tag wins
        # for an element that belongs to several groups.
        phys_tags = np.zeros(n_elems, dtype=np.int32)
        for next_tag, members in enumerate(poly.element_tags.values(), start=1):
            idx = member_indices(members, n_elems)
            unassigned = idx[phys_tags[idx] == 0]
            phys_tags[unassigned] = next_tag

    names: dict[str, tuple[int, int]] = {}
    # Physical tag → the dimensions already claimed for it, or None once a group
    # spanning several dimensions holds it. A physical tag is only unique within
    # a dimension, so two names may share one as long as each stays inside a
    # dimension of its own: that is how Gmsh itself writes a surface and a volume
    # group of the same id. A reader splits a shared tag by dimension, which
    # would tear apart a group whose members span more than one, so such a group
    # keeps its tag to itself.
    claimed: dict[int, set[int] | None] = {}
    unnamed: list[str] = []
    typeless: list[str] = []
    unreachable: list[str] = []
    for name, members in poly.element_tags.items():
        idx = member_indices(members, n_elems)
        if idx.size != np.asarray(members).size:
            # Members that index no element of this mesh cost the group the
            # part of itself they stood for, so say which group lost them.
            unreachable.append(name)
        if not idx.size:
            continue
        group_tags = np.unique(phys_tags[idx])
        # A name only survives if its members agree on one non-zero tag.
        if group_tags.size != 1 or group_tags[0] == 0:
            unnamed.append(name)
            continue
        tag = int(group_tags[0])
        # Members whose type Gmsh does not know are not written either, so they
        # place no constraint on the dimension declared for the group.
        dims = {
            _ELEMENT_DIM.get(ELEMENT_TYPES_INV.get(int(poly.element_types[i]), ""), -1)
            for i in idx
        } - {-1}
        if not dims:
            # Every member has a type Gmsh does not know, so there is no
            # dimension to declare and the elements are not written either.
            typeless.append(name)
            continue
        dim = max(dims)
        if tag not in claimed:
            claimed[tag] = {dim} if len(dims) == 1 else None
        else:
            holder = claimed[tag]
            if holder is None or len(dims) != 1 or dim in holder:
                unnamed.append(name)
                continue
            holder.add(dim)
        names[name] = (dim, tag)

    if unnamed:
        warnings.warn(
            f".msh: element tag group(s) {sorted(unnamed)} were not written to"
            " $PhysicalNames; their members must share one non-zero physical"
            " tag that no earlier group of the same dimension already uses.",
            stacklevel=3,
        )
    if typeless:
        warnings.warn(
            f".msh: element tag group(s) {sorted(typeless)} were not written to"
            " $PhysicalNames; no member has a Gmsh element type.",
            stacklevel=3,
        )
    if unreachable:
        warnings.warn(
            f".msh: element tag group(s) {sorted(set(unreachable))} name"
            " members that index no element of this mesh; those members were"
            " dropped.",
            stacklevel=3,
        )
    return phys_tags, names


def _sanitize_name(name: str) -> str:
    """Make a name safe inside a quoted record, group or data field alike."""
    cleaned = name.replace('"', "'").replace("\n", " ").replace("\r", " ")
    if cleaned != name:
        warnings.warn(
            f".msh: the name {name!r} contains characters that cannot be"
            f" quoted; written as {cleaned!r}.",
            stacklevel=2,
        )
    return cleaned
