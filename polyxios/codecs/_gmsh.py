"""Gmsh .msh v2 ASCII codec — read + write."""

from pathlib import Path
from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".msh"

# Gmsh v2 element type → (polyxios name, n_nodes)
_GMSH_TO_POLYXIOS: dict[int, tuple[str, int]] = {
    1: ("line", 2),
    2: ("triangle", 3),
    3: ("quad", 4),
    4: ("tetra", 4),
    5: ("hexahedron", 8),
    6: ("wedge", 6),
    7: ("pyramid", 5),
    21: ("quadratic_triangle", 6),
    29: ("quadratic_tetra", 10),
}
_POLYXIOS_TO_GMSH: dict[str, int] = {
    "line": 1,
    "triangle": 2,
    "quad": 3,
    "tetra": 4,
    "hexahedron": 5,
    "wedge": 6,
    "pyramid": 7,
}


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a Gmsh v2 .msh file.

    Parameters
    ----------
    path
        Path to the .msh file.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData
        Element physical tags stored in ``element_attrs["phys_tag"]``.

    Raises
    ------
    CodecError
        On missing ``$MeshFormat``, ``$Nodes``, or ``$Elements`` sections.
    """
    text = Path(path).read_text()
    sections = _split_sections(text)

    if "$MeshFormat" not in sections:
        raise CodecError(".msh: missing $MeshFormat section.")
    if "$Nodes" not in sections:
        raise CodecError(".msh: missing $Nodes section.")
    if "$Elements" not in sections:
        raise CodecError(".msh: missing $Elements section.")

    fmt_line = sections["$MeshFormat"][0].split()
    if not fmt_line[0].startswith("2"):
        raise CodecError(f".msh: only v2 ASCII supported; got version {fmt_line[0]}.")

    # --- Nodes ---
    node_lines = sections["$Nodes"]
    n_nodes = int(node_lines[0])
    node_map: dict[int, int] = {}
    coords: list[float] = []
    for i, ln in enumerate(node_lines[1 : n_nodes + 1]):
        parts = ln.split()
        node_map[int(parts[0])] = i
        coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.array(coords, dtype=np.float64).reshape(n_nodes, 3)

    # --- Elements ---
    elem_lines = sections["$Elements"]
    n_elems_total = int(elem_lines[0])
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    phys_tags: list[int] = []

    for ln in elem_lines[1 : n_elems_total + 1]:
        parts = ln.split()
        gmsh_type = int(parts[1])
        if gmsh_type not in _GMSH_TO_POLYXIOS:
            continue
        elem_name, n_nodes_elem = _GMSH_TO_POLYXIOS[gmsh_type]
        n_tags = int(parts[2])
        phys_tag = int(parts[3]) if n_tags > 0 else 0
        node_start = 3 + n_tags
        nodes = [node_map[int(parts[node_start + j])] for j in range(n_nodes_elem)]
        conn_list.extend(nodes)
        offsets_list.append(offsets_list[-1] + n_nodes_elem)
        types_list.append(ELEMENT_TYPES[elem_name])
        phys_tags.append(phys_tag)

    elem_attrs: dict[str, np.ndarray] = {}
    if phys_tags:
        elem_attrs["phys_tag"] = np.array(phys_tags, dtype=np.int32)

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        element_attrs=elem_attrs,
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Write PolyData to Gmsh v2 ASCII .msh format.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .msh path.
    """
    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    lines: list[str] = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$Nodes",
        str(n_verts),
    ]
    lines.extend(
        f"{i + 1} {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}"
        for i, v in enumerate(poly.vertices)
    )
    lines.append("$EndNodes")

    # Count writable elements
    writable: list[int] = []
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        if name in _POLYXIOS_TO_GMSH:
            writable.append(i)

    lines.append("$Elements")
    lines.append(str(len(writable)))
    for out_idx, ei in enumerate(writable):
        name = ELEMENT_TYPES_INV[int(poly.element_types[ei])]
        gmsh_type = _POLYXIOS_TO_GMSH[name]
        s, e = int(poly.offsets[ei]), int(poly.offsets[ei + 1])
        node_str = " ".join(str(poly.connectivity[s + j] + 1) for j in range(e - s))
        lines.append(f"{out_idx + 1} {gmsh_type} 2 0 0 {node_str}")
    lines.append("$EndElements")
    lines.append("")

    Path(path).write_text("\n".join(lines))


def _split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    content: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("$End"):
            if current is not None:
                sections[current] = content
            current = None
            content = []
        elif ln.startswith("$"):
            current = ln
            content = []
        elif current is not None:
            content.append(ln)
    return sections
