"""Kratos MDPA .mdpa ASCII codec — read + write."""

from pathlib import Path
import re
from typing import Any

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".mdpa"

# Kratos element class name suffix → (polyxios name, n_nodes)
_SUFFIX_RE = re.compile(r"(\d+)D(\d+)N$", re.IGNORECASE)

_N_NODES_TO_ELEM: dict[tuple[int, int], str] = {
    (2, 2): "line",
    (2, 3): "triangle",
    (2, 4): "quad",
    (3, 3): "triangle",
    (3, 4): "tetra",
    (3, 5): "pyramid",
    (3, 6): "wedge",
    (3, 8): "hexahedron",
}

_POLYXIOS_TO_KRATOS: dict[str, str] = {
    "line": "Element2D2N",
    "triangle": "Element3D3N",
    "quad": "Element3D4N",
    "tetra": "Element3D4N",
    "pyramid": "Element3D5N",
    "wedge": "Element3D6N",
    "hexahedron": "Element3D8N",
}

_NODES_PER_TYPE: dict[str, int] = {
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "tetra": 4,
    "pyramid": 5,
    "wedge": 6,
    "hexahedron": 8,
}


def _kratos_name_to_elem(name: str) -> tuple[str, int] | None:
    m = _SUFFIX_RE.search(name)
    if not m:
        return None
    dim, n = int(m.group(1)), int(m.group(2))
    elem = _N_NODES_TO_ELEM.get((dim, n))
    if elem is None:
        return None
    return elem, n


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a Kratos MDPA .mdpa file.

    Parameters
    ----------
    path
        Path to the .mdpa file.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData

    Raises
    ------
    CodecError
        If no ``Begin Nodes`` section is found.
    """
    lines = [
        ln.rstrip()
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("//")
    ]

    node_map: dict[int, int] = {}
    coords: list[float] = []
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []

    mode: str | None = None
    elem_info: tuple[str, int] | None = None

    for ln in lines:
        stripped = ln.strip()
        upper = stripped.upper()

        if upper.startswith("BEGIN NODES"):
            mode = "nodes"
            continue
        if upper.startswith("END NODES"):
            mode = None
            continue
        if upper.startswith("BEGIN ELEMENTS"):
            mode = "elements"
            klass = stripped.split()[-1] if len(stripped.split()) > 2 else ""
            elem_info = _kratos_name_to_elem(klass)
            continue
        if upper.startswith("END ELEMENTS"):
            mode = None
            elem_info = None
            continue
        if upper.startswith("BEGIN ") or upper.startswith("END "):
            mode = None
            elem_info = None
            continue

        if mode == "nodes":
            parts = stripped.split()
            if len(parts) < 4:
                continue
            nid = int(parts[0])
            node_map[nid] = len(coords) // 3
            coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])

        elif mode == "elements" and elem_info is not None:
            elem_name, n_nodes = elem_info
            parts = stripped.split()
            # format: elem_id prop_id n0 n1 n2 ...
            if len(parts) < 2 + n_nodes:
                continue
            nodes = [node_map[int(parts[2 + j])] for j in range(n_nodes)]
            conn_list.extend(nodes)
            offsets_list.append(offsets_list[-1] + n_nodes)
            types_list.append(ELEMENT_TYPES[elem_name])

    if not coords:
        raise CodecError(".mdpa: no 'Begin Nodes' section found.")

    n_verts = len(coords) // 3
    vertices = np.array(coords, dtype=np.float64).reshape(n_verts, 3)

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Write PolyData to Kratos MDPA .mdpa format.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .mdpa path.
    """
    n_elems = len(poly.element_types)

    groups: dict[str, list[int]] = {}
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        if name in _POLYXIOS_TO_KRATOS:
            klass = _POLYXIOS_TO_KRATOS[name]
            groups.setdefault(klass, []).append(i)

    lines: list[str] = [
        "Begin ModelPartData",
        "End ModelPartData",
        "",
        "Begin Properties 0",
        "End Properties",
        "",
        "Begin Nodes",
    ]
    lines.extend(
        f"{i + 1}  {v[0]:.10g}  {v[1]:.10g}  {v[2]:.10g}"
        for i, v in enumerate(poly.vertices)
    )
    lines.append("End Nodes")
    lines.append("")

    for klass, indices in groups.items():
        lines.append(f"Begin Elements {klass}")
        for out_idx, ei in enumerate(indices):
            s, e = int(poly.offsets[ei]), int(poly.offsets[ei + 1])
            node_str = "  ".join(
                str(poly.connectivity[s + j] + 1) for j in range(e - s)
            )
            lines.append(f"{out_idx + 1}  0  {node_str}")
        lines.append("End Elements")
        lines.append("")

    Path(path).write_text("\n".join(lines))
