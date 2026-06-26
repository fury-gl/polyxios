"""Abaqus .inp ASCII codec — read + write."""

from pathlib import Path
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".inp"

# Abaqus element type prefix → (polyxios name, n_nodes)
# Keys are upper-case prefixes matched with .startswith(); sorted longest-first so
# longer prefixes (e.g. C3D10) always match before any shorter prefix they share.
_INP_PREFIXES: list[tuple[str, str, int]] = sorted(
    [
        ("C3D4", "tetra", 4),
        ("C3D5", "pyramid", 5),
        ("C3D6", "wedge", 6),
        ("C3D8", "hexahedron", 8),
        ("C3D10", "quadratic_tetra", 10),
        ("C3D20", "quadratic_hexahedron", 20),
        ("STRI3", "triangle", 3),
        ("S3", "triangle", 3),
        ("S4", "quad", 4),
        ("CPS3", "triangle", 3),
        ("CPE3", "triangle", 3),
        ("CPS4", "quad", 4),
        ("CPE4", "quad", 4),
        ("T3D2", "line", 2),
        ("B31", "line", 2),
    ],
    key=lambda t: len(t[0]),
    reverse=True,
)

_POLYXIOS_TO_INP: dict[str, str] = {
    "triangle": "S3",
    "quad": "S4",
    "tetra": "C3D4",
    "pyramid": "C3D5",
    "wedge": "C3D6",
    "hexahedron": "C3D8",
    "line": "T3D2",
}


def _inp_type_info(type_str: str) -> tuple[str, int] | None:
    up = type_str.upper()
    for prefix, name, n in _INP_PREFIXES:
        if up.startswith(prefix):
            return name, n
    return None


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse an Abaqus .inp file.

    Parameters
    ----------
    path
        Path to the .inp file.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData

    Raises
    ------
    CodecError
        If no ``*Node`` section is found.
    """
    raw = [
        ln.split("**")[0].rstrip()
        for ln in Path(path).read_text().splitlines()
        if not ln.strip().startswith("**")
    ]

    # Join Abaqus continuation lines: data lines ending with ',' continue on next line
    lines: list[str] = []
    buf = ""
    for ln in raw:
        stripped_ln = ln.strip()
        if not stripped_ln:
            if buf:
                lines.append(buf)
                buf = ""
            continue
        if stripped_ln.startswith("*"):
            if buf:
                lines.append(buf)
                buf = ""
            lines.append(ln)
        elif stripped_ln.endswith(",") or buf:
            buf += stripped_ln
            if not stripped_ln.endswith(","):
                lines.append(buf)
                buf = ""
        else:
            lines.append(ln)
    if buf:
        lines.append(buf)

    node_map: dict[int, int] = {}
    coords: list[float] = []
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []

    mode: str | None = None
    elem_info: tuple[str, int] | None = None

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue

        if stripped.startswith("*"):
            parts = [p.strip() for p in stripped.split(",")]
            kw = parts[0].upper()
            if kw == "*NODE":
                mode = "node"
            elif kw == "*ELEMENT":
                mode = "element"
                elem_info = None
                for part in parts[1:]:
                    if "=" in part and part.upper().startswith("TYPE"):
                        type_str = part.split("=")[1].strip()
                        elem_info = _inp_type_info(type_str)
                        if elem_info is None:
                            warnings.warn(
                                f".inp: unrecognised element type '{type_str}';"
                                " skipping block",
                                stacklevel=2,
                            )
                        break
            else:
                mode = None
            continue

        if mode == "node":
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) < 3:
                continue
            node_id = int(parts[0])
            node_map[node_id] = len(coords) // 3
            if len(parts) >= 4:
                coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])
            else:
                # 2-D node: pad Z with 0
                coords.extend([float(parts[1]), float(parts[2]), 0.0])

        elif mode == "element" and elem_info is not None:
            elem_name, n_nodes = elem_info
            parts = [p.strip() for p in stripped.split(",")]
            try:
                nodes = [node_map[int(parts[1 + j])] for j in range(n_nodes)]
            except IndexError as exc:
                raise CodecError(
                    f".inp: element row has fewer than {n_nodes} node refs"
                ) from exc
            except KeyError as exc:
                raise CodecError(
                    f".inp: element refs undefined node {exc.args[0]}"
                ) from exc
            conn_list.extend(nodes)
            offsets_list.append(offsets_list[-1] + n_nodes)
            types_list.append(ELEMENT_TYPES[elem_name])

    if not coords:
        raise CodecError(".inp: no *Node section found.")

    n_verts = len(coords) // 3
    vertices = np.array(coords, dtype=np.float64).reshape(n_verts, 3)

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
    )


def write(poly: PolyData, path: Path | str) -> None:
    """Write PolyData to Abaqus .inp ASCII format.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .inp path.

    Raises
    ------
    CodecError
        If an element type id is unknown, or has no Abaqus write mapping.

    Notes
    -----
    Quadratic types readable by this codec (``quadratic_tetra``,
    ``quadratic_hexahedron``) have no write mapping and raise ``CodecError``.
    """
    n_elems = len(poly.element_types)

    groups: dict[str, list[int]] = {}
    for i in range(n_elems):
        type_id = int(poly.element_types[i])
        name = ELEMENT_TYPES_INV.get(type_id)
        if name is None:
            raise CodecError(f".inp: unknown element type id {type_id}")
        if name not in _POLYXIOS_TO_INP:
            raise CodecError(f".inp: no write mapping for element type '{name}'")
        inp_type = _POLYXIOS_TO_INP[name]
        groups.setdefault(inp_type, []).append(i)

    lines: list[str] = [
        "*Heading",
        "** exported by polyxios",
        "*Node",
    ]
    lines.extend(
        f"{i + 1}, {v[0]:.10g}, {v[1]:.10g}, {v[2]:.10g}"
        for i, v in enumerate(poly.vertices)
    )

    elem_id = 0
    for inp_type, indices in groups.items():
        lines.append(f"*Element, type={inp_type}")
        for ei in indices:
            elem_id += 1
            s, e = int(poly.offsets[ei]), int(poly.offsets[ei + 1])
            node_str = ", ".join(
                str(poly.connectivity[s + j] + 1) for j in range(e - s)
            )
            lines.append(f"{elem_id}, {node_str}")

    lines.append("")
    Path(path).write_text("\n".join(lines))
