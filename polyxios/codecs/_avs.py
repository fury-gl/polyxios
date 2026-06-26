"""AVS-UCD .avs ASCII codec — read + write."""

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".avs"

_AVS_TO_POLYXIOS: dict[str, tuple[str, int]] = {
    "line": ("line", 2),
    "tri": ("triangle", 3),
    "quad": ("quad", 4),
    "tet": ("tetra", 4),
    "pyr": ("pyramid", 5),
    "prism": ("wedge", 6),
    "hex": ("hexahedron", 8),
}
_POLYXIOS_TO_AVS: dict[str, str] = {
    "line": "line",
    "triangle": "tri",
    "quad": "quad",
    "tetra": "tet",
    "pyramid": "pyr",
    "wedge": "prism",
    "hexahedron": "hex",
}


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse an AVS-UCD .avs file.

    Parameters
    ----------
    path
        Path to the .avs file.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData

    Raises
    ------
    CodecError
        On malformed header or unknown element type.
    """
    if lazy:
        warnings.warn(
            ".avs: lazy=True is not supported; loading eagerly.", stacklevel=2
        )
    data_lines = [
        ln
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    if not data_lines:
        raise CodecError(".avs: empty file.")

    hdr = data_lines[0].split()
    if len(hdr) < 2:
        raise CodecError(".avs: first line must have n_nodes and n_elems.")
    try:
        n_nodes = int(hdr[0])
        n_elems = int(hdr[1])
    except ValueError as exc:
        raise CodecError(
            f".avs: non-integer header values: {data_lines[0]!r}."
        ) from exc
    if n_nodes < 0 or n_elems < 0:
        raise CodecError(
            f".avs: n_nodes and n_elems must be non-negative,"
            f" got {n_nodes} and {n_elems}."
        )

    if len(data_lines) < 1 + n_nodes + n_elems:
        raise CodecError(
            f".avs: file truncated — expected {1 + n_nodes + n_elems} data lines,"
            f" got {len(data_lines)}."
        )

    node_map: dict[int, int] = {}
    coords: list[float] = []
    for i, ln in enumerate(data_lines[1 : n_nodes + 1]):
        try:
            parts = ln.split()
            node_map[int(parts[0])] = i
            coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])
        except (IndexError, ValueError) as exc:
            raise CodecError(f".avs: malformed node line {i + 1}: {ln!r}.") from exc
    vertices = np.array(coords, dtype=np.float64).reshape(n_nodes, 3)

    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    mat_ids: list[int] = []

    for ln in data_lines[n_nodes + 1 : n_nodes + 1 + n_elems]:
        try:
            parts = ln.split()
            avs_type = parts[2].lower()
            mat_id_val = int(parts[1])
            if avs_type not in _AVS_TO_POLYXIOS:
                raise CodecError(f".avs: unknown element type {avs_type!r}.")
            elem_name, n_nodes_elem = _AVS_TO_POLYXIOS[avs_type]
            mat_ids.append(mat_id_val)
            nodes = []
            for j in range(n_nodes_elem):
                nid = int(parts[3 + j])
                if nid not in node_map:
                    raise CodecError(f".avs: element references undeclared node {nid}.")
                nodes.append(node_map[nid])
        except CodecError:
            raise
        except (IndexError, ValueError) as exc:
            raise CodecError(f".avs: malformed element line: {ln!r}.") from exc
        conn_list.extend(nodes)
        offsets_list.append(offsets_list[-1] + n_nodes_elem)
        types_list.append(ELEMENT_TYPES[elem_name])

    elem_attrs: dict[str, np.ndarray] = {}
    if mat_ids:
        elem_attrs["mat_id"] = np.array(mat_ids, dtype=np.int32)

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        element_attrs=elem_attrs,
    )


def write(poly: PolyData, path: Path | str, **opts: Any) -> None:
    """Write PolyData to AVS-UCD .avs ASCII format.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .avs path.

    Notes
    -----
    Elements whose type is absent from the AVS-UCD type map are skipped with a
    warning; only the referenced vertices are written (re-indexed from 1).
    When ``element_attrs["mat_id"]`` is absent, all elements are written with
    mat_id ``1``.
    """
    if opts:
        warnings.warn(
            f".avs write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )
    n_elems = len(poly.element_types)

    writable: list[int] = []
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        if name in _POLYXIOS_TO_AVS:
            writable.append(i)
        else:
            warnings.warn(
                f".avs: element type {name!r} not supported by AVS-UCD; skipping.",
                stacklevel=2,
            )

    mat_ids_arr = poly.element_attrs.get("mat_id") if poly.element_attrs else None
    if mat_ids_arr is not None and len(mat_ids_arr) != n_elems:
        raise CodecError(
            f".avs: element_attrs['mat_id'] length {len(mat_ids_arr)}"
            f" does not match element count {n_elems}"
            f" ({len(writable)} writable)."
        )

    # Re-index vertices to only those referenced by writable elements.
    referenced_sorted = sorted(
        {
            int(poly.connectivity[k])
            for ei in writable
            for k in range(int(poly.offsets[ei]), int(poly.offsets[ei + 1]))
        }
    )
    old_to_new = {old: new for new, old in enumerate(referenced_sorted)}
    new_verts = poly.vertices[referenced_sorted]

    lines: list[str] = [f"{len(new_verts)} {len(writable)} 0 0 0"]
    lines.extend(
        f"{i + 1} {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}" for i, v in enumerate(new_verts)
    )
    for out_idx, ei in enumerate(writable):
        name = ELEMENT_TYPES_INV[int(poly.element_types[ei])]
        avs_type = _POLYXIOS_TO_AVS[name]
        s, e = int(poly.offsets[ei]), int(poly.offsets[ei + 1])
        node_str = " ".join(
            str(old_to_new[int(poly.connectivity[s + j])] + 1) for j in range(e - s)
        )
        mat_id = int(mat_ids_arr[ei]) if mat_ids_arr is not None else 1
        lines.append(f"{out_idx + 1} {mat_id} {avs_type} {node_str}")
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
