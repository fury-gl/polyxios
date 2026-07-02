"""DOLFIN/FEniCS XML .xml codec — read + write."""

from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".xml"

_CELLTYPE_TO_POLYXIOS: dict[str, tuple[str, list[str]]] = {
    "interval": ("line", ["v0", "v1"]),
    "triangle": ("triangle", ["v0", "v1", "v2"]),
    "tetrahedron": ("tetra", ["v0", "v1", "v2", "v3"]),
    "quadrilateral": ("quad", ["v0", "v1", "v2", "v3"]),
    "hexahedron": ("hexahedron", ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]),
}
_POLYXIOS_TO_CELLTYPE: dict[str, str] = {
    "line": "interval",
    "triangle": "triangle",
    "tetra": "tetrahedron",
    "quad": "quadrilateral",
    "hexahedron": "hexahedron",
}


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse a DOLFIN XML .xml mesh file.

    Parameters
    ----------
    path
        Path to the DOLFIN XML file.
    lazy
        Ignored (XML format; always loads eagerly).

    Returns
    -------
    PolyData

    Raises
    ------
    CodecError
        On missing ``<mesh>`` element, unsupported cell type, or malformed
        vertex/cell attributes.
    """
    if lazy:
        warnings.warn(
            ".xml: lazy=True ignored; XML format always loads eagerly.",
            stacklevel=2,
        )

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise CodecError(f".xml: XML parse error: {exc}") from exc

    root = tree.getroot()
    # Some exporters omit the <dolfin> wrapper and use <mesh> as root directly.
    mesh_el = root.find("mesh") if root.tag != "mesh" else root
    if mesh_el is None:
        raise CodecError(".xml: no <mesh> element found.")

    celltype = mesh_el.get("celltype", "")
    if celltype not in _CELLTYPE_TO_POLYXIOS:
        raise CodecError(f".xml: unsupported celltype {celltype!r}.")

    elem_name, node_attrs = _CELLTYPE_TO_POLYXIOS[celltype]
    n_nodes_per = len(node_attrs)

    verts_el = mesh_el.find("vertices")
    if verts_el is None:
        raise CodecError(".xml: no <vertices> element.")

    coords: list[float] = []
    for expected_idx, v in enumerate(verts_el):
        idx_attr = v.get("index")
        if idx_attr is not None and int(idx_attr) != expected_idx:
            raise CodecError(
                f".xml: vertex index {int(idx_attr)} != expected {expected_idx}"
                f" (non-sequential indices not supported)."
            )
        for attr in ("x", "y"):
            val = v.get(attr)
            if val is None:
                raise CodecError(
                    f".xml: vertex (index={v.get('index', '?')!r}) missing"
                    f" '{attr}' attribute."
                )
            coords.append(float(val))
        coords.append(float(v.get("z", "0")))
    n_verts = len(coords) // 3
    declared_verts = verts_el.get("size")
    if declared_verts is not None and int(declared_verts) != n_verts:
        raise CodecError(
            f".xml: <vertices size={declared_verts!r}> but parsed {n_verts}."
        )
    vertices = np.array(coords, dtype=np.float64).reshape(n_verts, 3)

    cells_el = mesh_el.find("cells")
    if cells_el is None:
        raise CodecError(".xml: no <cells> element.")

    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    types_list: list[int] = []
    elem_code = ELEMENT_TYPES[elem_name]
    for expected_idx, cell in enumerate(cells_el):
        idx_attr = cell.get("index")
        if idx_attr is not None and int(idx_attr) != expected_idx:
            raise CodecError(
                f".xml: cell index {int(idx_attr)} != expected {expected_idx}"
                f" (non-sequential indices not supported)."
            )
        nodes: list[int] = []
        for attr in node_attrs:
            val = cell.get(attr)
            if val is None:
                raise CodecError(
                    f".xml: cell (index={cell.get('index', '?')!r}) missing"
                    f" '{attr}' attribute."
                )
            nodes.append(int(val))
        conn_list.extend(nodes)
        offsets_list.append(offsets_list[-1] + n_nodes_per)
        types_list.append(elem_code)

    n_cells = len(types_list)
    declared_cells = cells_el.get("size")
    if declared_cells is not None and int(declared_cells) != n_cells:
        raise CodecError(f".xml: <cells size={declared_cells!r}> but parsed {n_cells}.")

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
    )


def write(
    poly: PolyData,
    path: Path | str,
    *,
    dim: int | None = None,
) -> None:
    """Write PolyData to DOLFIN XML format.

    Only one element type is written per file. Mixed-type meshes should be
    split first; elements of other types are skipped with a warning.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output .xml path.
    dim
        Geometric dimension to embed in the ``<mesh dim="...">`` attribute.
        Inferred from vertex z-coordinates when None: 3 if any z != 0.0
        (exact comparison), otherwise 2. ``write`` always emits z, so
        round-tripped meshes preserve their original dim correctly.

    Raises
    ------
    CodecError
        If no supported element type is found.
    """
    n_elems = len(poly.element_types)
    n_verts = poly.vertices.shape[0]

    celltype_name: str | None = None
    elem_indices: list[int] = []
    skipped = 0
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        if name not in _POLYXIOS_TO_CELLTYPE:
            skipped += 1
            continue
        ct = _POLYXIOS_TO_CELLTYPE[name]
        if celltype_name is None:
            celltype_name = ct
        if ct == celltype_name:
            elem_indices.append(i)
        else:
            skipped += 1

    if celltype_name is None:
        raise CodecError(".xml: no supported element type found.")

    if skipped:
        warnings.warn(
            f".xml write: {skipped} element(s) skipped (unsupported type or"
            f" mixed type — only '{celltype_name}' written).",
            stacklevel=2,
        )

    _, node_attrs = _CELLTYPE_TO_POLYXIOS[celltype_name]
    if dim is None:
        dim = 3 if n_verts == 0 or np.any(poly.vertices[:, 2] != 0) else 2

    root = ET.Element("dolfin")
    mesh_el = ET.SubElement(root, "mesh", celltype=celltype_name, dim=str(dim))

    verts_el = ET.SubElement(mesh_el, "vertices", size=str(n_verts))
    for i, v in enumerate(poly.vertices):
        ET.SubElement(
            verts_el,
            "vertex",
            index=str(i),
            x=f"{v[0]:.10g}",
            y=f"{v[1]:.10g}",
            z=f"{v[2]:.10g}",
        )

    cells_el = ET.SubElement(mesh_el, "cells", size=str(len(elem_indices)))
    for out_idx, ei in enumerate(elem_indices):
        s = int(poly.offsets[ei])
        attrs: dict[str, str] = {"index": str(out_idx)}
        for j, attr in enumerate(node_attrs):
            attrs[attr] = str(int(poly.connectivity[s + j]))
        ET.SubElement(cells_el, celltype_name, **attrs)

    ET.indent(root)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
