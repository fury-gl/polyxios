"""DOLFIN/FEniCS XML .xml codec - read + write."""

import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from polyxios._dimension import has_solid_cells, mark_2d, output_dimension
from polyxios._element_types import ELEMENT_TYPES, ELEMENT_TYPES_INV
from polyxios._io import Source, open_read, open_write
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

# The XML prolog, comments and a doctype - internal subset included - may
# precede the root element.
_ROOT_TAG = re.compile(
    rb"\A\s*(?:<\?.*?\?>\s*|<!--.*?-->\s*|<!DOCTYPE[^\[>]*(?:\[.*?\])?\s*>\s*)*"
    rb"<(dolfin|mesh)[\s/>]",
    re.DOTALL,
)
# dolfin-convert writes its region markers under a <dolfin> root too, as a
# <mesh_function>, and Ogre's .mesh.xml has a <mesh> root of its own; only
# a <mesh> naming its cell type is one read() takes.
_MESH_TAG = re.compile(rb"<mesh\b[^>]*\bcelltype\s*=")


def sniff(head: bytes) -> bool:
    """Report whether a file's opening bytes look like DOLFIN XML.

    Parameters
    ----------
    head
        The file's first bytes.

    Returns
    -------
    bool
        True when the root element is ``<dolfin>``, or ``<mesh>`` as some
        exporters write it - the two roots :func:`read` accepts - and a
        ``<mesh>`` element in *head* declares a ``celltype``. A
        ``<mesh_function>`` file or a ``<mesh>`` with no cell type is not
        one.
    """
    head = head.removeprefix(b"\xef\xbb\xbf")
    return _ROOT_TAG.match(head) is not None and _MESH_TAG.search(head) is not None


def _int(text: str, *, what: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise CodecError(f".xml: {what} {text[:40]!r} is not an integer.") from exc


def _float(text: str, *, what: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise CodecError(f".xml: {what} {text[:40]!r} is not a number.") from exc


def read(path: Source, *, lazy: bool = False) -> PolyData:
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
        # Parsed from a handle rather than the path itself, so a buffer and a
        # gzip stream reach the parser the same way a plain file does.
        with open_read(path) as fh:
            tree = ET.parse(fh)
    except ET.ParseError as exc:
        raise CodecError(f".xml: XML parse error: {exc}") from exc
    except (LookupError, ValueError) as exc:
        # expat raises these for an encoding its prolog names but it cannot
        # decode: unknown, or multi-byte.
        raise CodecError(f".xml: XML encoding not readable: {exc}") from exc

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
    # A DOLFIN vertex spells z only in a 3-D mesh, so its absence is what says
    # the file is flat when <mesh> forgot to declare a dim.
    saw_z = False
    for expected_idx, v in enumerate(verts_el):
        idx_attr = v.get("index")
        if idx_attr is not None and _int(idx_attr, what="vertex index") != expected_idx:
            raise CodecError(
                f".xml: vertex index {idx_attr} != expected {expected_idx}"
                f" (non-sequential indices not supported)."
            )
        for attr in ("x", "y"):
            val = v.get(attr)
            if val is None:
                raise CodecError(
                    f".xml: vertex (index={v.get('index', '?')!r}) missing"
                    f" '{attr}' attribute."
                )
            coords.append(_float(val, what=f"vertex {expected_idx} {attr}"))
        z_attr = v.get("z")
        saw_z = saw_z or z_attr is not None
        coords.append(
            _float(z_attr, what=f"vertex {expected_idx} z")
            if z_attr is not None
            else 0.0
        )
    n_verts = len(coords) // 3
    declared_verts = verts_el.get("size")
    if (
        declared_verts is not None
        and _int(declared_verts, what="vertices size") != n_verts
    ):
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
        if idx_attr is not None and _int(idx_attr, what="cell index") != expected_idx:
            raise CodecError(
                f".xml: cell index {idx_attr} != expected {expected_idx}"
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
            node = _int(val, what=f"cell {expected_idx} {attr}")
            if not 0 <= node < n_verts:
                raise CodecError(
                    f".xml: cell {expected_idx} {attr}={node} is not one of the"
                    f" {n_verts} vertices."
                )
            nodes.append(node)
        conn_list.extend(nodes)
        offsets_list.append(offsets_list[-1] + n_nodes_per)
        types_list.append(elem_code)

    n_cells = len(types_list)
    declared_cells = cells_el.get("size")
    if (
        declared_cells is not None
        and _int(declared_cells, what="cells size") != n_cells
    ):
        raise CodecError(f".xml: <cells size={declared_cells!r}> but parsed {n_cells}.")

    # <mesh dim=> is DOLFIN's own geometric dimension and outranks the guess;
    # only a file that omitted it, or spelt it as something other than the
    # one digit a dimension is, is read off the vertices.
    declared_dim = (mesh_el.get("dim") or "").strip()
    read_dim = (
        int(declared_dim)
        if declared_dim.isascii() and declared_dim.isdigit() and len(declared_dim) == 1
        else (2 if n_verts and not saw_z else 3)
    )

    return PolyData(
        vertices=vertices,
        connectivity=np.array(conn_list, dtype=np.int32),
        offsets=np.array(offsets_list, dtype=np.int32),
        element_types=np.array(types_list, dtype=np.uint8),
        global_attrs=mark_2d(read_dim),
    )


def write(
    poly: PolyData,
    path: Source,
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
        Geometric dimension to embed in the ``<mesh dim="...">`` attribute,
        and the caller's own word when given. Inferred when None: 2 for a mesh
        whose z column is entirely zero, 3 for one that carries a z, for an
        empty mesh - DOLFIN's own default - and for a mesh of solid cells
        however flat it lies, DOLFIN needing its geometric dimension to be at
        least its topological one. ``write`` always emits z, so a
        round-tripped mesh keeps its coordinates whatever dim it declares.

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
            f" mixed type - only '{celltype_name}' written).",
            stacklevel=2,
        )

    _, node_attrs = _CELLTYPE_TO_POLYXIOS[celltype_name]
    if dim is None:
        # An empty mesh has no extent to read a dimension off, and DOLFIN's own
        # default is 3, so it keeps that rather than being called flat.
        dim = 3 if n_verts == 0 else output_dimension(poly, fmt=".xml", flat_default=2)
        if dim == 2 and has_solid_cells(poly):
            # DOLFIN needs its geometric dimension to be at least the
            # topological one, so a tetrahedron in a dim=2 mesh is one it
            # refuses. An explicit ``dim=`` is the caller's own word and is
            # left alone; this is only the inferred value.
            dim = 3

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
    with open_write(path) as fh:
        tree.write(fh, encoding="utf-8", xml_declaration=True)
