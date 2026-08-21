"""Medit ASCII mesh codec (INRIA ``.mesh``) - read + write.

A Medit file is a sequence of named sections: a mandatory
``MeshVersionFormatted``, a ``Dimension``, then entity blocks - ``Vertices``,
``Edges``, ``Triangles``, ``Quadrilaterals``, ``Tetrahedra``, ``Hexahedra``,
``Prisms``, ``Pyramids`` - each a count followed by that many records of node
indices and one trailing reference integer. ``End`` closes the file.

``.mesh`` is MFEM's extension too, so this codec claims it through the
registry's sniff hook: a file opening with ``MeshVersionFormatted`` resolves
here, one opening with ``MFEM mesh`` resolves to the MFEM codec, and a bare
``px.write(mesh, "out.mesh")`` still writes MFEM. Write a Medit file with
``fmt=".medit"``.

References are what a Medit file uses for surface and region labels, so they
land in ``element_attrs["ref"]`` and in one ``element_tags["ref_<n>"]`` group
per distinct value, which is what carries them into a ``.vtk``.
"""

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
from polyxios._io import Source, read_text, source_name, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError

EXTENSION: str = ".mesh"
# '.medit' is not a spelling found in the wild; it exists so a caller can name
# this codec on a write, where '.mesh' alone still means MFEM.
EXTENSIONS: tuple[str, ...] = (".mesh", ".medit")
# '.mesh' is MFEM's own extension as well, so it is shared rather than owned:
# the two are told apart by the keyword the file opens with.
SNIFF_EXTENSIONS: tuple[str, ...] = (".mesh",)
# Ahead of MFEM's: 'MeshVersionFormatted' opens a Medit file and nothing else.
SNIFF_PRIORITY: int = 0

# Section name → (polyxios element name, nodes per record). Medit numbers the
# corner nodes of every one of these the way VTK does, so no permutation
# applies; the higher-order sections, which do need one, are handled apart.
_SECTION_TO_ELEM: dict[str, tuple[str, int]] = {
    "EDGES": ("line", 2),
    "TRIANGLES": ("triangle", 3),
    "QUADRILATERALS": ("quad", 4),
    "TETRAHEDRA": ("tetra", 4),
    "PENTAHEDRA": ("wedge", 6),
    "PRISMS": ("wedge", 6),
    "HEXAHEDRA": ("hexahedron", 8),
    "PYRAMIDS": ("pyramid", 5),
}

# Written in this order, so a file's sections run from the lowest topological
# dimension up the way Medit's own writers emit them.
_ELEM_TO_SECTION: dict[str, str] = {
    "line": "Edges",
    "triangle": "Triangles",
    "quad": "Quadrilaterals",
    "tetra": "Tetrahedra",
    "pyramid": "Pyramids",
    "wedge": "Prisms",
    "hexahedron": "Hexahedra",
}
# Nodes per element, taken from the sections themselves so the two tables
# cannot drift apart.
_NODE_COUNT: dict[str, int] = dict(_SECTION_TO_ELEM.values())
_WRITE_ORDER: tuple[str, ...] = (
    "line",
    "triangle",
    "quad",
    "tetra",
    "pyramid",
    "wedge",
    "hexahedron",
)

# The higher-order sections. libMeshb's own documentation says the format
# fixes no node ordering for them - "there is as many HO nodes ordering in
# each kind of elements as there are programmers" - and leaves it to a
# companion ``*Ordering`` section the ASCII flavour rarely carries. Reading
# one without that section would mean guessing a permutation, so they are
# named in the warning and skipped instead.
_HIGHER_ORDER_SECTIONS: frozenset[str] = frozenset(
    {
        "EDGESP2",
        "TRIANGLESP2",
        "QUADRILATERALSQ2",
        "TETRAHEDRAP2",
        "PYRAMIDSP2",
        "PRISMSP2",
        "HEXAHEDRAQ2",
        "EDGESP3",
        "TRIANGLESP3",
        "TETRAHEDRAP3",
    }
)

# Sections that carry no mesh entities: skipped without a word, since a file
# is not worse for having them.
_QUIET_SECTIONS: frozenset[str] = frozenset(
    {
        "MESHVERSIONFORMATTED",
        "DIMENSION",
        "END",
        "CORNERS",
        "RIDGES",
        "REQUIREDVERTICES",
        "REQUIREDEDGES",
        "REQUIREDTRIANGLES",
        "REQUIREDQUADRILATERALS",
        "NORMALS",
        "TANGENTS",
        "NORMALATVERTICES",
        "NORMALATTRIANGLEVERTICES",
        "NORMALATQUADRILATERALVERTICES",
        "TANGENTATEDGEVERTICES",
        "TANGENTATVERTICES",
        "SUBDOMAINFROMGEOM",
        "ANGLEOFCORNERBOUND",
        "BOUNDINGBOX",
        "IDENTIFIER",
        "GEOMETRICSUPPORT",
        "TIME",
        "ITERATIONS",
    }
)

# A section name is a bare word; anything else on the line is its count, which
# Medit writers put on the line after the name and bamg puts on the same one.
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Tag group per distinct reference, the same shape the Nastran codec uses for
# property ids.
_REF_TAG_PREFIX: str = "ref_"
_REF_KEY: str = "ref"

_MAGIC: str = "MESHVERSIONFORMATTED"


def sniff(head: bytes) -> bool:
    """Report whether a file's opening bytes look like Medit ASCII.

    Parameters
    ----------
    head
        The file's first bytes, as handed over by the registry.

    Returns
    -------
    bool
        True when the first meaningful line names ``MeshVersionFormatted``.

    Notes
    -----
    Used to resolve ``.mesh``, which MFEM uses too. The test is deliberately
    narrow: the keyword is mandatory and opens no other format, so an MFEM
    file cannot pass it.
    """
    text = head.decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped.split()[0].upper() == _MAGIC
    return False


def _tokens(text: str) -> list[str]:
    """Return the file's whitespace-separated tokens, comments removed."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        if line.strip():
            out.extend(line.split())
    return out


def _checked_count(value: str, cap: int, what: str) -> int:
    """Parse a declared count, rejecting negatives and absurd values."""
    try:
        count = int(value)
    except ValueError as exc:
        raise CodecError(f".mesh: non-integer {what} count {value!r}.") from exc
    if count < 0:
        raise CodecError(f".mesh: negative {what} count {count}.")
    if count > cap:
        raise CodecError(f".mesh: {what} count {count} exceeds the safety cap {cap}.")
    return count


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a Medit ASCII mesh file (``.mesh``).

    Parameters
    ----------
    path
        Path to the ``.mesh`` file, or a file object.
    lazy
        Ignored (ASCII format; always loads eagerly).

    Returns
    -------
    PolyData
        Vertex references land in ``vertex_attrs["ref"]`` when any is
        non-zero; element references in ``element_attrs["ref"]`` and in one
        ``element_tags["ref_<n>"]`` group per distinct value.

    Raises
    ------
    CodecError
        On a file that does not open with ``MeshVersionFormatted``, a
        declared count that exceeds the records present or the safety caps, a
        malformed number, or an element referencing a vertex the file never
        declares.

    Notes
    -----
    The section name and its count may sit on one line or two - bamg writes
    ``Dimension 3`` on a single line where Medit puts the 3 on the next - and
    a file need not close with ``End``; both spellings read. Sections this
    codec does not decode are skipped, with a warning naming the ones that
    carried mesh entities.
    """
    if lazy:
        warnings.warn(
            ".mesh: lazy=True is not supported; loading eagerly.", stacklevel=2
        )

    text = read_text(path, encoding="utf-8-sig", errors="replace")
    tokens = _tokens(text)
    if not tokens:
        raise CodecError(f".mesh: '{source_name(path)}' is empty.")
    if tokens[0].upper() != _MAGIC:
        raise CodecError(
            f".mesh: '{source_name(path)}' does not open with"
            f" 'MeshVersionFormatted'; got {tokens[0]!r}."
        )

    dim = 3
    coords = np.zeros((0, 3), dtype=np.float64)
    vertex_refs = np.zeros(0, dtype=np.int32)
    conn: list[int] = []
    offsets: list[int] = [0]
    types: list[int] = []
    elem_refs: list[int] = []
    skipped: set[str] = set()
    higher_order: set[str] = set()

    cursor = 0
    n_tokens = len(tokens)
    while cursor < n_tokens:
        word = tokens[cursor]
        cursor += 1
        if not _NAME_RE.match(word):
            # A stray number between sections belongs to a section this codec
            # stepped over; it is not a name and cannot open one.
            continue
        name = word.upper()

        if name == "END":
            break
        if name == "MESHVERSIONFORMATTED":
            cursor += 1 if cursor < n_tokens else 0
            continue
        if name == "DIMENSION":
            if cursor >= n_tokens:
                raise CodecError(".mesh: Dimension names no value.")
            dim = _checked_count(tokens[cursor], 3, "dimension")
            cursor += 1
            if dim < 2:
                raise CodecError(f".mesh: Dimension {dim} is not a mesh.")
            continue

        if name == "VERTICES":
            if cursor >= n_tokens:
                raise CodecError(".mesh: Vertices names no count.")
            n_verts = _checked_count(tokens[cursor], MAX_SAFE_VERTICES, "vertex")
            cursor += 1
            width = dim + 1
            need = n_verts * width
            if cursor + need > n_tokens:
                raise CodecError(
                    f".mesh: Vertices declares {n_verts} but only"
                    f" {(n_tokens - cursor) // width} follow."
                )
            block = tokens[cursor : cursor + need]
            cursor += need
            try:
                values = np.array(block, dtype=np.float64).reshape(n_verts, width)
            except ValueError as exc:
                raise CodecError(f".mesh: malformed vertex record: {exc}.") from exc
            coords = np.zeros((n_verts, 3), dtype=np.float64)
            coords[:, :dim] = values[:, :dim]
            vertex_refs = values[:, dim].astype(np.int32)
            continue

        mapped = _SECTION_TO_ELEM.get(name)
        if mapped is None:
            if name in _HIGHER_ORDER_SECTIONS:
                higher_order.add(word)
            elif name not in _QUIET_SECTIONS:
                skipped.add(word)
            continue

        elem_name, n_nodes = mapped
        if cursor >= n_tokens:
            raise CodecError(f".mesh: {word} names no count.")
        n_elems = _checked_count(tokens[cursor], MAX_SAFE_ELEMENTS, "element")
        cursor += 1
        width = n_nodes + 1
        need = n_elems * width
        if cursor + need > n_tokens:
            raise CodecError(
                f".mesh: {word} declares {n_elems} but only"
                f" {(n_tokens - cursor) // width} follow."
            )
        block = tokens[cursor : cursor + need]
        cursor += need
        if not n_elems:
            continue
        try:
            records = np.array(block, dtype=np.int64).reshape(n_elems, width)
        except ValueError as exc:
            raise CodecError(f".mesh: malformed {word} record: {exc}.") from exc
        if len(conn) + n_elems * n_nodes > MAX_SAFE_CONN:
            raise CodecError(
                f".mesh: connectivity exceeds the safety cap {MAX_SAFE_CONN}."
            )
        nodes = records[:, :n_nodes]
        conn.extend(nodes.ravel().tolist())
        offsets.extend((offsets[-1] + np.arange(1, n_elems + 1) * n_nodes).tolist())
        types.extend([ELEMENT_TYPES[elem_name]] * n_elems)
        elem_refs.extend(records[:, n_nodes].tolist())

    if higher_order:
        warnings.warn(
            f".mesh: skipped the higher-order section(s) {sorted(higher_order)};"
            " the format fixes no node ordering for them, so reading one"
            " without its companion Ordering section would guess at the"
            " permutation.",
            stacklevel=2,
        )
    if skipped:
        warnings.warn(
            f".mesh: skipped the unsupported section(s) {sorted(skipped)}.",
            stacklevel=2,
        )

    if not coords.size and not conn:
        raise CodecError(f".mesh: '{source_name(path)}' holds no Vertices section.")

    connectivity = np.array(conn, dtype=np.int64)
    if connectivity.size:
        low, high = int(connectivity.min()), int(connectivity.max())
        if low < 1 or high > coords.shape[0]:
            # Medit vertex references are 1-based; a 0 or an overshoot would
            # wrap to a valid-looking index once shifted.
            raise CodecError(
                f".mesh: an element references vertex {low}..{high}, outside"
                f" 1..{coords.shape[0]}."
            )
    connectivity = (connectivity - 1).astype(np.int32)

    vertex_attrs: dict[str, np.ndarray] = {}
    if vertex_refs.size and vertex_refs.any():
        vertex_attrs[_REF_KEY] = vertex_refs

    element_attrs: dict[str, np.ndarray] = {}
    element_tags: dict[str, np.ndarray] = {}
    if elem_refs:
        refs = np.array(elem_refs, dtype=np.int32)
        # All-zero references are what a file has when it labels nothing, so
        # keeping them would invent an attribute and a tag group for every
        # unlabelled mesh.
        if refs.any():
            element_attrs[_REF_KEY] = refs
            element_tags = _ref_tags(refs)

    return PolyData(
        vertices=coords,
        connectivity=connectivity,
        offsets=np.array(offsets, dtype=np.int32),
        element_types=np.array(types, dtype=np.uint8),
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        element_tags=element_tags,
    )


def _ref_tags(refs: np.ndarray) -> dict[str, np.ndarray]:
    """Return one tag group per distinct reference, members ascending.

    A reference is what a Medit file uses for a surface or region label, and
    a label that stays a column of integers does not survive a conversion as
    a named group. One stable sort groups every value in a single pass.
    """
    order = np.argsort(refs, kind="stable").astype(np.int32)
    ranked = refs[order]
    starts = np.flatnonzero(np.concatenate(([True], ranked[1:] != ranked[:-1])))
    return {
        f"{_REF_TAG_PREFIX}{int(ref)}": members
        for ref, members in zip(ranked[starts], np.split(order, starts[1:]))
    }


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write PolyData to a Medit ASCII mesh file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output ``.mesh`` path. A bare ``.mesh`` write goes to the MFEM codec,
        so reaching here means the caller named ``fmt=".medit"`` or wrote to
        a ``.medit`` path.
    **opts
        ``float_fmt`` overrides the coordinate format specifier.

    Raises
    ------
    CodecError
        When an element does not carry the node count its section requires.

    Notes
    -----
    ``element_attrs["ref"]`` is written back as each record's trailing
    reference; failing that, references come from the ``ref_<n>`` groups in
    ``element_tags``, and failing that they are 0. Elements whose type has no
    Medit section are skipped with a warning. Sections are written in order
    of topological dimension, which is the order Medit's own writers use.
    """
    float_fmt = opts.pop("float_fmt", ".17g")
    if opts:
        warnings.warn(
            f".mesh write: unrecognized options {set(opts)}; ignored.", stacklevel=2
        )

    n_elems = len(poly.element_types)
    refs = _element_refs(poly, n_elems)

    groups: dict[str, list[int]] = {}
    skipped: set[str] = set()
    for i in range(n_elems):
        name = ELEMENT_TYPES_INV.get(int(poly.element_types[i]), "")
        if name in _ELEM_TO_SECTION:
            groups.setdefault(name, []).append(i)
        else:
            skipped.add(name or f"code {int(poly.element_types[i])}")
    if skipped:
        warnings.warn(
            f".mesh: no Medit section holds {sorted(skipped)}; those elements"
            " were skipped.",
            stacklevel=2,
        )

    lines: list[str] = ["MeshVersionFormatted 2", "", "Dimension 3", ""]

    n_verts = poly.vertices.shape[0]
    vrefs = poly.vertex_attrs.get(_REF_KEY)
    vertex_refs = (
        np.asarray(vrefs).astype(np.int64, copy=False)
        if vrefs is not None and np.asarray(vrefs).shape == (n_verts,)
        else np.zeros(n_verts, dtype=np.int64)
    )
    lines.append("Vertices")
    lines.append(str(n_verts))
    lines.extend(
        f"{v[0]:{float_fmt}} {v[1]:{float_fmt}} {v[2]:{float_fmt}}"
        f" {int(vertex_refs[i])}"
        for i, v in enumerate(poly.vertices)
    )
    lines.append("")

    connectivity = np.asarray(poly.connectivity)
    for name in _WRITE_ORDER:
        indices = groups.get(name)
        if not indices:
            continue
        expected = _NODE_COUNT[name]
        lines.append(_ELEM_TO_SECTION[name])
        lines.append(str(len(indices)))
        for ei in indices:
            start, end = int(poly.offsets[ei]), int(poly.offsets[ei + 1])
            if end - start != expected:
                raise CodecError(
                    f".mesh: element {ei} of type {name!r} has {end - start}"
                    f" node(s), expected {expected}."
                )
            nodes = " ".join(str(int(n) + 1) for n in connectivity[start:end])
            lines.append(f"{nodes} {int(refs[ei])}")
        lines.append("")

    lines.append("End")
    lines.append("")
    write_text(path, "\n".join(lines), encoding="utf-8")


def _element_refs(poly: PolyData, n_elems: int) -> np.ndarray:
    """Return one reference per element, from the attribute or the tag groups.

    Parameters
    ----------
    poly
        The mesh being written.
    n_elems
        How many elements it holds.

    Returns
    -------
    numpy.ndarray
        One int per element, zero where nothing names a reference.
    """
    stored = (poly.element_attrs or {}).get(_REF_KEY)
    if stored is not None:
        values = np.asarray(stored)
        if (
            values.ndim == 1
            and values.shape[0] == n_elems
            and values.dtype.kind in "iu"
        ):
            return values.astype(np.int64, copy=False)
        warnings.warn(
            f".mesh: element_attrs['{_REF_KEY}'] is not one integer per element;"
            " the references were taken from element_tags instead.",
            stacklevel=3,
        )

    refs = np.zeros(n_elems, dtype=np.int64)
    unnamed: set[str] = set()
    for name, members in (poly.element_tags or {}).items():
        match = re.fullmatch(rf"{_REF_TAG_PREFIX}(-?\d+)", name)
        if match is None:
            unnamed.add(name)
            continue
        picked = np.asarray(members).ravel()
        picked = picked[(picked >= 0) & (picked < n_elems)]
        refs[picked] = int(match.group(1))
    if unnamed:
        # A Medit record carries a number, not a name, so a group called
        # anything but 'ref_<n>' has nowhere to go. Numbering them here would
        # write labels the caller never chose.
        warnings.warn(
            f".mesh: element tag group(s) {sorted(unnamed)} are not named"
            " 'ref_<n>' and a Medit reference is a number; they were not"
            " written.",
            stacklevel=4,
        )
    return refs
