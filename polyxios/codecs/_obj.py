import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._io import Source, open_text, source_name, write_text
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".obj"


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse an OBJ file and return a PolyData.

    Parameters
    ----------
    path
        Path to the .obj file.
    lazy
        Not supported for OBJ - raises LazyReadError.

    Returns
    -------
    PolyData
        Parsed mesh data. ``vn`` and ``vt`` records land in ``vertex_attrs``
        as ``normals`` and ``texcoords``.

    Raises
    ------
    LazyReadError
        Always, if lazy=True.
    CodecError
        On a face index naming a vertex, texture coordinate or normal the
        file has not declared, or on a ``v``, ``vn`` or ``vt`` record that
        does not carry the components its directive needs.

    Notes
    -----
    OBJ indexes normals and texture coordinates **per face corner**, so a
    file may hold more of either than it holds vertices - a cube with a
    seam has two texture coordinates for one corner. polyxios stores one
    value per vertex, so a corner assigns to its vertex and a vertex named
    twice with different values keeps the last, which is warned about.

    A negative index counts back from what the file has declared so far, as
    the format defines; ``f -3 -2 -1`` is the last three vertices.

    A ``vt`` record may carry a third component, the depth of a volumetric
    texture. ``texcoords`` holds the two a surface uses, and the third is
    dropped.
    """
    if lazy:
        raise LazyReadError("OBJ format does not support lazy reads (ASCII only).")

    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    texcoords: list[list[float]] = []

    # face connectivity as list of (vertex_indices, normal_indices_or_None)
    face_vertices: list[list[int]] = []
    face_normals: list[list[int | None]] = []
    face_texcoords: list[list[int | None]] = []
    face_materials: list[str] = []

    # multi-group tag tracking: active groups at any point
    active_groups: list[str] = []
    element_tag_accumulator: dict[str, list[int]] = {}

    mtl_file: str | None = None
    object_name: str | None = None
    current_material = ""

    # Naming the source costs a path walk, so it is done once here rather
    # than per line; a message is spelled only when one is raised.
    source = source_name(path)

    with open_text(path, encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            directive = parts[0].lower()

            if directive == "v":
                vertices.append(
                    _record(parts, width=3, needs=3, source=source, line_no=line_no)
                )

            elif directive == "vn":
                normals.append(
                    _record(parts, width=3, needs=3, source=source, line_no=line_no)
                )

            elif directive == "vt":
                # A vt carries one to three values; a surface uses the first
                # two, and a missing v is the zero the format implies.
                texcoords.append(
                    _record(parts, width=2, needs=1, source=source, line_no=line_no)
                )

            elif directive == "f":
                v_idx, vt_idx, vn_idx = _parse_face(
                    parts[1:],
                    n_vertices=len(vertices),
                    n_texcoords=len(texcoords),
                    n_normals=len(normals),
                    source=source,
                    line_no=line_no,
                )
                face_vertices.append(v_idx)
                face_normals.append(vn_idx)
                face_texcoords.append(vt_idx)
                face_materials.append(current_material)
                face_idx = len(face_vertices) - 1
                for g in active_groups:
                    element_tag_accumulator.setdefault(g, []).append(face_idx)

            elif directive == "g":
                # "g name1 name2 ..." - all names become active groups. A bare
                # "g" is the format's way back to the unnamed default group,
                # which is the absence of a group rather than one more tag.
                active_groups = parts[1:] if len(parts) > 1 else []

            elif directive == "usemtl":
                current_material = parts[1] if len(parts) > 1 else ""

            elif directive == "mtllib":
                mtl_file = " ".join(parts[1:])

            elif directive == "o":
                object_name = " ".join(parts[1:])

    if not vertices:
        return PolyData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            connectivity=np.array([], dtype=np.int32),
            offsets=np.array([0], dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
        )

    verts_arr = np.array(vertices, dtype=np.float64)

    # Build CSR connectivity from face_vertices
    conn_list: list[int] = []
    offsets_list: list[int] = [0]
    type_codes: list[int] = []

    tri_code = ELEMENT_TYPES["triangle"]
    quad_code = ELEMENT_TYPES["quad"]
    poly_code = ELEMENT_TYPES["polygon"]

    for face in face_vertices:
        n = len(face)
        conn_list.extend(face)
        offsets_list.append(offsets_list[-1] + n)
        if n == 3:
            type_codes.append(tri_code)
        elif n == 4:
            type_codes.append(quad_code)
        else:
            type_codes.append(poly_code)

    connectivity = np.array(conn_list, dtype=np.int32)
    offsets = np.array(offsets_list, dtype=np.int32)
    element_types = np.array(type_codes, dtype=np.uint8)

    vertex_attrs: dict[str, np.ndarray] = {}
    folded = (
        ("normals", normals, face_normals, "normal"),
        ("texcoords", texcoords, face_texcoords, "texture coordinate"),
    )
    for key, values, face_indices, what in folded:
        if not values:
            continue
        # _per_vertex gives up on records it cannot line up with the
        # vertices; an attribute that is not there is what a caller can
        # handle, a None sitting where an array belongs is not.
        array = _per_vertex(
            values,
            face_vertices,
            face_indices,
            n_vertices=len(vertices),
            what=what,
        )
        if array is not None:
            vertex_attrs[key] = array

    element_attrs: dict[str, np.ndarray] = {}
    if any(m != "" for m in face_materials):
        element_attrs["material"] = np.array(face_materials, dtype=object)

    element_tags = {
        g: np.array(idxs, dtype=np.int32) for g, idxs in element_tag_accumulator.items()
    }

    global_attrs: dict[str, object] = {}
    if mtl_file is not None:
        global_attrs["mtl_file"] = mtl_file
    if object_name is not None:
        global_attrs["object_name"] = object_name

    return PolyData(
        vertices=verts_arr,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def write(poly: PolyData, path: Source, **opts: object) -> None:
    """Serialise PolyData to an OBJ file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    **opts
        Unused; accepted for API uniformity.

    Notes
    -----
    ``vertex_attrs['normals']`` and ``vertex_attrs['texcoords']`` are written
    as ``vn`` and ``vt`` records, one per vertex, and each face corner names
    the record sharing its vertex index.

    Each ``element_tags`` group becomes a ``g`` directive covering the run of
    faces that carries it. A face in no group is preceded by a bare ``g``, so
    it does not inherit the group of the face above it.
    """
    lines: list[str] = []

    lines.append("# Written by polyxios")

    global_attrs = poly.global_attrs
    if "object_name" in global_attrs:
        lines.append(f"o {global_attrs['object_name']}")
    if "mtl_file" in global_attrs:
        lines.append(f"mtllib {global_attrs['mtl_file']}")

    lines.extend(f"v {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}" for v in poly.vertices)

    n_verts = len(poly.vertices)

    vn_rows = None
    if "normals" in poly.vertex_attrs:
        vn_rows = _record_rows(
            poly.vertex_attrs["normals"], width=3, n_vertices=n_verts, what="vn"
        )
    if vn_rows is not None:
        lines.extend(f"vn {vn[0]:.10g} {vn[1]:.10g} {vn[2]:.10g}" for vn in vn_rows)

    uv_rows = None
    if "texcoords" in poly.vertex_attrs:
        uv_rows = _record_rows(
            poly.vertex_attrs["texcoords"], width=2, n_vertices=n_verts, what="vt"
        )
    if uv_rows is not None:
        lines.extend(f"vt {row[0]:.10g} {row[1]:.10g}" for row in uv_rows)

    # Build reverse tag map: element_idx - set of group names
    idx_to_groups: dict[int, list[str]] = {}
    for g, idxs in poly.element_tags.items():
        for i in idxs:
            idx_to_groups.setdefault(int(i), []).append(g)

    n_elems = len(poly.element_types)
    # A corner names a record only when the record was written; an attribute
    # dropped above must not leave the faces indexing it.
    has_normals = vn_rows is not None
    has_uv = uv_rows is not None
    has_material = "material" in poly.element_attrs

    current_groups: list[str] | None = None
    current_material: str | None = None

    for i in range(n_elems):
        start = int(poly.offsets[i])
        end = int(poly.offsets[i + 1])
        face_verts = poly.connectivity[start:end]

        # Emit group changes. A face in no group closes the run with a bare
        # "g": without it the reader would keep the last group active and
        # hand back a tag naming faces the mesh never put in it.
        groups = sorted(idx_to_groups.get(i, []))
        if groups != current_groups and (groups or current_groups):
            lines.append(f"g {' '.join(groups)}" if groups else "g")
            current_groups = groups

        # Emit material changes
        if has_material:
            mat = str(poly.element_attrs["material"][i])
            if mat != current_material:
                lines.append(f"usemtl {mat}")
                current_material = mat

        # Emit face (1-based indices). A corner names the uv and the normal
        # stored against its own vertex, which is what makes the file read
        # back into the arrays it was written from.
        corners = (int(vi) + 1 for vi in face_verts)
        if has_normals and has_uv:
            face_str = " ".join(f"{c}/{c}/{c}" for c in corners)
        elif has_normals:
            face_str = " ".join(f"{c}//{c}" for c in corners)
        elif has_uv:
            face_str = " ".join(f"{c}/{c}" for c in corners)
        else:
            face_str = " ".join(str(c) for c in corners)
        lines.append(f"f {face_str}")

    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")


def _record_rows(
    values: object, *, width: int, n_vertices: int, what: str
) -> np.ndarray | None:
    """Shape a vertex attribute into rows an OBJ record can hold.

    Parameters
    ----------
    values
        A ``vertex_attrs`` entry, one row per vertex.
    width
        Components an OBJ record of this kind carries: 3 for ``vn``, 2 for
        ``vt``.
    n_vertices
        Rows the mesh needs: a face corner names the record sharing its own
        vertex index, so a shorter array leaves corners pointing past the
        end of the record list.
    what
        ``'vn'`` or ``'vt'``, for the warning.

    Returns
    -------
    numpy.ndarray or None
        An ``(n_vertices, width)`` array. A row too short is padded with
        zeros rather than indexed past its end, and a value the format
        cannot spell - the NaN a reader leaves where no face named a
        record, an infinity - is written as zero: nothing indexes those
        rows, and ``vt nan nan`` is not a record another reader takes.

        None when the attribute holds no usable row per vertex, or holds
        something an OBJ record has no way to spell. Writing it anyway would
        emit faces indexing records that are not in the file, which no
        reader can take - this one included.
    """
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        # An attribute of strings or objects is a real thing to be carrying
        # - a label per vertex, say - and a vn record has no way to spell
        # it. Dropping it is the same answer as a wrong shape.
        warnings.warn(
            f".obj: vertex attribute for {what} holds values that are not"
            " numbers; not written.",
            stacklevel=3,
        )
        return None
    # A one-value-per-vertex attribute arrives one-dimensional; read it as a
    # column, which np.atleast_2d would turn into a single row instead.
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n_vertices:
        warnings.warn(
            f".obj: vertex attribute for {what} has shape {arr.shape}, not one"
            f" row per vertex ({n_vertices}); not written.",
            stacklevel=3,
        )
        return None
    if arr.shape[1] < width:
        arr = np.pad(arr, ((0, 0), (0, width - arr.shape[1])))
    return np.nan_to_num(arr[:, :width], nan=0.0, posinf=0.0, neginf=0.0)


def _record(
    parts: list[str], *, width: int, needs: int, source: str, line_no: int
) -> list[float]:
    """Read a ``v``, ``vn`` or ``vt`` record into a fixed-width row.

    Parameters
    ----------
    parts
        The line's whitespace-separated tokens, directive included.
    width
        Components the row must end up with, so every record of a kind is
        the same width - which is what folding them into a per-vertex array
        needs, and what keeps a ragged file from reaching numpy as a
        broadcast error naming nothing.
    needs
        Components the directive cannot do without. A ``vt`` may carry one
        value and mean zero for the other; a ``v`` carrying two is a
        truncated file, not a point.
    source
        File name, for an error message.
    line_no
        Line the record is on, for an error message.

    Returns
    -------
    list of float
        Exactly ``width`` values, zero-padded when the record carried fewer
        and truncated when it carried more - a ``vt`` may hold a third
        component that a surface has no use for.

    Raises
    ------
    CodecError
        If the record carries fewer than ``needs`` values, or one of them is
        not a number.
    """
    values = parts[1 : width + 1]
    if len(values) < needs:
        raise CodecError(
            f".obj: '{source}' line {line_no}: {parts[0]!r} carries"
            f" {len(parts) - 1} value(s); it needs at least {needs}."
        )
    try:
        row = [float(text) for text in values]
    except ValueError as exc:
        raise CodecError(
            f".obj: '{source}' line {line_no}: {parts[0]!r} carries"
            f" {' '.join(values)!r}, which is not a row of numbers."
        ) from exc
    row.extend([0.0] * (width - len(row)))
    return row


def _resolve_index(
    token: str, *, declared: int, what: str, source: str, line_no: int
) -> int:
    """Turn one OBJ index token into a 0-based index into what it names.

    Parameters
    ----------
    token
        The index as written: 1-based, or negative to count back from the
        last record declared so far.
    declared
        How many records of this kind the file has declared up to this line.
    what
        'vertex', 'texture coordinate' or 'normal', for the error message.
    source
        File name, for the error message.
    line_no
        Line the index is on, for the error message.

    Returns
    -------
    int
        The 0-based index.

    Raises
    ------
    CodecError
        If the token is not an integer, or names a record the file has not
        declared. Left alone it would index a numpy array from the end and
        put a different vertex in the mesh without a word.
    """
    try:
        value = int(token)
    except ValueError as exc:
        raise CodecError(
            f".obj: '{source}' line {line_no}: {what} index {token!r} is not a number."
        ) from exc

    if value == 0:
        raise CodecError(
            f".obj: '{source}' line {line_no}: {what} index 0 is not valid;"
            " the format numbers from 1, and counts back from -1."
        )

    idx = value - 1 if value > 0 else declared + value
    if not 0 <= idx < declared:
        raise CodecError(
            f".obj: '{source}' line {line_no}: {what} index {value} names a"
            f" record the file has not declared ({declared} so far)."
        )
    return idx


def _parse_face(
    tokens: list[str],
    *,
    n_vertices: int,
    n_texcoords: int,
    n_normals: int,
    source: str,
    line_no: int,
) -> tuple[list[int], list[int | None], list[int | None]]:
    """Parse OBJ face tokens into 0-based vertex, uv and normal index lists.

    Parameters
    ----------
    tokens
        Face token list after the 'f' directive.
    n_vertices, n_texcoords, n_normals
        How many of each record the file has declared so far, which is what
        a negative index counts back from and what an index is checked
        against.
    source
        File name, for an error message.
    line_no
        Line the face is on, for an error message.

    Returns
    -------
    tuple[list[int], list[int | None], list[int | None]]
        (vertex_indices, texcoord_indices, normal_indices), 0-based, with
        None where the corner named none.
    """
    v_idx: list[int] = []
    vt_idx: list[int | None] = []
    vn_idx: list[int | None] = []

    for tok in tokens:
        parts = tok.split("/")
        v_idx.append(
            _resolve_index(
                parts[0],
                declared=n_vertices,
                what="vertex",
                source=source,
                line_no=line_no,
            )
        )
        if len(parts) >= 2 and parts[1]:
            vt_idx.append(
                _resolve_index(
                    parts[1],
                    declared=n_texcoords,
                    what="texture coordinate",
                    source=source,
                    line_no=line_no,
                )
            )
        else:
            vt_idx.append(None)
        if len(parts) >= 3 and parts[2]:
            vn_idx.append(
                _resolve_index(
                    parts[2],
                    declared=n_normals,
                    what="normal",
                    source=source,
                    line_no=line_no,
                )
            )
        else:
            vn_idx.append(None)

    return v_idx, vt_idx, vn_idx


def _per_vertex(
    values: list[list[float]],
    face_vertices: list[list[int]],
    face_indices: list[list[int | None]],
    *,
    n_vertices: int,
    what: str,
) -> np.ndarray | None:
    """Fold per-corner OBJ records into one value per vertex.

    Parameters
    ----------
    values
        The ``vt`` or ``vn`` records, in file order.
    face_vertices
        Vertex indices per face.
    face_indices
        Indices into ``values`` per face corner, None where the corner named
        none.
    n_vertices
        How many vertices the mesh has.
    what
        'texture coordinate' or 'normal', for the warning.

    Returns
    -------
    numpy.ndarray or None
        One row per vertex, NaN where no corner named a value. None when the
        file declares records that nothing indexes and that cannot be paired
        with the vertices one for one either - there is no honest way to
        line them up, and inventing one is how a mesh ends up with someone
        else's texture.
    """
    width = len(values[0])
    indexed = any(idx is not None for face in face_indices for idx in face)

    if not indexed:
        if len(values) != n_vertices:
            warnings.warn(
                f".obj: {len(values)} {what}(s) for {n_vertices} vertices and no"
                f" face indexes them, so they cannot be matched up; dropped.",
                stacklevel=3,
            )
            return None
        return np.array(values, dtype=np.float64)

    out = np.full((n_vertices, width), np.nan, dtype=np.float64)
    conflicting = False
    for face_v, face_i in zip(face_vertices, face_indices):
        for vi, idx in zip(face_v, face_i):
            if idx is None:
                continue
            row = values[idx]
            if not conflicting and not np.isnan(out[vi, 0]):
                conflicting = bool(np.any(out[vi] != row))
            out[vi] = row

    if conflicting:
        warnings.warn(
            f".obj: a vertex is given more than one {what}, which a per-vertex"
            " array cannot hold; the last one written wins.",
            stacklevel=3,
        )
    return out
