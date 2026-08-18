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
        file has not declared.

    Notes
    -----
    OBJ indexes normals and texture coordinates **per face corner**, so a
    file may hold more of either than it holds vertices - a cube with a
    seam has two texture coordinates for one corner. polyxios stores one
    value per vertex, so a corner assigns to its vertex and a vertex named
    twice with different values keeps the last, which is warned about.

    A negative index counts back from what the file has declared so far, as
    the format defines; ``f -3 -2 -1`` is the last three vertices.
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

    with open_text(path, encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            directive = parts[0].lower()

            if directive == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif directive == "vn":
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif directive == "vt":
                texcoords.append(
                    [float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0]
                )

            elif directive == "f":
                v_idx, vt_idx, vn_idx = _parse_face(
                    parts[1:],
                    n_vertices=len(vertices),
                    n_texcoords=len(texcoords),
                    n_normals=len(normals),
                    where=f"'{source_name(path)}' line {line_no}",
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

    if "normals" in poly.vertex_attrs:
        vn_rows = _record_rows(poly.vertex_attrs["normals"], width=3)
        lines.extend(f"vn {vn[0]:.10g} {vn[1]:.10g} {vn[2]:.10g}" for vn in vn_rows)

    texcoords = poly.vertex_attrs.get("texcoords")
    if texcoords is not None:
        uv = _record_rows(texcoords, width=2)
        lines.extend(f"vt {row[0]:.10g} {row[1]:.10g}" for row in uv)

    # Build reverse tag map: element_idx - set of group names
    idx_to_groups: dict[int, list[str]] = {}
    for g, idxs in poly.element_tags.items():
        for i in idxs:
            idx_to_groups.setdefault(int(i), []).append(g)

    n_elems = len(poly.element_types)
    has_normals = "normals" in poly.vertex_attrs
    has_uv = texcoords is not None
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


def _record_rows(values: object, *, width: int) -> np.ndarray:
    """Shape a vertex attribute into rows an OBJ record can hold.

    Parameters
    ----------
    values
        A ``vertex_attrs`` entry, one row per vertex.
    width
        Components an OBJ record of this kind carries: 3 for ``vn``, 2 for
        ``vt``.

    Returns
    -------
    numpy.ndarray
        An ``(n, width)`` array. A row too short is padded with zeros rather
        than indexed past its end, and a value the format cannot spell -
        the NaN a reader leaves where no face named a record, an infinity -
        is written as zero: nothing indexes those rows, and ``vt nan nan``
        is not a record another reader takes.
    """
    arr = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if arr.shape[1] < width:
        arr = np.pad(arr, ((0, 0), (0, width - arr.shape[1])))
    return np.nan_to_num(arr[:, :width], nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_index(token: str, *, declared: int, what: str, where: str) -> int:
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
    where
        File and line, for the error message.

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
            f".obj: {where}: {what} index {token!r} is not a number."
        ) from exc

    if value == 0:
        raise CodecError(
            f".obj: {where}: {what} index 0 is not valid; the format numbers"
            " from 1, and counts back from -1."
        )

    idx = value - 1 if value > 0 else declared + value
    if not 0 <= idx < declared:
        raise CodecError(
            f".obj: {where}: {what} index {value} names a record the file has"
            f" not declared ({declared} so far)."
        )
    return idx


def _parse_face(
    tokens: list[str],
    *,
    n_vertices: int,
    n_texcoords: int,
    n_normals: int,
    where: str,
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
    where
        File and line, for an error message.

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
            _resolve_index(parts[0], declared=n_vertices, what="vertex", where=where)
        )
        if len(parts) >= 2 and parts[1]:
            vt_idx.append(
                _resolve_index(
                    parts[1],
                    declared=n_texcoords,
                    what="texture coordinate",
                    where=where,
                )
            )
        else:
            vt_idx.append(None)
        if len(parts) >= 3 and parts[2]:
            vn_idx.append(
                _resolve_index(parts[2], declared=n_normals, what="normal", where=where)
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
                stacklevel=2,
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
            stacklevel=2,
        )
    return out
