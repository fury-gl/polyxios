from pathlib import Path

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._types import PolyData
from polyxios.exceptions import LazyReadError

EXTENSION: str = ".obj"


def read(path: Path | str, *, lazy: bool = False) -> PolyData:
    """Parse an OBJ file and return a PolyData.

    Parameters
    ----------
    path
        Path to the .obj file.
    lazy
        Not supported for OBJ — raises LazyReadError.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    LazyReadError
        Always, if lazy=True.
    """
    if lazy:
        raise LazyReadError("OBJ format does not support lazy reads (ASCII only).")

    path = Path(path)

    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    texcoords: list[list[float]] = []

    # face connectivity as list of (vertex_indices, normal_indices_or_None)
    face_vertices: list[list[int]] = []
    face_normals: list[list[int | None]] = []
    face_materials: list[str] = []

    # multi-group tag tracking: active groups at any point
    active_groups: list[str] = []
    element_tag_accumulator: dict[str, list[int]] = {}

    mtl_file: str | None = None
    object_name: str | None = None
    current_material = ""

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
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
                v_idx, vn_idx = _parse_face(parts[1:])
                face_vertices.append(v_idx)
                face_normals.append(vn_idx)
                face_materials.append(current_material)
                face_idx = len(face_vertices) - 1
                for g in active_groups:
                    element_tag_accumulator.setdefault(g, []).append(face_idx)

            elif directive == "g":
                # "g name1 name2 ..." — all names become active groups
                if len(parts) > 1:
                    active_groups = parts[1:]
                else:
                    active_groups = ["default"]

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

    # Vertex normals: map face normals back to per-vertex (last-write wins)
    vertex_attrs: dict[str, np.ndarray] = {}
    if normals:
        vn_arr = np.full((len(vertices), 3), np.nan, dtype=np.float64)
        for _fi, (face_v, face_n) in enumerate(zip(face_vertices, face_normals)):
            for vi, ni in zip(face_v, face_n):
                if ni is not None and 0 <= ni < len(normals):
                    vn_arr[vi] = normals[ni]
        vertex_attrs["normals"] = vn_arr

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


def write(poly: PolyData, path: Path | str, **opts: object) -> None:
    """Serialise PolyData to an OBJ file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    **opts
        Unused; accepted for API uniformity.
    """
    path = Path(path)
    lines: list[str] = []

    lines.append("# Written by polyxios")

    global_attrs = poly.global_attrs
    if "object_name" in global_attrs:
        lines.append(f"o {global_attrs['object_name']}")
    if "mtl_file" in global_attrs:
        lines.append(f"mtllib {global_attrs['mtl_file']}")

    lines.extend(f"v {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}" for v in poly.vertices)

    if "normals" in poly.vertex_attrs:
        lines.extend(
            f"vn {vn[0]:.10g} {vn[1]:.10g} {vn[2]:.10g}"
            for vn in poly.vertex_attrs["normals"]
        )

    # Build reverse tag map: element_idx → set of group names
    idx_to_groups: dict[int, list[str]] = {}
    for g, idxs in poly.element_tags.items():
        for i in idxs:
            idx_to_groups.setdefault(int(i), []).append(g)

    n_elems = len(poly.element_types)
    has_normals = "normals" in poly.vertex_attrs
    has_material = "material" in poly.element_attrs

    current_groups: list[str] | None = None
    current_material: str | None = None

    if not poly.element_tags:
        lines.append("g default")

    for i in range(n_elems):
        start = int(poly.offsets[i])
        end = int(poly.offsets[i + 1])
        face_verts = poly.connectivity[start:end]

        # Emit group changes
        groups = sorted(idx_to_groups.get(i, []))
        if groups and groups != current_groups:
            lines.append("g " + " ".join(groups))
            current_groups = groups

        # Emit material changes
        if has_material:
            mat = str(poly.element_attrs["material"][i])
            if mat != current_material:
                lines.append(f"usemtl {mat}")
                current_material = mat

        # Emit face (1-based indices)
        if has_normals:
            face_str = " ".join(f"{int(vi) + 1}//{int(vi) + 1}" for vi in face_verts)
        else:
            face_str = " ".join(str(int(vi) + 1) for vi in face_verts)
        lines.append(f"f {face_str}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_face(tokens: list[str]) -> tuple[list[int], list[int | None]]:
    """Parse OBJ face tokens into 0-based vertex and normal index lists.

    Parameters
    ----------
    tokens
        Face token list after the 'f' directive.

    Returns
    -------
    tuple[list[int], list[int | None]]
        (vertex_indices, normal_indices) both 0-based.
    """
    v_idx: list[int] = []
    vn_idx: list[int | None] = []

    for tok in tokens:
        parts = tok.split("/")
        # OBJ uses 1-based; negative indices count from end (not supported here)
        vi = int(parts[0]) - 1
        v_idx.append(vi)
        if len(parts) >= 3 and parts[2]:
            vn_idx.append(int(parts[2]) - 1)
        else:
            vn_idx.append(None)

    return v_idx, vn_idx
