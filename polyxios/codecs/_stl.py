"""STL (Stereolithography) codec - binary and ASCII, read + write."""

import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._io import (
    Source,
    can_seek,
    open_block,
    open_read,
    open_write,
    read_bytes,
    source_size,
    write_text,
)
from polyxios._types import PolyData
from polyxios.exceptions import CodecError, LazyReadError

EXTENSION: str = ".stl"

# Binary STL layout constants
_HEADER_SIZE: int = 80
_BINARY_FACET_SIZE: int = 50  # 3*float32 normal + 3*3*float32 verts + uint16 attr

# The attribute word of a binary facet is where the tools that colour an STL
# put the colour, five bits per channel - but the two conventions in the wild
# disagree on both the channel order and the meaning of the top bit:
#
#   VisCAM / SolidView: bits 0-4 blue, 5-9 green, 10-14 red, and bit 15 set
#     when the word holds a colour at all.
#   Materialise Magics: bits 0-4 red, 5-9 green, 10-14 blue, and bit 15 clear
#     when the facet has a colour of its own, set when it takes the part's.
#
# Magics names itself in the 80-byte header, which is the only thing in the
# file that tells the two apart: it writes a ``COLOR=`` record there. A header
# without one reads as VisCAM, and VisCAM is what polyxios writes, since its
# spelling of "no colour here" is the zero word every other writer leaves.
#
# A zero word is that under Magics too: read to the letter it claims a facet
# colour of black, but it is what a writer that coloured nothing leaves behind,
# and reading an untouched file as uniformly black is the worse mistake.
_COLOR_VALID_BIT: int = 0x8000
_COLOR_MAX: float = 31.0
_COLOR_KEY: str = "colors"
_MAGICS_MARKER: bytes = b"COLOR="
# Bit position of red, green and blue under each convention.
_VISCAM_SHIFTS: tuple[int, int, int] = (10, 5, 0)
_MAGICS_SHIFTS: tuple[int, int, int] = (0, 5, 10)
# An integer colour column counts 0..255 the way every image format does; a
# float one runs 0..1. Scaling both as 0..1 turns an 8-bit colour white.
_INT_COLOR_MAX: float = 255.0


def read(
    path: Source,
    *,
    lazy: bool = False,
    merge_vertices: bool = True,
) -> PolyData:
    """Parse an STL file and return a PolyData of triangles.

    Parameters
    ----------
    path
        Path to the .stl file.
    lazy
        If True, skip vertex deduplication (merge_vertices is ignored).
        Useful for large files where deduplication overhead is significant.
        Not supported for ASCII STL.
    merge_vertices
        If True, deduplicate coincident vertices (default). Ignored when
        lazy=True.

    Returns
    -------
    PolyData
        Mesh with triangle elements only.

    Raises
    ------
    LazyReadError
        If lazy=True and the file is ASCII STL.
    CodecError
        On malformed STL data.
    """
    if lazy:
        file_size = source_size(path)
        with open_read(path) as fh:
            here = fh.tell()
            peek = fh.read(_HEADER_SIZE + 4)
            # A caller's handle is left where it was found: the lazy path
            # maps the file from its start, and a moved handle would leave
            # the caller reading from the middle of a file it never read.
            if can_seek(fh):
                fh.seek(here)
        if _is_ascii(peek, file_size=file_size):
            raise LazyReadError("STL ASCII format does not support lazy reads.")
        return _read_binary_lazy(path)

    raw = read_bytes(path)

    if _is_ascii(raw):
        vertices, normals = _read_ascii(raw)
        attrs = None
        header = b""
    else:
        vertices, normals, attrs = _read_binary(raw)
        header = raw[:_HEADER_SIZE]

    n_tris = vertices.shape[0]
    if n_tris == 0:
        return PolyData(
            vertices=np.empty((0, 3), dtype=np.float64),
            connectivity=np.array([], dtype=np.int32),
            offsets=np.zeros(1, dtype=np.int32),
            element_types=np.array([], dtype=np.uint8),
        )

    # vertices shape: (n_tris, 3, 3) - [tri, corner, xyz]
    if merge_vertices:
        flat = vertices.reshape(-1, 3)
        unique_verts, inv = _unique_rows_stable(flat)
        conn = inv.reshape(n_tris, 3)
    else:
        unique_verts = vertices.reshape(-1, 3)
        conn = np.arange(n_tris * 3, dtype=np.int32).reshape(n_tris, 3)

    tri_code = ELEMENT_TYPES["triangle"]
    connectivity = conn.astype(np.int32).ravel()
    offsets = np.arange(0, n_tris * 3 + 1, 3, dtype=np.int32)
    element_types = np.full(n_tris, tri_code, dtype=np.uint8)

    element_attrs: dict[str, np.ndarray] = {}
    if normals is not None:
        element_attrs["normals"] = normals
    colors = _decode_colors(attrs, header)
    if colors is not None:
        element_attrs[_COLOR_KEY] = colors

    return PolyData(
        vertices=unique_verts.astype(np.float64),
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        element_attrs=element_attrs,
    )


def write(poly: PolyData, path: Source, *, binary: bool = True) -> None:
    """Serialise PolyData to an STL file (triangles only).

    Non-triangle surface elements are skipped. Volume/line elements are
    also skipped.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    binary
        If True (default), write binary STL.
    """
    tri_code = ELEMENT_TYPES["triangle"]
    tri_indices = np.where(poly.element_types == tri_code)[0]

    if len(tri_indices) == 0:
        raise CodecError("STL requires triangle elements; none found in PolyData.")

    verts = poly.vertices

    # Vectorised gather - validate each triangle has exactly 3 connectivity slots
    sizes = poly.offsets[tri_indices + 1] - poly.offsets[tri_indices]
    if np.any(sizes != 3):
        raise CodecError("PolyData triangle element with connectivity != 3 vertices.")
    starts = poly.offsets[tri_indices]
    conn_indices = starts[:, None] + np.arange(3)
    vertex_indices = poly.connectivity[conn_indices]
    facet_verts = verts[vertex_indices].astype(np.float32)

    normals = _compute_normals(facet_verts)
    colors = _facet_colors(poly, tri_indices)

    if binary:
        _write_binary(path, facet_verts, normals, colors)
    else:
        if colors is not None:
            # ASCII STL has no field for a colour, so one carried in would be
            # dropped without a word by a plain write.
            warnings.warn(
                ".stl: ASCII STL has no field for a facet colour; the"
                " element_attrs['colors'] were not written. Pass binary=True"
                " to keep them.",
                stacklevel=2,
            )
        _write_ascii(path, facet_verts, normals)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_binary_lazy(path: Source) -> PolyData:
    """Read a binary STL without vertex deduplication.

    Skips merge_vertices step - useful for large files where deduplication
    overhead is significant. Data is eagerly copied; file is closed on return.

    The facets are copied out rather than handed back as a view, so nothing
    here needs a mapping to outlive the call: a file object with no
    descriptor is read into memory instead of being refused for a lazy read
    it would have served the same way.

    Normals are the values stored in the STL file (may be all-zero - the STL
    spec allows writers to omit them). Callers requiring unit normals should
    recompute from vertices.
    """
    facet_dt = np.dtype(
        [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
    )
    with open_block(path, fmt=".stl") as mm:
        if len(mm) < _HEADER_SIZE + 4:
            raise CodecError("Binary STL too short.")
        header = bytes(mm[:_HEADER_SIZE])
        n_tris = int(np.frombuffer(mm[_HEADER_SIZE : _HEADER_SIZE + 4], dtype="<u4")[0])
        data_start = _HEADER_SIZE + 4
        expected = data_start + n_tris * _BINARY_FACET_SIZE
        if len(mm) < expected:
            raise CodecError(
                f"Binary STL truncated: expected {expected} bytes, got {len(mm)}."
            )
        facets = np.frombuffer(
            memoryview(mm)[data_start : data_start + n_tris * _BINARY_FACET_SIZE],
            dtype=facet_dt,
        )
        normals = facets["normal"].copy()
        vertices = facets["verts"].reshape(-1, 3).copy()
        attrs = facets["attr"].copy()
        del facets  # release the view before the block goes

    tri_code = ELEMENT_TYPES["triangle"]
    connectivity = np.arange(n_tris * 3, dtype=np.int32)
    offsets = np.arange(0, n_tris * 3 + 1, 3, dtype=np.int32)
    element_types = np.full(n_tris, tri_code, dtype=np.uint8)

    element_attrs: dict[str, np.ndarray] = {"normals": normals}
    colors = _decode_colors(attrs, header)
    if colors is not None:
        element_attrs[_COLOR_KEY] = colors

    return PolyData(
        vertices=vertices.astype(np.float64),
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        element_attrs=element_attrs,
    )


def _decode_colors(attrs: np.ndarray | None, header: bytes) -> np.ndarray | None:
    """Return per-facet RGB from the attribute words, or None when none carry one.

    Parameters
    ----------
    attrs
        One uint16 attribute word per facet, or None for an ASCII file, which
        has no field for one.
    header
        The file's 80-byte header, which is where Magics says the words are
        its own; empty for an ASCII file.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_facets, 3)`` in 0..1, NaN on the facets whose word does not
        claim to hold a colour. None when no facet claims one, so an
        uncoloured file grows no attribute.

    Notes
    -----
    A zero word is no colour under either convention. Magics read to the
    letter says otherwise - bit 15 clear, so a facet colour of black - but a
    zero word is what a writer that coloured nothing leaves behind, and
    reading an untouched file as black from end to end is the worse mistake.
    """
    if attrs is None or attrs.size == 0:
        return None
    magics = _MAGICS_MARKER in header.upper()
    words = attrs.astype(np.uint16, copy=False)
    top = (words & _COLOR_VALID_BIT) != 0
    # Magics clears the top bit to claim the facet's own colour and sets it to
    # defer to the part's; everyone else sets it to claim one.
    valid = (~top & (words != 0)) if magics else top
    if not valid.any():
        return None
    shifts = _MAGICS_SHIFTS if magics else _VISCAM_SHIFTS
    live = words[valid]
    colors = np.full((words.size, 3), np.nan, dtype=np.float64)
    for channel, shift in enumerate(shifts):
        colors[valid, channel] = ((live >> shift) & 0x1F) / _COLOR_MAX
    return colors


def _encode_colors(colors: np.ndarray | None, n_facets: int) -> np.ndarray:
    """Return the attribute word for each facet, 0 where there is no colour.

    Written under the VisCAM convention the reader falls back on, since the
    header polyxios writes carries no Magics ``COLOR=`` record.
    """
    words = np.zeros(n_facets, dtype="<u2")
    if colors is None:
        return words
    values = np.asarray(colors, dtype=np.float64).reshape(n_facets, -1)[:, :3]
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        return words
    scaled = np.clip(np.rint(values[valid] * _COLOR_MAX), 0, 31).astype(np.uint16)
    packed = np.uint16(_COLOR_VALID_BIT)
    for channel, shift in enumerate(_VISCAM_SHIFTS):
        packed = packed | (scaled[:, channel] << shift)
    words[valid] = packed
    return words


def _is_ascii(raw: bytes, *, file_size: int | None = None) -> bool:
    """Return True if the raw bytes look like ASCII STL.

    Parameters
    ----------
    raw
        Raw bytes (may be a partial peek for lazy reads).
    file_size
        Actual file size in bytes. When provided, used instead of len(raw) for
        binary-size validation - required when raw is a partial read.
    """
    # Binary STL has an 80-byte header then a 4-byte triangle count.
    # ASCII STL starts with 'solid'. Some binary files also start with 'solid',
    # so cross-check with the declared triangle count.
    if not raw[:5].lower().startswith(b"solid"):
        return False
    size = file_size if file_size is not None else len(raw)
    if size < _HEADER_SIZE + 4:
        return True
    n_tris = int(np.frombuffer(raw[_HEADER_SIZE : _HEADER_SIZE + 4], dtype="<u4")[0])
    expected_size = _HEADER_SIZE + 4 + n_tris * _BINARY_FACET_SIZE
    # size < expected_size: too small to be valid binary → treat as ASCII.
    # size >= expected_size: valid binary (trailing data is allowed).
    # Heuristic limit: a corrupt n_tris can produce expected_size > size for a
    # valid binary file whose header starts with 'solid', misrouting it to the
    # ASCII parser. This edge case is undetected; CAD files with corrupt counts
    # may fail with a misleading parse error.
    return size < expected_size


def _read_binary(raw: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse binary STL bytes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (vertices, normals) where vertices is shape (n_tris, 3, 3) float32
        and normals is shape (n_tris, 3) float32.
    """
    if len(raw) < _HEADER_SIZE + 4:
        raise CodecError("Binary STL too short.")

    n_tris = int(np.frombuffer(raw[_HEADER_SIZE : _HEADER_SIZE + 4], dtype="<u4")[0])
    data_start = _HEADER_SIZE + 4
    expected = data_start + n_tris * _BINARY_FACET_SIZE
    if len(raw) < expected:
        raise CodecError(
            f"Binary STL truncated: expected {expected} bytes, got {len(raw)}."
        )

    # Layout per facet: 3 float32 normal, 9 float32 verts, 1 uint16 attr
    facet_dt = np.dtype(
        [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
    )
    facets = np.frombuffer(raw[data_start:expected], dtype=facet_dt)
    return facets["verts"].copy(), facets["normal"].copy(), facets["attr"].copy()


def _read_ascii(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse ASCII STL bytes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (vertices, normals) shape (n_tris, 3, 3) and (n_tris, 3).
    """
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise CodecError(f"ASCII STL contains non-ASCII bytes: {exc}") from exc
    lines = text.splitlines()

    verts_list: list[list[list[float]]] = []
    normals_list: list[list[float]] = []
    current_normal: list[float] = [0.0, 0.0, 0.0]
    current_verts: list[list[float]] = []

    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("facet normal"):
            # Consecutive facet normal lines: current_verts is reset, so any
            # incomplete prior facet is silently dropped (lenient parsing).
            parts = stripped.split()
            try:
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
            except (IndexError, ValueError) as exc:
                raise CodecError(f"Malformed facet normal line: {line!r}") from exc
            current_verts = []
        elif stripped.startswith("vertex"):
            parts = stripped.split()
            try:
                current_verts.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            except (IndexError, ValueError) as exc:
                raise CodecError(f"Malformed vertex line: {line!r}") from exc
        elif stripped.startswith("endfacet"):
            if len(current_verts) != 3:
                raise CodecError(
                    f"STL facet has {len(current_verts)} vertices, expected 3."
                )
            verts_list.append(current_verts)
            normals_list.append(current_normal)

    if not verts_list:
        return np.empty((0, 3, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    return (
        np.array(verts_list, dtype=np.float32),
        np.array(normals_list, dtype=np.float32),
    )


def _compute_normals(facet_verts: np.ndarray) -> np.ndarray:
    """Compute face normals from triangle vertex triplets.

    Parameters
    ----------
    facet_verts
        Shape (n_tris, 3, 3), float32.

    Returns
    -------
    np.ndarray
        Shape (n_tris, 3), float32 unit normals. Degenerate (zero-area)
        triangles get [0, 0, 1] as a stable fallback.
    """
    v0 = facet_verts[:, 0, :]
    v1 = facet_verts[:, 1, :]
    v2 = facet_verts[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    zero_mask = (norms < 1e-10).ravel()
    norms = np.where(norms < 1e-10, 1.0, norms)
    result = (normals / norms).astype(np.float32)
    result[zero_mask] = [0.0, 0.0, 1.0]
    return result


def _facet_colors(poly: PolyData, tri_indices: np.ndarray) -> np.ndarray | None:
    """Return the colour of each triangle written, or None when there is none.

    Parameters
    ----------
    poly
        The mesh being written.
    tri_indices
        Which of its elements become facets, in the order they are written.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n_facets, 3)`` in 0..1, or None when the mesh carries no
        usable colour attribute. An integer attribute is read as 0..255 and a
        floating point one as 0..1.
    """
    stored = (poly.element_attrs or {}).get(_COLOR_KEY)
    if stored is None:
        return None
    values = np.asarray(stored)
    if (
        values.ndim != 2
        or values.shape[0] != len(poly.element_types)
        or values.shape[1] < 3
        or values.dtype.kind not in "fiub"
    ):
        warnings.warn(
            f".stl: element_attrs['{_COLOR_KEY}'] is not three components per"
            " element; the colours were not written.",
            stacklevel=3,
        )
        return None
    picked = values[tri_indices].astype(np.float64, copy=False)
    # An integer column counts 0..255; scaling it as 0..1 saturates every
    # channel and writes a white mesh.
    if values.dtype.kind in "iu":
        picked = picked / _INT_COLOR_MAX
    return picked


def _write_binary(
    path: Source,
    facet_verts: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    n_tris = facet_verts.shape[0]
    facet_dt = np.dtype(
        [("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]
    )
    facets = np.zeros(n_tris, dtype=facet_dt)
    facets["normal"] = normals.astype("<f4")
    facets["verts"] = facet_verts.astype("<f4")
    facets["attr"] = _encode_colors(colors, n_tris)
    with open_write(path) as fh:
        hdr = b"Written by polyxios"
        fh.write(hdr + b"\x00" * (_HEADER_SIZE - len(hdr)))
        fh.write(np.array(n_tris, dtype="<u4").tobytes())
        fh.write(facets.tobytes())


def _write_ascii(path: Source, facet_verts: np.ndarray, normals: np.ndarray) -> None:
    n_tris = facet_verts.shape[0]
    n = normals.astype(np.float32)
    v = facet_verts.astype(np.float32)

    def _fmt3(prefix: str, arr: np.ndarray) -> np.ndarray:
        """Format (m, 3) float array as 'prefix x y z\n' strings."""
        return np.array([f"{prefix}{r[0]:.6e} {r[1]:.6e} {r[2]:.6e}\n" for r in arr])

    # Build (n_tris, 7) object array: one row per facet, one column per line.
    # Columns: facet-normal, outer-loop, vertex×3, endloop, endfacet.
    # Flattening avoids per-triangle Python iteration during the write.
    blocks = np.empty((n_tris, 7), dtype=object)
    blocks[:, 0] = _fmt3("  facet normal ", n)
    blocks[:, 1] = "    outer loop\n"
    blocks[:, 2] = _fmt3("      vertex ", v[:, 0, :])
    blocks[:, 3] = _fmt3("      vertex ", v[:, 1, :])
    blocks[:, 4] = _fmt3("      vertex ", v[:, 2, :])
    blocks[:, 5] = "    endloop\n"
    blocks[:, 6] = "  endfacet\n"

    text = "".join(
        ["solid polyxios\n", *blocks.ravel().tolist(), "endsolid polyxios\n"]
    )
    write_text(path, text, encoding="ascii")


def _unique_rows_stable(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique rows in first-occurrence order and inverse indices."""
    _, first_occ, inv = np.unique(arr, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_occ)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return arr[first_occ[order]], remap[inv].astype(np.int32)
