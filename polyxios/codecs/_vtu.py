from typing import Any

import numpy as np

from polyxios._element_types import (
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    POLYXIOS_TO_VTK,
    VTK_TO_POLYXIOS,
)
from polyxios._io import Source, write_text
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import (
    decode_da,
    format_attr_da,
    format_da,
    join_piece_attrs,
    parse_xml,
    piece_count,
    shaped_da,
    undecodable_type,
    vtk_type_to_np,
)
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vtu"


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a VTK UnstructuredGrid XML file (.vtu) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vtu file.
    lazy
        Deferred decoding is not supported; raises LazyReadError when True.

    Returns
    -------
    PolyData
        Parsed mesh data.

    Raises
    ------
    LazyReadError
        If lazy=True.
    """
    if lazy:
        raise LazyReadError("VTU lazy reads are not supported with frozen PolyData.")

    # The size comes back from the read itself: measuring the source
    # separately costs a whole decompression pass over a compressed one,
    # and a stream that cannot seek cannot be measured at all.
    (
        root,
        appended,
        header_type,
        big_endian,
        compressed,
        is_base64,
        file_size,
    ) = parse_xml(path)

    def _decode(elem):
        return decode_da(
            elem,
            big_endian=big_endian,
            appended=appended,
            header_type=header_type,
            compressed=compressed,
            is_base64=is_base64,
        )

    ug = root.find("UnstructuredGrid")
    if ug is None:
        raise ValueError("No <UnstructuredGrid> element found in VTU file.")

    all_vertices: list[np.ndarray] = []
    n_joined_points = 0
    all_connectivity: list[np.ndarray] = []
    all_offsets: list[int] = [0]
    all_types: list[int] = []
    all_vertex_attrs: dict[str, list[np.ndarray]] = {}
    all_element_attrs: dict[str, list[np.ndarray]] = {}

    for index, piece in enumerate(ug.findall("Piece")):
        n_points = piece_count(piece, "NumberOfPoints", fmt=".vtu")

        # Where this piece's points land in the joined array: its cells
        # index its own points from zero. Carried along rather than summed
        # per piece, which walks every piece read so far to answer the same
        # question a running count already holds.
        vert_offset = n_joined_points

        points_elem = piece.find("Points")
        if n_points > 0:
            # A piece that declares points and does not deliver them cannot
            # be dropped quietly: its cells index those points, and every
            # later piece is offset by how many there were.
            da = None if points_elem is None else points_elem.find("DataArray")
            # Asked before decoding, so an array of a type this reader has no
            # numbers for is reported as that rather than as an empty one.
            bad_type = None if da is None else undecodable_type(da)
            if bad_type is not None:
                raise CodecError(
                    f".vtu: Piece {index} declares {n_points} points but"
                    f" its Points array has type '{bad_type}', which holds"
                    " no numbers."
                )
            flat = np.array([]) if da is None else _decode(da)
            # The array has to hold whole tuples as well as enough of
            # them: a size that is not a multiple of the point count has
            # no shape to be read as, and reshape answers that with a
            # ValueError naming neither the file nor the Piece.
            if flat.size < n_points * 3 or flat.size % n_points:
                raise CodecError(
                    f".vtu: Piece {index} declares {n_points} points but"
                    f" its Points array holds {flat.size} values, which is"
                    f" not {n_points} tuples of three or more."
                )
            verts = flat.reshape(n_points, -1)[:, :3].astype(np.float64)
            all_vertices.append(verts)
            n_joined_points += n_points

        cells_elem = piece.find("Cells")
        if cells_elem is not None:
            conn_da = cells_elem.find("DataArray[@Name='connectivity']")
            off_da = cells_elem.find("DataArray[@Name='offsets']")
            types_da = cells_elem.find("DataArray[@Name='types']")

            if conn_da is not None and off_da is not None and types_da is not None:
                conn = _decode(conn_da).astype(np.int32) + vert_offset
                vtk_offsets = _decode(off_da).astype(np.int32)
                vtk_codes = _decode(types_da).astype(np.uint8)

                prev = all_offsets[-1]
                for i, end in enumerate(vtk_offsets):
                    start_local = int(vtk_offsets[i - 1]) if i > 0 else 0
                    end_local = int(end)
                    all_connectivity.append(conn[start_local:end_local])
                    prev = prev + (end_local - start_local)
                    all_offsets.append(prev)

                for code in vtk_codes:
                    name = VTK_TO_POLYXIOS.get(int(code), "empty_cell")
                    all_types.append(ELEMENT_TYPES.get(name, 0))

        pd_data = piece.find("PointData")
        if pd_data is not None:
            for da in pd_data:
                name = da.get("Name", "unknown")
                arr = _decode(da)
                if arr.size == 0:
                    continue
                arr = shaped_da(da, arr)
                all_vertex_attrs.setdefault(name, []).append(arr)

        cd_data = piece.find("CellData")
        if cd_data is not None:
            for da in cd_data:
                name = da.get("Name", "unknown")
                arr = _decode(da)
                if arr.size == 0:
                    continue
                arr = shaped_da(da, arr)
                all_element_attrs.setdefault(name, []).append(arr)

    vertices = (
        np.concatenate(all_vertices)
        if all_vertices
        else np.zeros((0, 3), dtype=np.float64)
    )
    connectivity = (
        np.concatenate(all_connectivity).astype(np.int32)
        if all_connectivity
        else np.array([], dtype=np.int32)
    )
    offsets = np.array(all_offsets, dtype=np.int32)
    element_types = np.array(all_types, dtype=np.uint8)

    validate_header(
        vertices.shape[0],
        len(element_types),
        len(connectivity),
        file_size,
        compressed=compressed,
    )

    vertex_attrs = join_piece_attrs(
        all_vertex_attrs, expected=vertices.shape[0], kind="point"
    )
    element_attrs = join_piece_attrs(
        all_element_attrs, expected=len(element_types), kind="cell"
    )

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise PolyData to a VTK UnstructuredGrid XML file (.vtu).

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path.
    binary
        If True (default), encode arrays as base64 binary.
    """
    binary: bool = bool(opts.get("binary", True))

    n_verts = poly.vertices.shape[0]
    n_elems = len(poly.element_types)

    vtk_types = np.array(
        [
            POLYXIOS_TO_VTK.get(ELEMENT_TYPES_INV.get(int(t), "empty_cell"), 0)
            for t in poly.element_types
        ],
        dtype=np.uint8,
    )
    vtk_offsets = poly.offsets[1:].astype(np.int32)

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">'
    )
    lines.append("  <UnstructuredGrid>")
    lines.append(f'    <Piece NumberOfPoints="{n_verts}" NumberOfCells="{n_elems}">')

    lines.append("      <Points>")
    lines.append(
        _da("", poly.vertices.ravel().astype(np.float64), "Float64", binary, 3, 10)
    )
    lines.append("      </Points>")

    lines.append("      <Cells>")
    lines.append(
        _da("connectivity", poly.connectivity.astype(np.int32), "Int32", binary, 1, 10)
    )
    lines.append(_da("offsets", vtk_offsets, "Int32", binary, 1, 10))
    lines.append(_da("types", vtk_types, "UInt8", binary, 1, 10))
    lines.append("      </Cells>")

    if poly.vertex_attrs:
        lines.append("      <PointData>")
        for name, arr in poly.vertex_attrs.items():
            lines.append(format_attr_da(name, arr, binary=binary, indent=10))
        lines.append("      </PointData>")

    if poly.element_attrs:
        lines.append("      <CellData>")
        for name, arr in poly.element_attrs.items():
            lines.append(format_attr_da(name, arr, binary=binary, indent=10))
        lines.append("      </CellData>")

    lines.append("    </Piece>")
    lines.append("  </UnstructuredGrid>")
    lines.append("</VTKFile>")

    write_text(path, "\n".join(lines), encoding="utf-8")


def _da(
    name: str,
    arr: np.ndarray,
    vtk_type: str,
    binary: bool,
    n_comp: int,
    indent: int,
) -> str:
    """Render one ``<DataArray>`` element.

    Parameters
    ----------
    name
        Array name; empty for the unnamed ``Points`` array.
    arr
        Values, flat or one row per tuple.
    vtk_type
        The type name the element declares. The values are cast to the dtype
        it names, so the bytes are what the header says they are on a
        big-endian machine as much as on a little-endian one.
    binary
        Base64 the raw bytes instead of spelling the numbers.
    n_comp
        Components per tuple.
    indent
        Spaces to prefix the line with.

    Returns
    -------
    str
        The ``<DataArray>`` line.
    """
    return format_da(
        name,
        arr,
        vtk_type=vtk_type,
        dtype=np.dtype("<" + (vtk_type_to_np(vtk_type) or "f8")),
        binary=binary,
        n_comp=n_comp,
        indent=indent,
    )
