from collections.abc import Sequence
from typing import Any

import numpy as np

from polyxios._globals import (
    globals_for_write,
    spell_globals,
    spell_text,
    text_for_write,
)
from polyxios._image import (
    GRID_GLOBALS,
    IMAGE_RESERVED_GLOBALS,
    ImageData,
    build_image,
    check_readable,
    grid_of,
    grid_rows,
    is_identity,
    lattice_mesh,
    triple_of,
    wider_whole_extent,
)
from polyxios._io import Source, source_name, write_text
from polyxios._tags import with_tag_masks
from polyxios._types import PolyData
from polyxios._warn import warn_caller
from polyxios.codecs._hdf5 import warn_unknown_opts
from polyxios.codecs._vtk_xml import (
    decode_da,
    extent_points,
    format_attr_da,
    format_field_data,
    parse_xml,
    read_field_data,
    shaped_da,
    sized_attrs,
    spellable_arrays,
    structured_cell_shape,
    whole_extent,
    xml_extent,
)
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vti"


def _direction(values: list[str] | None) -> np.ndarray | None:
    """Coerce a stored ``Direction`` into a ``(3, 3)`` matrix, or None for none.

    A file that spells no ``Direction`` has no rotation. One that spells
    anything but nine numbers is not refused, but its rotation is not guessed
    at: the warning names it, and the identity is taken.
    """
    if values is None:
        return None
    try:
        flat = [float(v) for v in values]
    except (TypeError, ValueError):
        warn_caller(
            f"{EXTENSION}: Direction holds {values!r}, which is not nine numbers; "
            "the identity is used instead."
        )
        return None
    if len(flat) != 9:
        warn_caller(
            f"{EXTENSION}: Direction spells {len(flat)} numbers where nine belong; "
            "the identity is used instead."
        )
        return None
    matrix = np.asarray(flat, dtype=np.float64).reshape(3, 3)
    return None if is_identity(matrix) else matrix


def _spell(value: float) -> str:
    """Spell an origin or a step at the width a double reads back at.

    An ImageData writes no coordinates: every point it holds is this origin
    plus a multiple of this step, so these six numbers are the whole geometry
    and a digit dropped here moves the mesh. ``'%.10g'`` kept ten of the
    seventeen a double carries - an origin of ``0.12345678901234`` came back
    2.6e-10 away and a step of ``1.0000000001234`` came back as a flat ``1``,
    with the error growing by one step per plane. ``repr`` of a Python float
    is the shortest decimal that reads back as that same double, which is
    what the rest of the VTK XML codecs spell their values with; a whole
    number keeps the bare spelling it always had rather than gaining a
    ``.0``, since that is what the format's own files use.

    Parameters
    ----------
    value
        One origin, spacing or direction component.

    Returns
    -------
    str
        The token to write.

    Examples
    --------
    >>> _spell(1.0)
    '1'
    >>> _spell(0.12345678901234)
    '0.12345678901234'
    """
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _read_grid(
    path: Source, *, stacklevel: int = 4, with_direction: bool, builds_vertices: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Parse a .vti into the parts of its ImageData and the keys that spell the grid.

    Returns
    -------
    tuple[dict, dict, dict]
        The image's parts, the keyword arguments of :class:`~polyxios.ImageData`;
        the grid keys - ``vti_origin`` / ``vti_spacing`` / ``vti_extent``, plus
        ``vti_whole_extent`` when the file names one - as the mesh reader carries
        them in ``global_attrs``; and the file's field data alone, in the order
        the file held it, which the mesh reader puts ahead of those keys.
        The image keeps ``vti_whole_extent`` when it names one wider than the
        piece, and parses the file's ``Direction`` only when ``with_direction``
        is set, so a malformed one warns on :func:`read_image` alone.
        ``stacklevel`` is how far above this frame a warning blamed on the
        reader's caller points: four for the caller of :func:`read` or
        :func:`read_image`.
    """
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

    # Decoding happens one frame below here, and read_field_data adds another
    # for the field data: the warnings are blamed on the caller of the reader.
    def _decoder(extra: int):
        def _decode(elem):
            return decode_da(
                elem,
                big_endian=big_endian,
                appended=appended,
                header_type=header_type,
                compressed=compressed,
                is_base64=is_base64,
                stacklevel=stacklevel + 2 + extra,
            )

        return _decode

    _decode = _decoder(0)
    _decode_field = _decoder(1)

    img = root.find("ImageData")
    if img is None:
        raise CodecError("No <ImageData> element found in VTI file.")

    # WholeExtent="x0 x1 y0 y1 z0 z1" gives the global grid dimensions
    whole_extent_str = img.get("WholeExtent", "0 1 0 1 0 1")
    origin_str = img.get("Origin", "0.0 0.0 0.0")
    spacing_str = img.get("Spacing", "1.0 1.0 1.0")
    direction_str = img.get("Direction")
    # A file spelling two numbers where three belong is malformed, but it
    # costs the reader nothing to take what is there and default the rest -
    # indexing off the end raised a bare IndexError from inside the parse.
    # The warnings are blamed on the caller of the reader, as the warning
    # helper finds it.
    origin = triple_of(origin_str.split(), default=0.0, what="Origin", fmt=EXTENSION)
    spacing = triple_of(spacing_str.split(), default=1.0, what="Spacing", fmt=EXTENSION)
    direction = (
        _direction(None if direction_str is None else direction_str.split())
        if with_direction
        else None
    )

    piece = img.find("Piece")
    if piece is None:
        raise CodecError("No <Piece> element found.")

    piece_extent_str = piece.get("Extent", whole_extent_str)
    pe = xml_extent(piece_extent_str, fmt=".vti", where="Extent")
    i0, i1, j0, j1, k0, k1 = pe
    nx, ny, nz = i1 - i0, j1 - j0, k1 - k0
    n_verts = extent_points((nx, ny, nz))
    # An extent flat along an axis is a sheet of quads or a run of lines, not
    # a grid of no cells: the cells the grid holds decide the shape of a
    # CellData array, so they are counted before the header is validated.
    n_cells, n_per_cell, cell_kind = structured_cell_shape(nx, ny, nz)

    # The file spells neither the points nor the cells: they are the extent
    # expanded, so the byte-size heuristic has nothing to weigh the counts
    # against and refused a bare 4 x 4 x 4 grid this codec had just written.
    validate_header(
        n_verts,
        n_cells,
        n_cells * n_per_cell,
        file_size,
        compressed=compressed,
        spells_vertices=False,
        spells_connectivity=False,
        builds_vertices=builds_vertices,
    )

    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    pd = piece.find("PointData")
    if pd is not None:
        for da in pd:
            arr = _decode(da)
            if arr.size > 0:
                point_data[da.get("Name", "unknown")] = shaped_da(
                    da, arr, stacklevel=stacklevel + 1
                )

    cd = piece.find("CellData")
    if cd is not None:
        for da in cd:
            arr = _decode(da)
            if arr.size > 0:
                cell_data[da.get("Name", "unknown")] = shaped_da(
                    da, arr, stacklevel=stacklevel + 1
                )

    # One frame deeper than the reader's own, for each helper that warns.
    vertex_attrs = sized_attrs(
        point_data, expected=n_verts, kind="point", stacklevel=stacklevel + 1
    )
    element_attrs = sized_attrs(
        cell_data, expected=n_cells, kind="cell", stacklevel=stacklevel + 1
    )

    # Whole-mesh metadata, under the grid the reader rebuilt: a file naming
    # one of the reader's own keys in its field data is describing the same
    # grid twice, and the grid is what the points were laid out on. The
    # dataset's own block is read last, as in every other codec here: a key
    # spelled on both levels means what the dataset says it means.
    field: dict[str, Any] = read_field_data(
        piece, _decode_field, stacklevel=stacklevel + 1
    )
    field |= read_field_data(img, _decode_field, stacklevel=stacklevel + 1)
    # The grid keys the reader spells win over any field data that uses their
    # names, as they do on the mesh path; the image keeps the other field data.
    global_attrs: dict[str, Any] = {
        k: v for k, v in field.items() if k not in GRID_GLOBALS
    }
    grid: dict[str, Any] = {
        "vti_origin": origin,
        "vti_spacing": spacing,
        "vti_extent": pe,
    }
    whole = img.get("WholeExtent")
    if whole:
        # The grid the piece belongs to, kept the way .vtr and .vts keep
        # theirs: dropped here, a piece of a .pvti went back out declaring
        # itself the whole domain.
        grid["vti_whole_extent"] = xml_extent(whole, fmt=EXTENSION, where="WholeExtent")
        kept_whole = wider_whole_extent(grid["vti_whole_extent"], pe)
        if kept_whole is not None:
            global_attrs["vti_whole_extent"] = kept_whole

    # The arrays arrive one row per point or cell; the image holds them laid
    # out along the grid, z slowest.
    point_lead = (nz + 1, ny + 1, nx + 1)
    cell_lead = (max(nz, 1), max(ny, 1), max(nx, 1))
    parts: dict[str, Any] = {
        "extent": tuple(pe),
        "origin": tuple(origin),
        "spacing": tuple(spacing),
        "direction": direction,
        "point_attrs": {
            k: v.reshape(point_lead + v.shape[1:]) for k, v in vertex_attrs.items()
        }
        if n_verts > 0
        else {},
        "cell_attrs": {
            k: v.reshape(cell_lead + v.shape[1:]) for k, v in element_attrs.items()
        }
        if n_cells > 0
        else {},
        "global_attrs": global_attrs,
    }
    return parts, grid, field


def read(path: Source, *, lazy: bool = False) -> PolyData:
    """Parse a VTK ImageData XML file (.vti) and return a PolyData.

    Parameters
    ----------
    path
        Path to the .vti file.
    lazy
        Deferred decoding is not supported; raises LazyReadError when True.

    Returns
    -------
    PolyData
        Uniform hex grid expanded from the ImageData metadata. The grid is
        laid axis-aligned whatever the file's ``Direction`` says, as it always
        has been; :func:`read_image` keeps the rotation.

    Raises
    ------
    LazyReadError
        If lazy=True.
    """
    if lazy:
        raise LazyReadError("VTI lazy reads are not supported with frozen PolyData.")
    parts, grid, field = _read_grid(path, with_direction=False, builds_vertices=True)
    # The mesh keeps the field data in the order the file held it, the grid's
    # keys after, as the mesh reader always has.
    poly = lattice_mesh(**{**parts, "global_attrs": field})
    poly.global_attrs.update(grid)
    return poly


def read_image(path: Source, **opts: Any) -> ImageData:
    """Parse a VTK ImageData XML file (.vti) and return an ImageData.

    Parameters
    ----------
    path
        Path to the .vti file.
    **opts
        No option is taken; any given is warned about and ignored.

    Returns
    -------
    ImageData
        The grid as the file spells it: extent, origin, spacing, the
        ``Direction`` when the file rotates it, and the arrays laid out along
        it. A one-component trailing axis is not kept: an array of
        ``(nz, ny, nx, 1)`` comes back ``(nz, ny, nx)``. ``global_attrs``
        holds the file's field data, and ``vti_whole_extent`` when the file's
        ``WholeExtent`` is wider than its ``Extent``.
    """
    warn_unknown_opts(opts, fmt=EXTENSION, what="read", stacklevel=4)
    return build_image(
        source_name(path),
        **_read_grid(path, with_direction=True, builds_vertices=False)[0],
    )


def _spelled_direction(direction: np.ndarray | None) -> np.ndarray | None:
    """The rotation to write as ``Direction``: None for none, or the identity.

    A file without the attribute reads back as the identity.
    """
    return None if direction is None or is_identity(direction) else direction


def write_image(
    image: ImageData, path: Source, *, binary: bool = True, **opts: Any
) -> None:
    """Serialise an ImageData to a VTK ImageData XML file (.vti).

    The grid is written as it stands: ``WholeExtent``, ``Origin``, ``Spacing``
    and, when it rotates the grid, ``Direction``, with the point and cell
    arrays laid out along it as ``(nz, ny, nx[, k])``. The mesh is never built,
    so a volume costs its samples and no coordinates or cells.

    Parameters
    ----------
    image
        The grid to write; its extent, origin, spacing and direction are
        written as they stand. An array whose trailing axis is one is written
        as the grid's ``(nz, ny, nx)`` shape, and reads back collapsed.
    path
        Output file path.
    binary
        If True (default), encode data as base64 binary.
    **opts
        No other option is taken; any given is warned about and ignored.

    Raises
    ------
    CodecError
        If the grid holds no point, or a numeric array is not laid out along the
        extent; a grid past the caps a reader holds a compact image to is a
        :class:`~polyxios.exceptions.ValidationError`, so no file is written that
        cannot be read back.

    Warns
    -----
    UserWarning
        For an option that is not recognised, and for each array dropped by
        name: one that is not numeric, one on a grid with no point or cell to
        hold it, or one whose name XML cannot spell; and for a global value
        that cannot be written.
    """
    warn_unknown_opts(opts, fmt=EXTENSION, what="write", stacklevel=4)
    nx, ny, nz = image.dimensions
    if min(nx, ny, nz) == 0:
        raise CodecError(
            f"{EXTENSION} write: the extent {tuple(image.extent)} holds no point,"
            " and an ImageData file cannot hold an empty grid."
        )
    check_readable(image)
    spans = image.spans
    n_cells = structured_cell_shape(*spans)[0]
    point_lead = (nz, ny, nx)
    cell_lead = tuple(max(s, 1) for s in (spans[2], spans[1], spans[0]))
    # Each warning is blamed on the caller of the public writer.
    points = grid_rows(
        image.point_attrs, point_lead, nx * ny * nz, fmt=EXTENSION, group="PointData"
    )
    cells = grid_rows(
        image.cell_attrs, cell_lead, n_cells, fmt=EXTENSION, group="CellData"
    )
    extent = list(image.extent)
    # The FieldData is rendered before the arrays, as write() renders a mesh's,
    # so the warnings of a grid and of a mesh come in the same order.
    field_lines = _field_lines(
        spell_globals(
            image.global_attrs,
            reserved=IMAGE_RESERVED_GLOBALS,
            fmt=EXTENSION,
            text=True,
            stacklevel=4,
        ),
        text=spell_text(image.global_attrs, reserved=IMAGE_RESERVED_GLOBALS),
        binary=binary,
        stacklevel=5,
    )
    _emit(
        path,
        extent=extent,
        whole=whole_extent(image.global_attrs.get("vti_whole_extent"), extent),
        origin=list(image.origin),
        spacing=list(image.spacing),
        direction=_spelled_direction(image.direction),
        point_arrays=spellable_arrays(
            points, fmt=EXTENSION, kind="point", stacklevel=4
        ),
        cell_arrays=spellable_arrays(cells, fmt=EXTENSION, kind="cell", stacklevel=4),
        field_lines=field_lines,
        binary=binary,
    )


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Serialise a hex PolyData to a VTK ImageData XML file (.vti).

    Parameters
    ----------
    poly
        PolyData to write. Must be a structured hex grid (same topology as
        produced by reading a .vti file).
    path
        Output file path.
    binary
        If True (default), encode data as base64 binary.
    """
    binary: bool = bool(opts.get("binary", True))
    ga = poly.global_attrs or {}
    extent, origin, spacing, kept = grid_of(poly, fmt=EXTENSION)
    # The grid this piece belongs to means something only while the piece is
    # still standing where it stood. An extent re-derived from the cells is
    # zero-based and says nothing about the old grid, so the stored one goes
    # with the extent it was read beside.
    whole = whole_extent(ga.get("vti_whole_extent") if kept else None, extent)
    # The FieldData is rendered before the arrays, so its warnings come in the
    # order the mesh's own metadata is read: the dropped globals, the names
    # no XML attribute can hold, and then the arrays.
    field_lines = _field_lines(
        globals_for_write(poly, reserved=GRID_GLOBALS, fmt=EXTENSION, text=True),
        text=text_for_write(poly, reserved=GRID_GLOBALS),
        binary=binary,
        stacklevel=4,
    )
    point_arrays = spellable_arrays(
        with_tag_masks(
            poly.vertex_attrs,
            poly.vertex_tags,
            poly.vertices.shape[0],
            fmt=EXTENSION,
            kind="point",
        ),
        fmt=EXTENSION,
        kind="point",
    )
    # A tag group travels as one column of ones and zeros named for it: the
    # channel holds one value per entity, and an element in two groups is
    # named by both columns, which one label per element cannot say.
    cell_arrays = spellable_arrays(
        with_tag_masks(
            poly.element_attrs,
            poly.element_tags,
            len(poly.element_types),
            fmt=EXTENSION,
            kind="cell",
        ),
        fmt=EXTENSION,
        kind="cell",
    )
    _emit(
        path,
        extent=extent,
        whole=whole,
        origin=origin,
        spacing=spacing,
        direction=None,
        point_arrays=point_arrays,
        cell_arrays=cell_arrays,
        field_lines=field_lines,
        binary=binary,
    )


def _field_lines(
    field: dict[str, np.ndarray],
    *,
    text: dict[str, tuple[str, ...]],
    binary: bool,
    stacklevel: int,
) -> list[str]:
    """Render an image piece's ``FieldData`` block, for :func:`_emit`.

    Called before the arrays are checked, so that its warnings come in the order
    the mesh's metadata is read. ``stacklevel`` is the caller's own, as for
    :func:`write`, and is one frame more here than in the caller.
    """
    return format_field_data(
        field,
        text=text,
        binary=binary,
        indent=4,
        fmt=EXTENSION,
        stacklevel=stacklevel + 1,
    )


def _emit(
    path: Source,
    *,
    extent: Sequence[int],
    whole: Sequence[int],
    origin: Sequence[float],
    spacing: Sequence[float],
    direction: np.ndarray | None,
    point_arrays: dict[str, Any],
    cell_arrays: dict[str, Any],
    field_lines: list[str],
    binary: bool,
) -> None:
    """Write the XML of one ImageData piece, the same for a mesh or a grid.

    Parameters
    ----------
    path
        Output file path.
    extent, whole, origin, spacing
        The piece's ``Extent``, the ``WholeExtent`` it belongs to, and the
        grid's origin and step.
    direction
        The rotation to spell as ``Direction``, or None to leave it out.
    point_arrays, cell_arrays
        The arrays to write, each one row per point or cell, already checked
        for names XML can hold.
    field_lines
        The ``FieldData`` block, rendered by :func:`_field_lines`.
    binary
        Whether to base64 the arrays.
    """
    ext_str = " ".join(str(v) for v in extent)
    whole_str = " ".join(str(v) for v in whole)
    orig_str = " ".join(map(_spell, origin))
    spac_str = " ".join(map(_spell, spacing))
    dir_attr = (
        ""
        if direction is None
        else f' Direction="{" ".join(map(_spell, direction.ravel()))}"'
    )

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">')
    lines.append(
        f'  <ImageData WholeExtent="{whole_str}" Origin="{orig_str}" '
        f'Spacing="{spac_str}"{dir_attr}>'
    )
    lines.extend(field_lines)
    lines.append(f'    <Piece Extent="{ext_str}">')

    if point_arrays:
        lines.append("      <PointData>")
        for name, arr in point_arrays.items():
            lines.append(format_attr_da(name, arr, binary=binary, indent=10))
        lines.append("      </PointData>")

    if cell_arrays:
        lines.append("      <CellData>")
        for name, arr in cell_arrays.items():
            lines.append(format_attr_da(name, arr, binary=binary, indent=10))
        lines.append("      </CellData>")

    lines.append("    </Piece>")
    lines.append("  </ImageData>")
    lines.append("</VTKFile>")

    write_text(path, "\n".join(lines), encoding="utf-8")
