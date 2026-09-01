from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np

from polyxios._element_types import ELEMENT_TYPES
from polyxios._io import Source, write_text
from polyxios._types import PolyData
from polyxios.codecs._vtk_xml import (
    decode_da,
    extent_points,
    extent_spans,
    format_attr_da,
    grid_axes,
    parse_xml,
    require_grid_order,
    require_structured_cells,
    shaped_da,
    sized_attrs,
    structured_cell_shape,
    structured_cells,
    structured_cells_fit,
    structured_spans_from_cells,
    vertices_match_axes,
    whole_extent,
    xml_extent,
)
from polyxios.exceptions import CodecError, LazyReadError
from polyxios.validate import validate_header

EXTENSION: str = ".vti"


def _triple(values: Any, *, default: float, what: str) -> list[float]:
    """Coerce a stored origin or spacing into exactly three floats.

    Parameters
    ----------
    values
        What the file or ``global_attrs`` held, or None when it held
        nothing. A bare number is read as the value of every axis, which is
        what a caller writing ``vti_spacing=0.5`` means by it.
    default
        The value each missing axis takes.
    what
        Name of the thing being read, for the warning a wrong length raises.

    Returns
    -------
    list of float
        Three floats. A shorter sequence is filled out with ``default``
        rather than left short: an ``Origin`` of two numbers is not an
        origin, and zipping against it drops the third axis silently. A
        longer one keeps its first three and warns - a mesh has three axes,
        and a fourth number describes nothing this codec can write.

    Warns
    -----
    UserWarning
        If more than three values are given, or if they are not numbers at
        all. The default is taken in the second case, which leaves the
        writer's own check against the vertices to fail and the origin and
        step re-derived from the mesh - rather than a well-formed file built
        on a stored value nothing could read.

    Examples
    --------
    >>> _triple([1.0, 2.0], default=0.0, what="Origin")
    [1.0, 2.0, 0.0]
    >>> _triple(0.5, default=1.0, what="Spacing")
    [0.5, 0.5, 0.5]
    """
    if values is None:
        return [default] * 3
    try:
        if np.ndim(values) == 0:
            # A bare number, or the zero-dimensional array one becomes when
            # it travels through NumPy: neither can be sliced.
            return [float(values)] * 3
        out = [float(v) for v in values[:3]]
        n_values = len(values)
    except (TypeError, ValueError):
        # Not three numbers and not one: a string, a run of names, something
        # with no length. Asked in a try rather than in an isinstance ladder
        # because the happy path pays nothing for it and the answer is the
        # same however it is wrong.
        warnings.warn(
            f"{EXTENSION}: {what} holds {values!r}, which is not a number or "
            "a run of them; the default is used instead.",
            UserWarning,
            stacklevel=3,
        )
        return [default] * 3
    if n_values > 3:
        warnings.warn(
            f"{EXTENSION}: {what} spells {n_values} numbers where three "
            "belong; the rest name no axis and are dropped.",
            UserWarning,
            stacklevel=3,
        )
    out.extend([default] * (3 - len(out)))
    return out


def _metadata_axes(
    origin: list[float], spacing: list[float], extent: Sequence[int]
) -> list[np.ndarray]:
    """Expand an origin, a step and an extent into the points they spell.

    Parameters
    ----------
    origin
        Where index zero sits on each axis.
    spacing
        The step between planes on each axis; may be negative.
    extent
        The ``[i0, i1, j0, j1, k0, k1]`` the grid runs over.

    Returns
    -------
    list of numpy.ndarray
        The x, y and z coordinates, built exactly as :func:`read` builds them,
        so a file that has not been touched compares equal to itself.
    """
    return [
        origin[axis]
        + np.arange(extent[2 * axis], extent[2 * axis + 1] + 1) * spacing[axis]
        for axis in range(3)
    ]


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
        One origin or spacing component.

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


def _evenly_spaced(u: np.ndarray, step: float) -> bool:
    """Say whether one step reproduces every plane on an axis.

    The slack is measured against the step and against what a double holds at
    this magnitude - never as a fraction of the coordinate. A relative
    tolerance is a fraction of where the axis sits rather than of how far
    apart its planes are, so an axis at ``x = 1e6`` was allowed half a
    millimetre of drift per plane: a visibly uneven lattice passed, and came
    back regularised with every plane past the second moved. The two bounds
    answer the two ways a plane can miss: by a fraction of the step, and by
    the last few bits of a coordinate that has no room to hold it.

    Parameters
    ----------
    u
        The axis's coordinates, at least two of them.
    step
        The single step being proposed for it.

    Returns
    -------
    bool
        True when ``u[0] + i * step`` is every plane the axis holds.

    Examples
    --------
    >>> _evenly_spaced(np.array([0.0, 1.0, 2.0]), 1.0)
    True
    >>> _evenly_spaced(np.array([1e6, 1e6 + 1.0, 1e6 + 2.001]), 1.0005)
    False
    """
    rebuilt = float(u[0]) + np.arange(len(u)) * step
    # The axis is monotone in the step, so its widest value is an endpoint and
    # the bound costs two lookups rather than a pass over the whole axis.
    magnitude = max(abs(float(rebuilt[0])), abs(float(rebuilt[-1])))
    atol = max(abs(step) * 1e-9, 8.0 * float(np.spacing(magnitude)))
    return bool(np.allclose(u, rebuilt, rtol=0.0, atol=atol))


def _uniform_steps(
    axes: Sequence[np.ndarray], spacing: list[float], *, fmt: str
) -> list[float]:
    """Measure the one step each axis of an ImageData is allowed.

    An ImageData holds no coordinates: it holds a step, and every plane on an
    axis is that step from the last. A lattice whose planes are not evenly
    spaced is a RectilinearGrid, and writing its first step as though it were
    the only one moves every point after the second - silently, since the
    file that comes back is a well-formed grid of the right size.

    Parameters
    ----------
    axes
        The x, y and z coordinates, as :func:`grid_axes` gives them.
    spacing
        The step to keep on an axis with no second plane to measure against.
    fmt
        Extension the message names.

    Returns
    -------
    list of float
        The step on each axis.

    Raises
    ------
    CodecError
        If an axis is not evenly spaced.
    """
    steps: list[float] = []
    for axis, (u, fallback) in enumerate(zip(axes, spacing, strict=True)):
        if len(u) < 2:
            # One plane, or none: there is no step to measure, and VTK reads
            # the same grid back whatever is written. Keep what came with the
            # mesh rather than inventing a number for it.
            steps.append(float(fallback))
            continue
        step = (float(u[-1]) - float(u[0])) / (len(u) - 1)
        if len(u) > 2 and not _evenly_spaced(u, step):
            raise CodecError(
                f"{fmt}: the planes on the {'xyz'[axis]} axis are not evenly "
                "spaced, and an ImageData holds one step per axis rather than "
                "a coordinate for each plane. Every plane past the second "
                "would move - write a .vtr instead, which spells its "
                "coordinates out."
            )
        steps.append(step)
    return steps


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
        Uniform hex grid expanded from the ImageData metadata.

    Raises
    ------
    LazyReadError
        If lazy=True.
    """
    if lazy:
        raise LazyReadError("VTI lazy reads are not supported with frozen PolyData.")

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

    img = root.find("ImageData")
    if img is None:
        raise ValueError("No <ImageData> element found in VTI file.")

    # WholeExtent="x0 x1 y0 y1 z0 z1" gives the global grid dimensions
    whole_extent_str = img.get("WholeExtent", "0 1 0 1 0 1")
    origin_str = img.get("Origin", "0.0 0.0 0.0")
    spacing_str = img.get("Spacing", "1.0 1.0 1.0")
    # A file spelling two numbers where three belong is malformed, but it
    # costs the reader nothing to take what is there and default the rest -
    # indexing off the end raised a bare IndexError from inside the parse.
    origin = _triple(origin_str.split(), default=0.0, what="Origin")
    spacing = _triple(spacing_str.split(), default=1.0, what="Spacing")

    piece = img.find("Piece")
    if piece is None:
        raise ValueError("No <Piece> element found.")

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
    )

    x = origin[0] + np.arange(i0, i1 + 1) * spacing[0]
    y = origin[1] + np.arange(j0, j1 + 1) * spacing[1]
    z = origin[2] + np.arange(k0, k1 + 1) * spacing[2]

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    connectivity, _, _ = structured_cells(nx, ny, nz)
    offsets = np.arange(n_cells + 1, dtype=np.int32) * n_per_cell
    element_types = np.full(n_cells, ELEMENT_TYPES[cell_kind], dtype=np.uint8)

    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    pd = piece.find("PointData")
    if pd is not None:
        for da in pd:
            arr = _decode(da)
            if arr.size > 0:
                point_data[da.get("Name", "unknown")] = shaped_da(da, arr)

    cd = piece.find("CellData")
    if cd is not None:
        for da in cd:
            arr = _decode(da)
            if arr.size > 0:
                cell_data[da.get("Name", "unknown")] = shaped_da(da, arr)

    vertex_attrs = sized_attrs(point_data, expected=n_verts, kind="point")
    element_attrs = sized_attrs(cell_data, expected=n_cells, kind="cell")

    global_attrs: dict[str, Any] = {
        "vti_origin": origin,
        "vti_spacing": spacing,
        "vti_extent": pe,
    }
    whole = img.get("WholeExtent")
    if whole:
        # The grid the piece belongs to, kept the way .vtr and .vts keep
        # theirs: dropped here, a piece of a .pvti went back out declaring
        # itself the whole domain.
        global_attrs["vti_whole_extent"] = xml_extent(
            whole, fmt=EXTENSION, where="WholeExtent"
        )

    return PolyData(
        vertices=vertices,
        connectivity=connectivity,
        offsets=offsets,
        element_types=element_types,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        global_attrs=global_attrs,
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
    origin = _triple(ga.get("vti_origin"), default=0.0, what="vti_origin")
    spacing = _triple(ga.get("vti_spacing"), default=1.0, what="vti_spacing")
    stored = ga.get("vti_extent")
    spans = extent_spans(stored) if stored is not None else None
    # Whatever ``global_attrs`` held, in the six ints the file spells - read
    # once here rather than twice below, and only where ``spans`` has already
    # said the six are there and are whole numbers.
    kept = [int(v) for v in stored] if spans is not None else None

    # The extent, origin and step a file was read with travel with the mesh,
    # so a grid that did not begin at zero - or that runs down an axis - goes
    # back exactly where it stood. They are checked against the mesh at the
    # point of writing rather than trusted, the way the entity ids are: a
    # transform since the read leaves them describing the grid the mesh used
    # to be, and each of the three fails differently. Pruning cells leaves
    # the extent counting points that are not in the file, moving the mesh
    # leaves the origin behind, scaling it leaves the step behind - and the
    # last two write a well-formed file holding a grid somewhere else.
    # Cheapest question first: the point count is three integers, the points
    # are compared against broadcast views that allocate nothing the size of
    # the mesh, and only a mesh that has answered both is walked cell by cell.
    if (
        kept is not None
        and spans is not None
        and extent_points(spans) == len(poly.vertices)
        and vertices_match_axes(poly.vertices, _metadata_axes(origin, spacing, kept))
        and structured_cells_fit(poly.connectivity, poly.element_types, spans)
    ):
        extent = kept
        # The grid this piece belongs to means something only while the piece
        # is still standing where it stood. An extent re-derived below is
        # zero-based and says nothing about the old grid, so the stored one
        # goes with the extent it was read beside.
        whole = ga.get("vti_whole_extent")
    else:
        whole = None
        # What the mesh is now, read off the cells: they are the grid whatever
        # the coordinates do, and the extent has to be theirs or the cells the
        # reader rebuilds are not the ones written.
        spans = structured_spans_from_cells(
            poly.vertices, poly.connectivity, poly.element_types, fmt=EXTENSION
        )
        require_structured_cells(
            poly.connectivity, poly.element_types, spans, fmt=EXTENSION
        )
        axes = grid_axes(poly.vertices, spans)
        require_grid_order(poly.vertices, axes, fmt=EXTENSION)
        spacing = _uniform_steps(axes, spacing, fmt=EXTENSION)
        origin = [
            float(u[0]) if len(u) else o for u, o in zip(axes, origin, strict=True)
        ]
        # An axis with no vertex at all ends at -1, which is how VTK spells an
        # empty extent and how the .vts and .vtr writers here spell one; a 0
        # would declare the one point an empty mesh does not hold.
        extent = [bound for span in spans for bound in (0, span)]

    ext_str = " ".join(str(v) for v in extent)
    whole_str = " ".join(str(v) for v in whole_extent(whole, extent))
    orig_str = " ".join(map(_spell, origin))
    spac_str = " ".join(map(_spell, spacing))

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">')
    lines.append(
        f'  <ImageData WholeExtent="{whole_str}" Origin="{orig_str}" '
        f'Spacing="{spac_str}">'
    )
    lines.append(f'    <Piece Extent="{ext_str}">')

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
    lines.append("  </ImageData>")
    lines.append("</VTKFile>")

    write_text(path, "\n".join(lines), encoding="utf-8")
