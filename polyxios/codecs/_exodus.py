"""Exodus II codec - Sandia's finite element database, a netCDF file.

An Exodus file is a netCDF file whose dimensions, variables and attributes
follow the names the Exodus II library fixes: ``num_nodes`` and ``coord``
for the geometry, one ``connect<i>`` table per element block with its
``elem_type`` attribute, ``node_ns<i>`` for the node sets, ``elem_ss<i>`` and
``side_ss<i>`` for the side sets, ``elem_els<i>`` for the element sets, and
the results - nodal, element and global variables - along a ``time_step``
record dimension. Cubit writes it, Sierra and MOOSE run on it, ParaView
reads it natively.

Element blocks and the sets become polyxios tag groups; a side set, which
names a side of a solid rather than an element, is read as the triangle or
quadrilateral that side describes, linked to its parent by ``face_parent``
and ``face_index``. The variables at one time step become attributes,
``_x``/``_y``/``_z`` and ``_0``/``_1``/... families folded into one array.
Node order is VTK's for every kind but HEX20, HEX27, WEDGE15 and WEDGE18,
whose edges Exodus walks bottom, vertical, top where VTK walks bottom, top,
vertical; those are permuted both ways.

The codec needs netCDF4, the ``polyxios[netcdf]`` extra, imported on first
use so every other format is free of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import errno
import functools
import os
from pathlib import Path
import tempfile
from types import ModuleType
from typing import Any
import warnings

import numpy as np

from polyxios._dimension import mark_2d, output_dimension, pad_to_3d
from polyxios._element_types import (
    ELEMENT_FACES,
    ELEMENT_TYPES,
    ELEMENT_TYPES_INV,
    MAX_SAFE_CONN,
    NODES_PER_ELEMENT,
)
from polyxios._faces import (
    FACE_INDEX_KEY,
    FACE_PARENT_KEY,
    parent_face_mask,
    parent_faces,
)
from polyxios._globals import globals_for_write, text_for_write
from polyxios._ids import IDS_KEY, ids_for_write, record_ids
from polyxios._io import (
    GZIP_SUFFIXES,
    Source,
    is_gzip,
    read_bytes,
    source_name,
    source_size,
    source_suffix,
    write_bytes,
)
from polyxios._optpkg import TripWire, optional_package
from polyxios._tags import member_indices, members_array
from polyxios._types import PolyData, make_polydata
from polyxios.codecs._hdf5 import is_hdf5
from polyxios.exceptions import (
    CodecError,
    LazyReadError,
    UnknownElementTypeError,
    UnsupportedFormatError,
)
from polyxios.validate import validate_header

EXTENSION: str = ".e"
EXTENSIONS: tuple[str, ...] = (".e", ".exo", ".ex2")

# The three classic netCDF signatures - CDF1, CDF2 (64-bit offset) and CDF5
# (64-bit data) - beside the HDF5 one a netCDF-4 file opens with.
_CDF_MAGICS: tuple[bytes, ...] = (b"CDF\x01", b"CDF\x02", b"CDF\x05")

_LEN_STRING: int = 33
_LEN_LINE: int = 81
_MAX_NAME: int = 256
_API_VERSION: float = 8.03

# What an Exodus element family with this many nodes reads as. The family is
# the type name with its trailing digits stripped, so ``HEX`` with twenty
# nodes per element and ``HEX20`` land the same; a count the family has no
# entry for - HEX9, WEDGE16, TETRA14, PYRAMID14 - is a shape polyxios cannot
# hold and is skipped with a warning naming it.
_FAMILIES: dict[str, dict[int, str]] = {
    "SPHERE": {1: "vertex"},
    "CIRCLE": {1: "vertex"},
    "BAR": {2: "line", 3: "quadratic_edge"},
    "BEAM": {2: "line", 3: "quadratic_edge"},
    "TRUSS": {2: "line", 3: "quadratic_edge"},
    "EDGE": {2: "line", 3: "quadratic_edge"},
    "ROD": {2: "line", 3: "quadratic_edge"},
    "LINE": {2: "line", 3: "quadratic_edge"},
    "TRI": {3: "triangle", 6: "quadratic_triangle", 7: "biquadratic_triangle"},
    "TRIANGLE": {3: "triangle", 6: "quadratic_triangle", 7: "biquadratic_triangle"},
    "TRISHELL": {3: "triangle", 6: "quadratic_triangle", 7: "biquadratic_triangle"},
    "QUAD": {4: "quad", 8: "quadratic_quad", 9: "biquadratic_quad"},
    "QUADRILATERAL": {4: "quad", 8: "quadratic_quad", 9: "biquadratic_quad"},
    "SHELL": {4: "quad", 8: "quadratic_quad", 9: "biquadratic_quad"},
    "TETRA": {4: "tetra", 10: "quadratic_tetra"},
    "TET": {4: "tetra", 10: "quadratic_tetra"},
    "PYRAMID": {5: "pyramid", 13: "quadratic_pyramid"},
    "PYRA": {5: "pyramid", 13: "quadratic_pyramid"},
    "WEDGE": {6: "wedge", 15: "quadratic_wedge", 18: "biquadratic_quadratic_wedge"},
    "PRISM": {6: "wedge", 15: "quadratic_wedge", 18: "biquadratic_quadratic_wedge"},
    "HEX": {8: "hexahedron", 20: "quadratic_hexahedron", 27: "triquadratic_hexahedron"},
    "HEXAHEDRON": {
        8: "hexahedron",
        20: "quadratic_hexahedron",
        27: "triquadratic_hexahedron",
    },
}

# The name a written block carries per polyxios type: the spelling Cubit
# uses, which every Exodus reader knows.
_WRITE_NAMES: dict[str, str] = {
    "vertex": "SPHERE",
    "line": "BAR2",
    "quadratic_edge": "BAR3",
    "triangle": "TRI3",
    "quadratic_triangle": "TRI6",
    "biquadratic_triangle": "TRI7",
    "quad": "QUAD4",
    "pixel": "QUAD4",
    "quadratic_quad": "QUAD8",
    "biquadratic_quad": "QUAD9",
    "tetra": "TETRA4",
    "quadratic_tetra": "TETRA10",
    "pyramid": "PYRAMID5",
    "quadratic_pyramid": "PYRAMID13",
    "wedge": "WEDGE6",
    "quadratic_wedge": "WEDGE15",
    "biquadratic_quadratic_wedge": "WEDGE18",
    "hexahedron": "HEX8",
    "voxel": "HEX8",
    "quadratic_hexahedron": "HEX20",
    "triquadratic_hexahedron": "HEX27",
}
_WRITE_SIZES: dict[int, int] = {
    int(ELEMENT_TYPES[name]): NODES_PER_ELEMENT[name] for name in _WRITE_NAMES
}

# Exodus families polyxios has no cell for at any node count: an arbitrary
# polygon or polyhedron, whose ``connect`` table is a flat node list with a
# count per element beside it. Skipped with a warning, never refused.
_UNHELD_FAMILIES: frozenset[str] = frozenset({"NSIDED", "NFACED"})

# One row per type code, a sentinel row past the last so a clipped lookup of
# an unknown code lands on "no entry".
_N_CODES: int = max(ELEMENT_TYPES.values()) + 2
_WRITE_SIZE_TABLE: np.ndarray = np.full(_N_CODES, -1, dtype=np.int64)
for _code, _size in _WRITE_SIZES.items():
    _WRITE_SIZE_TABLE[_code] = _size

# Entry ``i`` of a read order is the file position of the node VTK puts at
# ``i``. Exodus lists a hexahedron's twelve mid-edge nodes bottom ring,
# vertical edges, top ring and a wedge's nine the same way; VTK lists bottom,
# top, vertical. HEX27 puts the centroid first (node 21), then the face
# centres bottom, top, left, right, front, back (22 to 27, the manual's
# side-set table), where VTK orders them by axis - x-min, x-max, y-min,
# y-max, z-min, z-max - and the centroid last. A pixel and a voxel are a
# quad and a hexahedron with the corners in lattice order.
_READ_ORDER: dict[str, tuple[int, ...]] = {
    "quadratic_hexahedron": (*range(8), 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15),
    "triquadratic_hexahedron": (
        *range(8),
        *(8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15),
        *(23, 24, 25, 26, 21, 22, 20),
    ),
    "quadratic_wedge": (*range(6), 6, 7, 8, 12, 13, 14, 9, 10, 11),
    "biquadratic_quadratic_wedge": (
        *range(6),
        *(6, 7, 8, 12, 13, 14, 9, 10, 11),
        *(15, 16, 17),
    ),
}
_WRITE_ORDER: dict[str, tuple[int, ...]] = {
    name: tuple(order.index(i) for i in range(len(order)))
    for name, order in _READ_ORDER.items()
}
_WRITE_ORDER["pixel"] = (0, 1, 3, 2)
_WRITE_ORDER["voxel"] = (0, 1, 3, 2, 4, 5, 7, 6)

# Which of polyxios's faces Exodus side ``n`` names, per solid type: the
# entry at ``n - 1`` is the index into ELEMENT_FACES. Exodus numbers the
# lateral faces first and the ends last; polyxios puts the ends first.
# tests/codecs/test_exodus.py checks every entry against the nodes the
# Exodus side-set table gives the side.
_SIDE_ORDER: dict[str, tuple[int, ...]] = {
    "tetra": (0, 1, 2, 3),
    "quadratic_tetra": (0, 1, 2, 3),
    "hexahedron": (2, 3, 4, 5, 0, 1),
    "quadratic_hexahedron": (2, 3, 4, 5, 0, 1),
    "triquadratic_hexahedron": (2, 3, 4, 5, 0, 1),
    "wedge": (2, 3, 4, 0, 1),
    "quadratic_wedge": (2, 3, 4, 0, 1),
    "biquadratic_quadratic_wedge": (2, 3, 4, 0, 1),
    "pyramid": (1, 2, 3, 4, 0),
    "quadratic_pyramid": (1, 2, 3, 4, 0),
}
_MAX_SIDES: int = 6


def _voxel_side_order() -> tuple[int, ...]:
    """The hexahedron side table carried over to a voxel's own face list.

    A voxel goes out as a HEX8 with its corners reordered, so the side a
    face of it is named by is the side its corners land on in that HEX8.
    """
    hex_pos = {node: pos for pos, node in enumerate(_WRITE_ORDER["voxel"])}
    hex_faces = [set(face) for face in ELEMENT_FACES["hexahedron"]]
    order = [0] * len(_SIDE_ORDER["hexahedron"])
    for local, face in enumerate(ELEMENT_FACES["voxel"]):
        hex_local = hex_faces.index({hex_pos[n] for n in face})
        order[_SIDE_ORDER["hexahedron"].index(hex_local)] = local
    return tuple(order)


_SIDE_ORDER["voxel"] = _voxel_side_order()

# The same table as arrays, indexed by type code: ``_LOCAL_OF_SIDE[code, n-1]``
# is the ELEMENT_FACES index Exodus side ``n`` names, -1 for no such side;
# ``_SIDE_OF_LOCAL[code, local]`` is the side number, 0 for none; and
# ``_FACE_WIDTH[code, local]`` how many corners the face has.
_LOCAL_OF_SIDE: np.ndarray = np.full((_N_CODES, _MAX_SIDES), -1, dtype=np.int64)
_SIDE_OF_LOCAL: np.ndarray = np.zeros((_N_CODES, _MAX_SIDES), dtype=np.int64)
_FACE_WIDTH: np.ndarray = np.zeros((_N_CODES, _MAX_SIDES), dtype=np.int64)
for _name, _order in _SIDE_ORDER.items():
    _code = int(ELEMENT_TYPES[_name])
    for _side, _local in enumerate(_order):
        _LOCAL_OF_SIDE[_code, _side] = _local
        _SIDE_OF_LOCAL[_code, _local] = _side + 1
        _FACE_WIDTH[_code, _local] = len(ELEMENT_FACES[_name][_local])
_IS_SOLID: np.ndarray = (_LOCAL_OF_SIDE >= 0).any(axis=1)

# What a plain path's opening is checked against. An HDF5 superblock may sit
# past a user block at 512, 1024, 2048 and every doubling after; this much
# covers the sizes anyone writes.
_HEAD_BYTES: int = 1 << 16

_RESERVED_GLOBALS: frozenset[str] = frozenset({"was_2d", "title", "time"})
_RESERVED_ELEMENT_ATTRS: frozenset[str] = frozenset(
    {IDS_KEY, FACE_PARENT_KEY, FACE_INDEX_KEY}
)
_AXES: tuple[str, ...] = ("x", "y", "z")


# ----- netCDF plumbing ----------------------------------------------------------


@functools.cache
def _netcdf4() -> tuple[ModuleType | TripWire, bool]:
    """netCDF4 and whether it is there, imported on first use.

    The import is shielded from the ``numpy.ndarray size changed``
    RuntimeWarning a wheel built against an older numpy raises on load: it
    is the extension's own ABI note, harmless here, and a process that turns
    warnings into errors would otherwise lose the package to it.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="numpy.ndarray size changed", category=RuntimeWarning
        )
        return optional_package("netCDF4", min_version="1.6", extra="netcdf")


def _require_netcdf4(*, name: str, verb: str) -> ModuleType:
    netcdf4, have = _netcdf4()
    if not have:
        raise UnsupportedFormatError(
            f"'{name}': Exodus is a netCDF file, and {verb} it needs netCDF4:"
            ' pip install "polyxios[netcdf]".'
        )
    return netcdf4


def is_netcdf(head: bytes) -> bool:
    """Say whether a file's opening bytes are a netCDF signature.

    Parameters
    ----------
    head
        The first bytes of the file.

    Returns
    -------
    bool
        True for a classic netCDF file of any of the three flavours, or an
        HDF5 file, which is what a netCDF-4 file is.
    """
    return head[:4] in _CDF_MAGICS or is_hdf5(head)


def _plain_path(src: Source) -> bool:
    return isinstance(src, (str, os.PathLike)) and (
        source_suffix(src) not in GZIP_SUFFIXES
    )


@contextmanager
def _open_read(path: Source) -> Iterator[Any]:
    """Open an Exodus file to read, from a path, a buffer or a gzip member."""
    name = source_name(path)
    netcdf4 = _require_netcdf4(name=name, verb="reading")
    if _plain_path(path) and not is_gzip(path):
        with open(path, "rb") as fh:
            head = fh.read(_HEAD_BYTES)
        ok = is_netcdf(head)
        opener = functools.partial(netcdf4.Dataset, os.fspath(path), "r")
    else:
        data = read_bytes(path)
        ok = is_netcdf(data)
        opener = functools.partial(netcdf4.Dataset, name, "r", memory=data)
    if not ok:
        raise CodecError(f"'{name}': not a netCDF file, so not Exodus.")
    try:
        handle = opener()
    except OSError as exc:
        raise CodecError(f"'{name}': netCDF4 could not open it ({exc}).") from None
    handle.set_auto_mask(False)
    handle.set_auto_chartostring(False)
    try:
        with _quiet_netcdf4():
            yield handle
    finally:
        handle.close()


@contextmanager
def _quiet_netcdf4() -> Iterator[None]:
    """Silence the shape-setting DeprecationWarning netCDF4 1.7 raises on numpy 2.5.

    It is the library's own slicing code, not anything a caller can change,
    and a test suite that turns warnings into errors would otherwise refuse
    every Exodus file.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array",
            category=DeprecationWarning,
        )
        yield


@contextmanager
def _open_write(path: Source, *, fmt: str) -> Iterator[Any]:
    """Open an Exodus file to write, to a path, a buffer or a gzip target.

    A path is written under a temporary name beside it and moved into place
    on success. Anything else is written to a temporary file too and handed
    to the IO layer whole: the netCDF library's in-memory image is not the
    file it writes to disk, so the bytes a buffer receives are the bytes a
    path would.
    """
    name = source_name(path)
    netcdf4 = _require_netcdf4(name=name, verb="writing")
    if _plain_path(path):
        target = Path(path)
        try:
            fd, tmp = tempfile.mkstemp(
                dir=target.parent, prefix=target.name + ".", suffix=".partial"
            )
        except OSError as exc:
            # The temporary name would be the one reported; the caller asked
            # for the target, and that is the path whose directory is missing
            # or refuses the write.
            code = exc.errno if exc.errno is not None else errno.EIO
            raise type(exc)(code, os.strerror(code), os.fspath(target)) from None
        os.close(fd)
        partial = Path(tmp)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(partial, 0o666 & ~umask)
        try:
            handle = netcdf4.Dataset(os.fspath(partial), "w", format=fmt)
            try:
                with _quiet_netcdf4():
                    yield handle
            finally:
                handle.close()
            os.replace(partial, target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return
    with tempfile.TemporaryDirectory() as folder:
        partial = Path(folder) / "mesh.e"
        handle = netcdf4.Dataset(os.fspath(partial), "w", format=fmt)
        try:
            with _quiet_netcdf4():
                yield handle
        finally:
            handle.close()
        write_bytes(path, partial.read_bytes())


def _dim(handle: Any, key: str, default: int = 0) -> int:
    if key not in handle.dimensions:
        return default
    return int(handle.dimensions[key].size)


def _load(
    handle: Any, key: str, ctx: dict[str, Any], *, step: int | None = None
) -> np.ndarray | None:
    """A variable's values whole, or its row at ``step``, checked before the read.

    Every table an Exodus file holds - a set, a side set, a block's
    attributes, a variable - has its own dimension, and the library
    allocates what the dimension declares before it finds the file too
    short. The count is held to the file's size, and to the hard cap on a
    compressed file where the bytes say nothing, so a corrupt header costs
    an error and not the memory it asks for.

    Parameters
    ----------
    handle
        The open netCDF dataset.
    key
        The variable's name; it must be there.
    ctx
        The read context: ``name``, ``file_size`` and ``compressed``.
    step
        A record index to read alone, the leading dimension being the
        time step. None reads the whole variable.

    Returns
    -------
    numpy.ndarray or None
        The values. None when ``step`` names a row the variable has not.

    Raises
    ------
    CodecError
        If the variable declares more values than the file can hold, or
        more than the cap on any read whatever the file's size says.
    """
    var = handle.variables[key]
    shape = tuple(var.shape[1:] if step is not None else var.shape)
    n = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if n > MAX_SAFE_CONN:
        raise CodecError(
            f"'{ctx['name']}': '{key}' declares {n} values, past the cap of"
            f" {MAX_SAFE_CONN} on what is read."
        )
    if not ctx["compressed"] and n > ctx["file_size"]:
        raise CodecError(
            f"'{ctx['name']}': '{key}' declares {n} values, more than the"
            f" {ctx['file_size']}-byte file holds."
        )
    if step is None:
        return np.asarray(var[:])
    if var.ndim < 1 or var.shape[0] <= step:
        return None
    return np.asarray(var[step])


def _var(handle: Any, key: str, ctx: dict[str, Any]) -> np.ndarray:
    if key not in handle.variables:
        raise CodecError(f"'{ctx['name']}': the file holds no '{key}'.")
    values = _load(handle, key, ctx)
    assert values is not None
    return values


def _prop_ids(handle: Any, key: str, count: int, ctx: dict[str, Any]) -> list[int]:
    """The ids an ``*_prop1`` table gives its entities, ``1..count`` without one."""
    if key in handle.variables:
        ids = _var(handle, key, ctx).ravel()
        if ids.shape == (count,) and ids.dtype.kind in "iu":
            return ids.tolist()
    return list(range(1, count + 1))


def _texts(handle: Any, key: str, count: int, ctx: dict[str, Any]) -> list[str]:
    """Return a char table's rows as stripped strings, empty when absent."""
    if key not in handle.variables:
        return [""] * count
    table = np.atleast_2d(_var(handle, key, ctx).astype("S1"))[:count]
    # A row's bytes whole: a NUL-terminated name may be followed by the tail
    # of a longer one it was written over, and an ``S1`` scalar drops NUL.
    out = [
        row.tobytes().split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
        for row in table
    ]
    while len(out) < count:
        out.append("")
    return out


def _chars(names: list[str], width: int) -> np.ndarray:
    """Encode names into an ``(n, width)`` char table, NUL padded.

    A name longer than ``width - 1`` bytes is cut on a code point boundary,
    so what is written decodes back as text.
    """
    out = np.zeros((len(names), width), dtype="S1")
    for i, text in enumerate(names):
        raw = text.encode("utf-8")
        if len(raw) >= width:
            raw = raw[: width - 1].decode("utf-8", errors="ignore").encode("utf-8")
        out[i, : len(raw)] = np.frombuffer(raw, dtype="S1")
    return out


# ----- reading ----------------------------------------------------------------


def read(path: Source, *, lazy: bool = False, step: int = 0, **opts: Any) -> PolyData:
    """Read an Exodus II file and return a PolyData.

    Parameters
    ----------
    path
        Path to the ``.e`` / ``.exo`` / ``.ex2`` file, or an open binary
        file object.
    lazy
        Not supported; raises LazyReadError when True.
    step
        Which time step's variables to read, 0-based. A file with no steps
        holds no variables; a ``step`` other than 0 asked of it is warned
        about and ignored.
    **opts
        None are taken; any given are warned about and ignored.

    Returns
    -------
    PolyData
        The mesh. Every element block and element set is an element tag
        group named by its name or ``block_<id>`` / ``elemset_<id>``; every
        node set a vertex tag group named the same way; every side set a
        tag group over the faces it names, read as triangles and quads
        linked to their parent by ``face_parent`` and ``face_index``. The
        nodal, element and global variables at ``step`` are ``vertex_attrs``,
        ``element_attrs`` and ``global_attrs``, ``<name>_x/_y/_z`` and
        ``<name>_0/_1/...`` families folded into one array; the step's time
        is ``global_attrs["time"]``, the file's title
        ``global_attrs["title"]``, and ``node_num_map`` / ``elem_num_map``
        ``original_ids`` when they say something the index does not; the
        faces a side set adds carry no id in the file and are numbered on
        past the largest one, so the ids the file does hold survive a round
        trip. An empty block - the NULL block a mesher leaves for a set with
        nothing in it - reads as nothing, its name with it.

    Raises
    ------
    LazyReadError
        If ``lazy`` is set.
    UnsupportedFormatError
        If netCDF4 is not installed. The message names the extra that
        installs it.
    UnknownElementTypeError
        If a block names an element family Exodus does not define.
    CodecError
        If the file is not netCDF, holds no coordinates, a block's table or
        the legacy ``vals_nod_var`` table is not the shape its dimensions
        say, a connectivity or set names a node or element outside the
        file, a table declares more values than the file could hold, or
        ``step`` is past the last.

    Warns
    -----
    UserWarning
        For a block whose family has no shape of that node count, or is a
        ``NSIDED`` / ``NFACED`` block, which is skipped; for a side set on
        an element that is not a solid, naming a side number the element
        has no side of, or on an element of a skipped block, those sides
        skipped; for a variable that is not one value per entity at
        ``step``, skipped; for a variable name the file gives twice, the
        later one skipped; for a ``step`` asked of a file with no steps.
    """
    name = source_name(path)
    if lazy:
        raise LazyReadError(
            f"'{name}': Exodus lazy reads are not supported; the arrays live in"
            " netCDF variables and are decoded whole."
        )
    _warn_unknown_opts(opts, what="read")
    # Measured before the open, which reads a buffer to its end.
    file_size = source_size(path)
    with _open_read(path) as handle:
        ctx = {
            "name": name,
            "file_size": file_size,
            "compressed": str(handle.data_model).startswith("NETCDF4"),
        }
        n_steps = _dim(handle, "time_step")
        if n_steps and not 0 <= step < n_steps:
            raise CodecError(
                f"'{name}': step {step} is out of range; the file holds"
                f" {n_steps} time step(s)."
            )
        if not n_steps and step:
            warnings.warn(
                f"{EXTENSION} read: '{name}' holds no time step, so step={step}"
                " names nothing; ignored.",
                UserWarning,
                stacklevel=3,
            )
        vertices, dim = _read_coordinates(handle, ctx)
        n_verts = len(vertices)
        blocks, block_tags, block_attrs, file_to_mesh = _read_blocks(
            handle, n_verts, ctx
        )
        n_file = len(file_to_mesh)
        n_kept = int(np.count_nonzero(file_to_mesh >= 0))
        faces, face_tags, face_parent, face_local = _read_side_sets(
            handle, blocks, file_to_mesh, n_kept, ctx
        )
        n_faces = len(face_parent)
        n_elems = n_kept + n_faces
        element_tags = dict(block_tags)
        _merge_tags(element_tags, face_tags)
        _merge_tags(
            element_tags,
            _read_sets(handle, "els", "elem_els", "elemset", n_file, file_to_mesh, ctx),
        )
        vertex_tags = _read_sets(
            handle, "ns", "node_ns", "nodeset", n_verts, np.arange(n_verts), ctx
        )
        vertex_attrs: dict[str, np.ndarray] = {}
        # A block attribute is a value per block element; the faces the side
        # sets append after them have none.
        element_attrs: dict[str, np.ndarray] = {}
        for key, column in block_attrs.items():
            padded = np.full(n_elems, np.nan)
            padded[:n_kept] = column
            element_attrs[key] = padded
        global_attrs: dict[str, Any] = {}
        if n_steps:
            if "time_whole" in handle.variables:
                times = _var(handle, "time_whole", ctx)
                if times.size > step:
                    global_attrs["time"] = float(times[step])
            vertex_attrs.update(_read_nodal_vars(handle, step, n_verts, ctx))
            element_attrs.update(
                _read_element_vars(handle, step, blocks, file_to_mesh, n_elems, ctx)
            )
            global_attrs.update(_read_global_vars(handle, step, ctx))
        title = str(getattr(handle, "title", "") or "").strip("\x00").strip()
        if title:
            global_attrs["title"] = title
        if "node_num_map" in handle.variables:
            ids = _var(handle, "node_num_map", ctx)
            if ids.shape == (n_verts,):
                vertex_attrs.update(record_ids(ids, count=n_verts))
        if "elem_num_map" in handle.variables:
            ids = _var(handle, "elem_num_map", ctx)
            if ids.shape == (n_file,):
                ids = ids[file_to_mesh >= 0].astype(np.int64)
                if n_faces:
                    top = int(ids.max()) if ids.size else 0
                    ids = np.concatenate([ids, np.arange(top + 1, top + 1 + n_faces)])
                element_attrs.update(record_ids(ids, count=n_elems))
        if n_faces:
            parent_col = np.full(n_elems, -1, dtype=np.int32)
            index_col = np.full(n_elems, -1, dtype=np.int32)
            parent_col[n_kept:] = face_parent
            index_col[n_kept:] = face_local
            element_attrs[FACE_PARENT_KEY] = parent_col
            element_attrs[FACE_INDEX_KEY] = index_col
    global_attrs.update(mark_2d(dim))
    groups = [(ELEMENT_TYPES_INV[code], cells) for code, cells in blocks if len(cells)]
    groups.extend(faces)
    return make_polydata(
        vertices,
        groups,
        vertex_attrs=vertex_attrs,
        element_attrs=element_attrs,
        vertex_tags=vertex_tags,
        element_tags=element_tags,
        global_attrs=global_attrs,
    )


def _warn_unknown_opts(opts: dict[str, Any], *, what: str) -> None:
    if opts:
        warnings.warn(
            f"{EXTENSION} {what}: unrecognized options {set(opts)}; ignored.",
            UserWarning,
            stacklevel=3,
        )


def _read_coordinates(handle: Any, ctx: dict[str, Any]) -> tuple[np.ndarray, int]:
    name = ctx["name"]
    n_verts = _dim(handle, "num_nodes")
    dim = _dim(handle, "num_dim", 3)
    if dim not in (1, 2, 3):
        raise CodecError(f"'{name}': num_dim is {dim}, not 1, 2 or 3.")
    validate_header(n_verts, 0, 0, ctx["file_size"], compressed=ctx["compressed"])
    if "coord" in handle.variables:
        coords = _var(handle, "coord", ctx).astype(np.float64)
        if coords.shape != (dim, n_verts):
            raise CodecError(
                f"'{name}': 'coord' is {coords.shape}, not ({dim}, {n_verts})."
            )
        coords = coords.T
    else:
        columns = []
        for axis in _AXES[:dim]:
            key = f"coord{axis}"
            if key not in handle.variables:
                if n_verts == 0:
                    columns.append(np.zeros(0))
                    continue
                raise CodecError(f"'{name}': the file holds no 'coord' and no '{key}'.")
            column = _var(handle, key, ctx).astype(np.float64).ravel()
            if column.shape != (n_verts,):
                raise CodecError(f"'{name}': '{key}' holds {column.size} of {n_verts}.")
            columns.append(column)
        coords = np.column_stack(columns) if columns else np.zeros((0, dim))
    if dim == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(n_verts), np.zeros(n_verts)])
        return coords, 3
    return pad_to_3d(np.ascontiguousarray(coords), dim), dim


def _family(type_name: str) -> str:
    return type_name.upper().rstrip("0123456789")


def _read_blocks(
    handle: Any, n_verts: int, ctx: dict[str, Any]
) -> tuple[
    list[tuple[int, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
]:
    """Every element block in file order, its tag, its attributes, and where
    each file element landed in the mesh (-1 for a skipped one).

    A block of no elements - what the Exodus library writes for a NULL
    block: a status of 0 and neither dimensions nor table - lands nothing.
    """
    name = ctx["name"]
    n_blocks = _dim(handle, "num_el_blk")
    ids = _prop_ids(handle, "eb_prop1", n_blocks, ctx)
    names = _texts(handle, "eb_names", n_blocks, ctx)
    blocks: list[tuple[int, np.ndarray]] = []
    tags: dict[str, np.ndarray] = {}
    attrs: dict[str, list[tuple[int, int, np.ndarray]]] = {}
    positions: list[np.ndarray] = []
    skipped: list[str] = []
    n_elems = 0
    total_conn = 0
    for i in range(1, n_blocks + 1):
        count = _dim(handle, f"num_el_in_blk{i}")
        width = _dim(handle, f"num_nod_per_el{i}")
        total_conn += count * width
        validate_header(
            n_verts,
            n_elems + count,
            total_conn,
            ctx["file_size"],
            compressed=ctx["compressed"],
        )
        key = f"connect{i}"
        if key not in handle.variables:
            if count:
                raise CodecError(f"'{name}': block {i} has no '{key}' table.")
            positions.append(np.empty(0, dtype=np.int64))
            continue
        var = handle.variables[key]
        type_name = str(getattr(var, "elem_type", "") or "").strip("\x00").strip()
        family = _family(type_name)
        if family in _UNHELD_FAMILIES:
            skipped.append(type_name)
            positions.append(np.full(count, -1, dtype=np.int64))
            continue
        if family not in _FAMILIES:
            raise UnknownElementTypeError(EXTENSION, type_name or f"block {i}")
        ptype = _FAMILIES[family].get(width)
        if ptype is None:
            skipped.append(f"{type_name} ({width} nodes)")
            positions.append(np.full(count, -1, dtype=np.int64))
            continue
        cells = _var(handle, key, ctx)
        if cells.shape != (count, width) or cells.dtype.kind not in "iu":
            raise CodecError(
                f"'{name}': '{key}' is not the ({count}, {width}) integer table"
                " its dimensions declare."
            )
        cells = cells.astype(np.int64) - 1
        if cells.size and (cells.min() < 0 or cells.max() >= n_verts):
            raise CodecError(f"'{name}': '{key}' names a node outside 1..{n_verts}.")
        order = _READ_ORDER.get(ptype)
        if order is not None:
            cells = cells[:, list(order)]
        code = int(ELEMENT_TYPES[ptype])
        block_index = len(blocks)
        blocks.append((code, cells))
        positions.append(np.arange(n_elems, n_elems + count, dtype=np.int64))
        label = names[i - 1] or f"block_{ids[i - 1]}"
        tags[label] = np.union1d(tags.get(label, np.empty(0, np.int64)), positions[-1])
        for attr_name, column in _block_attributes(handle, i, count, ctx):
            attrs.setdefault(attr_name, []).append((block_index, n_elems, column))
        n_elems += count
    if skipped:
        warnings.warn(
            f"{EXTENSION} read: '{name}' holds block(s) of {sorted(set(skipped))},"
            " shapes polyxios does not hold; skipped with their sets and values.",
            UserWarning,
            stacklevel=4,
        )
    file_to_mesh = (
        np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
    )
    attr_columns: dict[str, np.ndarray] = {}
    for attr_name, pieces in attrs.items():
        column = np.full(n_elems, np.nan)
        for _, start, values in pieces:
            column[start : start + len(values)] = values
        attr_columns[attr_name] = column
    return blocks, tags, attr_columns, file_to_mesh


def _block_attributes(
    handle: Any, i: int, count: int, ctx: dict[str, Any]
) -> list[tuple[str, np.ndarray]]:
    n_attr = _dim(handle, f"num_att_in_blk{i}")
    key = f"attrib{i}"
    if not n_attr or key not in handle.variables:
        return []
    table = _var(handle, key, ctx).astype(np.float64)
    if table.shape != (count, n_attr):
        warnings.warn(
            f"{EXTENSION} read: '{key}' in '{ctx['name']}' is {table.shape}, not"
            f" ({count}, {n_attr}); skipped.",
            UserWarning,
            stacklevel=5,
        )
        return []
    names = _texts(handle, f"attrib_name{i}", n_attr, ctx)
    return [(names[k] or f"attrib{k + 1}", table[:, k].copy()) for k in range(n_attr)]


def _read_sets(
    handle: Any,
    prefix: str,
    table: str,
    fallback: str,
    count: int,
    to_mesh: np.ndarray,
    ctx: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Node sets or element sets as tag groups over mesh indices."""
    name = ctx["name"]
    n_sets = _dim(handle, f"num_{'node' if prefix == 'ns' else 'elem'}_sets")
    ids = _prop_ids(handle, f"{prefix}_prop1", n_sets, ctx)
    names = _texts(handle, f"{prefix}_names", n_sets, ctx)
    out: dict[str, np.ndarray] = {}
    for i in range(1, n_sets + 1):
        key = f"{table}{i}"
        if key not in handle.variables:
            continue
        members = _var(handle, key, ctx).ravel()
        if members.dtype.kind not in "iu":
            raise CodecError(f"'{name}': '{key}' is not an integer list.")
        members = members.astype(np.int64) - 1
        if members.size and (members.min() < 0 or members.max() >= count):
            raise CodecError(f"'{name}': '{key}' names an entity outside 1..{count}.")
        picked = to_mesh[members]
        picked = np.unique(picked[picked >= 0])
        label = names[i - 1] or f"{fallback}_{ids[i - 1]}"
        out[label] = np.union1d(out.get(label, np.empty(0, np.int64)), picked)
    return out


def _merge_tags(into: dict[str, np.ndarray], more: dict[str, np.ndarray]) -> None:
    for label, members in more.items():
        into[label] = np.union1d(into.get(label, np.empty(0, np.int64)), members)


def _read_side_sets(
    handle: Any,
    blocks: list[tuple[int, np.ndarray]],
    file_to_mesh: np.ndarray,
    n_kept: int,
    ctx: dict[str, Any],
) -> tuple[
    list[tuple[str, np.ndarray]],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    """The faces every side set names, appended after the block elements.

    Returns the face groups - triangles then quads - the tag each set puts
    on its faces, and for each face in its final order the parent element
    and which of its faces it is. A side named twice - by two sets, or twice
    by one - is one face in both groups, at the position of its first
    mention. A side of an element that is not a solid, a side number the
    element has no side of, or a side of an element in a skipped block has
    no face to read and is skipped with a warning naming the set.
    """
    name = ctx["name"]
    n_sets = _dim(handle, "num_side_sets")
    empty = np.empty(0, dtype=np.int64)
    if not n_sets:
        return [], {}, empty, empty
    ids = _prop_ids(handle, "ss_prop1", n_sets, ctx)
    names = _texts(handle, "ss_names", n_sets, ctx)
    n_file = len(file_to_mesh)
    codes = (
        np.concatenate([np.full(len(c), code, dtype=np.int64) for code, c in blocks])
        if blocks
        else empty
    )
    # Every mention as (mesh element, local face), sets laid end to end.
    mesh_parts: list[np.ndarray] = []
    local_parts: list[np.ndarray] = []
    spans: list[tuple[str, int, int]] = []
    not_solid: list[str] = []
    no_side: list[str] = []
    no_elem: list[str] = []
    total = 0
    for i in range(1, n_sets + 1):
        ekey, skey = f"elem_ss{i}", f"side_ss{i}"
        if ekey not in handle.variables or skey not in handle.variables:
            continue
        elems = _var(handle, ekey, ctx).ravel()
        sides = _var(handle, skey, ctx).ravel()
        if elems.dtype.kind not in "iu" or sides.dtype.kind not in "iu":
            raise CodecError(f"'{name}': '{ekey}' / '{skey}' are not integer lists.")
        if elems.shape != sides.shape:
            raise CodecError(f"'{name}': '{ekey}' and '{skey}' differ in length.")
        elems = elems.astype(np.int64) - 1
        sides = sides.astype(np.int64)
        if elems.size and (elems.min() < 0 or elems.max() >= n_file):
            raise CodecError(
                f"'{name}': '{ekey}' names an element outside 1..{n_file}."
            )
        label = names[i - 1] or f"sideset_{ids[i - 1]}"
        mesh = file_to_mesh[elems]
        live = mesh >= 0
        if not live.all():
            no_elem.append(label)
        mesh, sides = mesh[live], sides[live]
        code = codes[mesh]
        solid = _IS_SOLID[code]
        in_range = (sides >= 1) & (sides <= _MAX_SIDES)
        local = np.full(len(mesh), -1, dtype=np.int64)
        local[in_range] = _LOCAL_OF_SIDE[code[in_range], sides[in_range] - 1]
        if not solid.all():
            not_solid.append(label)
        if (solid & (local < 0)).any():
            no_side.append(label)
        keep = local >= 0
        mesh_parts.append(mesh[keep])
        local_parts.append(local[keep])
        spans.append((label, total, total + int(np.count_nonzero(keep))))
        total = spans[-1][2]
    if not_solid:
        warnings.warn(
            f"{EXTENSION} read: side set(s) {sorted(set(not_solid))} in '{name}'"
            " name a side of an element that is not a solid; those sides are"
            " skipped.",
            UserWarning,
            stacklevel=4,
        )
    if no_side:
        warnings.warn(
            f"{EXTENSION} read: side set(s) {sorted(set(no_side))} in '{name}'"
            " name a side number the element has no side of; those sides are"
            " skipped.",
            UserWarning,
            stacklevel=4,
        )
    if no_elem:
        warnings.warn(
            f"{EXTENSION} read: side set(s) {sorted(set(no_elem))} in '{name}'"
            " name a side of an element in a skipped block; those sides are"
            " skipped.",
            UserWarning,
            stacklevel=4,
        )
    if not total:
        return [], {label: empty for label, _, _ in spans}, empty, empty
    mesh_all = np.concatenate(mesh_parts)
    local_all = np.concatenate(local_parts)
    # One face per distinct (element, side), numbered in the order first met.
    key = mesh_all * _MAX_SIDES + local_all
    uniq, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[order] = np.arange(len(uniq))
    face_of = rank[inverse.ravel()]
    umesh = uniq[order] // _MAX_SIDES
    ulocal = uniq[order] % _MAX_SIDES
    ucode = codes[umesh]
    width = _FACE_WIDTH[ucode, ulocal]
    # Triangles are appended first, then quads, each in the order met.
    is_tri = width == 3
    n_tri = int(np.count_nonzero(is_tri))
    n_quad = len(uniq) - n_tri
    final = np.empty(len(uniq), dtype=np.int64)
    final[is_tri] = n_kept + np.arange(n_tri)
    final[~is_tri] = n_kept + n_tri + np.arange(n_quad)
    # Row of each mesh element in the stack of its type's blocks, so a face's
    # corners come from one fancy index per (type, side) pair.
    type_row = np.empty(n_kept, dtype=np.int64)
    parts: dict[int, list[np.ndarray]] = {}
    counted: dict[int, int] = {}
    start = 0
    for code, cells in blocks:
        n = len(cells)
        seen = counted.get(code, 0)
        type_row[start : start + n] = np.arange(seen, seen + n)
        counted[code] = seen + n
        parts.setdefault(code, []).append(cells)
        start += n
    stacks = {code: np.vstack(rows) for code, rows in parts.items()}
    tri = np.empty((n_tri, 3), dtype=np.int64)
    quad = np.empty((n_quad, 4), dtype=np.int64)
    for pair in np.unique(ucode * _MAX_SIDES + ulocal).tolist():
        code, local = divmod(pair, _MAX_SIDES)
        sel = (ucode == code) & (ulocal == local)
        corners = list(ELEMENT_FACES[ELEMENT_TYPES_INV[code]][local])
        ring = stacks[code][type_row[umesh[sel]]][:, corners]
        if len(corners) == 3:
            tri[final[sel] - n_kept] = ring
        else:
            quad[final[sel] - n_kept - n_tri] = ring
    groups: list[tuple[str, np.ndarray]] = []
    if n_tri:
        groups.append(("triangle", tri))
    if n_quad:
        groups.append(("quad", quad))
    out_tags = {
        label: np.unique(final[face_of[a:b]]) if b > a else empty
        for label, a, b in spans
    }
    # Parents in final face order: the position of face k is final[k].
    by_final = np.argsort(final)
    return groups, out_tags, umesh[by_final], ulocal[by_final]


def _fold(
    names: list[str], columns: list[np.ndarray], ctx: dict[str, Any], *, what: str
) -> dict[str, np.ndarray]:
    """Fold ``_x/_y/_z`` triples and ``_0/_1/...`` runs into one array each.

    A family is folded only when the file holds no plain variable of the
    base name: ``v`` beside ``v_x``, ``v_y``, ``v_z`` stays four variables,
    where folding would have overwritten one with the other. A name the
    file gives twice keeps its first column, the later one warned about and
    skipped.
    """
    held: dict[str, np.ndarray] = {}
    twice: list[str] = []
    for key, column in zip(names, columns):
        if key in held:
            twice.append(key)
        else:
            held[key] = column
    if twice:
        warnings.warn(
            f"{EXTENSION} read: {what} variable name(s) {sorted(set(twice))} in"
            f" '{ctx['name']}' are each given twice; the later one is skipped.",
            UserWarning,
            stacklevel=4,
        )
    out: dict[str, np.ndarray] = {}
    used: set[str] = set()
    for key in held:
        if key in used:
            continue
        if key.endswith("_x"):
            base = key[:-2]
            trio = [f"{base}_{axis}" for axis in _AXES]
            if base not in held and all(t in held and t not in used for t in trio):
                out[base] = np.column_stack([held[t] for t in trio])
                used.update(trio)
                continue
        if key.endswith("_0"):
            base = key[:-2]
            run: list[str] = []
            while f"{base}_{len(run)}" in held and f"{base}_{len(run)}" not in used:
                run.append(f"{base}_{len(run)}")
            if base not in held and len(run) > 1:
                out[base] = np.column_stack([held[t] for t in run])
                used.update(run)
                continue
        out[key] = held[key]
        used.add(key)
    return out


def _warn_misfit(bad: list[str], ctx: dict[str, Any], *, what: str, per: str) -> None:
    if bad:
        warnings.warn(
            f"{EXTENSION} read: {what} variable(s) {sorted(set(bad))} in"
            f" '{ctx['name']}' do not hold one value per {per} at the step read;"
            " skipped.",
            UserWarning,
            stacklevel=5,
        )


def _read_nodal_vars(
    handle: Any, step: int, n_verts: int, ctx: dict[str, Any]
) -> dict[str, np.ndarray]:
    n_vars = _dim(handle, "num_nod_var")
    if not n_vars:
        return {}
    names = [
        n or f"nod_var{i + 1}"
        for i, n in enumerate(_texts(handle, "name_nod_var", n_vars, ctx))
    ]
    columns: list[np.ndarray] = []
    kept: list[str] = []
    bad: list[str] = []
    legacy = None
    if "vals_nod_var" in handle.variables:
        var = handle.variables["vals_nod_var"]
        if var.ndim != 3 or var.shape[0] <= step or var.shape[1] < n_vars:
            raise CodecError(
                f"'{ctx['name']}': 'vals_nod_var' is {tuple(var.shape)}, not the"
                f" (steps, {n_vars}, {n_verts}) table its dimensions declare."
            )
        legacy = _load(handle, "vals_nod_var", ctx, step=step)
    for i in range(1, n_vars + 1):
        if legacy is not None:
            column = np.asarray(legacy[i - 1], dtype=np.float64)
        elif f"vals_nod_var{i}" in handle.variables:
            loaded = _load(handle, f"vals_nod_var{i}", ctx, step=step)
            if loaded is None:
                bad.append(names[i - 1])
                continue
            column = loaded.astype(np.float64)
        else:
            continue
        column = column.ravel()
        if column.shape != (n_verts,):
            bad.append(names[i - 1])
            continue
        kept.append(names[i - 1])
        columns.append(column)
    _warn_misfit(bad, ctx, what="nodal", per="node")
    return _fold(kept, columns, ctx, what="nodal")


def _read_element_vars(
    handle: Any,
    step: int,
    blocks: list[tuple[int, np.ndarray]],
    file_to_mesh: np.ndarray,
    n_elems: int,
    ctx: dict[str, Any],
) -> dict[str, np.ndarray]:
    n_vars = _dim(handle, "num_elem_var")
    if not n_vars:
        return {}
    names = [
        n or f"elem_var{i + 1}"
        for i, n in enumerate(_texts(handle, "name_elem_var", n_vars, ctx))
    ]
    n_blocks = _dim(handle, "num_el_blk")
    table = None
    if "elem_var_tab" in handle.variables:
        table = _var(handle, "elem_var_tab", ctx)
        if table.shape != (n_blocks, n_vars):
            table = None
    file_starts = []
    offset = 0
    for j in range(1, n_blocks + 1):
        count = _dim(handle, f"num_el_in_blk{j}")
        file_starts.append((offset, count))
        offset += count
    columns: list[np.ndarray] = []
    kept: list[str] = []
    bad: list[str] = []
    for i in range(1, n_vars + 1):
        column = np.full(n_elems, np.nan)
        found = False
        for j, (file_start, count) in enumerate(file_starts, start=1):
            if table is not None and not table[j - 1, i - 1]:
                continue
            key = f"vals_elem_var{i}eb{j}"
            if key not in handle.variables or not count:
                continue
            values = _load(handle, key, ctx, step=step)
            if values is None or values.ravel().shape != (count,):
                bad.append(f"{names[i - 1]} (block {j})")
                continue
            values = values.astype(np.float64).ravel()
            where = file_to_mesh[file_start : file_start + count]
            live = where >= 0
            column[where[live]] = values[live]
            found = True
        if found:
            kept.append(names[i - 1])
            columns.append(column)
    _warn_misfit(bad, ctx, what="element", per="element of the block")
    return _fold(kept, columns, ctx, what="element")


def _read_global_vars(handle: Any, step: int, ctx: dict[str, Any]) -> dict[str, Any]:
    n_vars = _dim(handle, "num_glo_var")
    if not n_vars or "vals_glo_var" not in handle.variables:
        return {}
    names = _texts(handle, "name_glo_var", n_vars, ctx)
    values = _load(handle, "vals_glo_var", ctx, step=step)
    values = np.empty(0) if values is None else values.astype(np.float64).ravel()
    if values.shape != (n_vars,):
        warnings.warn(
            f"{EXTENSION} read: 'vals_glo_var' in '{ctx['name']}' holds"
            f" {values.size} values for {n_vars} global variables; skipped.",
            UserWarning,
            stacklevel=4,
        )
        return {}
    folded = _fold(
        [n or f"glo_var{i + 1}" for i, n in enumerate(names)],
        [np.array([v]) for v in values.tolist()],
        ctx,
        what="global",
    )
    return {
        key: (float(arr.ravel()[0]) if arr.size == 1 else arr.ravel())
        for key, arr in folded.items()
    }


# ----- writing ----------------------------------------------------------------


def write(poly: PolyData, path: Source, **opts: Any) -> None:
    """Write a PolyData as an Exodus II file.

    Parameters
    ----------
    poly
        PolyData to write.
    path
        Output file path, or an open binary file object.
    **opts
        None are taken; any given are warned about and ignored.

    Raises
    ------
    UnsupportedFormatError
        If netCDF4 is not installed.

    Warns
    -----
    UserWarning
        Once for the element types Exodus has no block for - those elements,
        and the values and tags on them, are not written; once for the
        attributes that are not numeric values per entity; once for a
        variable name two attributes would share, the later dropped; once
        for the element attributes that hold a value on a face written as a
        side, which carries no value; once for the ``global_attrs`` that are
        neither numbers nor the title, and once for a ``title`` that is not
        text.

    Notes
    -----
    A tag group whose members are all of one type, and overlap no group
    already taken, is written as an element block of that name; every other
    element goes into a block per type, named ``triangle``, ``tetra`` and so
    on. A group whose members are all faces of solids - what a side set read
    from a file comes back as - is written as a side set, and its faces are
    not written as elements unless another group names them; every other
    element group is an element set, and every vertex group a node set.
    Vertex and element attributes are nodal and element variables at one
    time step, an ``(n, 3)`` array as ``<name>_x/_y/_z`` and any other
    ``(n, k)`` as ``<name>_0..<name>_{k-1}`` - an array of more axes is
    flattened to that first, and an ``(n, 1)`` one is ``<name>_0`` alone, so
    neither reads back in its shape; numeric ``global_attrs`` are global
    variables, an array of three values as ``<name>_x/_y/_z`` and any other
    array as ``<name>_0..``, ``title`` the file's title, ``time`` the
    step's time. The
    file is classic netCDF with 64-bit offsets, or CDF5 when an index needs
    64 bits, or netCDF-4 when the mesh is empty.
    """
    name = source_name(path)
    _require_netcdf4(name=name, verb="writing")
    _warn_unknown_opts(opts, what="write")
    dim = output_dimension(poly, fmt=EXTENSION)
    n_verts = len(poly.vertices)
    n_elems = len(poly.element_types)
    codes = np.asarray(poly.element_types)
    offsets = np.asarray(poly.offsets)
    conn = np.asarray(poly.connectivity)
    widths = np.diff(offsets)

    writable = _WRITE_SIZE_TABLE[np.clip(codes, 0, _N_CODES - 1)] == widths
    dropped = sorted(
        ELEMENT_TYPES_INV.get(int(c), f"type_{c}") for c in np.unique(codes[~writable])
    )
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: element type(s) {dropped} have no Exodus block"
            " (or hold a node count other than their type's); those elements and"
            " the values and tags on them are not written.",
            UserWarning,
            stacklevel=2,
        )

    side_sets, omitted = _side_set_groups(poly, writable)
    _warn_values_on_sides(poly, omitted)
    written = writable & ~omitted
    blocks, elem_sets = _partition(poly, written, side_sets)
    kept = (
        np.concatenate([index for _, _, index in blocks])
        if blocks
        else np.empty(0, np.int64)
    )
    position = np.full(n_elems, -1, dtype=np.int64)
    position[kept] = np.arange(len(kept))

    node_ids = ids_for_write(poly, kind="vertex", count=n_verts, fmt=EXTENSION)
    default = np.zeros(n_elems, dtype=np.int64)
    default[kept] = np.arange(1, len(kept) + 1)
    elem_ids = ids_for_write(
        poly, kind="element", count=n_elems, fmt=EXTENSION, default=default
    )[kept]

    nodal = _split_arrays(
        {k: v for k, v in (poly.vertex_attrs or {}).items() if k != IDS_KEY},
        n_verts,
        None,
        what="vertex",
    )
    element = _split_arrays(
        {
            k: v
            for k, v in (poly.element_attrs or {}).items()
            if k not in _RESERVED_ELEMENT_ATTRS
        },
        n_elems,
        kept,
        what="element",
    )
    numeric = globals_for_write(
        poly, reserved=_RESERVED_GLOBALS, fmt=EXTENSION, text=True
    )
    texts = text_for_write(poly, reserved=_RESERVED_GLOBALS)
    if texts:
        warnings.warn(
            f"{EXTENSION} write: global_attrs {sorted(texts)} are text, and an Exodus"
            " global variable is a number; dropped.",
            UserWarning,
            stacklevel=2,
        )
    globs: list[tuple[str, float]] = []
    for key, arr in numeric.items():
        flat = np.asarray(arr, dtype=np.float64).ravel()
        if flat.size == 1:
            globs.append((key, float(flat[0])))
        elif flat.size == 3:
            globs.extend((f"{key}_{axis}", float(v)) for axis, v in zip(_AXES, flat))
        else:
            globs.extend((f"{key}_{k}", float(v)) for k, v in enumerate(flat.tolist()))
    title = (poly.global_attrs or {}).get("title", "")
    if not isinstance(title, str):
        warnings.warn(
            f"{EXTENSION} write: global_attrs['title'] is {type(title).__name__},"
            " not text, and an Exodus title is text; dropped.",
            UserWarning,
            stacklevel=2,
        )
        title = ""
    time = (poly.global_attrs or {}).get("time", 0.0)
    try:
        time = float(np.asarray(time).ravel()[0]) if np.asarray(time).size else 0.0
    except (TypeError, ValueError):
        time = 0.0

    node_sets = [
        (str(label), member_indices(members, n_verts).astype(np.int64))
        for label, members in (poly.vertex_tags or {}).items()
    ]
    node_sets = [(label, np.unique(m)) for label, m in node_sets if m.size]
    elem_sets = [(label, position[m]) for label, m in elem_sets]
    elem_sets = [(label, np.unique(p[p >= 0])) for label, p in elem_sets]
    elem_sets = [(label, p) for label, p in elem_sets if p.size]
    sides = [
        (label, position[parents], side_numbers)
        for label, parents, side_numbers in side_sets
    ]

    all_names = (
        [label for label, _, _ in blocks]
        + [label for label, _ in node_sets]
        + [label for label, _ in elem_sets]
        + [label for label, _, _ in sides]
        + [key for key, _ in nodal]
        + [key for key, _ in element]
        + [key for key, _ in globs]
    )
    longest = max((len(n.encode("utf-8")) for n in all_names), default=0)
    len_name = min(_MAX_NAME, max(_LEN_STRING, longest + 1))
    if longest + 1 > _MAX_NAME:
        warnings.warn(
            f"{EXTENSION} write: a name longer than {_MAX_NAME - 1} bytes is cut to"
            " that; Exodus holds no longer name.",
            UserWarning,
            stacklevel=2,
        )

    big = (
        max(
            n_verts,
            len(kept),
            int(node_ids.max()) if n_verts else 0,
            int(elem_ids.max()) if len(kept) else 0,
        )
        >= 2**31
    )
    if n_verts == 0 or not blocks:
        nc_format = "NETCDF4"
    elif big:
        nc_format = "NETCDF3_64BIT_DATA"
    else:
        nc_format = "NETCDF3_64BIT_OFFSET"
    itype = "i8" if big else "i4"

    with _open_write(path, fmt=nc_format) as handle:
        handle.setncattr("api_version", np.float32(_API_VERSION))
        handle.setncattr("version", np.float32(_API_VERSION))
        handle.setncattr("floating_point_word_size", np.int32(8))
        handle.setncattr("file_size", np.int32(1))
        handle.setncattr("maximum_name_length", np.int32(len_name - 1))
        if big:
            handle.setncattr("int64_status", np.int32(1 | 2 | 4))
        handle.setncattr("title", title)
        handle.createDimension("len_string", _LEN_STRING)
        handle.createDimension("len_line", _LEN_LINE)
        handle.createDimension("len_name", len_name)
        handle.createDimension("four", 4)
        handle.createDimension("time_step", None)
        handle.createDimension("num_dim", dim)
        handle.createDimension("num_nodes", n_verts)
        handle.createDimension("num_elem", len(kept))

        def names_var(key: str, dim_name: str, labels: list[str]) -> None:
            var = handle.createVariable(key, "S1", (dim_name, "len_name"))
            if labels:
                var[:] = _chars(labels, len_name)

        coords = np.asarray(poly.vertices, dtype=np.float64)[:, :dim]
        coord = handle.createVariable("coord", "f8", ("num_dim", "num_nodes"))
        if n_verts:
            coord[:] = np.ascontiguousarray(coords.T)
        names_var("coor_names", "num_dim", list(_AXES[:dim]))

        handle.createDimension("num_el_blk", len(blocks))
        prop = handle.createVariable("eb_prop1", itype, ("num_el_blk",))
        prop.setncattr("name", "ID")
        status = handle.createVariable("eb_status", itype, ("num_el_blk",))
        if blocks:
            prop[:] = np.arange(1, len(blocks) + 1, dtype=np.int64)
            status[:] = np.ones(len(blocks), dtype=np.int64)
        names_var("eb_names", "num_el_blk", [label for label, _, _ in blocks])
        for j, (_, code, index) in enumerate(blocks, start=1):
            type_name = ELEMENT_TYPES_INV[code]
            width = _WRITE_SIZES[code]
            cells = conn[offsets[index][:, None] + np.arange(width)[None, :]]
            order = _WRITE_ORDER.get(type_name)
            if order is not None:
                cells = cells[:, list(order)]
            handle.createDimension(f"num_el_in_blk{j}", len(index))
            handle.createDimension(f"num_nod_per_el{j}", width)
            table = handle.createVariable(
                f"connect{j}", itype, (f"num_el_in_blk{j}", f"num_nod_per_el{j}")
            )
            table.setncattr("elem_type", _WRITE_NAMES[type_name])
            table[:] = cells.astype(np.int64) + 1

        if not np.array_equal(node_ids, np.arange(1, n_verts + 1)):
            handle.createVariable("node_num_map", itype, ("num_nodes",))[:] = node_ids
        if not np.array_equal(elem_ids, np.arange(1, len(kept) + 1)):
            handle.createVariable("elem_num_map", itype, ("num_elem",))[:] = elem_ids

        _write_sets(
            handle,
            "num_node_sets",
            "ns",
            "node_ns",
            "num_nod_ns",
            node_sets,
            itype,
            names_var,
        )
        _write_sets(
            handle,
            "num_elem_sets",
            "els",
            "elem_els",
            "num_ele_els",
            elem_sets,
            itype,
            names_var,
        )
        if sides:
            handle.createDimension("num_side_sets", len(sides))
            prop = handle.createVariable("ss_prop1", itype, ("num_side_sets",))
            prop.setncattr("name", "ID")
            prop[:] = np.arange(1, len(sides) + 1, dtype=np.int64)
            handle.createVariable("ss_status", itype, ("num_side_sets",))[:] = np.ones(
                len(sides), np.int64
            )
            names_var("ss_names", "num_side_sets", [label for label, _, _ in sides])
            for i, (_, parents, numbers) in enumerate(sides, start=1):
                handle.createDimension(f"num_side_ss{i}", len(parents))
                handle.createVariable(f"elem_ss{i}", itype, (f"num_side_ss{i}",))[:] = (
                    parents + 1
                )
                handle.createVariable(f"side_ss{i}", itype, (f"num_side_ss{i}",))[:] = (
                    numbers
                )

        handle.createVariable("time_whole", "f8", ("time_step",))[0] = time
        if nodal:
            handle.createDimension("num_nod_var", len(nodal))
            names_var("name_nod_var", "num_nod_var", [key for key, _ in nodal])
            for i, (_, column) in enumerate(nodal, start=1):
                var = handle.createVariable(
                    f"vals_nod_var{i}", "f8", ("time_step", "num_nodes")
                )
                var[0, :] = column
        if element:
            handle.createDimension("num_elem_var", len(element))
            names_var("name_elem_var", "num_elem_var", [key for key, _ in element])
            tab = handle.createVariable(
                "elem_var_tab", itype, ("num_el_blk", "num_elem_var")
            )
            if blocks:
                tab[:] = np.ones((len(blocks), len(element)), dtype=np.int64)
            start = 0
            for j, (_, _, index) in enumerate(blocks, start=1):
                for i, (_, column) in enumerate(element, start=1):
                    var = handle.createVariable(
                        f"vals_elem_var{i}eb{j}",
                        "f8",
                        ("time_step", f"num_el_in_blk{j}"),
                    )
                    var[0, :] = column[start : start + len(index)]
                start += len(index)
        if globs:
            handle.createDimension("num_glo_var", len(globs))
            names_var("name_glo_var", "num_glo_var", [key for key, _ in globs])
            var = handle.createVariable(
                "vals_glo_var", "f8", ("time_step", "num_glo_var")
            )
            var[0, :] = np.array([v for _, v in globs], dtype=np.float64)


def _write_sets(
    handle: Any,
    count_dim: str,
    prefix: str,
    table: str,
    size_dim: str,
    sets: list[tuple[str, np.ndarray]],
    itype: str,
    names_var: Any,
) -> None:
    if not sets:
        return
    handle.createDimension(count_dim, len(sets))
    prop = handle.createVariable(f"{prefix}_prop1", itype, (count_dim,))
    prop.setncattr("name", "ID")
    prop[:] = np.arange(1, len(sets) + 1, dtype=np.int64)
    handle.createVariable(f"{prefix}_status", itype, (count_dim,))[:] = np.ones(
        len(sets), np.int64
    )
    names_var(f"{prefix}_names", count_dim, [label for label, _ in sets])
    for i, (_, members) in enumerate(sets, start=1):
        handle.createDimension(f"{size_dim}{i}", len(members))
        handle.createVariable(f"{table}{i}", itype, (f"{size_dim}{i}",))[:] = (
            members + 1
        )


def _side_set_groups(
    poly: PolyData, writable: np.ndarray
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], np.ndarray]:
    """The tag groups that go back as side sets, and the faces they alone name.

    A group is a side set when every member still is the face of a solid it
    claims to be, and that solid has an Exodus side numbering. A face named
    by a side-set group and by no other group is not written as an element:
    it came in as a side and goes out as one. A face another group names too
    stays an element for that group's sake.
    """
    n_elems = len(poly.element_types)
    omitted = np.zeros(n_elems, dtype=bool)
    columns = parent_faces(poly)
    if columns is None or not poly.element_tags:
        return [], omitted
    parent_col, index_col = columns
    codes = np.asarray(poly.element_types)
    sides: list[tuple[str, np.ndarray, np.ndarray]] = []
    in_side = np.zeros(n_elems, dtype=np.int64)
    in_any = np.zeros(n_elems, dtype=np.int64)
    for label, members in poly.element_tags.items():
        held = members_array(members)
        picked = member_indices(held, n_elems)
        picked = np.unique(picked[writable[picked]]) if picked.size else picked
        in_any[picked] += 1
        if not picked.size:
            continue
        parents = parent_col[picked]
        locals_ = index_col[picked]
        if (parents < 0).any() or (parents >= n_elems).any():
            continue
        if (locals_ < 0).any() or (locals_ >= _MAX_SIDES).any():
            continue
        numbers = _SIDE_OF_LOCAL[np.clip(codes[parents], 0, _N_CODES - 1), locals_]
        if (numbers == 0).any() or not writable[parents].all():
            continue
        if not parent_face_mask(poly, picked, parents, locals_).all():
            continue
        in_side[picked] += 1
        sides.append((str(label), parents.astype(np.int64), numbers))
    omitted = (in_side > 0) & (in_side == in_any)
    return sides, omitted


def _warn_values_on_sides(poly: PolyData, omitted: np.ndarray) -> None:
    """Warn for the element attributes that hold a value on a face going out
    as a side: a side set names an element and a face number, nothing more,
    so the value has no place in the file."""
    if not omitted.any():
        return
    n_elems = len(poly.element_types)
    lost: list[str] = []
    for key, values in (poly.element_attrs or {}).items():
        if key in _RESERVED_ELEMENT_ATTRS:
            continue
        arr = np.asarray(values)
        if arr.dtype.kind not in "biuf" or arr.ndim == 0 or arr.shape[0] != n_elems:
            continue
        rows = arr[omitted].reshape(int(np.count_nonzero(omitted)), -1)
        if arr.dtype.kind == "f":
            if np.isfinite(rows).any():
                lost.append(str(key))
        elif rows.size:
            lost.append(str(key))
    if lost:
        warnings.warn(
            f"{EXTENSION} write: element attribute(s) {sorted(lost)} hold values"
            " on faces written as side sets, which carry no value; those"
            " values are not written.",
            UserWarning,
            stacklevel=3,
        )


def _partition(
    poly: PolyData,
    written: np.ndarray,
    side_sets: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[list[tuple[str, int, np.ndarray]], list[tuple[str, np.ndarray]]]:
    """Blocks as ``(label, code, mesh indices)`` and the groups left as sets.

    The block of leftovers per type is named after the type, or ``tetra_2``
    and up when a tag group already took the plain name, as a block, a set
    or a side set: the reader merges same-named groups of every kind into
    one tag, and the group's members would gain every leftover of that
    type.
    """
    n_elems = len(poly.element_types)
    codes = np.asarray(poly.element_types)
    side_labels = {label for label, _, _ in side_sets}
    taken = np.zeros(n_elems, dtype=bool)
    blocks: list[tuple[str, int, np.ndarray]] = []
    sets: list[tuple[str, np.ndarray]] = []
    for label, members in (poly.element_tags or {}).items():
        if str(label) in side_labels:
            continue
        picked = member_indices(members_array(members), n_elems)
        picked = np.unique(picked[written[picked]]) if picked.size else picked
        if not picked.size:
            continue
        kinds = np.unique(codes[picked])
        if len(kinds) == 1 and not taken[picked].any():
            taken[picked] = True
            blocks.append((str(label), int(kinds[0]), picked))
        else:
            sets.append((str(label), picked))
    used = {label for label, _, _ in blocks} | {label for label, _ in sets}
    used |= side_labels
    rest = np.flatnonzero(written & ~taken)
    for code in np.unique(codes[rest]).tolist():
        index = rest[codes[rest] == code]
        base = label = ELEMENT_TYPES_INV[int(code)]
        k = 2
        while label in used:
            label = f"{base}_{k}"
            k += 1
        used.add(label)
        blocks.append((label, int(code), index))
    return blocks, sets


def _split_arrays(
    attrs: dict[str, np.ndarray] | None,
    count: int,
    index: np.ndarray | None,
    *,
    what: str,
) -> list[tuple[str, np.ndarray]]:
    """One float column per variable, an ``(n, k)`` array split by suffix.

    A name two attributes land on - ``v`` split to ``v_x`` beside a plain
    ``v_x`` - is written once, from the attribute met first; the reader
    folds by name and could not tell the two apart.
    """
    out: list[tuple[str, np.ndarray]] = []
    dropped: list[str] = []
    clashed: list[str] = []
    taken: set[str] = set()

    def emit(label: str, column: np.ndarray) -> None:
        if label in taken:
            clashed.append(label)
            return
        taken.add(label)
        out.append((label, column))

    for key, values in (attrs or {}).items():
        arr = np.asarray(values)
        if arr.dtype.kind == "b":
            arr = arr.astype(np.float64)
        if arr.dtype.kind not in "iuf" or arr.ndim == 0 or arr.shape[0] != count:
            dropped.append(str(key))
            continue
        arr = arr.astype(np.float64, copy=False)
        if index is not None:
            arr = arr[index]
        if arr.ndim == 1:
            emit(str(key), arr)
            continue
        flat = arr.reshape(len(arr), -1)
        if flat.shape[1] == 3:
            for k, axis in enumerate(_AXES):
                emit(f"{key}_{axis}", flat[:, k])
        else:
            for k in range(flat.shape[1]):
                emit(f"{key}_{k}", flat[:, k])
    if dropped:
        warnings.warn(
            f"{EXTENSION} write: {what} attribute(s) {sorted(dropped)} are not one"
            " numeric value per entity; dropped.",
            UserWarning,
            stacklevel=3,
        )
    if clashed:
        warnings.warn(
            f"{EXTENSION} write: {what} variable name(s) {sorted(set(clashed))} are"
            " each claimed by two attributes; the later one is not written.",
            UserWarning,
            stacklevel=3,
        )
    return out
