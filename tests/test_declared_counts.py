"""What a reader does with a count no file could hold.

Every text format states its own sizes - ``POINTS n``, ``element vertex n``,
``N=`` - and a corrupt or hostile one states a size the file does not deliver.
The number is read before anything is allocated, so a reader that multiplies
it into a byte count or a dtype width before bounding it either allocates what
it was told to or hands back the error numpy raises about a shape, naming
neither the file nor the array.

This file is the matrix of that: one absurd count per format, and the promise
that what comes back is a polyxios error naming the file. It is exhaustive by
declaration rather than by construction - a format whose counts are implied by
its data (Abaqus, Nastran, OBJ, WKT, FLAC3D, MDPA, PERMAS) has no such header to
corrupt, and is listed in ``_NO_DECLARED_COUNT`` so the omission is on purpose,
as is one that is never read at all (SVG).
"""

from __future__ import annotations

from collections.abc import Callable
import re
import struct
import warnings

import numpy as np
import pytest

import polyxios
from polyxios.exceptions import CodecError, ValidationError

# Far past every safety cap, and past what a 32-bit byte count can hold.
BIG = 2**62

# One file per format, each declaring BIG of something it does not hold.
CORRUPT: dict[str, str] = {
    # A LAS 1.4 header of point format 0 whose 64-bit count is BIG; every
    # byte is below 128, so the text written is the bytes read.
    ".las": (
        b"LASF"
        + bytes(20)
        + bytes([1, 4])
        + bytes(64)
        + bytes(4)
        + (375).to_bytes(2, "little")
        + (375).to_bytes(4, "little")
        + bytes(4)
        + bytes([0])
        + (20).to_bytes(2, "little")
        + bytes(24)
        + bytes(24)
        + bytes(72)
        + bytes(8)
        + bytes(12)
        + BIG.to_bytes(8, "little")
        + bytes(120)
        + bytes(20)
    ).decode("ascii"),
    ".pcd": (
        f"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {BIG}\n"
        f"HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {BIG}\nDATA ascii\n0 0 0\n"
    ),
    ".pts": f"{BIG}\n0 0 0\n",
    ".ptx": (
        f"{BIG}\n1\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        "0 0 0 1\n"
    ),
    ".avs": f"3 {BIG} 0 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n1 1 tri 1 2 3\n",
    ".mesh": (
        "MFEM mesh v1.0\n\ndimension\n3\n\nelements\n1\n1 4 0 1 2 3\n\n"
        f"boundary\n0\n\nvertices\n{BIG}\n3\n0 0 0\n"
    ),
    ".medit": (
        "MeshVersionFormatted 1\nDimension\n3\nVertices\n3\n"
        f"0 0 0 0\n1 0 0 0\n0 1 0 0\nTriangles\n{BIG}\n1 2 3 0\nEnd\n"
    ),
    ".fluent": f"(2 3)\n(10 (0 1 {BIG:x} 0))\n(10 (1 1 1 1 3)(\n0 0 0\n))\n",
    ".msh": (
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        f"$Nodes\n{BIG}\n1 0 0 0\n$EndNodes\n"
        "$Elements\n1\n1 2 2 1 1 1 1 1\n$EndElements\n"
    ),
    ".node": f"{BIG} 3 0 0\n1 0 0 0\n",
    ".off": f"OFF\n3 {BIG} 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
    ".ply": (
        "ply\nformat ascii 1.0\n"
        f"element vertex {BIG}\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n"
    ),
    ".su2": f"NDIME= 3\nNPOIN= {BIG}\n0.0 0.0 0.0 0\n",
    ".tec": (
        'VARIABLES = "X" "Y" "Z"\n'
        f"ZONE N=3, E={BIG}, DATAPACKING=POINT, ZONETYPE=FETRIANGLE\n"
        "0 0 0\n1 0 0\n0 1 0\n1 2 3\n"
    ),
    ".ugrid": f"{BIG} 1 0 0 0 0 0\n0 0 0\n",
    ".vol": (
        "mesh3d\ndimension\n3\ngeomtype\n0\n\n"
        f"surfaceelements\n{BIG}\n1 1 0 0 3 1 2 3\n"
    ),
    ".vtk": (
        "# vtk DataFile Version 2.0\nx\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        f"POINTS {BIG} double\n0 0 0\n"
    ),
    ".vti": (
        '<?xml version="1.0"?>\n<VTKFile type="ImageData"><ImageData'
        f' WholeExtent="0 {BIG} 0 0 0 0" Origin="0 0 0" Spacing="1 1 1">'
        f'<Piece Extent="0 {BIG} 0 0 0 0"></Piece></ImageData></VTKFile>\n'
    ),
    ".vtp": (
        '<?xml version="1.0"?>\n<VTKFile type="PolyData"><PolyData>'
        f'<Piece NumberOfPoints="{BIG}" NumberOfPolys="0"><Points>'
        '<DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        "0 0 0</DataArray></Points></Piece></PolyData></VTKFile>\n"
    ),
    ".vtr": (
        '<?xml version="1.0"?>\n<VTKFile type="RectilinearGrid">'
        f'<RectilinearGrid WholeExtent="0 {BIG} 0 0 0 0">'
        f'<Piece Extent="0 {BIG} 0 0 0 0"><Coordinates>'
        '<DataArray type="Float64" format="ascii">0 1</DataArray>'
        '<DataArray type="Float64" format="ascii">0</DataArray>'
        '<DataArray type="Float64" format="ascii">0</DataArray>'
        "</Coordinates></Piece></RectilinearGrid></VTKFile>\n"
    ),
    ".vts": (
        '<?xml version="1.0"?>\n<VTKFile type="StructuredGrid">'
        f'<StructuredGrid WholeExtent="0 {BIG} 0 0 0 0">'
        f'<Piece Extent="0 {BIG} 0 0 0 0"><Points>'
        '<DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        "0 0 0</DataArray></Points></Piece></StructuredGrid></VTKFile>\n"
    ),
    ".vtu": (
        '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid">'
        f'<UnstructuredGrid><Piece NumberOfPoints="{BIG}" NumberOfCells="0">'
        '<Points><DataArray type="Float64" NumberOfComponents="3"'
        ' format="ascii">0 0 0</DataArray></Points></Piece>'
        "</UnstructuredGrid></VTKFile>\n"
    ),
    ".xdmf": (
        '<Xdmf Version="3.0"><Domain><Grid Name="g"><Geometry GeometryType="XYZ">'
        f'<DataItem Dimensions="{BIG} 3" Format="XML">0 0 0</DataItem></Geometry>'
        '<Topology TopologyType="Polyvertex" NumberOfElements="1">'
        '<DataItem Dimensions="1" NumberType="Int" Format="XML">0</DataItem>'
        "</Topology></Grid></Domain></Xdmf>"
    ),
    ".xml": (
        f'<dolfin><mesh celltype="tetrahedron" dim="3"><vertices size="{BIG}">'
        '<vertex index="0" x="0" y="0" z="0"/></vertices>'
        '<cells size="1"><tetrahedron index="0" v0="0" v1="0" v2="0" v3="0"/>'
        "</cells></mesh></dolfin>"
    ),
    ".dae": (
        '<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/'
        'COLLADASchema" version="1.4.1"><library_geometries><geometry id="g">'
        f'<mesh><source id="p"><float_array id="pa" count="{BIG}">0 0 0'
        f'</float_array><technique_common><accessor source="#pa" count="{BIG}" '
        'stride="3"><param name="X" type="float"/><param name="Y" type="float"/>'
        '<param name="Z" type="float"/></accessor></technique_common></source>'
        '<vertices id="v"><input semantic="POSITION" source="#p"/></vertices>'
        '<triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/>'
        "<p>0 0 0</p></triangles></mesh></geometry></library_geometries>"
        "</COLLADA>"
    ),
    ".gltf": (
        '{"asset":{"version":"2.0"},'
        '"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],'
        '"meshes":[{"primitives":[{"attributes":{"POSITION":0},"mode":4}]}],'
        f'"accessors":[{{"bufferView":0,"componentType":5126,"count":{BIG},"type":"VEC3"}}],'
        '"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],'
        '"buffers":[{"uri":"data:application/octet-stream;base64,'
        'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","byteLength":36}]}'
    ),
}

# Formats whose sizes are implied by the data rather than declared in a header,
# so there is no count to corrupt. Listed rather than left out, so a reader
# that grows a header of its own is noticed.
# The HDF5-backed formats declare their counts in attributes and index ranges
# beside datasets that carry their own size; one builder per format writes a
# tiny file whose declaration is BIG. Each needs h5py, and is skipped without.


def _corrupt_med(path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        mesh = f.create_group("ENS_MAA/m")
        mesh.attrs["ESP"] = 3
        coo = mesh.create_dataset("NOE/COO", data=np.zeros(3))
        coo.attrs["NBR"] = BIG


def _corrupt_cgns(path) -> None:
    h5py = pytest.importorskip("h5py")

    def node(parent, name, label, data):
        g = parent.create_group(name)
        g.attrs.create("name", np.bytes_(name), dtype="S33")
        g.attrs.create("label", np.bytes_(label), dtype="S33")
        g.attrs.create(
            "type", np.bytes_("I4" if data is not None else "MT"), dtype="S3"
        )
        if data is not None:
            g.create_dataset(" data", data=np.asarray(data))
        return g

    with h5py.File(path, "w") as f:
        base = node(f, "Base", "CGNSBase_t", np.array([3, 3], dtype=np.int32))
        zone = node(base, "Zone", "Zone_t", np.array([[1], [1], [0]], dtype=np.int32))
        grid = node(zone, "GridCoordinates", "GridCoordinates_t", None)
        for axis in ("CoordinateX", "CoordinateY", "CoordinateZ"):
            node(grid, axis, "DataArray_t", np.zeros(1))
        section = node(zone, "Cells", "Elements_t", np.array([2, 0], dtype=np.int32))
        node(
            section, "ElementRange", "IndexRange_t", np.array([1, BIG], dtype=np.int64)
        )
        node(
            section, "ElementConnectivity", "DataArray_t", np.array([1], dtype=np.int32)
        )


def _corrupt_h5m(path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        tstt = f.create_group("tstt")
        coords = tstt.create_dataset("nodes/coordinates", data=np.zeros((1, 3)))
        coords.attrs["start_id"] = 1
        sets = tstt.create_group("sets")
        table = sets.create_dataset(
            "list", data=np.array([[1, -1, -1, 8]], dtype=np.int64)
        )
        table.attrs["start_id"] = 2
        # Range-compressed contents: one run of BIG entities from handle 1.
        sets.create_dataset("contents", data=np.array([1, BIG], dtype=np.uint64))
        tags = tstt.create_group("tags/NAME")
        tags.create_dataset("id_list", data=np.array([2], dtype=np.uint64))
        tags.create_dataset("values", data=np.array([b"s"], dtype="S32"))


def _corrupt_vtkhdf(path) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 0], dtype=np.int64)
        root.attrs["Type"] = np.bytes_("UnstructuredGrid")
        root.create_dataset("NumberOfPoints", data=np.array([BIG], dtype=np.int64))
        root.create_dataset("NumberOfCells", data=np.array([0], dtype=np.int64))
        root.create_dataset(
            "NumberOfConnectivityIds", data=np.array([0], dtype=np.int64)
        )
        root.create_dataset("Points", data=np.zeros((1, 3)))
        root.create_dataset("Types", data=np.zeros(0, dtype=np.uint8))
        root.create_dataset("Connectivity", data=np.zeros(0, dtype=np.int64))
        root.create_dataset("Offsets", data=np.zeros(1, dtype=np.int64))


def _corrupt_exodus(path) -> None:
    # A netCDF header declares its dimensions up front, so a node count no
    # file could hold costs a few bytes to spell; CDF5 is the flavour whose
    # dimensions are 64-bit.
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as f:
        f.createDimension("num_dim", 3)
        f.createDimension("num_nodes", BIG)


CORRUPT_HDF5: dict[str, Callable] = {
    ".med": _corrupt_med,
    ".cgns": _corrupt_cgns,
    ".h5m": _corrupt_h5m,
    ".vtkhdf": _corrupt_vtkhdf,
    ".e": _corrupt_exodus,
}


@pytest.mark.parametrize("ext", sorted(CORRUPT_HDF5))
def test_a_count_no_hdf5_file_can_hold_is_refused(tmp_path, ext: str) -> None:
    path = tmp_path / f"corrupt{ext}"
    CORRUPT_HDF5[ext](path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((CodecError, ValidationError)) as excinfo:
            polyxios.read(path)

    assert re.search(r"\d{15,}", str(excinfo.value)), excinfo.value


_NO_DECLARED_COUNT: frozenset[str] = frozenset(
    {
        ".inp",
        ".bdf",
        ".obj",
        ".wkt",
        ".f3grid",
        ".mdpa",
        ".dato",
        ".post",
        ".stl",
        ".splat",
        # Bare columns: the count is the line count.
        ".xyz",
        ".vtm",
        # An index of datasets: every count is the named file's to declare.
        ".pvd",
        # Every vertex and triangle is an XML element of its own; no count.
        ".3mf",
        # Write-only: there is no reader for a count to reach.
        ".svg",
        # Every array is an HDF5 dataset of its own size; nothing declares one.
        ".hmf",
    }
)


@pytest.mark.parametrize("ext", sorted(CORRUPT))
def test_a_count_no_file_can_hold_is_refused(tmp_path, ext: str) -> None:
    path = tmp_path / f"corrupt{ext}"
    path.write_text(CORRUPT[ext])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises((CodecError, ValidationError)) as excinfo:
            polyxios.read(path)

    # The message names the size it refused, rather than being the shape or
    # length error numpy raises about an array nobody can name.
    assert re.search(r"\d{15,}", str(excinfo.value)), excinfo.value


def test_the_matrix_covers_every_format_that_declares_a_count() -> None:
    """A new codec cannot land without saying which side of this it is on."""
    covered = frozenset(CORRUPT) | frozenset(CORRUPT_HDF5) | _NO_DECLARED_COUNT
    # Aliases, meta-files and the binary-only flavours are covered by the
    # codec they resolve to; every other extension has to be accounted for.
    resolved = {
        ".fem": ".bdf",
        ".nas": ".bdf",
        ".dat": ".bdf",
        ".ele": ".node",
        ".meshb": ".medit",
        ".plt": ".tec",
        ".pvti": ".vti",
        ".pvtp": ".vtp",
        ".pvtr": ".vtr",
        ".pvts": ".vts",
        ".pvtu": ".vtu",
        ".glb": ".gltf",
        ".xmf": ".xdmf",
        ".exo": ".e",
        ".ex2": ".e",
        ".laz": ".las",
    }
    outstanding = {
        ext
        for ext in polyxios.supported_extensions()
        if ext not in covered and resolved.get(ext) not in covered
    }
    assert not outstanding


def test_a_corrupt_face_count_is_refused_not_built(tmp_path) -> None:
    """A binary PLY face declaring 2**31-1 vertices described a record wider
    than a C int can measure. The whole-block read built a dtype for it before
    bounding it against the file, and numpy answered with a ValueError about a
    tuple shape, naming neither the file nor the face."""
    path = tmp_path / "wide_face.ply"
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 3\nproperty float x\nproperty float y\n"
        b"property float z\n"
        b"element face 2\nproperty list int int vertex_indices\n"
        b"end_header\n"
    )
    body = (
        np.zeros(9, dtype="<f4").tobytes() + struct.pack("<i", 2**31 - 1) + b"\x00" * 8
    )
    path.write_bytes(header + body)

    with pytest.raises(CodecError, match="ends inside its 2 face record"):
        polyxios.read(path)
