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
its data (Abaqus, Nastran, OBJ, WKT, FLAC3D, MDPA) has no such header to
corrupt, and is listed in ``_NO_DECLARED_COUNT`` so the omission is on purpose.
"""

from __future__ import annotations

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
    ".avs": f"3 {BIG} 0 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n1 1 tri 1 2 3\n",
    ".mesh": (
        "MFEM mesh v1.0\n\ndimension\n3\n\nelements\n1\n1 4 0 1 2 3\n\n"
        f"boundary\n0\n\nvertices\n{BIG}\n3\n0 0 0\n"
    ),
    ".medit": (
        "MeshVersionFormatted 1\nDimension\n3\nVertices\n3\n"
        f"0 0 0 0\n1 0 0 0\n0 1 0 0\nTriangles\n{BIG}\n1 2 3 0\nEnd\n"
    ),
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
    ".xml": (
        f'<dolfin><mesh celltype="tetrahedron" dim="3"><vertices size="{BIG}">'
        '<vertex index="0" x="0" y="0" z="0"/></vertices>'
        '<cells size="1"><tetrahedron index="0" v0="0" v1="0" v2="0" v3="0"/>'
        "</cells></mesh></dolfin>"
    ),
}

# Formats whose sizes are implied by the data rather than declared in a header,
# so there is no count to corrupt. Listed rather than left out, so a reader
# that grows a header of its own is noticed.
_NO_DECLARED_COUNT: frozenset[str] = frozenset(
    {".inp", ".bdf", ".obj", ".wkt", ".f3grid", ".mdpa", ".stl", ".splat", ".vtm"}
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
    covered = frozenset(CORRUPT) | _NO_DECLARED_COUNT
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
    }
    outstanding = {
        ext
        for ext in polyxios.supported_extensions()
        if ext not in covered and resolved.get(ext) not in covered
    }
    assert not outstanding


def test_issue_1499_a_corrupt_face_count_is_refused_not_built(tmp_path) -> None:
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
