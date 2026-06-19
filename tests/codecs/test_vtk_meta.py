"""Tests for parallel/multi-block VTK meta-format stubs."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

import polyxios
from polyxios.exceptions import UnsupportedFormatError

# Minimal valid XML for each format — no geometry, just valid structure
_SAMPLES: dict[str, str] = {
    ".vtm": (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n'
        "  <vtkMultiBlockDataSet>\n"
        '    <DataSet index="0" file="piece0.vtu"/>\n'
        "  </vtkMultiBlockDataSet>\n"
        "</VTKFile>\n"
    ),
    ".pvtu": (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PUnstructuredGrid" version="0.1">\n'
        '  <PUnstructuredGrid GhostLevel="0">\n'
        '    <Piece Source="piece_0.vtu"/>\n'
        "  </PUnstructuredGrid>\n"
        "</VTKFile>\n"
    ),
    ".pvtp": (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PPolyData" version="0.1">\n'
        '  <PPolyData GhostLevel="0">\n'
        '    <Piece Source="piece_0.vtp"/>\n'
        "  </PPolyData>\n"
        "</VTKFile>\n"
    ),
    ".pvtr": (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PRectilinearGrid" version="0.1">\n'
        '  <PRectilinearGrid GhostLevel="0" WholeExtent="0 4 0 4 0 0">\n'
        '    <Piece Extent="0 2 0 4 0 0" Source="piece_0.vtr"/>\n'
        "  </PRectilinearGrid>\n"
        "</VTKFile>\n"
    ),
    ".pvts": (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PStructuredGrid" version="0.1">\n'
        '  <PStructuredGrid GhostLevel="0" WholeExtent="0 4 0 4 0 0">\n'
        '    <Piece Extent="0 2 0 4 0 0" Source="piece_0.vts"/>\n'
        "  </PStructuredGrid>\n"
        "</VTKFile>\n"
    ),
    ".pvti": (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PImageData" version="0.1">\n'
        '  <PImageData GhostLevel="0" WholeExtent="0 4 0 4 0 0"'
        '    Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="0 2 0 4 0 0" Source="piece_0.vti"/>\n'
        "  </PImageData>\n"
        "</VTKFile>\n"
    ),
}


@pytest.mark.parametrize("ext", list(_SAMPLES))
def test_read_raises_unsupported(ext: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w") as f:
        f.write(_SAMPLES[ext])
        tmp = f.name
    with pytest.raises(UnsupportedFormatError) as exc_info:
        polyxios.read(tmp)
    assert "read_parallel_vtk.py" in str(exc_info.value)
    assert ext in str(exc_info.value)


@pytest.mark.parametrize("ext", list(_SAMPLES))
def test_write_raises_not_implemented(ext: str) -> None:
    from polyxios import make_polydata

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    poly = make_polydata(verts, [("tetra", np.array([[0, 1, 2, 3]]))])
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        tmp = f.name
    with pytest.raises(NotImplementedError):
        polyxios.write(poly, tmp)


@pytest.mark.parametrize("ext", list(_SAMPLES))
def test_error_message_mentions_tutorial(ext: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w") as f:
        f.write(_SAMPLES[ext])
        tmp = f.name
    with pytest.raises(UnsupportedFormatError) as exc_info:
        polyxios.read(tmp)
    msg = str(exc_info.value)
    assert "read_parallel_vtk.py" in msg
    assert "sub-file" in msg.lower() or "meta" in msg.lower()
