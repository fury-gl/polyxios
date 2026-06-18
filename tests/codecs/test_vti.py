from __future__ import annotations

import tempfile

import numpy as np
import pytest

from polyxios.codecs._vti import read, write
from polyxios.exceptions import LazyReadError


def _synthetic_vti() -> object:
    """Write then read a 2×2×2 ImageData, return the PolyData."""
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 2 0 2 0 2" Origin="0 0 0" Spacing="1 1 1">\n'
        '    <Piece Extent="0 2 0 2 0 2">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    with open(tmp, "w") as fh:
        fh.write(xml)
    return read(tmp)


def test_read_basic() -> None:
    poly = _synthetic_vti()
    # 2×2×2 grid → 3×3×3 = 27 vertices, 8 hex cells
    assert len(poly.vertices) == 27
    assert len(poly.element_types) == 8
    assert poly.vertices.dtype == np.float64
    assert poly.vertices.shape[1] == 3


def test_roundtrip_ascii() -> None:
    poly = _synthetic_vti()
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=False)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-6)
    assert len(poly2.element_types) == len(poly.element_types)


def test_roundtrip_binary() -> None:
    poly = _synthetic_vti()
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    write(poly, tmp, binary=True)
    poly2 = read(tmp)
    np.testing.assert_allclose(poly2.vertices, poly.vertices, atol=1e-8)
    assert len(poly2.element_types) == len(poly.element_types)


def test_lazy_raises() -> None:
    poly = _synthetic_vti()
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    write(poly, tmp)
    with pytest.raises(LazyReadError):
        read(tmp, lazy=True)


def test_origin_and_spacing() -> None:
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        '  <ImageData WholeExtent="0 1 0 1 0 1" Origin="1 2 3" Spacing="0.5 0.5 0.5">\n'
        '    <Piece Extent="0 1 0 1 0 1">\n'
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    with open(tmp, "w") as fh:
        fh.write(xml)
    poly = read(tmp)
    # 1×1×1 grid → 8 vertices at origin + spacing
    assert len(poly.vertices) == 8
    np.testing.assert_allclose(poly.vertices.min(axis=0), [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(poly.vertices.max(axis=0), [1.5, 2.5, 3.5], atol=1e-12)


def test_cell_data_roundtrip() -> None:
    poly = _synthetic_vti()
    from polyxios._types import PolyData

    poly_with_attr = PolyData(
        vertices=poly.vertices,
        connectivity=poly.connectivity,
        offsets=poly.offsets,
        element_types=poly.element_types,
        vertex_attrs={},
        element_attrs={"pressure": np.arange(8, dtype=np.float64)},
        global_attrs=poly.global_attrs,
    )
    with tempfile.NamedTemporaryFile(suffix=".vti", delete=False) as f:
        tmp = f.name
    write(poly_with_attr, tmp, binary=False)
    poly2 = read(tmp)
    assert "pressure" in poly2.element_attrs
    np.testing.assert_allclose(
        poly2.element_attrs["pressure"], np.arange(8, dtype=np.float64), atol=1e-6
    )
