"""The plugin driven by napari itself, not by hand-built layer kwargs.

tests/test_napari.py feeds the writers what ``layer_data`` returns. napari
hands them ``layer.as_layer_data_tuple()`` instead - features as a DataFrame,
colours as RGBA, its own copy of the metadata - and only a real layer shows
whether the hooks agree with that. Skipped where napari is not installed.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

import polyxios
from polyxios import make_polydata

# napari and npe2 import pydantic models that still use deprecated hooks.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    components = pytest.importorskip("napari.components")
    npe2 = pytest.importorskip("npe2")

_MANIFEST = Path(polyxios.__file__).parent / "napari.yaml"

_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 3.0]],
)


@pytest.fixture
def viewer():
    # An editable install is invisible to npe2's discovery; registering the
    # manifest by hand puts this checkout's hooks behind napari either way.
    # Only a manifest registered here is unregistered after, so the global
    # plugin manager is left as the test found it.
    manager = npe2.PluginManager.instance()
    try:
        manager.get_manifest("polyxios")
        registered = False
    except KeyError:
        manager.register(npe2.PluginManifest.from_file(_MANIFEST))
        registered = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            yield components.ViewerModel()
    finally:
        if registered:
            manager.unregister("polyxios")


def _open(viewer, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return viewer.open(str(path), plugin=None)


def _save(layer, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return layer.save(str(path), plugin="polyxios")


def test_surface_opens_and_saves_through_napari(viewer, tmp_path):
    src = tmp_path / "m.vtp"
    polyxios.write(
        make_polydata(
            _VERTS,
            [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
            vertex_attrs={"temp": np.arange(4.0), "normals": np.ones((4, 3))},
        ),
        src,
    )
    [layer] = _open(viewer, src)
    assert layer._type_string == "surface"
    np.testing.assert_allclose(layer.data[0], _VERTS[:, ::-1])

    out = tmp_path / "out.vtp"
    _save(layer, out)
    back = polyxios.read(out)
    np.testing.assert_allclose(back.vertices, _VERTS)
    np.testing.assert_array_equal(back.vertex_attrs["temp"], np.arange(4.0))
    np.testing.assert_array_equal(back.vertex_attrs["normals"], np.ones((4, 3)))


def test_surface_without_values_saves_none_through_napari(viewer, tmp_path):
    # napari fills a Surface given no values with ones.
    src = tmp_path / "m.off"
    polyxios.write(make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))]), src)
    [layer] = _open(viewer, src)
    out = tmp_path / "out.ply"
    _save(layer, out)
    assert "values" not in polyxios.read(out).vertex_attrs


def test_points_open_and_save_through_napari(viewer, tmp_path):
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], np.float32)
    src = tmp_path / "c.pcd"
    polyxios.write(
        make_polydata(
            _VERTS,
            [],
            vertex_attrs={"intensity": np.arange(1.0, 5.0), "colors": colors},
        ),
        src,
    )
    [layer] = _open(viewer, src)
    assert layer._type_string == "points"
    assert list(layer.features.columns) == ["intensity"]

    out = tmp_path / "out.pcd"
    _save(layer, out)
    back = polyxios.read(out)
    np.testing.assert_allclose(back.vertices, _VERTS)
    np.testing.assert_array_equal(back.vertex_attrs["intensity"], [1, 2, 3, 4])
    np.testing.assert_allclose(back.vertex_attrs["colors"], colors, atol=1 / 255)


def test_integer_values_keep_their_dtype_through_napari(viewer, tmp_path):
    src = tmp_path / "m.vtp"
    labels = np.array([1, 2, 3, 4], dtype=np.int32)
    polyxios.write(
        make_polydata(
            _VERTS,
            [("triangle", np.array([[0, 1, 2]]))],
            vertex_attrs={"label": labels},
        ),
        src,
    )
    [layer] = _open(viewer, src)
    out = tmp_path / "out.vtp"
    _save(layer, out)
    back = polyxios.read(out).vertex_attrs["label"]
    assert back.dtype == np.int32
    np.testing.assert_array_equal(back, labels)


def test_upper_case_extension_opens_through_napari(viewer, tmp_path):
    # npe2 lower-cases the extension before matching it to the patterns.
    src = tmp_path / "PART.STL"
    polyxios.write(
        make_polydata(_VERTS, [("triangle", np.array([[0, 1, 2]]))]), src, fmt=".stl"
    )
    [layer] = _open(viewer, src)
    assert layer._type_string == "surface"
    assert layer.name == "PART"
