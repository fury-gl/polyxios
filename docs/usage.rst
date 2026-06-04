Usage
=====

Basic I/O
---------

.. code-block:: python

    import polyxios as px

    # Read any supported format
    mesh = px.read("brain.vtk")

    # Inspect
    print(mesh.vertices.shape)      # (n_verts, 3)
    print(len(mesh.element_types))  # number of elements

    # Write to a different format
    px.write(mesh, "brain.ply")
    px.write(mesh, "brain.vtp")

Format-specific options
-----------------------

.. code-block:: python

    px.write(mesh, "brain.vtk", binary=True)
    px.write(mesh, "brain.ply", binary=True, endian="little")

Lazy loading
------------

For large meshes (gigabytes of binary data), pass ``lazy=True``. polyxios
memory-maps the file and only loads the pages you actually touch — the rest
stays on disk until needed.

.. code-block:: python

    # File is opened but data is not loaded into RAM yet
    mesh = px.read("huge_brain.vtk", lazy=True)

    # Only the vertices are pulled from disk here
    first_vertex = mesh.vertices[0]

    # Element connectivity is still on disk until you access it

Lazy loading is supported for binary ``.vtk`` and ``.ply`` files. ASCII
formats load eagerly (the whole file must be parsed to extract values).

Supported formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 10 15

   * - Format
     - Extension
     - Read
     - Write
     - Lazy load
   * - VTK Legacy
     - ``.vtk``
     - ✓
     - ✓
     - binary only
   * - VTK RectilinearGrid
     - ``.vtr``
     - ✓
     - ✓
     - —
   * - VTK PolyData
     - ``.vtp``
     - ✓
     - ✓
     - —
   * - Wavefront OBJ
     - ``.obj``
     - ✓
     - ✓
     - —
   * - Stanford PLY
     - ``.ply``
     - ✓
     - ✓
     - binary only

Transforms
----------

.. code-block:: python

    from polyxios.transforms import (
        pipeline,
        merge,
        filter_element_type,
        remove_orphan_vertices,
    )

    # Compose transforms into a single function
    clean = pipeline(
        filter_element_type(keep="triangle"),
        remove_orphan_vertices,
    )
    result = clean(mesh)

    # Merge two meshes into one
    combined = merge(mesh_a, mesh_b)

Plugin system
-------------

Any third-party package can register a new format — no fork required.

**Step 1 — write a codec:**

.. code-block:: python

    # mypackage/stl_codec.py
    from polyxios._registry import Codec
    from polyxios._types import PolyData

    def read(path, *, lazy=False) -> PolyData:
        ...

    def write(poly: PolyData, path, **opts) -> None:
        ...

    def register():
        return ".stl", Codec(read, write)

**Step 2 — declare an entry point** in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."polyxios.codecs"]
    stl = "mypackage.stl_codec:register"

After ``pip install mypackage``, polyxios picks up ``.stl`` automatically:

.. code-block:: python

    mesh = px.read("model.stl")   # works out of the box
