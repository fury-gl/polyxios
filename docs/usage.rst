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


Command Line Interface (pxios)
------------------------------

polyxios comes with a command-line interface ``pxios`` to quickly fetch, list, convert, and visualize 3D models.

Available subcommands:

*   ``pxios list``: Lists all available remote assets grouped by package that can be fetched.
*   ``pxios fetch <filename|extension>``: Downloads and caches a model file (e.g., ``bunny.obj``) or an entire extension pack zip (e.g., ``obj`` or ``.obj``).
*   ``pxios convert <input_file> <output_file>``: Converts a model file from one format to another directly in a single process.
*   ``pxios viz [filename]``: Visualizes a local or cached model file using the FURY library.
    *   ``--list``: Lists all locally cached files (can be filtered by ``--ext``).
    *   ``--ext EXT``: Filter cached files or use as extension fallback when no filename is given.
    *   ``--lines``: Render line elements using ``actor.line`` instead of rendering as a point cloud.

Example commands:

.. code-block:: bash

    # List all fetchable remote models
    pxios list

    # Fetch a single model
    pxios fetch bunny.obj

    # Fetch a whole extension folder package zip
    pxios fetch vtk

    # Convert a mesh file
    pxios convert bunny.obj bunny.vtk

    # Visualize a model
    pxios viz bunny.obj


Lazy loading
------------

For large meshes (gigabytes of binary data), pass ``lazy=True``. polyxios
memory-maps the file and only loads the pages you actually touch - the rest
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
     - -
   * - VTK PolyData
     - ``.vtp``
     - ✓
     - ✓
     - -
   * - Wavefront OBJ
     - ``.obj``
     - ✓
     - ✓
     - -
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

Any third-party package can register a new format - no fork required.

**Step 1 - write a codec:**

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

**Step 2 - declare an entry point** in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."polyxios.codecs"]
    stl = "mypackage.stl_codec:register"

After ``pip install mypackage``, polyxios picks up ``.stl`` automatically:

.. code-block:: python

    mesh = px.read("model.stl")   # works out of the box
