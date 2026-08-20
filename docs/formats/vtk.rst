.. _format-vtk:

VTK Legacy
==========

.. rst-class:: px-badges

``.vtk`` ``read + write`` ``lazy: binary only``

Summary of the specification
----------------------------

The legacy VTK format is a single-dataset serial file with a five-line ASCII preamble - version banner, title, ``ASCII`` or ``BINARY`` data mode, and a ``DATASET`` keyword naming the geometry type. What follows depends on that type: an unstructured grid lists ``POINTS``, then ``CELLS`` as connectivity lists prefixed by their point count, then a parallel ``CELL_TYPES`` array of integer type codes. Point and cell attributes arrive afterwards in ``POINT_DATA`` / ``CELL_DATA`` sections as named ``SCALARS``, ``VECTORS`` or ``FIELD`` arrays. In ``BINARY`` mode the arrays are raw big-endian values packed straight after their declaration line.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - banner
     - # vtk DataFile Version x.y
   * - data mode
     - ASCII or BINARY (binary payload is big-endian)
   * - dataset types
     - STRUCTURED_POINTS, STRUCTURED_GRID, RECTILINEAR_GRID, POLYDATA, UNSTRUCTURED_GRID
   * - connectivity
     - CELLS <n> <size> followed by CELL_TYPES <n> integer codes
   * - attributes
     - POINT_DATA / CELL_DATA with SCALARS, COLOR_SCALARS, VECTORS, NORMALS, TENSORS, TEXTURE_COORDINATES, FIELD

.. rst-class:: px-speclink

`Read the full VTK Legacy specification ↗ <https://examples.vtk.org/site/VTKFileFormats/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.vtk")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Binary bodies can be memory-mapped instead of loaded:

.. code-block:: python

    mesh = px.read("big.vtk", lazy=True)

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vtk")

Format-specific options:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``binary``
     - ``False``
     - Write a BINARY body instead of ASCII.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Binary files can be memory-mapped with ``lazy=True``; ASCII files must be parsed end to end before any value is available.
- Cell type codes are mapped to polyxios element types, so a file mixing triangles, quads and tetrahedra keeps every group separate.
- Point and cell data arrays are carried through as named vertex and element attributes rather than being dropped on read.
- ``SCALARS``, ``VECTORS``, ``NORMALS``, ``TENSORS``, ``COLOR_SCALARS``, ``TEXTURE_COORDINATES`` and ``FIELD`` sections are all read, in every dataset type - unstructured, polydata, structured points, structured grid and rectilinear grid alike. A ``LOOKUP_TABLE`` definition is a palette rather than a value per point, so it becomes no attribute, but it is counted past so the arrays after it are still found. A keyword outside that set stops the scan, and says so. ``COLOR_SCALARS`` is the one attribute whose type its own line does not name: one unsigned char per component in a binary file, a float in 0..1 in an ASCII one. The byte is scaled onto 0..1, so the same colour reads back the same from either flavour.
- An attribute section that declares more values than the file holds raises ``CodecError`` naming the array, rather than an ``IndexError`` naming nothing in ASCII or a reshape failure naming nothing in binary.
- ``STRUCTURED_POINTS`` keeps ``DIMENSIONS``, ``ORIGIN`` and ``SPACING`` in ``global_attrs`` (``vtk_dimensions``, ``vtk_origin``, ``vtk_spacing``); ``STRUCTURED_GRID`` and ``RECTILINEAR_GRID`` keep ``DIMENSIONS``. The points are expanded into an explicit array, so without those the grid behind them would be lost.
- Those ``vtk_*`` entries are read-only: ``write`` always emits an ``UNSTRUCTURED_GRID`` and does not consume them.
- ``CELL_DATA`` is read from the structured datasets as well as the unstructured ones. An array whose declared length matches neither the points nor the cells of the grid the header describes is dropped with a warning naming it, rather than reaching ``PolyData`` as a validation error about lengths.
- A structured grid extends along whichever axes its ``DIMENSIONS`` declare: ``3 1 3`` is a sheet of quads in the x-z plane, not a run of lines, and a column along ``y`` or ``z`` is indexed with its own stride.
- VTK 5.1 cells - the default since VTK 9.0 - are read wherever they appear: ``CELLS`` in an unstructured grid and ``POLYGONS``, ``LINES``, ``VERTICES`` or ``TRIANGLE_STRIPS`` in polydata. The two numbers on such a line are the length of the ``OFFSETS`` array and the length of ``CONNECTIVITY``, so the mesh holds one cell fewer than the first of them; the offsets are counted up to the ``CONNECTIVITY`` keyword, so a file spelling that line either way is read. ``write(..., vtk_version="5.1")`` declares the offsets length, which is what VTK's own reader expects.
- A ``METADATA`` block - component names and information keys, written after every array by VTK 4.2 and later - is stepped over rather than read as an array. It is text even in a binary file, and it appears between the entries of a ``FIELD`` block as well as after a section.
- A ``RECTILINEAR_GRID`` takes its grid from its coordinate arrays: the points are their outer product, so a ``DIMENSIONS`` header that disagrees with them is warned about and ignored.

.. seealso::

   :doc:`index` - the full format table.
