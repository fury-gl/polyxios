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
     - POINT_DATA / CELL_DATA with SCALARS, VECTORS, NORMALS, FIELD

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

.. seealso::

   :doc:`index` - the full format table.
