.. _format-vts:

VTK StructuredGrid
==================

.. rst-class:: px-badges

``.vts`` ``read + write`` ``eager``

Summary of the specification
----------------------------

``.vts`` is the XML serial form of a VTK StructuredGrid: a curvilinear grid with an explicit coordinate per node but implicit connectivity. A ``<VTKFile type="StructuredGrid">`` root holds a ``<StructuredGrid>`` whose ``WholeExtent`` gives the index range on each axis as ``i0 i1 j0 j1 k0 k1``. Each ``<Piece>`` restates its own ``Extent`` and carries a ``<Points>`` ``DataArray`` listing every node's coordinates in i-fastest order. There is no connectivity array: the cells are the hexahedra implied by the extent. ``<PointData>`` and ``<CellData>`` hold named attribute arrays, encoded the same way as any other VTK XML file.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - <VTKFile type="StructuredGrid">
   * - extent
     - WholeExtent="i0 i1 j0 j1 k0 k1"
   * - points
     - explicit coordinates, i-fastest ordering
   * - connectivity
     - implicit; cells follow from the extent
   * - encodings
     - ascii, base64, appended (raw or base64), optionally zlib-compressed

.. rst-class:: px-speclink

`Read the full VTK XML specification ↗ <https://examples.vtk.org/site/VTKFileFormats/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("curvilinear.vts")
    mesh.element_types     # hexahedra expanded from the extent

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vts")                 # base64 payloads (default)
    px.write(mesh, "out.vts", binary=False)   # inline ASCII

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: px-spec-table

   * - option
     - meaning
   * - ``binary``
     - ``True`` (the default) writes base64-encoded payloads; ``False`` writes inline ASCII.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- ``<FieldData>`` is the mesh's own metadata rather than any point's or cell's: it is read from the dataset element and from a ``<Piece>`` alike, and written back from ``global_attrs``. The block holds arrays and nothing else, so a scalar written from one comes back as a one-element array, and a value no numeric array holds - a string, a mapping - is dropped with a warning naming the key.
- The ``vts_*`` grid entries are spelled from the grid itself on the way out, so they never travel as field data; every other ``global_attrs`` entry does.
- Multi-component attributes are cut into tuples with ``NumberOfComponents``, so an ``(n, 3)`` vector reads back with its shape rather than as ``3n`` rows.
- The implicit grid is expanded to explicit connectivity on read, so the resulting :class:`~polyxios.PolyData` carries real elements rather than an extent.
- That expansion is what makes a structured file cost the same as an unstructured one in memory; a large extent expands to a large connectivity array.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.
- Header counts are validated against the file size before any array is allocated.
- Attributes are written in the type their array is held in, so an integer identifier keeps every digit rather than being rounded through a double.
- An extent flat along an axis - an image one voxel deep - is a sheet of quads, and one flat along two axes is a run of lines. Only a fully three-dimensional extent expands to hexahedra; reading a flat one as a grid of no cells leaves every ``CellData`` array belonging to nothing.
- The points are written out in the mesh's own order, so they need not be a lattice: a warped block, a cylindrical shell and an aerofoil O-grid are all StructuredGrids, and holding those is what the format is for. The extent is read off the cells, which are a grid whatever the coordinates do.
- The cells are the one thing the file does not carry. A mesh whose elements are not the ones the extent reads back - a mix of types, tetrahedra over grid points, or hexahedra in another order - raises :class:`~polyxios.exceptions.CodecError` at the point of writing rather than being silently swapped for the grid's own cells, taking its ``CellData`` with it. Write a :doc:`vtu`, which carries an arbitrary mesh and its own cells.
- A mesh with no vertices writes the extent VTK spells an empty grid with, ``0 -1 0 -1 0 -1``, and reads back as a mesh with none.
- An extent that ends before it starts on any axis holds no points, so it holds no cells either - ``0 -1 0 2 0 2`` reads as an empty mesh rather than as four quads whose corners name points the file never spelled.
- A StructuredGrid carries its points, so the extent and the ``<Points>`` array have to agree on how many. The column count is read off the two, so a file that disagrees used to come back as a mesh of the wrong width - a point of no coordinates at all - and fail later on a shape nothing in the file explained; it now raises :class:`~polyxios.exceptions.CodecError` naming both counts. An array wider than three components keeps its first three, as before.
- The extent of the file a mesh was read from travels with it, so a grid that did not begin at zero is written back where it stood - but only while it still describes the mesh. One a transform has moved the mesh out from under is re-derived from the cells rather than trusted.

.. seealso::

   :doc:`index` - the full format table.
   :doc:`vtr` - the axis-aligned rectilinear sibling.
