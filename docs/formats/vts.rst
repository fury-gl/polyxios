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

- Multi-component attributes are cut into tuples with ``NumberOfComponents``, so an ``(n, 3)`` vector reads back with its shape rather than as ``3n`` rows.
- The implicit grid is expanded to explicit hexahedral connectivity on read, so the resulting :class:`~polyxios.PolyData` carries real elements rather than an extent.
- That expansion is what makes a structured file cost the same as an unstructured one in memory; a large extent expands to a large connectivity array.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.
- Header counts are validated against the file size before any array is allocated.

.. seealso::

   :doc:`index` - the full format table.
   :doc:`vtr` - the axis-aligned rectilinear sibling.
