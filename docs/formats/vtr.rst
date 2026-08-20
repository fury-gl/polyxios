.. _format-vtr:

VTK RectilinearGrid
===================

.. rst-class:: px-badges

``.vtr`` ``read + write`` ``eager``

Summary of the specification
----------------------------

``.vtr`` is one of the VTK XML serial formats. A ``<VTKFile type="RectilinearGrid">`` root wraps a piece whose extent is declared as six integer indices, and whose geometry is three independent coordinate arrays - one per axis - rather than an explicit point list. Data arrays live in ``<PointData>`` and ``<CellData>`` and may be stored inline as ASCII, inline as base64, or appended as a single raw block referenced by byte offset, optionally zlib-compressed.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - <VTKFile type="RectilinearGrid" version="1.0" byte_order="...">
   * - geometry
     - <Coordinates> with three <DataArray>, one per axis
   * - extent
     - WholeExtent / Extent as six integers: x0 x1 y0 y1 z0 z1
   * - storage
     - format="ascii" | "binary" (base64) | "appended" with offsets
   * - compression
     - optional vtkZLibDataCompressor header

.. rst-class:: px-speclink

`Read the full VTK RectilinearGrid specification ↗ <https://examples.vtk.org/site/VTKFileFormats/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.vtr")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vtr")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The implied point grid is expanded to explicit vertices on read, so a rectilinear file behaves like any other mesh downstream.
- Appended and base64 payloads are decoded eagerly - the XML container has no seekable layout for mmap, so ``lazy=True`` has no effect.
- Multi-component attributes declare and honour ``NumberOfComponents``, so an ``(n, 3)`` vector survives a round trip rather than coming back as ``3n`` rows.
- Binary arrays are written in the type their ``<DataArray>`` declares, so an integer attribute stays an integer one.

.. seealso::

   :doc:`index` - the full format table.
