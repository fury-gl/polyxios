.. _format-vti:

VTK ImageData
=============

.. rst-class:: px-badges

``.vti`` ``read + write`` ``eager``

Summary of the specification
----------------------------

``.vti`` is the XML serial form of a VTK ImageData: a uniform grid stored entirely as metadata. A ``<VTKFile type="ImageData">`` root holds an ``<ImageData>`` carrying ``WholeExtent``, ``Origin`` and ``Spacing`` - the index range on each axis, the coordinate of the first node, and the constant step between nodes. There are no coordinates in the file at all; every node position follows from ``origin + index * spacing``, and the cells are the hexahedra implied by the extent. Only ``<PointData>`` and ``<CellData>`` arrays carry actual payload, encoded like any other VTK XML file.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - <VTKFile type="ImageData">
   * - extent
     - WholeExtent="i0 i1 j0 j1 k0 k1"
   * - geometry
     - Origin + Spacing; no coordinate array
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

    mesh = px.read("volume.vti")
    mesh.vertices          # materialised from origin + index * spacing

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vti")                 # base64 payloads (default)
    px.write(mesh, "out.vti", binary=False)   # inline ASCII

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
- Both the coordinates and the connectivity are materialised on read. A file of a few hundred bytes can expand to a large in-memory mesh, because the extent is all it takes to describe one.
- ``Origin`` defaults to ``0 0 0`` and ``Spacing`` to ``1 1 1`` when the attributes are absent.
- A ``<Piece>`` may restate a sub-extent of ``WholeExtent``; the piece's own extent is what gets expanded.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.
- Attributes are written in the type their array is held in, so an integer identifier keeps every digit rather than being rounded through a double.
- An extent flat along an axis - an image one voxel deep - is a sheet of quads, and one flat along two axes is a run of lines. Only a fully three-dimensional extent expands to hexahedra; reading a flat one as a grid of no cells leaves every ``CellData`` array belonging to nothing.

.. seealso::

   :doc:`index` - the full format table.
   :doc:`vts` - the curvilinear structured sibling.
