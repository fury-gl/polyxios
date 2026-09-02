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

- A tag group has no set of its own in this format, so it travels as one ``PointData`` or ``CellData`` column of ones and zeros named ``polyxios_tag_<group>``. An entity in two groups is named by both columns, which a format spelling one reference per entity cannot say. On the way in, a column with that name holding whole numbers is read back as the group; one holding anything else stays an attribute, since a member rounded into place names the wrong entity.
- ``<FieldData>`` is the mesh's own metadata rather than any point's or cell's: it is read from the dataset element and from a ``<Piece>`` alike, and written back from ``global_attrs``. The block holds arrays and nothing else, so a scalar written from one comes back as a one-element array, and a value no numeric array holds - a string, a mapping - is dropped with a warning naming the key.
- The ``vti_*`` grid entries are spelled from the grid itself on the way out, so they never travel as field data; every other ``global_attrs`` entry does.
- Multi-component attributes are cut into tuples with ``NumberOfComponents``, so an ``(n, 3)`` vector reads back with its shape rather than as ``3n`` rows.
- Both the coordinates and the connectivity are materialised on read. A file of a few hundred bytes can expand to a large in-memory mesh, because the extent is all it takes to describe one.
- ``Origin`` defaults to ``0 0 0`` and ``Spacing`` to ``1 1 1`` when the attributes are absent.
- A ``<Piece>`` may restate a sub-extent of ``WholeExtent``; the piece's own extent is what gets expanded.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.
- Attributes are written in the type their array is held in, so an integer identifier keeps every digit rather than being rounded through a double.
- An extent flat along an axis - an image one voxel deep - is a sheet of quads, and one flat along two axes is a run of lines. Only a fully three-dimensional extent expands to hexahedra; reading a flat one as a grid of no cells leaves every ``CellData`` array belonging to nothing.
- The writer holds a uniform grid and nothing else, and says so rather than spelling a file its own reader refuses. The extent is read off the cells, which are a grid whatever the coordinates do; the vertices then have to be that grid expanded, in the order the file reads them back - x fastest, z slowest - since the file holds no points of its own, and every axis has to be evenly spaced, since it holds one step per axis rather than a coordinate per plane. A mesh that is none of those raises :class:`~polyxios.exceptions.CodecError`. Write a :doc:`vtr` for a lattice whose planes are unevenly spaced, a :doc:`vts` for points that are not a lattice at all, or a :doc:`vtu` for an arbitrary mesh.
- Cells that are not the grid's own - tetrahedra over grid points, or the same hexahedra in another order - raise as well. The file carries no connectivity, so they would be dropped on the way back in and their ``CellData`` with them.
- A mesh with no vertices writes the extent VTK spells an empty grid with, ``0 -1 0 -1 0 -1``, and reads back as a mesh with none.
- An extent that ends before it starts on any axis holds no points, so it holds no cells either - ``0 -1 0 2 0 2`` reads as an empty mesh rather than as four quads whose corners name points the file never spelled.
- The extent, origin and step of the file a mesh was read from travel with it, so a grid that did not begin at zero - or that steps down an axis - goes back exactly where it stood. All three are checked against the vertices at the point of writing rather than trusted: pruning cells leaves the extent counting points that are not in the file, moving the mesh leaves the origin behind, and scaling it leaves the step behind. Whichever no longer describes the mesh is re-derived from it.

.. seealso::

   :doc:`index` - the full format table.
   :doc:`vts` - the curvilinear structured sibling.
