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

- A tag group has no set of its own in this format, so it travels as one ``PointData`` or ``CellData`` column of ones and zeros named ``polyxios_tag_<group>``. An entity in two groups is named by both columns, which a format spelling one reference per entity cannot say. On the way in, a column with that name holding whole numbers is read back as the group; one holding anything else stays an attribute, since a member rounded into place names the wrong entity.
- ``<FieldData>`` is the mesh's own metadata rather than any point's or cell's: it is read from the dataset element and from a ``<Piece>`` alike, and written back from ``global_attrs``. A key both levels spell is the dataset's, which is the file's own answer for the mesh where a piece's is one piece's. The block holds arrays and nothing else, so a scalar written from one comes back as a one-element array, and every axis past the first is a component.
- ``<FieldData>`` holds a ``String`` array beside its numeric ones, so a ``global_attrs`` value that is text - a name, a title, a solver's own label - is written as one and comes back the string it was; a list of strings is one array of several tuples. A value that is neither numbers nor text - a mapping, a ragged list - is dropped with a warning naming the key.
- The ``vtr_*`` grid entries are spelled from the grid itself on the way out, so they never travel as field data; every other ``global_attrs`` entry does.
- The implied point grid is expanded to explicit vertices on read, so a rectilinear file behaves like any other mesh downstream.
- Appended and base64 payloads are decoded eagerly, and ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError` rather than pretending otherwise - the XML container has no seekable layout for mmap.
- Multi-component attributes declare and honour ``NumberOfComponents``, so an ``(n, 3)`` vector survives a round trip rather than coming back as ``3n`` rows.
- Binary arrays are written in the type their ``<DataArray>`` declares, so an integer attribute stays an integer one.
- An extent flat along an axis - an image one voxel deep - is a sheet of quads, and one flat along two axes is a run of lines. Only a fully three-dimensional extent expands to hexahedra; reading a flat one as a grid of no cells leaves every ``CellData`` array belonging to nothing.
- The writer holds a lattice and nothing else, and says so rather than spelling a file its own reader refuses. The extent is read off the cells, and the three coordinate arrays are then read off the vertices a stride at a time - so an axis that runs downwards keeps its direction rather than being sorted round under its own point data. A mesh whose vertices are not that lattice expanded, in the order the file reads them back, raises :class:`~polyxios.exceptions.CodecError`; so do cells that are not the grid's own, which the file carries no connectivity for and would drop with their ``CellData``. Write a :doc:`vts` for points that are not a lattice, or a :doc:`vtu` for an arbitrary mesh.
- A mesh with no vertices writes the extent VTK spells an empty grid with, ``0 -1 0 -1 0 -1``, and reads back as a mesh with none.
- An extent that ends before it starts on any axis holds no points, so it holds no cells either - ``0 -1 0 2 0 2`` reads as an empty mesh rather than as four quads whose corners name points the file never spelled.
- The extent counts the planes and the coordinate arrays spell them, and nothing in the file makes the two agree. An array of a different length raises :class:`~polyxios.exceptions.CodecError` naming its axis: the cells, the offsets and every ``PointData`` array are sized off the extent, so a longer one used to expand into a mesh whose connectivity covered part of itself and whose attributes covered none of it. Every axis is asked, including one the extent gives no plane at all - the point count is a product and goes to zero on such an axis, but the vertices are built from the coordinates, so a file spelling one anyway came back as a mesh the extent called empty. An axis the file leaves no array for at all takes its one plane at zero, which is how a 2-D grid writes its z.
- The extent of the file a mesh was read from travels with it, so a block that did not begin at zero is written back at the indices it stood on rather than slid to the origin - which is what a ``.pvtr`` assembling it next to its neighbours reads. It is checked against the mesh at the point of writing rather than trusted: one a transform has moved the mesh out from under is re-derived from the cells.

.. seealso::

   :doc:`index` - the full format table.
