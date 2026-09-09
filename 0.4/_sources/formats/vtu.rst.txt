.. _format-vtu:

VTK UnstructuredGrid
====================

.. rst-class:: px-badges

``.vtu`` ``read + write`` ``eager``

Summary of the specification
----------------------------

``.vtu`` is the XML serial form of a VTK UnstructuredGrid: an arbitrary mix of cell types in one dataset. A ``<VTKFile type="UnstructuredGrid">`` root holds an ``<UnstructuredGrid>`` with one or more ``<Piece>`` elements, each declaring ``NumberOfPoints`` and ``NumberOfCells``. A piece carries ``<Points>`` with a three-component coordinate ``DataArray``, and ``<Cells>`` with three named arrays - ``connectivity``, ``offsets`` and ``types`` - where ``types`` holds one VTK cell type code per cell. ``<PointData>`` and ``<CellData>`` hold named attribute arrays. Every ``DataArray`` is inline ASCII, inline base64, or a reference into a single appended binary blob, optionally zlib-compressed.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - <VTKFile type="UnstructuredGrid">
   * - pieces
     - one or more <Piece NumberOfPoints= NumberOfCells=>
   * - cells
     - connectivity, offsets and types DataArrays
   * - cell types
     - one VTK type code per cell
   * - encodings
     - ascii, base64, appended (raw or base64), optionally zlib-compressed
   * - indices
     - 0-based

.. rst-class:: px-speclink

`Read the full VTK XML specification ↗ <https://examples.vtk.org/site/VTKFileFormats/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("grid.vtu")
    mesh.element_types     # mixed cell types, mapped from the VTK codes

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vtu")                 # base64 payloads (default)
    px.write(mesh, "out.vtu", binary=False)   # inline ASCII

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: px-spec-table

   * - option
     - meaning
   * - ``binary``
     - ``True`` (the default) writes base64-encoded payloads; ``False`` writes inline ASCII, which is larger but diffable.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- A tag group has no set of its own in this format, so it travels as one ``PointData`` or ``CellData`` column of ones and zeros named ``polyxios_tag_<group>``. An entity in two groups is named by both columns, which a format spelling one reference per entity cannot say. On the way in, a column with that name holding whole numbers is read back as the group; one holding anything else stays an attribute, since a member rounded into place names the wrong entity.
- ``<FieldData>`` is the mesh's own metadata rather than any point's or cell's: it is read from the dataset element and from a ``<Piece>`` alike, and written back from ``global_attrs``. A key both levels spell is the dataset's, which is the file's own answer for the mesh where a piece's is one piece's. The block holds arrays and nothing else, so a scalar written from one comes back as a one-element array, and every axis past the first is a component.
- ``<FieldData>`` holds a ``String`` array beside its numeric ones, so a ``global_attrs`` value that is text - a name, a title, a solver's own label - is written as one and comes back the string it was; a list of strings is one array of several tuples. A value that is neither numbers nor text - a mapping, a ragged list - is dropped with a warning naming the key.
- Multiple ``<Piece>`` elements are concatenated into one :class:`~polyxios.PolyData`, with each piece's connectivity shifted by the running vertex count.
- A piece that declares points and does not deliver them raises :class:`~polyxios.exceptions.CodecError`; its cells would index points that are not there, and every later piece would be shifted by the count that never arrived.
- A point or cell array carried by only some of the pieces is dropped with a warning: joined short, its rows would sit against the wrong points from the second piece on.
- A ``Points`` array of a type that holds no numbers - ``type="String"``, or any type this reader does not know - raises :class:`~polyxios.exceptions.CodecError` naming the type.
- VTK cell type codes with no polyxios equivalent are dropped rather than guessed at.
- Attributes are written in the type their array is held in, so an integer identifier keeps every digit rather than being rounded through a double.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`; the payload may be compressed or base64-encoded, neither of which can be memory-mapped.
- Header counts are validated against the file size before any array is allocated.

.. seealso::

   :doc:`index` - the full format table.
   :doc:`vtp` - the XML PolyData sibling.
