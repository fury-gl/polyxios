.. _format-vtp:

VTK PolyData
============

.. rst-class:: px-badges

``.vtp`` ``read + write`` ``lazy: raw appended``

Summary of the specification
----------------------------

``.vtp`` is the VTK XML format for surface and curve geometry. A ``<Piece>`` declares its point and cell counts as attributes, then carries ``<Points>`` plus up to four cell containers - ``<Verts>``, ``<Lines>``, ``<Strips>`` and ``<Polys>`` - each expressed as a ``connectivity`` array and an ``offsets`` array rather than the legacy size-prefixed lists. Attributes travel in ``<PointData>`` and ``<CellData>`` with the same inline, base64 or appended storage choices as the other XML formats.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - <VTKFile type="PolyData">
   * - cell containers
     - Verts, Lines, Strips, Polys
   * - connectivity
     - paired connectivity + offsets DataArrays (0-based)
   * - storage
     - ascii, base64 binary, or appended raw block
   * - attributes
     - PointData / CellData, one named DataArray each

.. rst-class:: px-speclink

`Read the full VTK PolyData specification ↗ <https://examples.vtk.org/site/VTKFileFormats/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.vtp")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vtp")                 # base64 payloads (default)
    px.write(mesh, "out.vtp", binary=False)   # inline ASCII
    px.write(mesh, "out.vtp", appended=True)  # raw appended section, mappable

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: px-spec-table

   * - option
     - meaning
   * - ``binary``
     - ``True`` (the default) writes base64-encoded payloads; ``False`` writes inline ASCII, which is larger but diffable.
   * - ``appended``
     - ``True`` writes every array as one raw ``<AppendedData encoding="raw">`` section after the XML, the layout VTK itself writes by default: a third smaller than base64, and the one a lazy read can map. Field data stays inline.

Lazy reading
------------

A file whose arrays sit in a raw, uncompressed appended section can be
mapped instead of loaded:

.. code-block:: python

    mesh = px.read("model.vtp", lazy=True)
    mesh.vertices.flags.writeable   # False: the array is the file's own bytes

The vertices, connectivity and every point and cell array are read-only
views of the mapping, in the dtype and byte order the file holds. The offsets
and element types are derived from the file rather than stored in it, so
those two are built in memory, as is a connectivity the file declares as
floats: an index is a whole number, so it is cast to integers the way an
eager read casts it. A file of several pieces, or a piece holding
more than one of ``Verts``, ``Lines``, ``Strips`` and ``Polys``, is joined by
copying. A file that keeps its arrays inline, base64-encoded or
zlib-compressed has no bytes on disk in the shape an array needs, and
``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError` naming
which.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- A tag group has no set of its own in this format, so it travels as one ``PointData`` or ``CellData`` column of ones and zeros named ``polyxios_tag_<group>``. An entity in two groups is named by both columns, which a format spelling one reference per entity cannot say. On the way in, a column with that name holding whole numbers is read back as the group; one holding anything else stays an attribute, since a member rounded into place names the wrong entity.
- ``<FieldData>`` is the mesh's own metadata rather than any point's or cell's: it is read from the dataset element and from a ``<Piece>`` alike, and written back from ``global_attrs``. A key both levels spell is the dataset's, which is the file's own answer for the mesh where a piece's is one piece's. The block holds arrays and nothing else, so a scalar written from one comes back as a one-element array, and every axis past the first is a component.
- ``<FieldData>`` holds a ``String`` array beside its numeric ones, so a ``global_attrs`` value that is text - a name, a title, a solver's own label - is written as one and comes back the string it was; a list of strings is one array of several tuples. A value that is neither numbers nor text - a mapping, a ragged list - is dropped with a warning naming the key.
- Triangle strips are expanded into individual triangles on read; writing emits polygons rather than re-striping.
- Each cell container becomes its own element group, so lines and polygons in one file stay distinguishable.
- A piece that declares points and does not deliver them raises :class:`~polyxios.exceptions.CodecError`; its cells would index points that are not there, and every later piece would be shifted by the count that never arrived.
- A point or cell array carried by only some of the pieces is dropped with a warning: joined short, its rows would sit against the wrong points from the second piece on.
- A ``Points`` array of a type that holds no numbers - ``type="String"``, or any type this reader does not know - raises :class:`~polyxios.exceptions.CodecError` naming the type.
- Attributes are written in the type their array is held in, so an integer identifier keeps every digit rather than being rounded through a double.
- Cell offsets that run backwards, or reach past the end of a section's connectivity, raise :class:`~polyxios.exceptions.CodecError` naming the piece and section; a file like that describes no cells, and used to come back with some of them silently missing.

.. seealso::

   :doc:`index` - the full format table.
