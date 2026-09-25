.. _format-ply:

Stanford PLY
============

.. rst-class:: px-badges

``.ply`` ``read + write`` ``lazy: binary vertices``

Summary of the specification
----------------------------

PLY - the Stanford Triangle Format - stores a mesh as a list of named elements, typically ``vertex`` and ``face``, each declared in an ASCII header along with its property names and scalar types. The body that follows is either ASCII text or a packed binary block in the byte order the header names. Because the header is self-describing, PLY can carry arbitrary per-vertex and per-face attributes: colours, normals, confidence, intensity.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - magic / header
     - ply, then format ascii | binary_little_endian | binary_big_endian 1.0
   * - structure
     - element <name> <count> declarations, each followed by its property lines
   * - scalar types
     - char uchar short ushort int uint float double, plus list <count-type> <item-type>
   * - faces
     - property list uchar int vertex_indices - 0-based, arbitrary polygon size
   * - comments
     - comment lines anywhere in the header; obj_info for producer metadata
   * - published by
     - Greg Turk, Stanford University

.. rst-class:: px-speclink

`Read the full Stanford PLY specification ↗ <https://paulbourke.net/dataformats/ply/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.ply")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

A binary body's vertex block can be memory-mapped instead of loaded:

.. code-block:: python

    mesh = px.read("big.ply", lazy=True)
    mesh.vertices.flags.writeable   # False: a view of the file, in its own dtype

The vertex block is one run of fixed-width records, so the coordinates are
one ``(n, 3)`` array striding from record to record - when ``x``, ``y`` and
``z`` sit side by side in one floating type, which is how every writer lays
them out - and every other scalar vertex property is a strided column of its
own, all read-only views of the mapping in the dtype the file holds. A file
spelling its coordinates as integers, or laying the three out apart or in
differing types, has them converted to float64 the way an eager read does,
and a vertex element carrying a list property has no fixed record width, so
its whole block is decoded into copies. Faces and edges are decoded either
way: a face list is prefixed by its own count, so nothing on disk is the
flat connectivity.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.ply")

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
     - Write a packed binary body instead of ASCII.
   * - ``endian``
     - ``"little"``
     - Byte order of the binary body; "big" emits binary_big_endian.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Vertex properties beyond x/y/z - colour, normals, confidence, intensity - are preserved as named vertex attributes rather than dropped.
- Lazy loading applies to a binary body's vertex block only; faces are decoded, and an ASCII file must be parsed in full before any value is available.
- Index widths are checked against the declared vertex count, so a mesh too large for the header's list type raises instead of truncating.
- Line elements travel as ``element edge`` with ``vertex1`` / ``vertex2``, the spelling the spec gives them, rather than as a two-vertex face list a reader would take for a degenerate polygon. On read, ``vertex_index1`` / ``vertex_index2`` and a bare pair of integer properties are accepted too, and the edges land after the faces so a per-face attribute keeps lining up with its faces, whatever order the header declares the two elements in. An element block is read in the order the header names it, since that is the order it sits in the file; one this codec has no place for costs its own records and nothing else.
- An ``element edge`` record carries the same element properties a face does, so a value the mesh held on a line survives the trip both ways. A property only one of the two elements declares is NaN over the other, the format spelling no missing value.
- An element index no vertex answers to is refused rather than read into a mesh nothing can draw.
- PLY spells no 64-bit integer, so a column of one is written at the narrowest type that holds the values it actually carries - ``int`` when they fit a signed 32-bit field, ``double`` when they do not. The header and the record are taken from the same decision, so a field's declared width is always the width written.
- A face's vertex count is declared ``uchar``, as almost every PLY file does, and widens to ``ushort`` or ``uint`` for a mesh carrying a polygon of more than 255 vertices - a count the narrower type cannot spell.
- A face record is a flat ring of vertices and PLY spells no other shape, so an element that is not one - a ``tetra``, a ``quadratic_triangle`` - keeps its vertices and loses the type it was: a reader names a record by how many vertices it holds, so it comes back a triangle at three, a quad at four and a polygon otherwise. The elements are still written, and the types they lose are named in a warning rather than dropped quietly.
- An integer attribute is written in full in the ASCII flavour rather than through a float format, which would turn a large one into ``1.23456789e+13``: not a token a reader expecting the declared integer property accepts.

.. seealso::

   :doc:`index` - the full format table.
