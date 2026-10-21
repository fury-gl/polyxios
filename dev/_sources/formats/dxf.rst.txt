.. _format-dxf:

DXF
===

.. rst-class:: px-badges

``.dxf`` ``read + write`` ``eager``

Summary of the specification
----------------------------

DXF - Autodesk's Drawing Exchange Format - is how AutoCAD drawings travel to every other CAD, CAM and GIS tool. A file is a flat run of group code / value pairs, a code saying what the value is: ``0`` starts an entity, ``8`` names its layer, ``10``/``20``/``30`` are its first point. The pairs are split into sections: ``HEADER`` variables, ``TABLES`` of layers and styles, ``BLOCKS`` of named entity groups, and the ``ENTITIES`` that draw the model. An ``INSERT`` places a block by position, scale, rotation and extrusion, optionally as a grid of copies. The mesh geometry lives in a few entities - ``3DFACE``, polyface and polygon-mesh ``POLYLINE``, the subdivision ``MESH`` of AutoCAD 2010 - next to lines, polylines and points. Planar entities are spelled in an object coordinate system derived from their extrusion direction. The same pairs can be stored as text or in a binary flavour.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - layout
     - a group code line, then its value line; ``SECTION`` ... ``ENDSEC``, closed by ``EOF``
   * - binary
     - the sentinel ``AutoCAD Binary DXF\r\n\x1a\0``, one-byte codes in R12, two-byte from R13, values typed by code
   * - faces
     - ``3DFACE`` (3 or 4 corners), ``SOLID``/``TRACE`` (corners 1, 2, 4, 3, in OCS)
   * - meshes
     - ``POLYLINE`` flag 64 (polyface: coordinate vertices, then faces of up to four 1-based indices), flag 16 (M x N polygon mesh), ``MESH``
   * - curves
     - ``LINE``, ``POINT``, 3D and 2D ``POLYLINE``, ``LWPOLYLINE``; a bulge (``42``) bends a segment into an arc
   * - blocks
     - ``BLOCK`` base point + entities + ``ENDBLK``; ``INSERT`` with ``41-43`` scale, ``50`` rotation, ``70/71`` columns and rows, ``44/45`` spacing, ``210-230`` extrusion
   * - units
     - ``$INSUNITS`` in the header, from R2000

.. rst-class:: px-speclink

`DXF reference ↗ <https://help.autodesk.com/view/OARX/2024/ENU/?guid=GUID-235B22E0-A567-4CF6-92D3-38A2306D73F3>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("plant.dxf")               # ASCII or binary, any version
    mesh.element_tags                         # one group per layer
    mesh.global_attrs                         # {"dxf_version": "AC1032", "unit": "millimeter"}

The model-space entities that are mesh geometry are read in file order: ``3DFACE``, ``SOLID`` and ``TRACE`` as triangles and quads (a fourth corner equal to the third makes a triangle), polyface meshes as their faces - three or four indices, a two-index face as a line - polygon meshes as quads, wrapping where they are closed, ``MESH`` as triangles, quads and polygons, 3D and 2D ``POLYLINE`` and ``LWPOLYLINE`` as poly lines (a closed one repeats its first vertex, and one of two vertices is a line), ``LINE`` as lines and ``POINT`` as vertices. Planar entities are taken out of their object coordinate system by the arbitrary axis algorithm. Every ``INSERT`` adds its block's geometry, nested inserts included, under its base point, scale, rotation, column and row array and extrusion, with the faces of a mirroring insert turned back to face outward.

DXF entities share no vertices, so vertices at equal coordinates are merged into one, in the order they first appear. Every element is in the element tag group of its layer, in the order the layers first come; an entity on layer ``0`` inside a block takes the layer of the insert placing it. Layer names are compared without case, as DXF compares every symbol name, so ``Walls`` and ``WALLS`` are one group, named as the layer is first spelled. ``$ACADVER`` lands in ``global_attrs["dxf_version"]`` and ``$INSUNITS`` in ``global_attrs["unit"]`` under the names 3MF and AMF use. Text is decoded by ``$DWGCODEPAGE`` before R2007 and as UTF-8 after, and ``\U+XXXX`` escapes are resolved.

Entities that are not mesh geometry - circles, arcs, text, hatches, dimensions - and paper-space entities are passed over, the former with one warning naming them by count, as is a ``3DFACE``, ``SOLID`` or ``TRACE`` missing one of its first three corners; a bulge is read as a straight segment with a warning, and a block defined twice keeps its first definition with a warning. A group code that is not an integer, a number that is not one, a file ending inside a section or a pair, a block with no ``ENDBLK``, a polyface face naming a vertex its mesh lacks, a polygon mesh or ``MESH`` not holding what it declares, an insert naming a missing block or one that inserts itself, an insert placing a vertex past the float range, and inserts expanding past what the file's size can describe - each copy counted as its vertices and connectivity entries and 16 more, so a block copied millions of times is refused however small it is - each raise :class:`~polyxios.exceptions.CodecError`. What follows the first ``0``/``EOF`` pair, such as a DOS end-of-file byte or NUL padding, is ignored, as in a binary file. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.dxf")

The writer spells AutoCAD R12 (``AC1009``), the version every DXF reader opens. Triangles, quads, pixels and polygons of up to four corners are faces of polyface meshes, consecutive faces on one layer sharing one mesh of at most 8191 faces, so neither of R12's 16-bit counts overflows; a larger polygon or a triangle strip is split into triangles, a quadratic element keeps its corners. Lines are ``LINE``, poly lines 3D ``POLYLINE`` - closed when their last vertex is their first, and one of two vertices a ``LINE``, as it reads back - and vertices ``POINT``; volume elements have no entity and are dropped with a warning, as are a polygon, strip or poly line with too few vertices to draw; a mesh of nothing else is refused, as is a vertex that is not finite. Each element tag group is a layer, declared in the ``LAYER`` table; an element in two groups is on the first, groups whose names differ only in case share the first one's layer, with a warning, and the untagged elements are on layer ``0``. A layer name outside ASCII is written as ``\U+XXXX`` escapes; one holding a control character is refused. The characters no DXF version allows in a layer name - ``< > / \ " : ; ? * =`` and the backquote - become ``_``, and white space around a name, which every reader strips, is dropped, and a name left empty is ``_``, all with a warning. ``global_attrs["unit"]`` is written as ``$INSUNITS`` when DXF has a code for it; vertex and element attributes have no place in R12 and are not written, nor are the vertices no element uses. Coordinates are written as the shortest decimal that reads back to the same double.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- **Vertex order follows the faces.** Each polyface mesh lists the vertices its faces use, so a round trip returns the vertices in the order the faces first reach them.
- **Equal coordinates are one vertex.** Two distinct vertices at one point come back as one; DXF has no way to tell them apart.
- **Untagged elements come back on layer 0.** A mesh written with no tags reads back with one group, ``0``.
- **Inserts are applied, not kept.** A block placed a hundred times reads as a hundred copies; the blocks are not recoverable from what comes back.
- **No attributes.** DXF has no per-vertex or per-element data; colours, line types and thicknesses are not read.
- **$INSUNITS in R12.** The variable dates from R2000 and ezdxf leaves it out of the R12 files it writes, but reads it back from an R12 header with nothing for its auditor to report; the writer keeps it so the unit round-trips through DXF.
- **Layer names are not made R12-strict.** AutoCAD up to R14 limited names to 31 upper-case characters of ``A-Z 0-9 $ _ -``. The writer keeps longer names, lower case and ``\U+XXXX`` escapes all the same: readers since R13 take them, ezdxf reads them from R12 and writes R12 that way itself, and renaming every layer would lose the tag names a round trip keeps. A file meant for an R12-era Autodesk product needs its names translated first - ezdxf's ``r12strict`` does it.
- **Layer names ignore case.** ``Walls`` and ``WALLS`` are one layer on read and on write.

.. seealso::

   :doc:`index` - the full format table.
