.. _format-fluent:

ANSYS Fluent
============

.. rst-class:: px-badges

``.msh`` ``.fluent`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A Fluent mesh is a run of parenthesised sections, each opening with an index that says what it holds: ``(0 "...")`` a comment, ``(1 "...")`` the header, ``(2 3)`` the dimension, ``(10 ...)`` nodes, ``(12 ...)`` cells, ``(13 ...)`` faces and ``(45 ...)`` the type and name of a zone. The index is decimal, as is the zone id a ``(45 ...)`` record names; every field of a section header - zone id, range, type - and every value of a cell or face body is hexadecimal. A node, cell or face section carries a ``(zone-id first-index last-index type ...)`` header and, unless it is the zone-0 declaration of the total, a body: coordinates for a node zone, one type per cell for a mixed cell zone, and for a face zone one record per face naming its nodes and the cell on either side, ``0`` where there is none. A cell has no node list of its own - the mesh is described by its faces - so a reader assembles each cell from the faces that bound it. Adding ``2000`` or ``3000`` to a section index names the single- or double-precision binary flavour of the same section, whose body is raw bytes closed by an ``End of Binary Section`` marker.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - ``(index ...)`` sections; comments, header, dimension, nodes, cells, faces, zones
   * - numbers
     - decimal section indices; hexadecimal header fields, counts and connectivity; decimal coordinates
   * - nodes
     - ``(10 (zone first last type dimension)( x y z ... ))``
   * - cells
     - ``(12 (zone first last type element-type))``; a body lists one type per cell when mixed
   * - faces
     - ``(13 (zone first last bc-type face-type)( n0 n1 ... c0 c1 ))``; count-prefixed when mixed or polygonal
   * - zones
     - ``(45 (id type name)())``, the id in decimal
   * - binary
     - ``2010`` / ``3010``, ``2012`` / ``3012``, ``2013`` / ``3013``, raw little-endian bodies; a double-precision section's integers 32-bit or 64-bit

.. rst-class:: px-speclink

`Fluent mesh file format ↗ <https://www.afs.enea.it/project/neptunius/docs/fluent/html/ug/node1464.htm>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("grid.msh")
    mesh.element_tags                          # one entry per zone: fluid, wall, inlet, ...
    mesh.global_attrs["fluent_zone_types"]     # each tag's zone type word
    mesh.element_attrs["face_parent"]          # the cell each boundary face bounds
    mesh.element_attrs["face_index"]           # which of that cell's faces it is

Writing
-------

.. code-block:: python

    px.write(mesh, "out.msh", fmt="fluent")
    px.write(mesh, "out.fluent")

``.msh`` on its own is written as Gmsh; ``fmt="fluent"`` or the ``.fluent`` spelling names this codec. It takes no format-specific options; any given is warned about and ignored.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- A cell is assembled from its faces as the tetrahedron, hexahedron, wedge or pyramid - in the plane the triangle, quadrilateral or polygon - they close, so its node order is the assembler's rather than anything the file spelled. The cell comes back right-handed whatever the faces' own orientation, since the handedness is read off the coordinates. A cell whose faces close none of those - a polyhedron, a hanging-node refinement, a face list with a hole - is dropped with a warning, and a zone naming only dropped cells leaves no tag.
- Every face of a zone that is not interior - a boundary, a periodic pair, a fan, an interface - comes back as an element of its own; a face born of a non-conformal interface, whose type is spelled with 1000 added, is read by the type under it. It is: a triangle, quadrilateral, polygon or, in the plane, a line, after the cells. It carries ``element_attrs["face_parent"]`` and ``["face_index"]`` naming the cell it bounds and which of that cell's faces it is, the way an Abaqus surface does; in the plane, where an edge has no such numbering, the columns are left out. Interior faces are implied by the cells and are not elements.
- Each zone is an ``element_tags`` entry, named by its ``(45 ...)`` record or ``<type>-<id>`` without one - ``wall-5``, ``fluid-2`` - and the zone type word travels under ``global_attrs["fluent_zone_types"]``. The record spells the zone id in decimal where the section headers spell it in hex; an id that names no zone in decimal is retried as hex, for files from other writers.
- The writer spells the standard face-based ASCII file, its faces outward of each cell - a left-handed cell is mirrored on the way out, since a face record's node order says which side its normal points to: one cell zone per element tag group holding cells, ``fluid`` for the rest, each declaring its element type - or listing one type per cell when the group mixes them - then the faces of every cell, interior faces in one zone with the cell on either side, and the boundary in a zone per group naming the matching face elements, ``wall`` for the rest. The zone type is taken from ``fluent_zone_types`` when the mesh remembers one, and otherwise from a tag named the way Fluent names its zones - ``pressure-outlet-7`` opens with a boundary word. Nodes are numbered as the mesh holds them; cells are renumbered by zone, in order of each zone's first cell, so a mesh whose groups are contiguous keeps its order.
- A face element - a triangle or quadrilateral beside solids, a line beside planar cells - is written only as the side of a cell it matches by node set, under the zone its group names; one matching an interior face is a named interior zone, the way a baffle is. One that is no side of a written cell, or repeats one, is dropped with a warning, since the format holds a face only between cells. An element of any other type - a line among solids, a quadratic element, a vertex - is dropped with a warning too.
- A Fluent zone holds each cell or face once, so an element in two groups stays with the first, with a warning; a group naming both cells and faces becomes two zones, the face zone with a ``-faces`` suffix. A face shared by three cells is refused, since no Fluent file holds one.
- A mesh with a solid is a three-dimensional file. One of triangles, quadrilaterals and polygons is two-dimensional, which Fluent holds in the plane only, so a surface whose vertices carry a third coordinate is refused rather than flattened; a two-dimensional file reads back with a zero third coordinate and ``global_attrs["was_2d"]``.
- Both flavours of the node, cell and face sections are read - ASCII, and binary in single or double precision, count-prefixed face records included, the payload beginning on the line after the opening parenthesis the way Fluent writes it or on the same byte. A double-precision cell or face section holds 32-bit integers when Fluent wrote it and 64-bit ones from a writer that widened them with the floats, and nothing in the header says which, so the width under which the block closes on its marker is the one read. The explicit node lists another reader writes under a typed cell zone are read as those lists. A binary section of another kind, a cell tree or a periodic-shadow list, is stepped over by its end marker; a polyhedral cell zone carrying a body is refused. The writer spells ASCII only.
- Nothing in the format carries per-entity data or node sets, so ``vertex_attrs``, ``element_attrs``, ``vertex_tags`` and the rest of ``global_attrs`` are not written. A zone name that cannot sit on a record - whitespace, parentheses, quotes - has those replaced by ``_``, with a warning.
- A declared count past the file's own size or the safety caps is refused before any array is sized; a face naming a node or cell the file does not hold, two zones claiming one range, a node never given coordinates and a malformed number are each refused with the line named.
- ``.msh`` is shared with Gmsh and resolved by content: a file opening with a parenthesised section reads here, one opening with ``$MeshFormat`` reads as :doc:`gmsh`. ``lazy=True`` warns and loads eagerly.

.. seealso::

   :doc:`index` - the full format table.
