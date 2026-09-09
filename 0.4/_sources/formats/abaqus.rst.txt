.. _format-abaqus:

Abaqus
======

.. rst-class:: px-badges

``.inp`` ``read + write`` ``eager``

Summary of the specification
----------------------------

An Abaqus input deck is a keyword-driven text file. Lines beginning with ``*`` introduce a keyword with comma-separated parameters; the data lines that follow belong to it until the next keyword. A mesh needs only two: ``*NODE``, listing an id and its coordinates per line, and ``*ELEMENT, TYPE=...``, listing an element id and its node ids. Named collections are declared with ``*NSET`` and ``*ELSET``, and ``**`` starts a comment.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - keywords
     - ``*NODE``, ``*ELEMENT``, ``*NSET``, ``*ELSET``, ``*PART``, ``*INCLUDE``
   * - node line
     - id, x, y, z
   * - element line
     - id, n1, n2, ... (count implied by TYPE)
   * - element types
     - C3D4, C3D8, CPS3, S4R, ... mapped to polyxios element types
   * - comments
     - lines starting with ``**``; keywords are case-insensitive

.. rst-class:: px-speclink

`Read the full Abaqus specification ↗ <https://help.3ds.com/2023/english/dssimulia_established/SIMACAEKEYRefMap/simakey-c-gen.htm>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.inp")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.inp")

``element_type=`` maps a polyxios element name to the Abaqus card written for
it, so a deck can ask for reduced integration where the mesh only says
``hexahedron``:

.. code-block:: python

    px.write(mesh, "out.inp", element_type={"hexahedron": "C3D8R"})

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- A ``*SURFACE`` names a side of an element, which the deck holds no element for. polyxios has no face set of its own - a mesh is vertices and elements - so the face is read as the triangle or quadrilateral it describes, tagged with the surface's name, the way the UGRID, SU2 and Netgen readers hand back their boundary faces. Two element attributes say what it is a face of: ``face_parent``, the element, and ``face_index``, which of that element's faces, both -1 for an element that is not a face. Abaqus numbers a solid's faces in its own order, so ``S1`` of a C3D4 - its base - is the face polyxios numbers last.
- Writing puts the ``*Surface`` back, with the ``*Elset`` of parents it needs marked ``internal`` the way Abaqus marks its own. An internal set a ``*Surface`` names is read for the surface's sake and then dropped, so a round trip does not grow a tag group per face label; one no surface names is a group the deck's author wrote, and is kept. A face whose vertices are no longer its parent's - a transform dropped or reordered the solids - is written as the ordinary element it has become, and a group mixing faces with other elements stays an ``*Elset``, since a deck cannot name one group both ways.
- ``SPOS`` and ``SNEG`` name a side of a shell, which is the element itself, so the surface's name goes on that element rather than on a duplicate of it. A ``*Surface, type=NODE`` becomes a vertex tag, and goes back out as an ``*Nset`` of the same name: a node set and a node surface are the same members under two cards, and only the card the deck used says which.
- A ``*HEADING`` card is the deck's own title, and it is read into ``global_attrs["abaqus_heading"]`` and written back from there. polyxios writes its own banner as a comment, so a deck with no heading of its own does not grow one over a round trip. A ``<FieldData>`` block holds text, so the title reaches a ``.vtu``, ``.vtp``, ``.vti``, ``.vtr`` or ``.vts`` and comes home; a legacy ``.vtk`` ``FIELD`` block spells numbers only, and drops it with a warning.
- Node ids need not be contiguous or sorted; they are remapped to a dense 0-based index. Several ``*NODE`` blocks accumulate, and a repeated id restates that node rather than adding another.
- ``*NSET`` and ``*ELSET`` names become vertex and element tags, whether declared on the block itself or as a standalone card, with or without ``GENERATE``; an entity in several sets stays in all of them. Abaqus matches a set name without regard to case, so a body naming ``TOP`` reaches the set declared as ``Top``, and the same name in two cases is one set.
- Element cards are matched on their base name, so the modifier suffixes - ``R`` reduced integration, ``H`` hybrid, ``I``, ``M``, ``T``, ``P``, and the shell degree-of-freedom numbers - resolve to the same element: ``CPS8R`` reads as ``CPS8``.
- ``*SYSTEM`` transforms every node block that follows it until the next ``*SYSTEM``, which with no data lines restores the global system.
- ``*INCLUDE`` is resolved against the including file's directory. A path that leaves that directory, or nesting deeper than eight files, is refused - an input deck is untrusted input. A deck read from a buffer has no directory, so it cannot use ``*INCLUDE``.
- Every ``*PART`` / ``*INSTANCE`` is merged into one mesh, each read under its own node numbering and tagged by its name. An instance that only places its part carries no nodes of its own and shares the part's numbering.
- A set carrying ``INSTANCE=`` is numbered by that instance rather than by whatever numbering is in force where the card sits, which is what lets an assembly keep its sets outside the instance they name. One naming an instance the deck never defines is reported.
- A ``GENERATE`` range wider than the deck has ids is resolved by walking the ids rather than the range: the card names two numbers and nothing bounds their distance.
- A ``*PART`` sets the deck's own numbering aside rather than replacing it, so a set out past ``*End Part`` still reaches the nodes the deck defined before it.
- On write, a tag member that indexes no node or element of the mesh is dropped and reported: a set naming an id no card defines is a deck Abaqus refuses to load. A float column is refused whole rather than rounded, since rounding an index moves a label onto another entity.
- Analysis keywords (steps, materials, boundary conditions) are skipped rather than treated as errors, and an unrecognised element card is warned about and skipped rather than failing the read.
- A node card spells the third coordinate only in a 3-D model, so a deck where none does is a plane: the vertices are padded with a zero z and ``global_attrs["was_2d"]`` records the fact, which is what writes the cards back out two columns wide. Abaqus takes a node's dimensionality from the element referencing it, so a two-column deck is written under the planar cards - ``CPS3``, ``CPS4``, ``T2D2`` - rather than the ``S3``/``S4`` shells a 3-D deck gets. A mesh holding a type with no planar card, or an ``element_type=`` override naming a 3-D one, stays three-dimensional. A mesh whose vertices have since left the plane is written in three with a warning.

.. seealso::

   :doc:`index` - the full format table.
