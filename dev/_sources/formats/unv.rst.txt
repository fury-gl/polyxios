.. _format-unv:

I-DEAS universal file
=====================

.. rst-class:: px-badges

``.unv`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A universal file is plain ASCII made of numbered datasets. Each opens with a line holding ``-1`` in its first six columns and the dataset number on the next, and closes with another ``-1`` line. Records are fixed-format Fortran lines: integers in ten-column fields, reals in ``D25.16`` or ``E13.5``. The mesh lives in three datasets. 2411 lists each node as a label record (label, two coordinate systems, colour) and a coordinate record. 2412 lists each element as a record naming its label, its FE descriptor - the solver element, such as a thin-shell triangle or a parabolic solid tetrahedron - and its node count, then its node labels eight to a line; a beam carries an orientation record before its nodes. 2467 and 2477 name permanent groups whose members are typed: 7 for a node, 8 for an element. Results are 55 (data at nodes) and 2414 (analysis data at nodes, on elements or at nodes on elements). Labels are free.

The format is I-DEAS's, kept on by Siemens NX and Simcenter, and written by Salome, Gmsh and many test-data tools; the same file is often called UFF.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - datasets between ``-1`` lines, each opening with its number
   * - nodes
     - 2411 (double), 781, 15 (single, one line per node)
   * - elements
     - 2412: label, FE descriptor, property, material, colour, node count
   * - groups
     - 2467 / 2477 / 2435 / 2452, members typed 7 (node) or 8 (element)
   * - results
     - 55 at nodes, 2414 at nodes or on elements, real or complex
   * - indices
     - free labels, positive
   * - second order
     - corner and mid-edge nodes alternate around each face

.. rst-class:: px-speclink

`Universal file datasets (UC-SDRL) ↗ <https://www.ceas3.uc.edu/sdrluff/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.unv")
    mesh.vertex_tags                      # the nodes of each group, by name
    mesh.element_tags                     # the elements of each group, by name
    mesh.vertex_attrs["DISPLACEMENT"]     # a 2414 or 55 dataset at nodes, one row per node
    mesh.vertex_attrs["original_ids"]     # the node labels the file spelled, when not 1..n

Writing
-------

.. code-block:: python

    px.write(mesh, "out.unv")

The file holds a 2411 node dataset in double precision, a 2412 element dataset in the mesh's element order, a 2477 group per tag name (a name past the record's 40 columns is written whole, with a warning, since a reader holding to fixed columns cuts it) and a 2414 dataset per attribute. A surface goes out with the thin-shell descriptors (91, 92, 94, 95), a volume with the solid ones (111, 118, 112, 113, 115, 116), a line as a linear beam (21, 24) and a vertex as a lumped mass (161). There are no options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The FE descriptor is a solver element rather than a geometry, so it is read as its family - beam, surface, tetrahedron, wedge, hexahedron, point - and the node count picks the geometry: a plane-stress, plate, membrane, axisymmetric or shell triangle all read as a triangle, and written again land as a thin shell. A descriptor and node count naming no geometry here - a spring, a rigid element, a cubic shell - is skipped with a warning, and a group naming one of its elements drops that member.
- Second order elements list their nodes around the element, corner and mid-edge alternating, bottom face, vertical edges, then top face; polyxios lists corners first. The reader and writer permute between the two, and Gmsh reads the writer's second order elements back as the ones it was given.
- Nodes and elements are renumbered from zero on the way in, and the file's own labels are kept under ``original_ids`` when they say something the index does not; the writer spells them back when it still can. A label spelled twice - by a kept element and one skipped for its descriptor too - not positive or wider than 64 bits, or an element naming a node the file never declares, is refused. A coordinate may carry a Fortran ``D`` exponent, or a three-digit exponent with no letter (``1.00000-100``).
- Coordinates are read as written, whatever coordinate system a node names; the coordinate system datasets (2420, 18) are not applied.
- A group becomes a vertex tag for the nodes it names and an element tag for the elements, both under its name, so a group of elements alone has no vertex tag. Groups of one name add to one tag, and a name's blanks and NUL padding are stripped; a group with no name is called ``group_<number>``, with ``_2``, ``_3`` added when another group of the file, before or after it, takes that name. The writer joins the vertex and element tags of one name into one group.
- A 2414 dataset is named by its name record, else its first ID line; a 55 by its first ID line. A name already taken gets ``_2``, ``_3``, so the time steps of one result each keep theirs. ``original_ids`` is kept for the labels, so a result named so reads as ``original_ids_2``. A name the writer cannot spell as given - one spanning lines, blank, the ``-1`` delimiter, or ``NONE`` for a result, which reads as no name - is written changed, with a warning, and two tags or two attributes of a kind cleaning to one name stay apart, the second taking ``_2``. An entity no record names reads as NaN, both parts NaN for complex data - the format's way of saying it has no data - and the writer leaves out a row that is all NaN, a complex value counting as NaN only when both its parts are. Complex data reads as complex; integer data as integers unless an entity is missing, then as float, with a warning when a value is past 2\ :sup:`53`, which a float rounds. An integer or boolean attribute is written as integer data and comes back int64, exactly; a value too wide for the ten-column field widens it, which a reader holding to fixed columns misreads. Every other attribute is written as doubles. One value per entity is marked a scalar, three a vector, and any other count of unknown kind, so six values are never shown as a rotation.
- 2414 element data whose value count changes from one element to the next, data at nodes on elements or at points, and datasets 56, 57, 780 and the pre-2435 groups are skipped with a warning. Header, units and coordinate system datasets are passed over silently, as is every dataset number this codec does not know.
- Binary datasets (a number spelled ``2411b``) are refused, as is a dataset with no closing ``-1``, a line outside every dataset and a record holding the wrong number of fields.
- Result values are written with an ``E`` exponent and coordinates with a ``D``, as the format's Fortran readers expect, so readers outside Fortran that know no ``D`` still parse the results. ``global_attrs`` are not written.
- ``lazy=True`` warns and loads eagerly; the format is ASCII.
