.. _format-cgns:

CGNS
====

.. rst-class:: px-badges

``.cgns`` ``read + write`` ``eager`` ``needs h5py``

Summary of the specification
----------------------------

The CFD General Notation System describes a mesh and its solutions as a tree of typed nodes, the *Standard Interface Data Structures*, and stores that tree in HDF5 (or, historically, in ADF). In the HDF5 mapping every node is a group carrying its ``name``, its ``label`` - the SIDS type, ``CGNSBase_t``, ``Zone_t``, ``Elements_t`` - and the ``type`` of its value, ``I4``, ``R8``, ``C1`` or ``MT`` for none; the value itself is a dataset called `` data`` with a leading space, its dimensions the CGNS ones reversed since CGNS counts in Fortran order. A base holds the cell and physical dimensions and any number of zones. An unstructured zone holds its coordinates under ``GridCoordinates`` and its cells in ``Elements_t`` sections, each one element type - ``TETRA_4``, ``HEXA_20``, ``NGON_n`` - or ``MIXED`` with a type code ahead of each cell, over a consecutive ``ElementRange`` of element numbers shared across the zone's sections. A structured zone holds its coordinates on a lattice and its cells are implied. Boundary conditions under ``ZoneBC`` name their entities in a ``PointList`` or ``PointRange`` at a ``GridLocation``; solutions under ``FlowSolution_t`` hold one ``DataArray_t`` per variable at ``Vertex`` or ``CellCenter``.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - HDF5: root `` format`` and `` hdf5version``; ``CGNSLibraryVersion``; ``CGNSBase_t`` → ``Zone_t`` → ``GridCoordinates_t``, ``Elements_t``, ``ZoneBC_t``, ``FlowSolution_t``
   * - node
     - group with ``name``, ``label``, ``type`` (33, 33 and 3 characters) and ``flags``; value in `` data``
   * - zone
     - `` data`` = ``[[vertices], [cells], [boundary vertices]]``; ``ZoneType`` ``Unstructured`` or ``Structured``
   * - sections
     - `` data`` = ``[ElementType, ElementSizeBoundary]``; ``ElementRange``; ``ElementConnectivity`` 1-based; ``ElementStartOffset`` for ``NGON_n``, ``NFACE_n`` and ``MIXED`` since CGNS 4
   * - element types
     - ``NODE BAR_2 BAR_3 BAR_4 TRI_3 TRI_6 QUAD_4 QUAD_8 QUAD_9 TETRA_4 TETRA_10 PYRA_5 PYRA_13 PENTA_6 PENTA_15 PENTA_18 HEXA_8 HEXA_20 HEXA_27 NGON_n`` read; ``PYRA_14``, ``NFACE_n`` and the cubic and quartic kinds skipped
   * - boundaries
     - ``BC_t`` with ``GridLocation`` and ``PointList`` or ``PointRange``, or the ``ElementList`` / ``ElementRange`` CGNS 2 spelled (read only)
   * - solutions
     - ``FlowSolution_t`` with ``GridLocation`` and ``DataArray_t`` children, ``(n, k)`` stored as ``(k, n)``

.. rst-class:: px-speclink

`CGNS Standard Interface Data Structures ↗ <https://cgns.github.io/standard/SIDS/CGNS_docs_current/sids/index.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("wing.cgns")
    mesh.element_tags["wall"]                  # a BC at FaceCenter
    mesh.vertex_attrs["Pressure"]              # a FlowSolution at Vertex
    mesh.global_attrs["zone_name"]

    mesh = px.read("multi.cgns", zone="blk-2")  # one of several zones
    mesh = px.read("multi.cgns", base="Fluid")  # a base other than the first

Bases, zones and sections are found by their ``label``, whatever they are named. Every zone of the base is read, and when there are several they are merged into one mesh, each zone's elements tagged with the zone's name - the zones being the first thing a multi-zone file is opened for; ``zone=`` picks one and ``base=`` picks the base, a name the file does not hold refused listing the ones it does. Sections are read in element-number order and split by type, a ``MIXED`` section by the code ahead of each cell, with or without ``ElementStartOffset``; an ``NGON_n`` section reads as polygons in either the CGNS 4 layout or the older one with a node count ahead of each. A structured zone expands into explicit hexahedra, quadrilaterals or lines the way ``.vts`` does, and its solutions flatten off the lattice in the same order. A boundary condition at ``Vertex`` is a ``vertex_tags`` group and one at any other location an ``element_tags`` group, its numbers turned into mesh indices; a ``PointRange`` is spanned, and on a structured zone a range or list of index triples is spanned over the lattice's vertices or cells. A condition spelled with an ``ElementList`` or ``ElementRange``, as CGNS 2 did, names elements whatever its location says. A solution at ``Vertex`` is a ``vertex_attrs`` entry and one at any other location an ``element_attrs`` one: it spans every element of its location's dimension - a cell-centred solution the volume cells, a face-centred one the faces - or every element when its length says so, or the entities its own ``PointList`` or ``PointRange`` names; several solutions can each fill part of one attribute, and what none covers is NaN. ``DataArray_t`` and ``Descriptor_t`` nodes under a ``UserDefinedData_t`` on the base are ``global_attrs``, as are the base's and the zone's names. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.

The codec needs `h5py <https://www.h5py.org/>`_, which is optional - ``pip install "polyxios[hdf5]"``. Without it :class:`~polyxios.exceptions.UnsupportedFormatError` spells that command.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.cgns")
    px.write(mesh, "out.cgns", base_name="Fluid", zone_name="wing")
    px.write(mesh, "out.cgns", compression="gzip", compression_opts=4)

One base of the mesh's own cell and physical dimensions, one unstructured zone, one section per element type over consecutive element numbers, the types in ascending polyxios code order with the polygons in their place among them as an ``NGON_n`` section with ``ElementStartOffset``. Every node carries the ``name``, ``label``, ``type`` and ``flags`` the CGNS library reads, the root its `` format`` and `` hdf5version``, and a ``CGNSLibraryVersion`` says 4.2. ``vertex_tags`` are ``BC_t`` nodes at ``Vertex`` and ``element_tags`` at ``FaceCenter`` when every member is a face below the cells, ``EdgeCenter`` for an edge below a surface mesh, ``CellCenter`` otherwise; each with a ``PointList``. ``vertex_attrs`` go under a ``FlowSolution_t`` at ``Vertex``; ``element_attrs`` under one at ``CellCenter`` over the cells of the mesh's own dimension - as many as the zone declares, which is what a CGNS reader expects there - and, for the faces or edges below them, under one at ``FaceCenter`` or ``EdgeCenter`` with a ``PointList`` naming them. Numeric ``global_attrs`` are ``DataArray_t`` nodes and text ones ``Descriptor_t`` nodes under a ``UserDefinedData_t`` on the base; ``zone_name`` and ``base_name`` name the zone and the base, the options winning over the globals. A tag, attribute or global name that has to change - a node name is 32 characters, holds no ``/`` and cannot repeat among its siblings - is warned about once, with what it became. ``compression`` and ``compression_opts`` go to h5py for every dataset of more than one value; a file without them is what the CGNS library writes.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Node order follows the SIDS, which lists a quadratic hexahedron's vertical mid-edge nodes before its top ones where VTK lists the top ones first, and a quadratic wedge's likewise; the 27-node hexahedron's face centres are listed by face where VTK lists them by axis. ``HEXA_20``, ``HEXA_27``, ``PENTA_15`` and ``PENTA_18`` are permuted on the way in and out; every other type's order is VTK's already. A pixel and a voxel go out as the quad and hexahedron they are.
- Cells read by ascending polyxios type code, sections merged, whatever order the file's sections come in, so a boundary-face section ahead of the cells does not put the faces first. A ``PYRA_14``, an ``NFACE_n`` section of polyhedra, or a cubic or quartic kind is skipped on the way in with one warning naming it; a type code the SIDS does not define raises :class:`~polyxios.exceptions.UnknownElementTypeError`. An element type CGNS has no code for - a triangle strip, a Lagrange cell - is dropped on the way out with a warning, the values and tags over the elements cut to match.
- Integer ``I4`` and ``I8`` datasets are both read; the writer uses ``I4`` until a count outgrows it, an unsigned value the ``I4`` or ``I8`` it fits, and refuses one above what ``I8`` holds. A CGNS ``(n, k)`` solution array reads as ``(k, n)`` and is transposed back, so a vector stays ``(n, 3)``; a solution whose length fits neither the vertices nor the elements of its location is skipped with a warning. A boolean attribute is written as ``I4``. A node name is 32 characters, so two attributes alike that far, or one named ``GridLocation``, are told apart by a ``_2`` within the width; a zone named like one of its own boundary conditions tags the merged mesh as ``name_2``.
- A physical dimension of 2 pads the coordinates and flags ``global_attrs["was_2d"]``; a mesh so flagged that stayed in the plane writes two coordinates back. A structured zone's boundary at ``Vertex`` or ``CellCenter`` is spanned over the lattice; one at a face-centred location names faces the expanded mesh does not hold and is skipped. A zone declaring more or fewer vertices than its coordinates hold, a section numbered from below one, or two sections claiming one element number are refused naming the culprit; an empty section is walked past.
- The written file spells what ``cgnscheck`` probes first - `` hdf5version``, the node attributes, the typed `` data`` under every node - which the file this codec replaces did not; a BC goes out as ``BCTypeUserDefined``.

.. seealso::

   :doc:`index` - the full format table.
