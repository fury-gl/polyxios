.. _format-exodus:

Exodus II
=========

.. rst-class:: px-badges

``.e`` ``.exo`` ``.ex2`` ``read + write`` ``eager`` ``needs netCDF4``

Summary of the specification
----------------------------

Exodus II is Sandia's finite element database: a netCDF file whose dimensions, variables and attributes carry the names the Exodus library fixes. ``num_nodes`` and ``coord`` (``num_dim`` rows of coordinates, or ``coordx`` / ``coordy`` / ``coordz``) hold the geometry; the elements come in blocks, one ``connect<i>`` table per block with an ``elem_type`` attribute naming the shape - ``HEX8``, ``TETRA10``, ``SHELL4``, ``BAR2`` - and a 1-based node list per element; ``node_ns<i>`` lists the node sets, ``elem_ss<i>`` with ``side_ss<i>`` the side sets - an element and which of its sides - and ``elem_els<i>`` the element sets, each with an id in ``*_prop1`` and a name in ``*_names``. Results live along a ``time_step`` record dimension: ``vals_nod_var<i>`` per nodal variable, ``vals_elem_var<i>eb<j>`` per element variable and block, ``vals_glo_var`` for the whole-mesh values, named in ``name_nod_var``, ``name_elem_var`` and ``name_glo_var``, with the times in ``time_whole``. Cubit writes it, Sierra and MOOSE run on it, ParaView reads it natively.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - container
     - netCDF: classic (``CDF\x01``), 64-bit offset (``CDF\x02``), CDF5 (``CDF\x05``) or netCDF-4 (HDF5)
   * - geometry
     - ``coord`` ``(num_dim, num_nodes)`` float64, or ``coordx`` / ``coordy`` / ``coordz``
   * - elements
     - ``connect<i>`` ``(num_el_in_blk<i>, num_nod_per_el<i>)`` 1-based, ``elem_type`` names the shape; ``eb_prop1`` ids, ``eb_names`` names, ``attrib<i>`` per-element block attributes
   * - sets
     - ``node_ns<i>``, ``elem_ss<i>`` + ``side_ss<i>``, ``elem_els<i>``, each with ``*_prop1`` and ``*_names``
   * - results
     - ``time_whole``; ``vals_nod_var<i>``, ``vals_elem_var<i>eb<j>`` (``elem_var_tab`` says which block holds which), ``vals_glo_var``
   * - numbering
     - ``node_num_map``, ``elem_num_map``: the ids the entities carry outside the file

.. rst-class:: px-speclink

`Exodus II manual ↗ <https://sandialabs.github.io/seacas-docs/exodusII-new.pdf>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("out.e")
    mesh.element_tags["steel"]                # a block, by its name
    mesh.vertex_tags["clamped"]               # a node set
    mesh.element_tags["load_face"]            # a side set, read as faces
    mesh.vertex_attrs["disp"]                 # disp_x, disp_y, disp_z folded
    mesh = px.read("out.e", step=4)           # the fifth time step

Every element block is read in file order and becomes an element tag group named by its name, or ``block_<id>`` when the file names it nothing; a block's attributes - ``attrib<i>`` - are element attributes, NaN on the other blocks. A node set is a vertex tag group and an element set an element tag group, named the same way (``nodeset_<id>``, ``elemset_<id>``). A side set names a side of a solid rather than an element, so it is read the way the :doc:`Abaqus <abaqus>` reader reads a ``*Surface``: as the triangle or quadrilateral that side describes, appended after the blocks, tagged with the set's name and linked to its parent by the ``face_parent`` and ``face_index`` element attributes. A side named twice is one face in both groups; a side of a shell or a bar, or a side number the element has no side of, has no face to read and is skipped with a warning naming the set. An empty block - the NULL block Cubit leaves for a set with nothing in it - reads as nothing, its name with it.

The variables at one time step - the first, or ``step=`` - are the attributes: nodal variables ``vertex_attrs``, element variables ``element_attrs`` (NaN on a block the ``elem_var_tab`` leaves out), global variables ``global_attrs``. Exodus holds scalars only, so a ``<name>_x`` / ``<name>_y`` / ``<name>_z`` triple folds back into one ``(n, 3)`` array and a ``<name>_0`` / ``<name>_1`` / ... run into an ``(n, k)`` one. The step's time is ``global_attrs["time"]`` and the file's title ``global_attrs["title"]``; ``node_num_map`` and ``elem_num_map`` land in ``original_ids`` when they say something the index does not - the faces a side set adds carry no id in the file and are numbered on past the largest, so the ids the file holds survive a round trip. A ``num_dim`` of 2 pads the coordinates and flags the mesh two-dimensional.

The shape is the block's family and its node count together - ``HEX`` with twenty nodes per element reads as ``HEX20`` does - so a file that spells ``TETRA`` for a ten-node block reads. A family Exodus does not define raises :class:`~polyxios.exceptions.UnknownElementTypeError`; a count the family has no VTK cell for - ``HEX9``, ``WEDGE16``, ``PYRAMID14``, ``TETRA14`` - is skipped with a warning, its values and set members with it. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.

The codec needs `netCDF4 <https://unidata.github.io/netcdf4-python/>`_, which is optional - ``pip install "polyxios[netcdf]"``. Without it :class:`~polyxios.exceptions.UnsupportedFormatError` spells that command.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.e")

Exodus partitions the elements into blocks of one shape each, so the writer has to choose them: a tag group whose members are all of one type and overlap no group already taken goes out as a block of its name, and every element left over goes into a block per type, named ``triangle``, ``tetra`` and so on - ``tetra_2`` when a group already took the plain name, so the two never read back as one tag. A group whose members are all faces of solids - what a side set reads back as - goes out as a side set, and its faces are not written as elements unless another group names them; every other element group is an element set and every vertex group a node set. The variables are written at one time step: an ``(n, 3)`` attribute as ``<name>_x`` / ``<name>_y`` / ``<name>_z``, any other ``(n, k)`` as ``<name>_0`` to ``<name>_{k-1}`` - an array of more axes flattened to that first, an ``(n, 1)`` one as ``<name>_0`` alone, so neither reads back in its shape - integers and booleans as doubles. A name two attributes would land on (``v`` split beside a plain ``v_x``) is written once, from the first, with a warning; a value on a face that goes out as a side has no place in the file and is dropped with a warning too. Numeric ``global_attrs`` are global variables, ``title`` the file's title, ``time`` the step's time; any other text is dropped with a warning. The file is classic netCDF with 64-bit offsets, CDF5 when an index needs 64 bits, netCDF-4 for an empty mesh.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- **Node order.** VTK's for every shape but the quadratic hexahedra and wedges: Exodus lists a ``HEX20``'s mid-edge nodes bottom ring, vertical edges, top ring where VTK lists bottom, top, vertical, and a ``WEDGE15`` the same way; a ``HEX27`` puts the centroid first and the face centres bottom, top, left, right, front, back - nodes 22 to 27 of the manual's side-set table - where VTK orders them by axis with the centroid last. Both are permuted on the way in and back on the way out, so a reader that trusted the file's order would hand back a hexahedron whose top edges sit on its vertical ones.
- **Side numbering.** Exodus counts a solid's lateral faces first and the ends last; polyxios's ``ELEMENT_FACES`` puts the ends first. The table between them is checked in the tests against the manual's side-set node lists, so a side set read here names the face the manual names.
- **Blocks come back as tags.** A mesh written with no tags reads back with one tag per block - ``triangle``, ``tetra`` - because the file holds the partition and the reader reports what the file holds. A mesh whose tags partition it by type goes out under those names and comes back under them.
- **A pixel and a voxel** go out as the quad and hexahedron they are, corners reordered.
- **Buffers.** A file object reads through the netCDF library's in-memory open and writes through a temporary file, so a buffer receives the same bytes a path would. A ``.gz`` name is compressed on the way out and read on the way in like any other format.
- Every element type the file holds a block for is read; a ``NSIDED`` or ``NFACED`` block - polygons and polyhedra - is not, and is skipped with a warning.

.. seealso::

   :doc:`index` - the full format table.
