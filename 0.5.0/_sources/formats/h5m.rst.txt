.. _format-h5m:

H5M (MOAB)
==========

.. rst-class:: px-badges

``.h5m`` ``read + write`` ``eager`` ``needs h5py``

Summary of the specification
----------------------------

H5M is the native file of MOAB, the Mesh-Oriented datABase used by DAGMC and the SIGMA toolkit. Everything sits under one HDF5 group, ``tstt``: the coordinates in ``nodes/coordinates``, one group per element type under ``elements`` - ``Tri3``, ``Tet4``, ``Hex8``, named for the shape and its node count - each holding a ``connectivity`` table, and ``sets``, the meshsets MOAB builds its materials, boundary conditions, geometric entities and partitions out of. Every entity has a *handle id*: the nodes are numbered from the ``start_id`` on their coordinates, then each element group from its own, then the sets, and both the connectivity and the sets speak in those ids. Values on the entities are *dense tags*, one dataset per name under the owner's ``tags`` group, each also declared under ``tstt/tags`` with its datatype and class; a *sparse tag* names its entities in an ``id_list`` beside its ``values``, which is how a set carries its ``NAME``, ``MATERIAL_SET``, ``NEUMANN_SET`` or ``DIRICHLET_SET`` number.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - HDF5: ``tstt/{nodes,elements/<Type>,sets,tags,elemtypes,history}``; ``tstt`` carries ``max_id``
   * - nodes
     - ``nodes/coordinates`` ``(n, 3)`` float64 with ``start_id``; ``nodes/tags/<name>`` dense
   * - elements
     - ``elements/<Type>/connectivity`` ``(n, k)`` of handle ids, ``start_id``, ``element_type`` from the ``elemtypes`` enum; ``tags/<name>`` dense
   * - types
     - ``Edge2 Edge3 Edge4 Tri3 Tri6 Tri7 Quad4 Quad8 Quad9 Tet4 Tet10 Pyramid5 Pyramid13 Prism6 Prism15 Prism18 Hex8 Hex20 Hex27``; ``Polygon``, ``Polyhedron``, ``Knife`` skipped
   * - sets
     - ``sets/list`` ``(n, 4)`` of ``[contents end, children end, parents end, flags]``; ``contents`` of handle ids, ``(start, count)`` pairs when the range bit is set
   * - tags
     - ``tags/<name>`` with ``class`` (bit, sparse, dense, mesh), a committed ``type``, and ``id_list`` + ``values`` for a sparse one

.. rst-class:: px-speclink

`MOAB HDF5 file format ↗ <https://www.mcs.anl.gov/~fathom/moab-docs/html/h5mmain.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("dagmc.h5m")
    mesh.element_tags["mat:steel"]            # a set with a NAME tag
    mesh.element_tags["material_2"]           # one with a MATERIAL_SET number only
    mesh.vertex_attrs["original_ids"]         # the GLOBAL_ID tag, when it is not 1..n

Element groups are read by ascending polyxios type code and their connectivity turned from handle ids into mesh indices through the ``start_id`` of the nodes. Dense tags on the nodes are ``vertex_attrs`` and dense tags on the elements ``element_attrs``, a tag one element type carries and another does not laid over every element with NaN where it is absent; a ``GLOBAL_ID`` tag other than ``1..n`` lands in ``original_ids``. Each set becomes a tag group over the nodes and the elements it holds - a set holding both becomes one group of each - named by its ``NAME`` tag when it has one and by its ``MATERIAL_SET``, ``NEUMANN_SET`` or ``DIRICHLET_SET`` number otherwise, as ``material_<n>`` and the like; a set with none of those is not a group, and an empty one is an empty group over the vertices when it is Dirichlet, over the elements otherwise. Range-compressed contents are expanded. A mesh-wide tag with a value is a ``global_attrs`` entry. A sparse tag set over the nodes or the elements themselves rather than over sets is not read; the ones met are named in one warning. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.

The codec needs `h5py <https://www.h5py.org/>`_, which is optional - ``pip install "polyxios[hdf5]"``. Without it :class:`~polyxios.exceptions.UnsupportedFormatError` spells that command.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.h5m")
    px.write(mesh, "out.h5m", compression="gzip", compression_opts=4)
    px.write(mesh, "out.h5m", global_ids=False)

One group per element type, the types in ascending polyxios code order, each element's handle id following the last, the ``elemtypes`` enum and ``max_id`` spelled as MOAB expects. ``vertex_attrs`` and ``element_attrs`` are dense tags, a vector as a record of ``k`` values per entity rather than an ``(n, k)`` table, each declared under ``tstt/tags``; a ``GLOBAL_ID`` numbering the nodes and the elements from one in the written order is added unless ``global_ids=False``, and ``original_ids`` are written under it when the mesh carries them. A MOAB tag has one type, so a vertex attribute and an element attribute of one name and different types go out as two tags, the elements' under ``<name>__cell``; one named like the ``NAME`` or set-kind tag the meshsets carry goes out as ``<name>__attr``, and a numeric global named like a dense tag as ``<name>__global``. Each such declaration carries a ``polyxios_name`` attribute that the reader undoes. The tag groups are meshsets, one per group, each carrying a ``NAME`` tag with the group's name and - so ``mbconvert`` and friends list them - a ``DIRICHLET_SET`` number for a vertex group and a ``MATERIAL_SET`` number for an element group. Numeric ``global_attrs`` are mesh tags with a ``global`` value. ``compression`` and ``compression_opts`` go to h5py for every dataset of more than one value.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Node order is MOAB's canonical one, which is VTK's for every type both name but the wedges and the quadratic hexahedra: MOAB's prism is VTK's wedge with its triangles turned over, and MOAB numbers a hexahedron's vertical edges before its top ring where VTK numbers them after, its 27-node face centres by face where VTK goes by axis. Those are permuted on the way in and out, by the tables MOAB's own VTK reader uses. A pixel and a voxel go out as the quad and hexahedron they are. A vertex element has no MOAB group - the nodes are entities in their own right - and is dropped on the way out with a warning, as is a polygon, a strip or a Lagrange cell; a ``Polygon``, ``Polyhedron`` or ``Knife`` group is skipped on the way in with a warning naming it.
- A mesh whose elements of one type are interleaved with others - one block per Gmsh entity, say - is written as one group per type and reads back grouped, which is the only shape MOAB has for it.
- A dense tag whose length is not the owner's, or whose values are strings or records, is not an attribute; a tag whose shape differs from one element type to the next is skipped with a warning. A set's ``NAME`` is read from its 32-byte opaque value up to the first NUL.
- The written ``history`` names polyxios and the time; the file is otherwise what MOAB's own writer spells, but MOAB itself has not been run against it.

.. seealso::

   :doc:`index` - the full format table.
