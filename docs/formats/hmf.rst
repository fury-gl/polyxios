.. _format-hmf:

HMF
===

.. rst-class:: px-badges

``.hmf`` ``read + write`` ``eager`` ``needs h5py``

Summary of the specification
----------------------------

HMF is XDMF's data model in one HDF5 file, with no XML beside it - the experimental format meshio invented for the case XDMF serves badly, a mesh whose arrays all live in the HDF5 file anyway and whose XML only says where. The file calls itself ``hmf`` and versions itself ``0.1-alpha`` in two root attributes, and holds one ``domain`` group with one ``grid``. The grid holds a ``Geometry`` dataset of ``(n, dim)`` coordinates with a ``GeometryType`` of ``XYZ``, ``XY`` or ``X``; one ``Topology<k>`` dataset per element type, each an ``(n, k)`` index table whose ``TopologyType`` is XDMF's name for the cell - ``Triangle``, ``Tetrahedron_10``, ``Polyvertex``; and the arrays over the nodes and the cells under ``NodeAttributes`` and ``CellAttributes``, one dataset per name. polyxios adds ``NodeSets`` and ``CellSets`` for its tag groups and ``Attributes`` for the mesh-wide values, each only when the mesh has some.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - HDF5: root attrs ``type="hmf"``, ``version="0.1-alpha"``; ``domain/grid/{Geometry,Topology<k>,NodeAttributes,CellAttributes}``
   * - geometry
     - ``Geometry`` ``(n, dim)`` with ``GeometryType`` ``XYZ`` / ``XY`` / ``X``
   * - topology
     - ``Topology<k>`` ``(n, nodes)`` 0-based with ``TopologyType``, the XDMF names - ``Polyvertex`` through ``Hexahedron_27``
   * - attributes
     - ``NodeAttributes/<name>``, ``CellAttributes/<name>``, one value per entity in the order the topologies are numbered
   * - polyxios extension
     - ``NodeSets/<name>``, ``CellSets/<name>`` as index arrays; ``Attributes/<name>`` for mesh-wide numbers and text

.. rst-class:: px-speclink

`HMF in meshio ↗ <https://github.com/nschloe/meshio/tree/main/src/meshio/hmf>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("out.hmf")
    mesh.element_tags["skin"]                 # CellSets/skin
    mesh.global_attrs["case"]                 # Attributes/case

The topologies are read in the order they are numbered - ``Topology10`` after ``Topology9``, not after ``Topology1`` - so the elements keep the order the file gave them and the cell attributes line up. ``NodeAttributes`` are ``vertex_attrs``, ``CellAttributes`` ``element_attrs``, ``NodeSets`` and ``CellSets`` the tag groups and ``Attributes`` the ``global_attrs``. A file that does not call itself ``hmf`` is refused by name, as is a topology XDMF does not name, one of the wrong width for its type, or one naming a node the geometry does not hold. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.

The codec needs `h5py <https://www.h5py.org/>`_, which is optional - ``pip install "polyxios[hdf5]"``. Without it :class:`~polyxios.exceptions.UnsupportedFormatError` spells that command.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.hmf")
    px.write(mesh, "out.hmf", compression="gzip", compression_opts=4)

One ``Topology<k>`` per element type, the types in ascending polyxios code order; a pixel and a voxel go out as the quadrilateral and hexahedron they are, and polygons all of one width as a ``Polygon`` table - polygons of differing widths have no ``(n, k)`` table to go in and are dropped with a warning. The ``Geometry`` is ``XYZ``, or ``XY`` for a mesh that came from a two-dimensional file and stayed in the plane. ``compression`` and ``compression_opts`` go to h5py for every dataset of more than one value.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The topology names and the node order are XDMF's, so what :doc:`xdmf` says of them holds here: VTK's order for every named kind, and an element type XDMF has no name for dropped on the way out with a warning per type, the values and tags over the elements cut to match.
- A mesh whose types are interleaved is written grouped by type and reads back grouped. An attribute whose length fits neither the nodes nor the cells is skipped with a warning naming it.
- The format calls itself experimental and versions itself ``0.1-alpha``; this codec reads and writes that version. The sets and mesh-wide attributes are polyxios' own addition to it, written only when the mesh has some: the originator's reader asserts on a key it does not know, so a mesh without tag groups or ``global_attrs`` is written in exactly the layout it reads, and one with them reads back here and in any reader that walks the keys it knows.

.. seealso::

   :doc:`index` - the full format table.
