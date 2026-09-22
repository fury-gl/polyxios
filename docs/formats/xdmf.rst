.. _format-xdmf:

XDMF
====

.. rst-class:: px-badges

``.xdmf`` ``.xmf`` ``read + write`` ``eager`` ``time series``

Summary of the specification
----------------------------

The eXtensible Data Model and Format splits a dataset in two. The *light data* is an XML document: an ``<Xdmf>`` root holding a ``<Domain>``, which holds ``<Grid>`` elements. A uniform grid names its ``<Topology>`` - the element type and count, or ``Mixed`` with a type code ahead of each cell - and its ``<Geometry>`` - ``XYZ`` or ``XY`` coordinates, or the axes and spacing of a lattice - then any number of ``<Attribute>`` elements centred on ``Node``, ``Cell`` or ``Grid``, ``<Set>`` elements naming nodes or cells, and ``<Information>`` pairs. A collection grid holds other grids, spatially - several meshes side by side - or temporally, each child a step carrying a ``<Time Value>``. The *heavy data* is every array, and each ``<DataItem>`` says where its own lives: inline in the XML (``Format="XML"``), in an HDF5 file as ``file.h5:/dataset`` (``Format="HDF"``, the usual case and the reason the format exists), or in a raw file at a byte offset (``Format="Binary"``). A ``DataItem`` may also be a ``HyperSlab`` selecting from another, or a ``Reference`` to one elsewhere in the document by XPath, and a grid may borrow another's topology and geometry through an XInclude pointer rather than spelling them again.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - XML: ``<Xdmf><Domain><Grid>`` with ``<Topology>``, ``<Geometry>``, ``<Attribute>``, ``<Set>``, ``<Time>``, ``<Information>``
   * - topology
     - ``TopologyType`` naming the cell kind - ``Triangle``, ``Hexahedron_20``, ``Polyline NodesPerElement="2"`` - or ``Mixed``; ``2DSMesh`` … ``3DCoRectMesh`` for a lattice
   * - geometry
     - ``XYZ`` / ``XY`` coordinates, ``X_Y_Z`` split axes, ``VXVYVZ`` rectilinear axes, ``ORIGIN_DXDYDZ``
   * - heavy data
     - ``<DataItem Dimensions NumberType Precision Format>``: inline text, ``file.h5:/path``, or a binary file with ``Seek`` and ``Endian``
   * - time series
     - ``<Grid GridType="Collection" CollectionType="Temporal">`` of grids, each with ``<Time Value>``
   * - sharing
     - ``<DataItem Reference="XML">`` by XPath; ``<xi:include xpointer="…">`` for a grid's topology and geometry

.. rst-class:: px-speclink

`XDMF Model and Format ↗ <https://www.xdmf.org/index.php/XDMF_Model_and_Format>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("result.xdmf")                  # the first step of a series
    mesh = px.read("result.xdmf", step=-1)         # the last
    mesh.global_attrs["time"]                      # that step's <Time Value>
    mesh.vertex_tags, mesh.element_tags            # <Set> elements
    mesh.global_attrs["case"]                      # <Information Name="case">

    from polyxios import helper
    times, meshes = helper.read_time_series("result.xdmf")

:func:`polyxios.read` hands back one mesh, always. Every uniform grid at the chosen step is read - a spatial collection flat, several grids beside one another merged - and where there are several, each grid's elements are tagged with the grid's ``Name``, so a domain split into parts keeps saying which was which. A temporal collection is read at one step, the first without ``step=``; :func:`polyxios.helper.read_time_series` reads them all. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`: the arrays live wherever the file says, and are decoded whole.

The HDF5 flavour needs `h5py <https://www.h5py.org/>`_, which is optional - ``pip install "polyxios[hdf5]"``. Without it the inline and binary flavours still read, and a file naming an HDF5 sidecar raises :class:`~polyxios.exceptions.UnsupportedFormatError` spelling that command.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.xdmf")                          # arrays in out.h5, beside it
    px.write(mesh, "out.xdmf", data_format="xml")       # arrays inline
    px.write(mesh, "out.xdmf", data_format="binary")    # arrays in out.bin
    px.write(mesh, "out.xdmf", compression="gzip", compression_opts=4)
    px.write(mesh, "out.xdmf", time=0.5)

    helper.write_time_series(((t, solve(t)) for t in times), "run.xdmf")

``data_format`` says where the arrays go: ``"hdf"``, the default, writes them to an HDF5 file named after the XDMF one - ``out.h5`` beside ``out.xdmf``, never in the working directory - and needs h5py; ``"binary"`` writes them all to one ``out.bin`` beside it, each at its own ``Seek``; ``"xml"`` keeps them inline, which is the one flavour that writes to a file object or a ``.gz`` path, since the other two need a beside. ``compression`` and ``compression_opts`` go to h5py for every dataset. ``time`` writes the grid's ``<Time Value>``; without it, a number under ``global_attrs["time"]`` does. Any other option is warned about and ignored.

:func:`polyxios.helper.write_time_series` takes ``(time, mesh)`` pairs from any iterable, a generator included, and writes each step as it arrives: the first step spells the topology and geometry, and every later one includes them by pointer rather than spelling them again, so the mesh is in the file once. A step whose vertices moved writes its own geometry; one whose elements differ is refused, being another mesh rather than another step.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- A mesh of one element type writes a topology named for it, the free-size kinds with ``NodesPerElement``; a mesh mixing types writes ``Mixed``, one type code ahead of each cell - and after the code of a ``Polyvertex``, ``Polyline`` or ``Polygon``, its node count, which is how the Xdmf library and VTK spell those and what lets a line sit beside a triangle. A pixel and a voxel go out as the quadrilateral and hexahedron they are, corners reordered. The quadratic and higher-order kinds - ``Edge_3`` through ``Hexahedron_27`` - are read and written in VTK's node order, which is the order every other reader takes them in; ``Hexahedron_64`` and up read as ``lagrange_hexahedron`` with the file's own order. An element type XDMF has no name for - a triangle strip, a polyhedron, a Lagrange triangle - is dropped with a warning per type, and the values and tags over the elements are cut to match; a polyhedron in a mixed topology is stepped over on the way in the same way.
- ``<Attribute>`` elements centred on ``Node`` and ``Cell`` are ``vertex_attrs`` and ``element_attrs``, their shape the DataItem's, so a vector stays ``(n, 3)`` and a tensor ``(n, 3, 3)``; the ``AttributeType`` is written from the shape and not read. One centred on ``Grid`` lands in ``global_attrs`` as an array, an ``<Information Name Value>`` as text - a repeated name as a list of them - and a ``<Time Value>`` under ``global_attrs["time"]``. On the way out a number under that key is the ``<Time>`` and a text is an ``<Information>``; a ``time=`` option wins over either, and a text it displaces is warned about. An attribute centred on a face or an edge, or holding a finite element function, is neither one value per point nor one per cell and is skipped with a warning, as is one whose length matches neither count. ``vertex_tags`` and ``element_tags`` are ``<Set>`` elements of ``Node`` and ``Cell`` type; a face or edge set is skipped.
- Every ``DataItem`` flavour is read: inline text, HDF5 by ``file.h5:/dataset``, a binary file at a ``Seek`` in either ``Endian``, a ``HyperSlab`` over any of those, and a ``Reference`` to another item by XPath. The sidecar a file names is found beside it; a spelled path that leaves the file's own directory - a ``..`` or an absolute one - is refused, while a symlink beside the file is followed wherever it leads, that being the usual layout on a cluster. A ``DataItem`` declaring more values than it holds is refused naming the count; a count past the safety caps is refused before anything is sized.
- A grid may include another's topology and geometry with ``xi:include xpointer="xpointer(//Grid[@Name="…"]/*[self::Topology or self::Geometry])"``, the way a solver's time series does. The pointer's steps may name a grid, count one (``Grid[1]``) or start from ``/Xdmf/Domain``; a pointer of another form, or an include of another document, is skipped with a warning. A series reads an included array once and hands each step its own copy. A file holding more than one temporal collection is read at the first; the others are named in a warning and left. A grid named by such a pointer and lying beside a temporal collection rather than in it is the series' shared mesh, not a mesh of its own, and is not read twice.
- A lattice topology - ``2DSMesh``, ``3DRectMesh``, ``3DCoRectMesh`` and the rest - is expanded into explicit hexahedra or quadrilaterals, the way ``.vts``, ``.vtr`` and ``.vti`` are. Its ``Dimensions`` run slowest axis first, ``nz ny nx``, and so do the origin and spacing of ``ORIGIN_DXDYDZ``, which is how VTK reads them. The writer spells every mesh unstructured.
- A planar geometry (``XY``) is padded to three columns and flagged ``global_attrs["was_2d"]``; a mesh so flagged that stayed in the plane writes ``XY`` back.
- Several grids merged keep each grid's own points, so a point block two grids share through a ``Reference`` comes back once per grid; :func:`~polyxios.transforms.merge_duplicate_vertices` joins them.

.. seealso::

   :doc:`index` - the full format table.
