.. _format-vtkhdf:

VTKHDF
======

.. rst-class:: px-badges

``.vtkhdf`` ``read + write`` ``time series`` ``polyxios[hdf5]``

Summary of the specification
----------------------------

VTKHDF is VTK's own HDF5 layout and the file ParaView is moving its readers and writers to: one HDF5 file, a ``/VTKHDF`` group at its root, and two attributes on that group - ``Version``, two integers, and ``Type``, the VTK data object it holds. An ``UnstructuredGrid`` keeps ``Points``, ``Connectivity``, ``Offsets`` and ``Types`` datasets, a ``PolyData`` its points and four cell groups - ``Vertices``, ``Lines``, ``Polygons``, ``Strips`` - each with a connectivity and offsets of its own, an ``ImageData`` no points at all, only the ``WholeExtent``, ``Origin``, ``Spacing`` and ``Direction`` attributes its lattice is expanded from. Every type keeps its arrays under ``PointData``, ``CellData`` and ``FieldData``. The file is partitioned - ``NumberOfPoints`` and ``NumberOfCells`` hold one entry per part, each part's cells indexing its own points - and, when a ``Steps`` group is there, temporal: every dataset is the steps' arrays laid end to end, and ``Steps`` holds the time of each step and where its points, cells and arrays begin.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - ``/VTKHDF`` with ``Version`` (``[2, 0]``) and ``Type`` attributes
   * - UnstructuredGrid
     - ``NumberOfPoints`` ``NumberOfCells`` ``NumberOfConnectivityIds`` per part; ``Points`` ``Types`` ``Connectivity`` ``Offsets``
   * - PolyData
     - ``NumberOfPoints``, ``Points``; ``Vertices`` ``Lines`` ``Polygons`` ``Strips`` groups, each with its counts, ``Connectivity`` and ``Offsets``
   * - ImageData
     - ``WholeExtent`` ``Origin`` ``Spacing`` ``Direction`` attributes; arrays shaped ``(nz, ny, nx[, k])``, an axis of size one kept or left out
   * - arrays
     - ``PointData/<name>`` ``CellData/<name>`` ``FieldData/<name>``
   * - time series
     - ``Steps`` with ``NSteps``, ``Values``, ``PartOffsets``, ``NumberOfParts``, ``PointOffsets``, ``CellOffsets``, ``ConnectivityIdOffsets``, ``PointDataOffsets/<name>``, ``CellDataOffsets/<name>``, ``FieldDataOffsets/<name>``, ``FieldDataSizes/<name>``

.. rst-class:: px-speclink

`VTKHDF file format ↗ <https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/index.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("result.vtkhdf")              # the first step of a series
    mesh = px.read("result.vtkhdf", step=-1)     # the last
    mesh.global_attrs["vtkhdf_type"]             # "UnstructuredGrid", "PolyData" or "ImageData"
    mesh.global_attrs["time"]                    # that step's time, when the file holds steps

    from polyxios import helper
    times, meshes = helper.read_time_series("result.vtkhdf")

:func:`polyxios.read` hands back one mesh, always. An ``UnstructuredGrid`` reads as its cells are, its VTK type codes mapped to polyxios's; a ``PolyData`` reads its four cell groups in VTK's order - vertices, lines, polygons, strips - with each cell typed by its point count, a ``vertex`` or ``poly_vertex``, a ``line`` or ``poly_line``, a ``triangle``, ``quad`` or ``polygon``, a ``triangle_strip``; an ``ImageData`` is expanded into the hexahedra, quadrilaterals or lines its extent spans, the way :doc:`vti` is, with ``vti_origin``, ``vti_spacing`` and ``vti_extent`` in ``global_attrs`` so it writes back as a ``.vti``, and a ``Direction`` that is not the identity applied to the points and kept as ``vtkhdf_direction``. ``PointData`` and ``CellData`` arrays are ``vertex_attrs`` and ``element_attrs`` in the file's own dtypes, ``FieldData`` arrays ``global_attrs`` - a one-value array as the value, a string array as text - and the VTK type the file declared is ``global_attrs["vtkhdf_type"]``. Every partition at the chosen step is merged into the one mesh, and where there are several, each part's elements are tagged ``part_<k>``. A temporal file is read at one step, the first without ``step=``; :func:`polyxios.helper.read_time_series` reads them all, each carrying its time under ``global_attrs["time"]``.

Only the rows a step needs are read from each dataset, so the other steps of a long series stay on disk. Every count the file declares is checked against the dataset that has to hold it before anything is allocated, and a connectivity id past the part's points, offsets that run backwards or reach past the ids, or a step naming parts the file does not have, is refused naming the file and the dataset.

The format needs `h5py <https://www.h5py.org/>`_, which is optional - ``pip install "polyxios[hdf5]"``. Without it a read or a write raises :class:`~polyxios.exceptions.UnsupportedFormatError` spelling that command.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.vtkhdf")                          # UnstructuredGrid
    px.write(mesh, "out.vtkhdf", polydata=True)           # PolyData: vertices, lines, polygons, strips
    px.write(mesh, "out.vtkhdf", time=0.5)                # a Steps group of one step
    px.write(mesh, "out.vtkhdf", compression="gzip", compression_opts=4)

    helper.write_time_series(((t, solve(t)) for t in times), "run.vtkhdf")

``polydata`` picks the VTK type: ``True`` writes a ``PolyData`` and refuses a mesh holding a cell one cannot - a tetrahedron, a hexahedron - naming it, ``False`` an ``UnstructuredGrid``; without it, ``global_attrs["vtkhdf_type"]`` decides, so a file read as a ``PolyData`` is written back as one, and anything else is an ``UnstructuredGrid``. The points are written as ``float64``, the cell types as VTK's codes, the connectivity and offsets as ``int64``. Every numeric vertex and element attribute goes out under its name and dtype, a boolean as ``uint8`` and an array of more than two dimensions flattened to one row per entity; a tag group travels as one ``polyxios_tag_<name>`` column of ones and zeros, the way the VTK XML codecs write theirs. A number or an array under ``global_attrs`` is a ``FieldData`` array, a text a string array. ``time`` writes a ``Steps`` group of one step at that time; without it, a number under ``global_attrs["time"]`` does. ``compression`` and ``compression_opts`` go to h5py for every dataset.

:func:`polyxios.helper.write_time_series` takes ``(time, mesh)`` pairs from any iterable, a generator included, and appends each step as it arrives: the cells are written once, the points again only for a step whose vertices moved, and every ``PointData``, ``CellData`` and ``FieldData`` array once per step. A step whose elements differ from the first's is refused, being another mesh rather than another step; so is one that lacks an array the first step had, or brings a new one, since VTK reads every step of a series through the same array names.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- An ``ImageData`` is read and never written: a lattice goes out as the ``UnstructuredGrid`` its cells are. Read one and write it as a :doc:`vti` to keep it a lattice.
- A ``Type`` polyxios does not read - ``OverlappingAMR``, ``MultiBlockDataSet``, ``PartitionedDataSetCollection``, ``HyperTreeGrid`` - raises :class:`~polyxios.exceptions.UnsupportedFormatError` naming it, and a ``Version`` past 2 :class:`~polyxios.exceptions.CodecError`. A ``Version`` of 1 reads as leniently as 2: the layouts 2.0 defined - ``PolyData``, several parts, ``Steps`` - are taken as written when a 1.x file holds them.
- A ``PointData`` or ``CellData`` array that is not numeric, or holds fewer rows than the step needs, is dropped with a warning naming it, the answer the VTK XML readers give an array of the wrong length. An array with no entry under ``PointDataOffsets`` in a temporal file is read from row zero, which is where an array the file keeps once for every step sits.
- A temporal ``ImageData`` keeps its steps along a leading axis of each array - ``(steps, nz, ny, nx[, k])`` - and its offsets count along it, which is what VTK's reader asks for; an ``UnstructuredGrid`` or ``PolyData`` lays the steps end to end along the row axis.
- Every element type VTK has a code for is written; a code polyxios has no name for is dropped with a warning, and the values and tags over it cut to match.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`: the arrays live in HDF5 datasets that h5py decodes whole.
- Verified against VTK's own writer and reader: the test suite, when VTK is installed - the interop CI job installs it - reads what ``vtkHDFWriter`` writes - an ``UnstructuredGrid``, a ``PolyData``, an ``ImageData``, a partitioned grid and a temporal series - and has ``vtkHDFReader`` read back the files polyxios writes, the series' time steps included.

.. seealso::

   :doc:`pvd` - ParaView's collection of datasets over time.
   :doc:`xdmf` - the other HDF5-backed VTK family format, with its XML beside it.
   :doc:`vtu` - the same UnstructuredGrid in XML.
   :doc:`index` - the full format table.
