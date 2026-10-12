.. _format-pvd:

PVD (ParaView collection)
=========================

.. rst-class:: px-badges

``.pvd`` ``read + write`` ``time series``

Summary of the specification
----------------------------

A ``.pvd`` file is ParaView's collection: a short XML document that holds no geometry and names the files that do. A ``<VTKFile type="Collection">`` root holds a ``<Collection>`` of ``<DataSet>`` elements, and each names a file - a ``.vtu``, ``.vtp``, ``.vti``, a parallel ``.pvtu`` index, any VTK file - by a path relative to the collection, the ``timestep`` it belongs to, and the ``part`` and ``group`` it is when one step is split across several files. ParaView reads the collection as one dataset over time; a solver writes one by adding a line per step.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - ``<VTKFile type="Collection" version="0.1">``
   * - datasets
     - ``<Collection>`` of ``<DataSet timestep= group= part= file=/>``
   * - files
     - any VTK dataset, by a path relative to the collection
   * - time
     - ``timestep``, one number per dataset; several datasets may share one

.. rst-class:: px-speclink

`ParaView data formats ↗ <https://docs.paraview.org/en/latest/UsersGuide/understandingData.html#pvd-files>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("run.pvd")                    # the first time step
    mesh = px.read("run.pvd", step=-1)           # the last
    mesh.global_attrs["time"]                    # that step's timestep

    from polyxios import helper
    times, meshes = helper.read_time_series("run.pvd")

:func:`polyxios.read` hands back one mesh, always. The collection's time steps are its distinct ``timestep`` values in ascending order, a dataset spelling none at zero; the first is read without ``step=``. Every dataset at that step is read through its own codec and merged into the one mesh - a ``.pvtu`` or other parallel index through :func:`polyxios.helper.read_multiblock` - and where there are several, tagged the way ParaView groups them: the datasets of one ``group`` share its tag, which names every element of every one of them, and a dataset naming no group is ``part_<n>``, ``n`` its place among that step's datasets counted from zero - so the untagged datasets of every step are ``part_0``, ``part_1`` and on, whatever ``part`` they spell. A name already in use - a tag a dataset carries, or another group's - is left to it, and the collection's gets a number appended. The datasets' ``global_attrs`` are merged, a key two of them spell differently warned about and the later one's value kept. The step's ``timestep`` is ``global_attrs["time"]``, over whatever the dataset spelled itself. A subdirectory an index written on Windows spells with a backslash is found beside it too. :func:`polyxios.helper.read_time_series` reads every step.

``lazy=True`` is handed to each dataset's reader: a step held in one file comes back the way that reader hands it out - views of a ``.vtu`` with a raw appended section - while a step split across several files is merged, which copies. It is handed on as asked, so a dataset whose codec cannot map its file - a ``.vtkhdf`` - raises :class:`~polyxios.exceptions.LazyReadError` from inside the collection, the way it would read on its own.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.pvd")                              # out.pvd and out_0.vtu beside it
    px.write(mesh, "out.pvd", format=".vtp", binary=False) # the dataset's codec and options
    px.write(mesh, "out.pvd", time=0.5)

    helper.write_time_series(((t, solve(t)) for t in times), "run.pvd")

:func:`polyxios.write` is the one-step collection: the mesh is written beside the index as ``<stem>_0`` with the extension ``format`` names, ``.vtu`` without it, and every other option goes to that codec. ``time`` is the step's ``timestep``; without it a number under ``global_attrs["time"]`` is, and zero otherwise. :func:`polyxios.helper.write_time_series` writes one dataset per step, ``<stem>_0``, ``<stem>_1`` and on, as the steps arrive from any iterable; the meshes need not share their elements, every step being a file of its own, but two steps at one time are refused, since a collection reads every dataset at one time as parts of one mesh.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The collection is found and written by path: a file object, or a ``.gz`` name, raises :class:`~polyxios.exceptions.CodecError`, since the datasets are found beside the index and a handle has no beside.
- A dataset whose path resolves outside the collection's own directory - ``../shared/step_0.vtu`` - is refused, the way the multi-block helpers refuse one for a ``.vtm`` or ``.pvtu``: an index is data, and one naming a file outside its own directory asks for a file the caller did not. A collection naming another collection is refused too.
- A series refused part-way - two steps at one time, a dataset its codec will not write - leaves the datasets written so far beside the index's path and no index; a shorter series written over a longer one leaves the earlier run's later datasets beside the new index. A read follows the index, so neither is read.
- The index spells the time, so the dataset's own ``time`` is left out of the file it goes into, and reads back from the index.
- A ``timestep`` that is not a finite number is refused on read, ``nan`` and ``inf`` among them, and a step at such a time is refused on write the same way, so no index is written that the reader would then reject.
- Whatever the dataset's codec keeps is what the collection keeps: a ``.vtu`` carries every attribute and tag group, a ``.vtp`` only what a PolyData holds, and a ``.vti`` a lattice. The mesh a step reads back as is the one its codec hands out - a tetrahedron written to a ``.vtp`` comes back as the four-sided polygon that file holds.

.. seealso::

   :doc:`vtkhdf` - VTK's own HDF5 file, with its time steps inside.
   :doc:`xdmf` - the other time series format polyxios holds.
   :doc:`index` - the full format table.
