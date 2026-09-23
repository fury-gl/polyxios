.. _format-med:

MED
===

.. rst-class:: px-badges

``.med`` ``read + write`` ``eager`` ``needs h5py``

Summary of the specification
----------------------------

MED is the mesh and field file of the Salome platform and of code_aster: one HDF5 file with a fixed tree. The meshes live under ``ENS_MAA``, each a group named for the mesh and carrying the space dimension, the mesh dimension, a description and the axis names as attributes; under it a computation step - ``-0000000000000000001-0000000000000000001`` for a mesh that does not step - holds the nodes (``NOE``) and the cells by geometry (``MAI/TR3``, ``MAI/HE8`` and so on, named for the shape and its node count). Coordinates and connectivity are flat datasets stored component-major - every X, then every Y; every first node, then every second - with a ``NBR`` attribute counting the entities. Every node and every cell carries a *family* number, and the families under ``FAS/<mesh>`` each belong to any number of named *groups*: that is how a MED mesh spells its sets, node families numbered upward and cell families downward, family zero belonging to nothing. Fields live under ``CHA``, one group per field with its component count and names, one subgroup per time step, and under each step the values on the nodes (``NOE``), on the cells of one geometry (``MAI.HE8``), at the nodes of each cell (``NOE.HE8``) or at Gauss points named under ``GAUSS``; a field over part of a mesh names a profile under ``PROFILS``.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - HDF5: ``INFOS_GENERALES`` (version), ``ENS_MAA/<mesh>/<step>/{NOE,MAI}``, ``FAS/<mesh>``, ``CHA/<field>/<step>``, ``PROFILS``, ``GAUSS``
   * - nodes
     - ``NOE/COO`` flat, component-major; ``FAM`` family per node; ``NUM`` optional numbering
   * - cells
     - ``MAI/<GEO>/NOD`` flat, node-slot-major, 1-based; ``FAM``; ``NUM``; ``GEO`` attribute = dimension × 100 + node count
   * - geometries
     - ``PO1 SE2 SE3 SE4 TR3 TR6 TR7 QU4 QU8 QU9 TE4 T10 PY5 P13 PE6 P15 P18 HE8 H20 H27``, plus polygons and polyhedra
   * - families
     - ``FAS/<mesh>/{NOEUD,ELEME}/<family>`` with ``NUM`` and ``GRO/NOM``, the group names as 80-character rows
   * - fields
     - ``CHA/<name>`` with ``NCO``, ``NOM``, ``TYP``; per step ``NOE``, ``MAI.<GEO>``, ``NOE.<GEO>``; values in ``<profile>/CO``, ``NBR`` × ``NGA`` × ``NCO``

.. rst-class:: px-speclink

`MED file format ↗ <https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.med")
    mesh.global_attrs["mesh_name"]            # the mesh's name in the file
    mesh.vertex_tags, mesh.element_tags       # the MED groups
    mesh.element_attrs["SIEF_ELGA"]           # a field on the cells

    mesh = px.read("results.med", mesh="Fluid domain")   # one of several
    mesh = px.read("results.med", step=3)                # the fields' fourth step

Every mesh in the file is read, and when there are several they are merged into one, each mesh's elements tagged with the mesh's name; ``mesh=`` picks one by name, and a name the file does not hold is refused listing the ones it does. Each MED group becomes a tag group whose members are the nodes or cells of every family listing it, so a cell in two groups is in both; a family belonging to no group keeps its number as ``family_<n>``. A field on the nodes is a ``vertex_attrs`` entry and a field on the cells an ``element_attrs`` one, laid over every element with NaN where the field does not reach a geometry; values at Gauss points or at the nodes of each cell keep their axis. Fields are read at one step, the first without ``step=``, and the step's time lands in ``global_attrs["time"]``. Node and cell numbers other than ``1..n`` land in ``original_ids``. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.

The codec needs `h5py <https://www.h5py.org/>`_, which is optional - ``pip install "polyxios[hdf5]"``. Without it :class:`~polyxios.exceptions.UnsupportedFormatError` spells that command.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.med")
    px.write(mesh, "out.med", mesh_name="plate", time=0.5)
    px.write(mesh, "out.med", compression="gzip", compression_opts=4)

The file is MED 3.0, which every Salome since 7 and every code_aster since 11 reads; the nine attributes the MED library's mesh-info call reads are all spelled, and each cell group carries its ``GEO`` code. ``mesh_name`` names the mesh inside the file, ``global_attrs["mesh_name"]`` failing that and ``"mesh"`` failing both. Tag groups become MED groups: each distinct combination of groups an entity belongs to is a family, and the family belongs to every group in the combination, which is what Salome itself does. ``vertex_attrs`` are ``NOEU`` fields and ``element_attrs`` are fields with one ``MAI.<GEO>`` part per cell type; a third axis the size of the cell's node count goes out as ``NOE.<GEO>`` values. A vertex attribute and an element attribute sharing a name, a component count and a value type are one field over the nodes and the cells, which is how MED spells such a field; sharing a name and nothing else, or holding a ``/`` an HDF5 link cannot, the second is filed under a name that fits (``v_2``, ``a_b``) with a warning saying so. ``time`` is the fields' time step, ``global_attrs["time"]`` failing that; without either the fields carry no step. An element attribute that is NaN on every element is not written - a MED field would turn that "no value" into one - and is warned about. ``compression`` and ``compression_opts`` go to h5py for every dataset of more than one value; the MED library reads a filtered dataset as it reads any other. A file written to a path is written under a temporary name and moved into place, so a write that fails leaves nothing half done.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Node order is Salome's, which is VTK's for every geometry MED names but the 27-node hexahedron, whose six face centres MED lists bottom, front, right, back, left, top and VTK by axis; those are permuted on the way in and out, and per-node (``NOE.<GEO>``) values with them. A pixel and a voxel go out as the quad and hexahedron they are, corners reordered and per-node values with them. Polygons, polyhedra and any other geometry polyxios has no element type for are skipped on the way in with one warning naming them, and an element type MED has no geometry for - a polygon, a triangle strip, a Lagrange cell - is dropped on the way out with one warning per type, the values and tags over the elements cut to match.
- Cells read by ascending polyxios type code whatever order the file lists its geometry groups in, so a mesh already grouped by type comes back in the order it was written. A mesh whose types are interleaved is written grouped, one ``MAI`` group per geometry, and reads back grouped.
- A family with no ``GRO`` subgroup - Gmsh writes one per elementary entity outside any physical group - reads as ``family_<n>``; a file with no ``FAS`` at all reads with no tags. A structured (grid) mesh is refused by name; only unstructured meshes are read.
- A field's part on a geometry the mesh does not hold, or holding the wrong count for one it does, is skipped with a warning naming the field; a field whose parts disagree on shape across geometries is skipped whole, and one holding both cell and per-node values on a geometry keeps the cell values. A profiled field is spread over its whole support, NaN where the profile does not reach, however far it reaches; a profile naming an entity past the support is refused. Gauss-point values are read but not written: writing them needs a localisation the mesh does not carry, so an element attribute whose third axis is neither one nor the cell's node count is dropped with a warning.
- A MED file holds meshes, families and fields and nothing mesh-wide, so ``global_attrs`` other than ``mesh_name`` and ``time`` are dropped with a warning. A space dimension of 2 pads the coordinates and flags ``global_attrs["was_2d"]``; a mesh so flagged that stayed in the plane writes two coordinates back. The mesh dimension written is the cells' - a shell in 3-D space is ``DIM=2, ESP=3`` - which is what the MED library expects.
- Counts are checked against the datasets: a ``NBR`` larger than what its dataset holds is refused naming both, one smaller reads the first entities whole - the block is shaped by what it holds before it is cut, coordinates and connectivity being component-major - and a node index outside ``1..n`` is refused before anything is built. A ``FAM`` or ``NUM`` shorter than its entities is ignored with a warning. ``step=`` must be a whole number from zero.

.. seealso::

   :doc:`index` - the full format table.
