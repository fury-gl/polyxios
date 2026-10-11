.. _format-avs:

AVS-UCD
=======

.. rst-class:: px-badges

``.avs`` ``read + write`` ``eager``

Summary of the specification
----------------------------

AVS Unstructured Cell Data begins with a five-integer counts line: numbers of nodes, cells, node data values, cell data values and model data values. Node records follow - an id and three coordinates each - then one line per cell giving its id, material id, a cell type keyword (``tri``, ``quad``, ``tet``, ``hex``, ``prism``, ``pyr``, ``line``, ``pt``) and its node ids. Optional data sections at the end declare component counts and labels before the per-node or per-cell values.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - counts line
     - nnodes ncells ndata_node ndata_cell ndata_model
   * - node line
     - id x y z
   * - cell line
     - id mat_id type n1 n2 ...
   * - cell types
     - pt line tri quad tet pyr prism hex
   * - data sections
     - component counts, then labelled value blocks

.. rst-class:: px-speclink

`Read the full AVS-UCD specification ↗ <https://lanl.github.io/LaGriT/pages/docs/read_avs.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.avs")
    mesh.vertices          # (n, 3)
    mesh.element_types     # one type code per element

Writing
-------

.. code-block:: python

    px.write(mesh, "out.avs")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The per-cell material id is read into ``element_attrs["mat_id"]`` and written back from there, so material groups survive a round trip; a mesh without one is written with material ``1`` throughout.
- Only the geometry is read: the node, cell and model data sections are skipped, and the writer declares none.
- Node ids are remapped to 0-based indices and the originals are not kept; the writer numbers the nodes from 1 again.
- ``pt`` cells are not read: a file holding one is refused with :class:`~polyxios.exceptions.CodecError`, as is any other cell type outside ``line``, ``tri``, ``quad``, ``tet``, ``pyr``, ``prism`` and ``hex``. On write, an element with no AVS type - a vertex, a polygon, a quadratic cell - is skipped with a warning, and only the vertices the written cells use go out.

.. seealso::

   :doc:`index` - the full format table.
