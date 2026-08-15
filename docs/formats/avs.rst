.. _format-avs:

AVS-UCD
=======

.. rst-class:: px-badges

``.avs`` ``read + write`` ``eager``

Summary of the specification
----------------------------

AVS Unstructured Cell Data begins with a five-integer counts line: numbers of nodes, cells, node data values, cell data values and model data values. Node records follow — an id and three coordinates each — then one line per cell giving its id, material id, a cell type keyword (``tri``, ``quad``, ``tet``, ``hex``, ``prism``, ``pyr``, ``line``, ``pt``) and its node ids. Optional data sections at the end declare component counts and labels before the per-node or per-cell values.

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
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.avs")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The per-cell material id is preserved as an element tag, so material groups survive a round trip.
- Node and cell data sections are read into named vertex and element attributes.
- 1-based node ids are remapped to 0-based indices; the originals are kept as vertex tags.

.. seealso::

   :doc:`index` — the full format table.
