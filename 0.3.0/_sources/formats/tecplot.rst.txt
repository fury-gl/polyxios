.. _format-tecplot:

Tecplot ASCII
=============

.. rst-class:: px-badges

``.tec`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A Tecplot ASCII data file opens with an optional ``TITLE`` and a ``VARIABLES`` list naming every column, then one or more ``ZONE`` records. A finite-element zone declares its node and element counts (``N=``, ``E=``), its element shape (``ZONETYPE=FETRIANGLE``, ``FEQUADRILATERAL``, ``FETETRAHEDRON``, ``FEBRICK``) and its data packing. ``DATAPACKING=POINT`` interleaves all variables per node; ``DATAPACKING=BLOCK`` writes each variable's full column in turn. The connectivity list follows the nodal data.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - header
     - TITLE = "..." then VARIABLES = "X" "Y" "Z" ...
   * - zone record
     - ZONE T="name", N=..., E=..., ZONETYPE=..., DATAPACKING=...
   * - zone types
     - FETRIANGLE, FEQUADRILATERAL, FETETRAHEDRON, FEBRICK
   * - packing
     - POINT (per node) or BLOCK (per variable)
   * - connectivity
     - 1-based node indices, one element per line
   * - other suffixes
     - .dat is read via fmt=".tec"

.. rst-class:: px-speclink

`Read the full Tecplot ASCII specification ↗ <https://tecplot.azureedge.net/products/360/current/360-data-format.pdf>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.tec")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.tec")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Both POINT and BLOCK packing are read; finite-element zones are supported, ordered (structured) zones are not.
- Variables beyond the coordinate columns are read as named vertex attributes, so solution fields survive the round trip.
- ``.dat`` files are recognised when the codec is named explicitly: ``px.read("flow.dat", fmt=".tec")``.

.. seealso::

   :doc:`index` - the full format table.
