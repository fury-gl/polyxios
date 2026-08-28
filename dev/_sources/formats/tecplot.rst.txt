.. _format-tecplot:

Tecplot ASCII
=============

.. rst-class:: px-badges

``.tec .dat`` ``read + write`` ``eager``

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
     - .dat resolves by content; .plt is binary and not read

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
- A zone declaring only ``X`` and ``Y`` is 2-D: the vertices are padded with a zero z and ``global_attrs["was_2d"]`` records the fact, which is what writes the zone back out with two coordinate variables rather than three. Tecplot names the coordinates by position, so a mesh carrying a variable of its own named ``Z`` keeps three: the zone would otherwise read back with that variable as its z. A mesh whose vertices have since left the plane is written in three with a warning, and a flat mesh of ``TETRAHEDRON`` or ``BRICK`` cells keeps three coordinate variables whatever the flag says - Tecplot takes the node count per element from ``ET`` and the coordinate count from ``VARIABLES``, and the two have to agree.
- ``.dat`` is shared with Nastran, LS-DYNA and plain ASCII tables, so it is resolved by looking
  inside the file: a ``.dat`` opening with ``TITLE = "``, ``VARIABLES =``, ``ZONE``, ``FILETYPE =``
  or ``DATASETAUXDATA`` lands here. An unquoted ``TITLE =`` decides nothing - Nastran case
  control spells its title the same way - so the line under it settles the question. ``px.read("flow.dat", fmt=".tec")`` still forces the issue,
  and writing to ``.dat`` needs ``fmt=".tec"`` because an output file has no content to inspect.
- A variable that is not finite throughout is not written. The format spells no missing value, so a NaN - which is what a field covering part of a mesh carries over the rest of it - would go out as the token ``nan``, and a zone holding one is a zone no reader loads.
- Binary Tecplot (``.plt``) is registered so it fails with a clear message rather than not
  resolving at all; only the ASCII flavour is parsed.

.. seealso::

   :doc:`index` - the full format table.
