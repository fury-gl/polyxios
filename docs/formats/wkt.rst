.. _format-wkt:

Well-Known Text
===============

.. rst-class:: px-badges

``.wkt`` ``read + write`` ``eager``

Summary of the specification
----------------------------

WKT is the OGC text encoding for vector geometry: a type keyword followed by parenthesised coordinate lists. ``POLYGON`` takes one or more rings, the first being the exterior boundary and the rest holes; ``MULTIPOLYGON`` nests one level deeper; ``TRIANGLE``, ``POLYHEDRALSURFACE`` and ``TIN`` describe surfaces directly. Coordinates are whitespace-separated within a tuple and comma-separated between tuples. The EWKT extension prefixes a spatial reference as ``SRID=4326;``, and ``Z`` / ``M`` suffixes declare extra ordinates.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - geometry types
     - POINT, LINESTRING, POLYGON, MULTI*, TRIANGLE, TIN, POLYHEDRALSURFACE
   * - polygon rings
     - first ring exterior, subsequent rings are holes
   * - coordinates
     - x y [z [m]], space-separated in a tuple, comma-separated between
   * - dimensions
     - Z / M / ZM suffixes on the type keyword
   * - EWKT
     - optional SRID=<n>; prefix

.. rst-class:: px-speclink

`Read the full Well-Known Text specification ↗ <https://libgeos.org/specifications/wkt/>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.wkt")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.wkt")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- 2D geometry is padded to z = 0 so the vertex array is always (n, 3).
- Interior rings are preserved as element attributes rather than being merged into the exterior boundary or dropped.
- An EWKT ``SRID=`` prefix is parsed and then dropped — polyxios carries no coordinate reference system.

.. seealso::

   :doc:`index` — the full format table.
