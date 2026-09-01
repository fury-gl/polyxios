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

- A 2-D file's coordinates are padded with a zero z, so the mesh is 3-D like every other one polyxios holds, and ``global_attrs["was_2d"]`` records the fact. The ``Z`` suffix follows it. The writer puts it back so long as the vertices have stayed in the plane; a mesh whose vertices have since left it is written in three with a warning.
- Interior rings are preserved as element attributes rather than being merged into the exterior boundary or dropped.
- An EWKT ``SRID=`` prefix is parsed and then dropped - polyxios carries no coordinate reference system.
- The ISO/SQL-MM surface family reads too: ``TRIANGLE`` as one triangle, ``TIN`` as a set of them, ``POLYHEDRALSURFACE`` as a set of polygons. A ``TIN`` patch that is not a triangle, or a ``TRIANGLE`` carrying an interior ring, is refused rather than read as something else.
- ``EMPTY`` is a legal geometry for every type, including the surface family, and parses to an empty mesh rather than an error.
- ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`; the geometry is spelled as text and has to be parsed before it holds numbers.

.. seealso::

   :doc:`index` - the full format table.
