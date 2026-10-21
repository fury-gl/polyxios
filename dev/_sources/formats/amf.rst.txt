.. _format-amf:

AMF
===

.. rst-class:: px-badges

``.amf`` ``read + write`` ``eager``

Summary of the specification
----------------------------

AMF - the Additive Manufacturing File Format, ISO/ASTM 52915 - was the standards bodies' answer to STL before 3MF took its place. It is one XML document, stored as it is or inside a ZIP archive under the same ``.amf`` extension. Its ``<object>`` elements each hold a ``<mesh>``: one ``<vertices>`` list and one or more ``<volume>`` blocks of ``<triangle>`` entries, a volume being a region of one material. A ``<color>`` can sit on a material, an object, a volume, a vertex or a triangle, the most specific winning; a ``<constellation>`` places objects, or other constellations, by the displacement and rotation of each ``<instance>``. The document carries a ``unit`` - millimetres unless it says otherwise - and ``<metadata>`` entries by type. Slic3r and PrusaSlicer read and write it, and older CAD exporters still do.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - container
     - plain XML, or a ZIP archive holding the one document
   * - geometry
     - ``<object id>`` → ``<mesh>`` → ``<vertices>`` of ``<vertex><coordinates><x/><y/><z/>``, then ``<volume materialid>`` of ``<triangle><v1/><v2/><v3/>``
   * - colour
     - ``<color><r/><g/><b/><a/>``, 0..1, on material, object, volume, vertex or triangle; a channel may be a formula in x, y, z
   * - placement
     - ``<constellation id>`` of ``<instance objectid>`` with ``deltax deltay deltaz`` and ``rx ry rz`` in degrees
   * - curved triangles
     - ``<edges>`` of tangent vectors bend a triangle's edges
   * - units
     - ``millimeter``, ``inch``, ``feet``, ``meter``, ``micron``

.. rst-class:: px-speclink

`ISO/ASTM 52915 ↗ <https://www.iso.org/standard/74640.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("part.amf")                # plain or zipped
    mesh.element_tags                         # one group per object
    mesh.element_attrs["colors"]              # RGBA per triangle, NaN where none
    mesh.vertex_attrs["normals"]              # when the vertices carry them

Every object's triangles, from all its volumes, form an element tag group named by its ``name`` metadata, or ``object_<id>``; each object keeps its own vertices. A triangle's colour is its own, else its volume's, else its object's, else that of its volume's material, read into ``element_attrs["colors"]`` as RGBA, NaN on the triangles none of them colours. A volume naming a material the document lacks takes no colour from it, with a warning; the void material ``0`` the specification reserves needs no definition and warns about nothing. Vertex colours and normals land in ``vertex_attrs["colors"]`` and ``vertex_attrs["normals"]`` the same way. The ``unit`` and every root ``<metadata>`` entry land in ``global_attrs``. A name or a metadata value keeps the white space at its ends, unless that white space holds a line break: then it is the indentation of a pretty-printed document, and dropped.

A document with constellations is read through them: each constellation no instance names is a root, every instance places its object - or nested constellation - rotated about x, then y, then z, then moved, and an object no constellation names is placed once where it stands. Objects and constellations are numbered apart, as files in the wild number them, and an instance naming an id both use names the object.

A vertex that is not one set of three coordinates, a triangle that is not one each of ``<v1>``, ``<v2>`` and ``<v3>``, a triangle naming a vertex its object lacks, an instance naming an object the document lacks, two objects or two constellations of one id, a constellation containing itself - every constellation is checked, roots or not - constellations placing more vertices and triangles than the document's size on disk can describe - each placement counted as 16 on top of its object's, and 16 again for each constellation that holds it on the way to a root, so a chain doubling a one-vertex object is refused like one doubling a large one, and a zipped document buys no more than its compressed bytes - constellations placing a vertex past the float range, a coordinate that is not a finite number, and an XML entity declaration each raise :class:`~polyxios.exceptions.CodecError`. So does a ZIP archive Python's ``zipfile`` cannot open, or a member that is encrypted, stored in a method ``zipfile`` lacks, or that declares more than 256 MiB inflated at over 200 times its stored size, the rule 3MF follows; of several ``.amf`` members the first is read, with a warning. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.amf")
    px.write(mesh, "out.amf", unit="inch", compress=True)

The writer shares 3MF's mapping. Triangles go out as they are, quads, pixels, polygons and triangle strips are split into triangles, a quadratic surface element keeps its corners, and everything else is dropped with a warning; a mesh with elements and no surface is refused, as is a vertex that is not finite. Every element tag group becomes an object of its name, the untagged triangles one last unnamed object, which reads back as ``object_<id>``; an element in two groups stays with the first, and a vertex two objects share is written to both. ``element_attrs["colors"]`` goes on each triangle, ``vertex_attrs["colors"]`` and ``vertex_attrs["normals"]`` on each vertex, a colour channel outside 0..1 clipped with a warning. The unit is ``unit=``, else ``global_attrs["unit"]``, else millimetres; 3MF's ``foot`` is written as ``feet``, and a mesh unit AMF lacks is written as millimetres with a warning, the vertices scaled to match when it is a length - 3MF's ``centimeter``, DXF's ``yard``. The global attributes named as AMF names its metadata - ``name``, ``description``, ``url``, ``author``, ``company``, ``cad``, ``revision``, ``tolerance``, ``volume`` - are written as ``<metadata>``.

Format-specific options
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``unit``
     - ``None``
     - On write: ``millimeter``, ``inch``, ``feet``, ``meter`` or ``micron``.
       Falls back to ``global_attrs["unit"]``, then to ``millimeter``.
   * - ``compress``
     - ``False``
     - On write: put the document in a ZIP archive, with a fixed timestamp
       so two writes are byte-identical.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- **Volumes merge into their object.** The material regions of one object come back as one tag group; a volume's material survives only as the colour it gave its triangles.
- **Formulas are not evaluated.** A colour channel spelled as an expression in x, y and z reads as NaN, with one warning per document.
- **Curved triangles read flat.** ``<edges>`` is skipped with a warning, as are ``<texmap>``, ``<texture>`` and ``<composite>``, everything inside them with them.
- **The unit is the attribute.** A ``<metadata type="unit">`` does not override the ``unit`` attribute of ``<amf>``.
- **Rotations are degrees.** The specification spells them so; a file a slicer wrote in radians places its instances at the wrong angle.
- **Instances are applied, not kept.** An object placed three times reads as three copies of its triangles.

.. seealso::

   :doc:`3mf` - the format that replaced it, read and written with the same mapping.
