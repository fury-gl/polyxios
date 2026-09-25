.. _format-3mf:

3MF
===

.. rst-class:: px-badges

``.3mf`` ``read + write`` ``eager``

Summary of the specification
----------------------------

3MF - the 3D Manufacturing Format of the 3MF Consortium - is the file a slicer takes in place of an STL. It is a ZIP package in the Open Packaging Conventions layout: ``[Content_Types].xml`` names the part types, ``_rels/.rels`` points at the model part, and the model part - ``3D/3dmodel.model`` by convention - is one XML document. Its ``<resources>`` hold ``<object>`` elements, each either a ``<mesh>`` of ``<vertex x y z>`` and ``<triangle v1 v2 v3>`` entries or a ``<components>`` assembly of other objects placed under a ``transform``; ``<basematerials>`` names the materials and their display colours. The ``<build>`` says which objects are printed and where, one ``<item objectid transform>`` each. The model carries a ``unit`` - millimetres unless it says otherwise - and a handful of named ``<metadata>`` entries. Every slicer reads it, Windows and macOS preview it, and Cura, PrusaSlicer and Bambu Studio save their projects in it.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - container
     - ZIP; ``_rels/.rels`` relates the package to its model part, ``[Content_Types].xml`` types the parts
   * - model part
     - ``<model unit xml:lang>`` in ``http://schemas.microsoft.com/3dmanufacturing/core/2015/02``
   * - geometry
     - ``<object id type name>`` → ``<mesh>`` → ``<vertices>`` of ``<vertex x y z>``, ``<triangles>`` of ``<triangle v1 v2 v3 pid p1 p2 p3>``
   * - assemblies
     - ``<object>`` → ``<components>`` → ``<component objectid transform>``, nested to any depth
   * - transform
     - twelve numbers, a 4x3 affine matrix in row-vector order; the last three are the translation
   * - materials
     - ``<basematerials id>`` of ``<base name displaycolor>``, ``#RRGGBB`` or ``#RRGGBBAA``; the materials extension adds ``<colorgroup>`` and textures
   * - build
     - ``<build>`` of ``<item objectid transform>``; an object of type ``other`` is never built
   * - units
     - ``micron``, ``millimeter``, ``centimeter``, ``inch``, ``foot``, ``meter``

.. rst-class:: px-speclink

`3MF core specification ↗ <https://github.com/3MFConsortium/spec_core/blob/master/3MF%20Core%20Specification.md>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("part.3mf")
    mesh.element_tags                         # one group per built object
    mesh.element_attrs["colors"]              # RGBA per triangle, NaN where none
    mesh.global_attrs["unit"]                 # "millimeter"

The build is read the way a printer would see it: every item's object, a mesh outright or an assembly through its components, with each ``transform`` composed on the way down and applied to the vertices. A transform that mirrors turns each triangle inside out, so the winding of a mirrored placement is reversed to keep the surface facing outward. Each mesh object's triangles become an element tag group named by the object's ``name``, or ``object_<id>`` when it has none; an object placed twice - two components of one part - contributes twice under the one name. An object the build never places is not read.

A triangle's material is its colour: a ``pid``/``p1`` naming a ``<basematerials>`` entry or a materials-extension ``<colorgroup>`` entry reads into ``element_attrs["colors"]`` as RGBA in 0..1, the object's ``pid``/``pindex`` standing in for a triangle that names none, NaN on the triangles with no colour at all. A texture, composite or multi-property is not a colour and reads as NaN. The model's ``unit`` and every ``<metadata>`` entry land in ``global_attrs`` under their own names. ``lazy=True`` raises :class:`~polyxios.exceptions.LazyReadError`: a ZIP member has to be inflated before it can be parsed.

The model part is found through ``_rels/.rels``; a package without one is read through the conventional ``3D/3dmodel.model``. A file that is not a ZIP, a package with no model part, a triangle naming a vertex the object does not hold, an item naming an object the model does not hold, an object that contains itself, and a transform that is not twelve numbers each raise :class:`~polyxios.exceptions.CodecError` naming the fault.

Writing
-------

.. code-block:: python

    px.write(mesh, "out.3mf")
    px.write(mesh, "out.3mf", unit="inch")

3MF holds triangles, so the writer works in them: a triangle goes out as it is, a quad, pixel, polygon or triangle strip is split into triangles, and a quadratic surface element keeps its corners. Lines, vertices and volume elements have no triangle to be written as and are dropped with a warning; a mesh with elements and no surface at all is refused. Every element tag group goes out as an ``<object>`` of its name, in the order of the groups, and the triangles no group claims as one last unnamed object. A 3MF object owns its triangles, so an element in two groups stays with the first and the second is told so; each object also carries its own vertex list, so a vertex two objects share is written to both. ``element_attrs["colors"]`` - three or four columns, 0..1 as floats or 0..255 as integers - becomes one ``<basematerials>`` with a ``<base>`` per distinct colour, each triangle naming its own; a NaN row names none. The unit is the ``unit=`` argument, else ``global_attrs["unit"]``, else millimetres; a mesh unit the format lacks - ``mm`` from another reader, say - is written as millimetres with a warning, while a ``unit=`` the format lacks is refused. A global attribute named as the specification names its metadata - ``Title``, ``Designer``, ``Description``, ``Copyright``, ``LicenseTerms``, ``Rating``, ``CreationDate``, ``ModificationDate``, ``Application`` - is written as ``<metadata>``. Every other attribute has no place in the format and is not written. Two writes of one mesh are byte-identical: the ZIP entries carry a fixed timestamp.

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
     - On write: the model's unit of length, one of ``micron``,
       ``millimeter``, ``centimeter``, ``inch``, ``foot``, ``meter``. Falls
       back to ``global_attrs["unit"]``, then to ``millimeter``.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- **Objects come back as tags.** A mesh written with no tags reads back with one group, ``object_1``, because the file holds an object and the reader reports what the file holds. A mesh whose tags partition it goes out under those names and comes back under them.
- **Transforms are applied, not kept.** A part placed three times reads as three copies of its triangles; the placements are not recoverable from what comes back, and a rewrite holds three objects' worth of triangles in one.
- **One colour per triangle.** 3MF lets a triangle name a property per corner (``p1``, ``p2``, ``p3``); only ``p1`` is read, since a colour per corner has no element to land on. The material's name is not kept, only its colour.
- **Coordinates are written in full.** Each coordinate is written as the shortest decimal that reads back to the same double, so a round trip through the file changes nothing.
- **Buffers.** A file object is read and written like a path: the whole package is read into memory, and a ``.gz`` name is compressed on the way out and read on the way in like any other format.
- **The model part is parsed as a stream.** It is fed to the parser as it inflates, and each ``<vertex>`` and ``<triangle>`` goes straight into a numeric buffer rather than into the tree, so a mesh of ten million vertices - two gigabytes of XML - reads in the memory its numbers take, not thirty times that. Inflated whole, a part past a gigabyte is more than the XML parser will take in one buffer.
- The materials and production extensions are read only as far as ``<colorgroup>``; a beam lattice, a slice stack or a texture is not geometry the reader returns.

.. seealso::

   :doc:`index` - the full format table.
