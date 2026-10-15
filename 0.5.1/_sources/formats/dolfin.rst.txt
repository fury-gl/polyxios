.. _format-dolfin:

DOLFIN / FEniCS XML
===================

.. rst-class:: px-badges

``.xml`` ``read + write`` ``eager``

Summary of the specification
----------------------------

The legacy DOLFIN mesh format is a small XML tree: a ``<dolfin>`` root containing a ``<mesh>`` whose ``celltype`` attribute names the element (``interval``, ``triangle``, ``tetrahedron``) and whose ``dim`` gives the geometric dimension. Inside, ``<vertices>`` and ``<cells>`` declare their sizes and hold one ``<vertex>`` or one ``<triangle>`` element per entity, each with an explicit index and its coordinates or node references. Everything is indexed, 0-based and explicit.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - root
     - <dolfin xmlns:dolfin="...">
   * - mesh element
     - <mesh celltype="triangle" dim="2">
   * - vertices
     - <vertices size="n"> with <vertex index x y z>
   * - cells
     - <cells size="n"> with <triangle index v0 v1 v2> etc.
   * - homogeneity
     - one cell type per file
   * - status
     - superseded by XDMF in modern FEniCS

.. rst-class:: px-speclink

`Read the full DOLFIN / FEniCS XML specification ↗ <https://people.sc.fsu.edu/~jburkardt/data/dolfin_xml/dolfin_xml.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.xml")
    mesh.vertices          # (n, 3)
    mesh.element_types     # one type code per element

Writing
-------

.. code-block:: python

    px.write(mesh, "out.xml")
    px.write(mesh, "out.xml", dim=2)

Format-specific options
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``dim``
     - ``None``
     - On write: the geometric dimension written as ``<mesh dim>``. Inferred
       when not given: 2 for a mesh of no solid cells whose vertices lie in the
       ``z = 0`` plane, 3 otherwise. Every vertex is written with its ``z``
       whatever ``dim`` says.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The cell types read and written are ``interval``, ``triangle``, ``quadrilateral``, ``tetrahedron`` and ``hexahedron``; any other ``celltype`` is refused with :class:`~polyxios.exceptions.CodecError`.
- The format holds a single cell type per file. Writing a mixed mesh keeps the type of its first writable element and skips every other element with a warning; a mesh with no element of a supported type is refused.
- A 2-D file's coordinates are padded with a zero z, so the mesh is 3-D like every other one polyxios holds, and ``global_attrs["was_2d"]`` records the fact. The ``dim`` attribute follows it. The writer puts it back so long as the vertices have stayed in the plane; a mesh whose vertices have since left it is written in three with a warning. An inferred ``dim`` is 3 for a mesh of solid cells however flat it lies, DOLFIN needing its geometric dimension to be at least its topological one; a ``dim=`` given by the caller is left alone.
- Only the ``<vertices>`` and ``<cells>`` are read: mesh function, mesh value collection and domain data blocks are ignored, so no attribute or tag comes back.

.. seealso::

   :doc:`index` - the full format table.
