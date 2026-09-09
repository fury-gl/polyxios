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
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.xml")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The format holds a single cell type per file; writing a mixed mesh raises rather than dropping the elements that do not fit.
- A 2-D file's coordinates are padded with a zero z, so the mesh is 3-D like every other one polyxios holds, and ``global_attrs["was_2d"]`` records the fact. The ``dim`` attribute follows it. The writer puts it back so long as the vertices have stayed in the plane; a mesh whose vertices have since left it is written in three with a warning. An inferred ``dim`` is 3 for a mesh of solid cells however flat it lies, DOLFIN needing its geometric dimension to be at least its topological one; a ``dim=`` given by the caller is left alone.
- Mesh function and mesh value collection blocks are read as attributes where present.

.. seealso::

   :doc:`index` - the full format table.
