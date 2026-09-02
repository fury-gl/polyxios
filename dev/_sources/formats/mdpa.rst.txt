.. _format-mdpa:

Kratos MDPA
===========

.. rst-class:: px-badges

``.mdpa`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A Kratos ``.mdpa`` file is a flat list of ``Begin <Section>`` / ``End <Section>`` blocks, and everything the format holds lives in one of them. Nothing is counted in a header, so a block runs until its ``End``. ``Nodes`` carries ``id x y z``, one node per line, numbered freely. An ``Elements`` block is headed by a Kratos *element class* - ``Begin Elements Element3D4N`` - and each line spells ``id property_id <nodes>``; a ``Conditions`` block has the same shape and its own numbering. ``Properties`` blocks carry material and solver settings, ``NodalData`` and ``ElementalData`` carry one variable per block keyed by entity id, ``ModelPartData`` carries whole-mesh values, and a ``SubModelPart`` names the nodes, elements and conditions of a group - and may nest other parts inside itself.

The element class name is a registered Kratos class rather than a geometry. The generic spelling is ``Element<space dim>D<node count>N``, and an application registers its own element behind the same suffix: ``SmallDisplacementElement3D4N``, ``SurfaceLoadCondition3D3N``, ``VMS3D4N``. Kratos also uses explicit geometry names in its own tables - ``Tetrahedra3D4``, ``Quadrilateral3D4``, ``Prism3D6``.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - Begin <Section> … End <Section>, no counts
   * - node record
     - id x y z
   * - cell record
     - id property_id <nodes>
   * - cell blocks
     - Elements <class>, Conditions <class>
   * - indices
     - 1-based, free numbering
   * - data blocks
     - NodalData, ElementalData, ModelPartData
   * - groups
     - SubModelPart, nestable
   * - comments
     - ``//`` to end of line

.. rst-class:: px-speclink

`Kratos Multiphysics input data ↗ <https://github.com/KratosMultiphysics/Kratos/wiki/Input-data>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.mdpa")
    mesh.element_tags                        # one entry per SubModelPart
    mesh.vertex_attrs["original_ids"]        # the node ids the file spelled
    mesh.element_attrs["mdpa_property_id"]   # the Properties block each cell points at
    mesh.global_attrs                        # the ModelPartData entries

Writing
-------

.. code-block:: python

    px.write(mesh, "out.mdpa")

The writer recognises no options; any that are passed are warned about and ignored.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The class name and the geometry do not identify each other: ``Element3D4N`` is a tetrahedron, and a quadrilateral sitting in space would want the same name. polyxios writes the class name with the element's *topological* dimension - a quadrilateral is ``Element2D4N`` whatever plane it lies in - which makes the pair unique and keeps a round trip exact. Reading is wider than writing: the ``<n>D<m>N`` suffix is read off whatever name carries it, and the explicit geometry names are recognised too.
- ``Conditions`` are read as elements, after the elements, because they are cells of the same mesh. That merges two id spaces, so a file whose condition id repeats an element id comes back without ``original_ids`` - a numbering with a duplicate in it is one no writer can spell back. Conditions are written under ``Elements``: which cells a solver should treat as boundary is a modelling choice the mesh does not carry.
- Node ordering follows Kratos, which numbers its higher-order nodes the way GiD does. That agrees with polyxios for every type but two: a 20-node hexahedron and a 15-node wedge list their vertical mid-edge nodes before the ones on the top face, and both are permuted on the way in and back on the way out.
- An element of a type Kratos has no unambiguous class name for - a 3-node line, whose ``Element2D3N`` is how a triangle is spelled - is dropped on write with a warning, as is one whose node count does not match its type. A tag naming a dropped element leaves it out of its ``SubModelPart``: Kratos refuses to load a file whose part names an element the file never declared.
- A data block spells a vector two ways and both are read: a ``Variable<Vector>`` declares its length (``[3] (1.0,2.0,3.0)``), while an ``array_1d<double,3>`` - what ``DISPLACEMENT``, ``VELOCITY`` and the rest are - does not. A matrix (``[3,3] ((…),(…),(…))``) is more than one value per entity, which no attribute column holds, so its block is passed over with a warning.
- Each ``SubModelPart`` becomes a ``vertex_tags`` and/or ``element_tags`` entry named after it, nested parts included. Kratos asks only that siblings differ, so two parts under different parents may carry one name; the second gets a ``_2`` suffix and a warning rather than its members poured into the first. A part contributing no member of its own - one that only groups its children - claims no name and becomes no tag.
- ``element_attrs["mdpa_property_id"]`` appears only when some cell points at something other than property 0; a file using one property says nothing a writer could not reproduce. The contents of a ``Properties`` block do not travel - only the id.
- A ``ModelPartData`` value carries no type of its own, so one spelling a number or ``true``/``false`` comes back as one: a mesh that wrote the string ``"42"`` reads it back as the integer 42.
- A ``NodalData`` or ``ElementalData`` block naming a variable polyxios keeps for itself - ``original_ids``, ``mdpa_property_id`` - is read under a ``_2`` suffix and warned about, rather than buried under the key or left posing as it. A block naming no variable at all is read as ``unnamed``, also with a warning.
- ``Geometries``, ``ConditionalData`` and ``Tables`` blocks are not read, and each is warned about.
- ``lazy=True`` warns and loads eagerly; the format is ASCII.

.. seealso::

   :doc:`index` - the full format table.
