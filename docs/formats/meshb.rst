.. _format-meshb:

Medit binary
============

.. rst-class:: px-badges

``.meshb`` ``read + write`` ``always mmapped``

Summary of the specification
----------------------------

``.meshb`` is the binary form of the INRIA Medit mesh format. The file is a stream of keyword-indexed fields: an integer keyword code, the byte position of the next field, then that field's data. Codes identify the version and dimension first, then typed entity blocks - ``Vertices``, ``Edges``, ``Triangles``, ``Quadrilaterals``, ``Tetrahedra``, ``Hexahedra`` - each a count followed by fixed-width records of node indices plus a trailing reference (tag) integer. The version code fixes whether floats are 32- or 64-bit.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - layout
     - keyword code, next-field offset, then the field payload
   * - header codes
     - MeshVersionFormatted, Dimension
   * - entity blocks
     - Vertices, Edges, Triangles, Quadrilaterals, Tetrahedra, Hexahedra
   * - record shape
     - node indices (1-based) + one reference integer
   * - float width
     - set by the version code: 32-bit or 64-bit
   * - ascii sibling
     - .mesh - the same keywords in text form

.. rst-class:: px-speclink

`Read the full Medit binary specification ↗ <https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.meshb")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

A path is memory-mapped rather than loaded, always - there is no eager mode
to ask for and no ``lazy=`` to pass:

.. code-block:: python

    mesh = px.read("big.meshb")

Writing
-------

.. code-block:: python

    px.write(mesh, "out.meshb")

This codec takes no format-specific options.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- The trailing reference integer on each entity becomes ``element_attrs["ref"]`` and one ``element_tags["ref_<n>"]`` group per distinct value, which is how Medit files carry surface and region labels. On write, references come from that attribute, failing that from the ``ref_<n>`` groups; a group named anything else has nowhere to go in a record that carries a number, so it is reported rather than numbered on the caller's behalf.
- A record spells its reference in one signed 32-bit field, so a label outside that range is reported rather than written narrowed - narrowing one wraps it onto another label. ``vertex_attrs["ref"]`` is checked the same way, and a float or wrong-length column is refused rather than truncated or left to a broadcast error.
- A ``Dimension 2`` file's vertices are padded with a zero z, so the mesh is 3-D like every other one polyxios holds, and ``global_attrs["was_2d"]`` records the fact so the writer restores it. A mesh whose vertices have since left the plane is written in three with a warning, and a flat mesh of solid cells keeps three whatever the flag says, a ``Tetrahedra`` section under ``Dimension 2`` being a file no reader loads.
- ``Edges``, ``Prisms`` and ``Pyramids`` are decoded alongside the triangles, quadrilaterals, tetrahedra and hexahedra.
- The higher-order sections (``TrianglesP2``, ``TetrahedraP2``, ``HexahedraQ2``, ...) are stepped over. The format fixes no node ordering for high-order elements - libMeshb's own documentation defers it to a companion ``*Ordering`` section - so reading one would mean guessing a permutation, and a silently bent element is worse than a skipped one.
- Field offsets are validated against the file size before allocation, so a truncated or hostile file raises instead of over-allocating.
- A path is mapped and a file object read into memory, whichever ``lazy=`` says, so ``lazy=True`` warns and changes nothing while ``lazy=False`` is the default and passes without comment. The format is binary throughout - there is no ASCII flavour under this extension to load eagerly, which is what the flag would otherwise choose between.

.. seealso::

   :doc:`index` - the full format table.
