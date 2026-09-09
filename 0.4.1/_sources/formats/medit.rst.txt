.. _format-medit:

Medit ASCII
===========

.. rst-class:: px-badges

``.mesh`` ``.medit`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A Medit file is a flat sequence of named sections. ``MeshVersionFormatted`` opens it and ``Dimension`` says whether the coordinates are 2-D or 3-D. Each entity section - ``Vertices``, ``Edges``, ``Triangles``, ``Quadrilaterals``, ``Tetrahedra``, ``Prisms``, ``Hexahedra``, ``Pyramids`` - names a count and then that many records, each holding its node indices (1-based) followed by one integer reference. The reference is how a Medit file carries a surface or region label. ``End`` closes the file, though not every writer emits it.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - opening keyword
     - ``MeshVersionFormatted`` (mandatory; it is what tells this format from MFEM)
   * - vertex record
     - x y [z] ref
   * - element record
     - n1 n2 ... ref, indices 1-based
   * - sections read
     - ``Vertices``, ``Edges``, ``Triangles``, ``Quadrilaterals``, ``Tetrahedra``, ``Prisms`` / ``Pentahedra``, ``Hexahedra``, ``Pyramids``
   * - comments
     - from ``#`` to end of line; keywords are case-insensitive

.. rst-class:: px-speclink

`Read the full Medit specification ↗ <https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>`__

Sharing ``.mesh`` with MFEM
---------------------------

``.mesh`` is MFEM's extension too, so neither format owns it. Reading resolves by content: a file opening with ``MeshVersionFormatted`` lands here, one opening with ``MFEM mesh`` lands in :doc:`mfem`. An output file has no content to look at, so a bare write keeps its historical meaning and emits MFEM; name this codec to write Medit.

.. code-block:: python

    import polyxios as px

    px.read("bamg.mesh")                     # Medit, by its header
    px.read("beam.mesh")                     # MFEM, by its header

    px.write(mesh, "out.mesh")               # MFEM, as it has always been
    px.write(mesh, "out.mesh", fmt=".medit")  # Medit
    px.write(mesh, "out.medit")              # Medit, no fmt= needed

Reading
-------

.. code-block:: python

    mesh = px.read("model.mesh")
    mesh.element_attrs["ref"]     # per-element reference
    mesh.element_tags["ref_10"]   # the elements labelled 10

Writing
-------

.. code-block:: python

    px.write(mesh, "out.medit", float_fmt=".17g")

``float_fmt`` overrides the coordinate format specifier.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- References become both ``element_attrs["ref"]`` and one ``element_tags["ref_<n>"]`` group per distinct value, so a region label survives a conversion as a named group rather than as an anonymous column. All-zero references label nothing and are not kept.
- On write, references come from ``element_attrs["ref"]``, failing that from the ``ref_<n>`` groups. A group named anything else has nowhere to go - a Medit record carries a number, not a name - so it is reported rather than numbered on the caller's behalf, as is one whose members are not element indices.
- ``vertex_attrs["ref"]`` goes the same way onto the vertex records. A float column is refused rather than truncated, for the reason a label is a number the file names exactly: rounding one relabels the vertex it stands for.
- References are held at 64 bits. The ASCII format puts no ceiling on one, and narrowing a reference either wraps it onto another label or saturates it onto the largest.
- A section's count may sit on the keyword's own line or the line after it: bamg writes ``Dimension 3`` where Medit writes the 3 on the next line, and both read. A file need not close with ``End``.
- A 2-D file's vertices are padded with a zero z, so the mesh is 3-D like every other one polyxios holds. The fact is kept in ``global_attrs["was_2d"]`` and written back, so a 2-D file does not come out as a flat 3-D one that a reader expecting a plane refuses; a mesh whose vertices have since left the plane is written in three with a warning, and a flat mesh of solid cells keeps three whatever the flag says, a ``Tetrahedra`` section under ``Dimension 2`` being a file no reader loads.
- The higher-order sections (``TrianglesP2``, ``TetrahedraP2``, ``HexahedraQ2``, ...) are skipped with a warning naming them. The format fixes no node ordering for high-order elements - libMeshb's own documentation says there are as many orderings as there are programmers, and defers to a companion ``*Ordering`` section - so reading one would mean guessing a permutation, and a silently bent element is worse than a skipped one.
- Sections carrying no geometry (``Corners``, ``Ridges``, ``Required*``, ``Normals``, ...) are stepped over without a word; an unrecognised one is named in a warning.
- A file declares its vertices once. A second ``Vertices`` section is refused rather than allowed to replace the block the elements already read index into - which, when the two counts agree, would move every element onto other geometry without a single index going out of range.
- A record carries one reference, so an element two ``ref_<n>`` groups both name keeps the later group's and the other is reported. Which group comes later is the order the mesh holds them in, not one the caller chose.

.. seealso::

   :doc:`meshb` - the binary flavour of the same format.

   :doc:`mfem` - the other format using ``.mesh``.

   :doc:`index` - the full format table.
