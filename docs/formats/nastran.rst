.. _format-nastran:

Nastran
=======

.. rst-class:: px-badges

``.bdf .nas .fem .dat`` ``read + write`` ``eager``

Summary of the specification
----------------------------

A Nastran bulk data file is a deck of fixed-column cards. Each card starts with a name in the first field and continues across eight further fields. Three field widths coexist: small field (8 columns), large field (16 columns, signalled by a ``*`` on the card name), and free field (comma-separated). ``GRID`` cards define points; ``CTRIA3``, ``CQUAD4``, ``CTETRA``, ``CHEXA`` and friends define elements by referencing grid ids and a property id, and continuation lines carry any fields that overflow the first card.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - field widths
     - small (8 col), large (16 col, name*), free (comma-separated)
   * - points
     - GRID  id  cp  x1  x2  x3
   * - elements
     - CTRIA3, CQUAD4, CTETRA, CHEXA, CPENTA, CBAR ...
   * - grouping
     - property id (PID) on each element card
   * - continuation
     - + / * continuation markers on overflowing cards
   * - other suffixes
     - .dat resolves by content (GRID / CEND / BEGIN BULK / SOL)

.. rst-class:: px-speclink

`Read the full Nastran specification ↗ <https://pynastran-git.readthedocs.io/en/latest/quick_start/bdf_overview.html>`__

Reading
-------

.. code-block:: python

    import polyxios as px

    mesh = px.read("model.bdf")
    mesh.vertices          # (n, 3)
    mesh.element_types     # element groups found in the file

Writing
-------

.. code-block:: python

    px.write(mesh, "out.bdf")

Format-specific options:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``large_field_grid``
     - ``False``
     - Emit GRID cards in large (16-column) field form for full float precision.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- All three field widths are read; writing emits free field, with large-field ``GRID`` cards on request.
- ``.dat`` is shared with Tecplot, LS-DYNA and plain ASCII tables, so it is resolved by looking
  inside the file: a ``$`` comment banner is stepped over and a ``GRID`` card, ``CEND``,
  ``BEGIN BULK``, ``SOL`` or ``NASTRAN`` statement lands the deck here.
  ``px.read("model.dat", fmt=".bdf")`` still forces the issue, and writing to ``.dat`` needs
  ``fmt=".bdf"`` because an output file has no content to inspect.
- Property ids become element tags, so the deck's grouping is preserved.

.. seealso::

   :doc:`index` - the full format table.
