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
  ``BEGIN BULK``, ``NASTRAN`` or a ``SOL`` naming a solution number or name lands the deck
  here - a numeric table whose first row happens to read ``SOL 1.0 2.0`` does not.
  ``px.read("model.dat", fmt=".bdf")`` still forces the issue, and writing to ``.dat`` needs
  ``fmt=".bdf"`` because an output file has no content to inspect.
- A real field is written in whichever legal spelling fits it: the plain form first, then the implicit-exponent shorthand the format allows (``1.234-10`` for ``1.234E-10``, ``1.2346+7`` for ``1.2346E+07``), which buys two or three significant digits in an eight-column field. A value below one drops its leading zero when that column is a significant digit, since bulk data reads ``.5`` as ``0.5``. A spelling that reads back as the value itself is always preferred to a longer one that does not, so ``1e7`` goes out as ``1.E+07`` rather than as seven nines. A value that would round up past the largest double is stepped one digit toward zero instead of being refused, so no finite coordinate fails to write.
- Property ids become element tags, so the deck's grouping is preserved.
- A card name does not say the element's order, so the shape comes from the grid points the card carries: ``CTETRA`` with ten grid points is a ``quadratic_tetra``, with four a ``tetra``. Counting stops at the first blank field, because Nastran lets a mid-side grid point be omitted and that card is a linear element.
- ``CPENTA`` runs the prism's three vertical edges before its top ring where VTK runs them after, so those mid-side nodes are permuted on read and back on write. Every other card already agrees with VTK.
- One-dimensional cards (``CBAR``, ``CBEAM``, ``CROD``, ``CONROD``, ``CTUBE``, ``CBUSH``, ``CGAP``, ``CBEND``, ``CVISC``) read as lines rather than being skipped; ``CONROD`` names a material where the others name a property, so its grid points sit one field earlier.
- A shell's ``ZOFFS`` becomes ``element_attrs["zoffs"]`` when any card carries one, and is written back on the shell cards that have a field for it: ``CTRIA3``, ``CTRIAR``, ``CQUAD4``, ``CQUADR``, ``CTRIA6`` and ``CQUAD8``.
- Nastran and VTK number the corner grid points alike, but not every mid-side one. ``CPENTA`` runs the prism's vertical edges before its top ring and ``CHEXA`` does the same with the brick's, where VTK runs the verticals last; ``CTRIAX6`` interleaves corners and mid-side nodes rather than listing the corners first. All three are permuted on the way in and back on the way out.
- ``CTRIAX6`` names a material in the field where the shells name a property, so its second field is not read as a property id. A ``CBUSH`` grounding its second end names one grid point and no element, and is skipped with a warning rather than sinking the deck.

.. seealso::

   :doc:`index` - the full format table.
