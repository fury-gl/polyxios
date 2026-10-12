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
    mesh.element_types     # one type code per element

Writing
-------

.. code-block:: python

    px.write(mesh, "out.bdf")

Format-specific options
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56
   :class: px-spec-table

   * - Option
     - Default
     - Effect
   * - ``field_format``
     - ``"free"``
     - ``"free"`` writes comma-separated cards; ``"large"`` writes ``GRID*`` cards in
       large (16-column) field form. Element cards stay free field either way.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- All three field widths are read; writing emits free field, with large-field ``GRID`` cards on request (``field_format="large"``).
- ``.dat`` is shared with Tecplot, LS-DYNA and plain ASCII tables, so it is resolved by looking
  inside the file: a ``$`` comment banner is stepped over and a ``GRID`` card, ``CEND``,
  ``BEGIN BULK``, ``NASTRAN`` or a ``SOL`` naming a solution number or name lands the deck
  here - a numeric table whose first row happens to read ``SOL 1.0 2.0`` does not.
  ``px.read("model.dat", fmt=".bdf")`` still forces the issue, and writing to ``.dat`` needs
  ``fmt=".bdf"`` because an output file has no content to inspect.
- Free-field coordinates are written with the shortest text that round-trips the double exactly, so polyxios reads them back bit for bit; a magnitude below one whose short spelling needs an exponent goes out in fixed point instead. A solver keeps only the first eight characters of a free-field entry, so a longer value is read rounded there, with a warning on write. ``field_format="large"`` fits each coordinate in sixteen columns, using the implicit-exponent shorthand the format allows (``1.234-10`` for ``1.234E-10``) when it saves digits; a value that still does not fit is rounded on the way out, with a warning.
- Property ids land in ``element_attrs["pid"]`` and each distinct one becomes an element tag ``pid_<id>``, so the deck's grouping is preserved.
- A card name does not say the element's order, so the shape comes from the grid points the card carries: ``CTETRA`` with ten grid points is a ``quadratic_tetra``, with four a ``tetra``. Counting stops at the first blank field, because Nastran lets a mid-side grid point be omitted and that card is a linear element.
- The axisymmetric cards read as their planar counterparts: ``CTRAX3``, ``CTRAX6``, ``CTRIAX6``, ``CQUADX``, ``CQUADX4`` and ``CQUADX8``, alongside ``CSHEAR``.
- One-dimensional cards (``CBAR``, ``CBEAM``, ``CROD``, ``CONROD``, ``CTUBE``, ``CBUSH``, ``CGAP``, ``CBEND``, ``CVISC``) read as lines rather than being skipped; ``CONROD`` names a material where the others name a property, so its grid points sit one field earlier.
- A shell's ``ZOFFS`` becomes ``element_attrs["zoffs"]`` when any card carries one, and is written back on the shell cards that have a field for it: ``CTRIA3``, ``CTRIAR``, ``CQUAD4``, ``CQUADR``, ``CTRIA6`` and ``CQUAD8``.
- Nastran and VTK number the corner grid points alike, but not every mid-side one. ``CPENTA`` runs the prism's vertical edges before its top ring and ``CHEXA`` does the same with the brick's, where VTK runs the verticals last; ``CTRIAX6`` interleaves corners and mid-side nodes rather than listing the corners first. All three are permuted on the way in; ``CPENTA`` and ``CHEXA`` are permuted back on the way out, while a quadratic triangle is always written as ``CTRIA6``, whose order already agrees with VTK.
- ``CTRIAX6`` names a material in the field where the shells name a property, so its second field is not read as a property id. A ``CBUSH`` grounding its second end names one grid point and no element, and is skipped with a warning of its own rather than being counted among the cards this codec has no shape for.

.. seealso::

   :doc:`index` - the full format table.
