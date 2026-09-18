.. _format-svg:

SVG
===

.. rst-class:: px-badges

``.svg`` ``write only``

Summary of the specification
----------------------------

Scalable Vector Graphics is the W3C's XML picture format: a ``<svg>`` root whose ``viewBox`` names the drawing's coordinate extent and whose ``width`` and ``height``, when present, say how large to render it. Shapes are children of the root; a ``<path>`` is spelled by its ``d`` attribute as a sequence of commands - ``M`` moves, ``L`` draws a line, ``Z`` closes the outline - and a ``<g>`` groups shapes under shared presentation attributes such as ``stroke`` and ``fill``. The y axis runs downward. Coordinates are unitless user space, scaled to the rendered size through the ``viewBox``.

An SVG is a picture, not a mesh. It carries no third coordinate, no element type and no attribute, so the mesh cannot be recovered from it: polyxios writes the format and refuses to read it.

Specification at a glance
-------------------------

.. list-table::
   :widths: 28 72
   :class: px-spec-table

   * - structure
     - XML: ``<svg viewBox="…">`` holding one ``<g>`` of ``<path>`` elements
   * - element record
     - ``<path class="<type>" d="M x y L x y … Z">``, one per element type
   * - coordinates
     - the mesh's own units, y mirrored, the top-left corner at the origin
   * - style
     - ``fill="none" stroke="black" stroke-width="…"`` on the group
   * - size
     - ``viewBox`` always; ``width`` / ``height`` on request

.. rst-class:: px-speclink

`SVG 1.1 (W3C) ↗ <https://www.w3.org/TR/SVG11/>`__

Reading
-------

Not supported. :func:`polyxios.read` raises :class:`~polyxios.exceptions.UnsupportedFormatError` naming the file, so a pipeline that tries to round-trip through ``.svg`` fails at once rather than at the next step.

Writing
-------

.. code-block:: python

    import polyxios as px

    px.write(mesh, "out.svg")
    px.write(mesh, "out.svg", plane="xz", width=800, stroke_width=0.02)
    px.write(mesh, "out.svg", float_fmt=".3f")

``plane`` picks the projection: ``"xy"`` (the default), ``"xz"`` or ``"yz"``, the first axis running rightward and the second upward. ``width`` sets the rendered width in CSS pixels, the height following from the mesh's aspect ratio; without it the picture fills whatever displays it. ``stroke_width`` is the line weight in the mesh's own units and defaults to a hundredth of the picture's shorter side. ``float_fmt`` is the format specifier for every number written, ``.6g`` by default; one whose fill character is an XML delimiter (``"``, ``<``, ``>``, ``&``) is refused, since every number lands in an attribute. Any other option is warned about and ignored.

Quirks worth knowing
--------------------

.. rst-class:: px-quirks

- Coordinates go out in the mesh's own units, shifted so the picture's top-left corner is the origin and the upward axis mirrored, since SVG's y grows downward; the ``viewBox`` is the mesh's extent padded by half a stroke so an outline on the boundary is not clipped. Passing ``width`` scales the picture without touching the coordinates.
- Lines and faces are drawn: a line, polyline, triangle, quadrilateral, pixel, polygon or triangle strip, and the higher-order kinds of each through their corner nodes, a triangle strip as one ring per triangle. Every other element type - a vertex, a solid cell - is skipped with a warning, one per type, so a volume mesh writes as an empty picture with a warning per cell type; extract its surface first. An element of a free-size type with too few nodes to outline is skipped the same way. A vertex coordinate that is not finite is refused outright: it would leave the ``viewBox`` undefined and the whole picture with it.
- Each drawn type is one ``<path>`` carrying the type's name as its ``class``, its elements as subpaths in mesh order, so a stylesheet can colour triangles apart from quadrilaterals. ``vertex_attrs``, ``element_attrs``, the tags and ``global_attrs`` have no spelling in a picture and are not written.
- The coordinate the projection drops is expected to be constant across the mesh. When it is not, the picture is still written and a warning says which axis varies and that the drawing is a projection. A flat mesh off the origin's plane - every vertex at ``z=7`` - is flat and warns about nothing.
- A picture with no extent in one direction - a single horizontal line - takes its stroke from the other; one with no extent at all takes a stroke of one unit. Both still get a ``viewBox`` with area, from the padding.
- ``.svg.gz`` is written compressed like every other format. ``.svgz``, the conventional name for a gzipped SVG, is not registered.

.. seealso::

   :doc:`index` - the full format table.
