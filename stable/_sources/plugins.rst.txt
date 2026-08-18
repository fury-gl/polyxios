Plugin system
=============

.. meta::
   :description: Teach polyxios a new 3D file format from a third-party package. Two functions and an entry point register a codec by extension - no fork and no pull request.


Any third-party package can teach polyxios to read and write a new format - no
fork required, no pull request needed.

Step 1 - write a codec
----------------------

Two functions, nothing more:

.. code-block:: python

    # mypackage/abc_codec.py
    from polyxios._registry import Codec
    from polyxios._types import PolyData

    def read(path, *, lazy=False) -> PolyData:
        ...

    def write(poly: PolyData, path, **opts) -> None:
        ...

    def register():
        return ".abc", Codec(read, write)

Step 2 - declare an entry point
-------------------------------

In your ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."polyxios.codecs"]
    abc = "mypackage.abc_codec:register"

After ``pip install mypackage``, polyxios picks up ``.abc`` automatically - no
configuration, no restart needed:

.. code-block:: python

    mesh = px.read("model.abc")   # works out of the box

.. note::

   Extensions are resolved in lower case, so register ``".abc"`` rather than
   ``".ABC"`` - a key the resolver cannot reach is worse than none.
