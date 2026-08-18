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

Reading a path or a buffer
--------------------------

``path`` is whatever the caller passed ``read()`` or ``write()``: a path, or
an open file object. Opening it yourself would refuse the second, so use the
helpers in :mod:`polyxios._io`, which take either:

.. code-block:: python

    from polyxios._io import read_text, write_text

    def read(path, *, lazy=False) -> PolyData:
        text = read_text(path, encoding="utf-8", errors="replace")
        ...

    def write(poly: PolyData, path, **opts) -> None:
        write_text(path, "\n".join(lines), encoding="utf-8")

``read_bytes`` / ``write_bytes`` are the binary pair, ``open_read`` /
``open_write`` yield a binary handle for a codec that streams, ``open_text``
yields one to iterate line by line, and ``open_block`` gives a binary parser
the whole file at once - mapped for a path, read into memory for a buffer.
``source_name`` and ``source_suffix`` are what an error message should quote,
since a buffer has no ``Path`` to ask.

Going through the helpers is also what gives a codec gzip for free: the
decompression happens inside them, so a codec that opened the path itself
would be the only format in the library that cannot read a ``.gz``.

A codec that genuinely needs a file on disk - a format split across sibling
files, say - calls ``require_path(path, fmt=..., reason=...)``, which returns
a ``Path`` or refuses the buffer with a message naming why.
