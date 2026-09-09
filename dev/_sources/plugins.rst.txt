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

Extensions several formats share
--------------------------------

An extension no format owns outright - ``.dat`` belongs to Tecplot, Nastran,
LS-DYNA and plain ASCII tables alike - is resolved by looking inside the file.
Among the codecs polyxios ships, one competing for such an extension lists it
under ``SNIFF_EXTENSIONS`` and supplies ``sniff(head: bytes) -> bool``, which
answers whether the opening bytes look like its format; ``SNIFF_PRIORITY``
orders the attempts, lowest first, so a narrow test runs ahead of a broad one.
The extension then resolves to a dispatcher that delegates to the first codec
recognising the file, and names the candidates when none does. Writing needs an
owner, an output file having no content to inspect: the codec that owns the
extension keeps the writes by declaring ``SNIFF_DEFAULT_WRITER``, and without
one a bare write raises and asks for ``fmt=``.

An entry point is a claim, not a competitor. It registers one extension, it is
registered last, and it replaces whatever held that key - a dispatcher
included: installing a codec for ``.dat`` is a deliberate choice by the person
installing it, and it wins over polyxios' own guess at the file's content.

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
a ``Path`` or refuses the buffer with a message naming why. It refuses a
compressed one too: a codec opening its own files is not going through the
layer that unwraps gzip, so a ``.gz`` there would be parsed as text.

Pass ``reading=True`` on the way in, where the file already holds bytes to
look at, so a file compressed without being renamed is caught by its content
the way it is everywhere else. Leave it unset on the way out: a destination
holds either nothing or the file about to be replaced, and reading that one
would refuse a plain write for the sake of whatever happened to be sitting
there. ``require_path`` only ever sees the path the caller named, so a codec
that then opens siblings of its own has to check those itself - ``is_gzip``
takes a path and answers in four bytes.
