.. _changes:

=========
Changelog
=========

.. _changes_0.1.0:

0.1.0 (upcoming)
-----------------

First public release of **polyxios**.

New features
~~~~~~~~~~~~

- Plugin-based codec registry via Python entry points — third-party packages
  can register mesh formats without patching polyxios.
- VTK legacy (``.vtk``) and XML (``.vtu``, ``.vtp``) reader/writer with
  ASCII and binary (raw + appended) encoding.
- VTR appended format support.
- MFEM mesh codec (``.mesh``).
- MEDIT mesh codec (``.mesh``).
- ``polyxios convert`` and ``polyxios visualize-mesh`` CLI commands.
- Lazy / memory-mapped loading for binary formats (``read(..., lazy=True)``).
- ``polyxios.__version__`` exposes the full version string including the git
  commit hash for development builds (e.g. ``0.1.0.dev0+git20260623.101006a``).
