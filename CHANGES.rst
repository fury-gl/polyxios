.. _changes:

=========
Changelog
=========

.. _changes_0.3.0:

0.3.0 (upcoming)
----------------

(No entries yet.)

.. _changes_0.2.0:

0.2.0 (2026-06-25)
------------------

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

GitHub stats for 2026/05/26 - 2026/06/25 (tag: None)

These lists are automatically generated and may be incomplete or contain duplicates.

The following 4 authors contributed 47 commits.

* Maharshi Gor
* Praneeth Shetty
* Serge Koudoro
* skoudoro


We closed a total of 12 issues, 12 pull requests and 0 regular issues.

Pull Requests (12):

* :ghpull:`12`: NF: add MFEM mesh codec (.mesh)
* :ghpull:`9`: NF: Handle the other vtk formats
* :ghpull:`11`: MNT: update pre-commit hooks
* :ghpull:`10`: BF/NF: fix PLY binary reader + add SPLAT codec and compressed 3DGS support
* :ghpull:`8`: BF: handling VTK files improvements
* :ghpull:`4`: Fix: vtk codec to read ascii polydata and support v1.0
* :ghpull:`7`: CI: Avoid cron job on fork
* :ghpull:`6`: MNT: update pre-commit hooks
* :ghpull:`3`: NF: Adding Data Fetcher
* :ghpull:`5`: MNT: update pre-commit hooks
* :ghpull:`2`: DOC: Replace arrow and em-dashes with dash
* :ghpull:`1`: NF: initial framework from polyxios

Issues (0):
