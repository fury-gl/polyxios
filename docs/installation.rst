Installation
============

.. meta::
   :description: Install polyxios with pip or conda, or build it from source with spin. Requires Python 3.12+ and NumPy; the Cython accelerators are optional and pure-Python fallbacks ship with the package.


pip (recommended)
-----------------

.. code-block:: bash

    pip install polyxios

conda
-----

polyxios is packaged on conda-forge:

.. code-block:: bash

    conda install -c conda-forge polyxios

Development install
-------------------

Clone the repo, then use `spin <https://github.com/scientific-python/spin>`_
to manage the build:

.. code-block:: bash

    pip install spin
    spin setup       # first-time: upstream remote + build deps (libomp on macOS)
    spin install     # compiled install
    spin install -e  # editable install - source changes reflected immediately

Dependencies
------------

- Python >= 3.12
- NumPy >= 1.24

Optional:

- Cython >= 3.0 (compiled hot-paths; pure Python fallbacks included)
- h5py >= 3.0, for the HDF5 heavy data of :doc:`XDMF <formats/xdmf>` files
  and for the formats that are HDF5 files outright - :doc:`MED <formats/med>`,
  :doc:`CGNS <formats/cgns>`, :doc:`H5M <formats/h5m>` and
  :doc:`HMF <formats/hmf>` - ``pip install "polyxios[hdf5]"``. Without it
  XDMF's inline and binary flavours still read and write, and an HDF5 file or
  reference raises :class:`~polyxios.exceptions.UnsupportedFormatError`
  naming the extra.
- netCDF4 >= 1.6, for :doc:`Exodus II <formats/exodus>` files -
  ``pip install "polyxios[netcdf]"``. Without it the format raises
  :class:`~polyxios.exceptions.UnsupportedFormatError` naming the extra.
