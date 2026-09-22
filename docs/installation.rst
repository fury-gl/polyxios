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
- h5py >= 3.0, for the HDF5 heavy data of :doc:`XDMF <formats/xdmf>` files -
  ``pip install "polyxios[hdf5]"``. Without it the inline and binary flavours
  still read and write, and an HDF5 reference raises
  :class:`~polyxios.exceptions.UnsupportedFormatError` naming the extra.
