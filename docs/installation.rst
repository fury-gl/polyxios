Installation
============

pip (recommended)
-----------------

.. code-block:: bash

    pip install polyxios

Development install
-------------------

Clone the repo, then use `spin <https://github.com/scientific-python/spin>`_
to manage the build:

.. code-block:: bash

    pip install spin
    spin setup       # first-time: upstream remote + build deps (libomp on macOS)
    spin install     # compiled install
    spin install -e  # editable install — source changes reflected immediately

Dependencies
------------

- Python >= 3.11
- NumPy >= 1.24

Optional:

- Cython >= 3.0 (compiled hot-paths; pure Python fallbacks included)
