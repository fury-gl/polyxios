# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# cython: language_level=3

"""Cython hot-paths for VTP base64 decoding and byte-swapping."""

import base64
import numpy as np
cimport numpy as cnp

cnp.import_array()


cpdef cnp.ndarray decode_base64_array(bytes b64_data, str dtype_str, bint big_endian):
    """Decode a VTP base64-encoded binary data array.

    Parameters
    ----------
    b64_data
        Base64-encoded bytes (with 4-byte header).
    dtype_str
        Numpy dtype string (e.g. 'f8', 'i4').
    big_endian
        If True, byte-swap from big-endian to native.

    Returns
    -------
    np.ndarray
        Decoded array in native byte order.
    """
    raw = base64.b64decode(b64_data)
    # First 4 or 8 bytes = uncompressed length (UInt32 or UInt64)
    if len(raw) <= 4:
        return np.array([], dtype=dtype_str)
    data_bytes = raw[4:]  # skip the 4-byte header
    endian = '>' if big_endian else '<'
    arr = np.frombuffer(data_bytes, dtype=endian + dtype_str)
    return arr.astype(dtype_str)
