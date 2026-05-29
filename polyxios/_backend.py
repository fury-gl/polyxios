"""
_backend.py — dispatch between compiled _core and pure-Python _core_fallback
=============================================================================

All other modules import from here, never from _core or _core_fallback directly.
"""

try:
    from polyxios._core import (  # type: ignore[import]
        build_csr,
        compact_vertex_indices,
        has_orphan_vertices,
    )

    HAS_CYTHON = True
except ImportError:
    from polyxios._core_fallback import (  # type: ignore[assignment]
        build_csr,
        compact_vertex_indices,
        has_orphan_vertices,
    )

    HAS_CYTHON = False

__all__ = [
    "HAS_CYTHON",
    "build_csr",
    "compact_vertex_indices",
    "has_orphan_vertices",
]
