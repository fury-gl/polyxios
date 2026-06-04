cimport numpy as cnp

cpdef void build_csr(
    list element_groups,
    int[:] out_connectivity,
    int[:] out_offsets,
    unsigned char[:] out_types,
) except *

cpdef bint has_orphan_vertices(int n_verts, int[:] connectivity) except -1

cpdef int[:] compact_vertex_indices(int[:] connectivity, int n_verts)
