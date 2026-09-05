from typing import Final

# Hard caps - prevent unbounded allocation from corrupt headers (meshio #1562)
MAX_SAFE_VERTICES: Final[int] = 500_000_000
MAX_SAFE_ELEMENTS: Final[int] = 2_000_000_000
MAX_SAFE_CONN: Final[int] = 8_000_000_000

# The cap above is the only one a format that describes its points rather than
# writing them can be held to, and on its own it is far too loose for one: an
# ImageData spells its whole geometry in an origin, a step and six indices, so
# a 250-byte file can declare half a billion points and be expanded into the
# twelve gigabytes of vertices they come to. Every other format has to spend
# bytes on a point before the reader allocates one, which is what the file-size
# heuristic in ``validate_header`` weighs; there is no such evidence here, and
# nothing between the header and the allocation but a number.
#
# A hundred million points is past any grid that expands into a mesh a machine
# can work with - the vertices alone are 2.4 GB, and the connectivity of the
# hexahedra over them another 3.2 GB - and far below the point where a header
# a few bytes long can ask for the whole of memory.
MAX_IMPLIED_VERTICES: Final[int] = 100_000_000

# Canonical polyxios element type codes: str name - uint8 code.
# These are polyxios's own codes - do NOT use VTK integers directly.
ELEMENT_TYPES: Final[dict[str, int]] = {
    "empty_cell": 0,
    "vertex": 1,
    "poly_vertex": 2,
    "line": 3,
    "poly_line": 4,
    "triangle": 5,
    "triangle_strip": 6,
    "polygon": 7,
    "pixel": 8,
    "quad": 9,
    "tetra": 10,
    "voxel": 11,
    "hexahedron": 12,
    "wedge": 13,
    "pyramid": 14,
    "pentagonal_prism": 15,
    "hexagonal_prism": 16,
    "quadratic_edge": 17,
    "quadratic_triangle": 18,
    "quadratic_quad": 19,
    "quadratic_polygon": 20,
    "quadratic_tetra": 21,
    "quadratic_hexahedron": 22,
    "quadratic_wedge": 23,
    "quadratic_pyramid": 24,
    "biquadratic_quad": 25,
    "triquadratic_hexahedron": 26,
    "quadratic_linear_quad": 27,
    "quadratic_linear_wedge": 28,
    "biquadratic_quadratic_wedge": 29,
    "biquadratic_quadratic_hexahedron": 30,
    "biquadratic_triangle": 31,
    "cubic_line": 32,
    "convex_point_set": 33,
    "polyhedron": 34,
    "parametric_curve": 35,
    "parametric_surface": 36,
    "parametric_tri_surface": 37,
    "parametric_quad_surface": 38,
    "parametric_tetra_region": 39,
    "parametric_hex_region": 40,
    "higher_order_edge": 41,
    "higher_order_triangle": 42,
    "higher_order_quad": 43,
    "higher_order_polygon": 44,
    "higher_order_tetrahedron": 45,
    "higher_order_wedge": 46,
    "higher_order_pyramid": 47,
    "higher_order_hexahedron": 48,
    "lagrange_curve": 49,
    "lagrange_triangle": 50,
    "lagrange_quadrilateral": 51,
    "lagrange_tetrahedron": 52,
    "lagrange_hexahedron": 53,
    "lagrange_wedge": 54,
    "lagrange_pyramid": 55,
    "bezier_curve": 56,
    "bezier_triangle": 57,
}

# Reverse lookup: uint8 code - str name
ELEMENT_TYPES_INV: Final[dict[int, str]] = {v: k for k, v in ELEMENT_TYPES.items()}

# Node count per element type; -1 means variable-length (uses offsets array).
NODES_PER_ELEMENT: Final[dict[str, int]] = {
    "empty_cell": 0,
    "vertex": 1,
    "poly_vertex": -1,
    "line": 2,
    "poly_line": -1,
    "triangle": 3,
    "triangle_strip": -1,
    "polygon": -1,
    "pixel": 4,
    "quad": 4,
    "tetra": 4,
    "voxel": 8,
    "hexahedron": 8,
    "wedge": 6,
    "pyramid": 5,
    "pentagonal_prism": 10,
    "hexagonal_prism": 12,
    "quadratic_edge": 3,
    "quadratic_triangle": 6,
    "quadratic_quad": 8,
    "quadratic_polygon": -1,
    "quadratic_tetra": 10,
    "quadratic_hexahedron": 20,
    "quadratic_wedge": 15,
    "quadratic_pyramid": 13,
    "biquadratic_quad": 9,
    "triquadratic_hexahedron": 27,
    "quadratic_linear_quad": 6,
    "quadratic_linear_wedge": 12,
    "biquadratic_quadratic_wedge": 18,
    "biquadratic_quadratic_hexahedron": 24,
    "biquadratic_triangle": 7,
    "cubic_line": 4,
    "convex_point_set": -1,
    "polyhedron": -1,
    "parametric_curve": -1,
    "parametric_surface": -1,
    "parametric_tri_surface": -1,
    "parametric_quad_surface": -1,
    "parametric_tetra_region": -1,
    "parametric_hex_region": -1,
    "higher_order_edge": -1,
    "higher_order_triangle": -1,
    "higher_order_quad": -1,
    "higher_order_polygon": -1,
    "higher_order_tetrahedron": -1,
    "higher_order_wedge": -1,
    "higher_order_pyramid": -1,
    "higher_order_hexahedron": -1,
    "lagrange_curve": -1,
    "lagrange_triangle": -1,
    "lagrange_quadrilateral": -1,
    "lagrange_tetrahedron": -1,
    "lagrange_hexahedron": -1,
    "lagrange_wedge": -1,
    "lagrange_pyramid": -1,
    "bezier_curve": -1,
    "bezier_triangle": -1,
}

# VTK type integer - polyxios type name. Covers all VTK types 0–76.
# Any VTK code not in this dict - UnknownElementTypeError, never IndexError/KeyError.
VTK_TO_POLYXIOS: Final[dict[int, str]] = {
    0: "empty_cell",
    1: "vertex",
    2: "poly_vertex",
    3: "line",
    4: "poly_line",
    5: "triangle",
    6: "triangle_strip",
    7: "polygon",
    8: "pixel",
    9: "quad",
    10: "tetra",
    11: "voxel",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
    15: "pentagonal_prism",
    16: "hexagonal_prism",
    # 17-20: undefined in VTK
    21: "quadratic_edge",
    22: "quadratic_triangle",
    23: "quadratic_quad",
    24: "quadratic_tetra",
    25: "quadratic_hexahedron",
    26: "quadratic_wedge",
    27: "quadratic_pyramid",
    28: "biquadratic_quad",
    29: "triquadratic_hexahedron",
    30: "quadratic_linear_quad",
    31: "quadratic_linear_wedge",
    32: "biquadratic_quadratic_wedge",
    33: "biquadratic_quadratic_hexahedron",
    34: "biquadratic_triangle",
    35: "cubic_line",
    36: "quadratic_polygon",
    # 37-40: undefined in VTK
    41: "convex_point_set",
    42: "polyhedron",
    # 43-50: undefined in VTK
    51: "parametric_curve",
    52: "parametric_surface",
    53: "parametric_tri_surface",
    54: "parametric_quad_surface",
    55: "parametric_tetra_region",
    56: "parametric_hex_region",
    # 57-59: undefined in VTK
    60: "higher_order_edge",
    61: "higher_order_triangle",
    62: "higher_order_quad",
    63: "higher_order_polygon",
    64: "higher_order_tetrahedron",
    65: "higher_order_wedge",
    66: "higher_order_pyramid",
    67: "higher_order_hexahedron",
    68: "lagrange_curve",
    69: "lagrange_triangle",
    70: "lagrange_quadrilateral",
    71: "lagrange_tetrahedron",
    72: "lagrange_hexahedron",
    73: "lagrange_wedge",
    74: "lagrange_pyramid",
    75: "bezier_curve",
    76: "bezier_triangle",
}

# Reverse: polyxios name - VTK integer (only for types that have a VTK mapping)
POLYXIOS_TO_VTK: Final[dict[str, int]] = {v: k for k, v in VTK_TO_POLYXIOS.items()}

# Element type codes that represent 2-D surface geometry renderable as triangles.
SURFACE_ELEMENT_TYPES: Final[frozenset[int]] = frozenset(
    {
        ELEMENT_TYPES["triangle"],
        ELEMENT_TYPES["triangle_strip"],
        ELEMENT_TYPES["polygon"],
        ELEMENT_TYPES["pixel"],
        ELEMENT_TYPES["quad"],
        # Quadratic surface elements - linearized to corner nodes for rendering.
        ELEMENT_TYPES["quadratic_triangle"],
        ELEMENT_TYPES["biquadratic_triangle"],
        ELEMENT_TYPES["quadratic_quad"],
        ELEMENT_TYPES["biquadratic_quad"],
    }
)

# Corner node count for quadratic surface elements.
# When rendering, only the first N nodes (corner nodes) are used.
QUADRATIC_SURFACE_CORNERS: Final[dict[int, int]] = {
    ELEMENT_TYPES["quadratic_triangle"]: 3,
    ELEMENT_TYPES["biquadratic_triangle"]: 3,
    ELEMENT_TYPES["quadratic_quad"]: 4,
    ELEMENT_TYPES["biquadratic_quad"]: 4,
}

# Element type codes that represent 1-D line geometry.
LINE_ELEMENT_TYPES: Final[frozenset[int]] = frozenset(
    {
        ELEMENT_TYPES["line"],
        ELEMENT_TYPES["poly_line"],
    }
)

# Topological dimension of each element type: the dimension of the element
# itself, not of the space it sits in. A triangle in 3-D is still 2-D.
# Every code in ELEMENT_TYPES must appear here - a missing entry would read
# as a point cloud, which is wrong in a way nothing downstream can detect.
TOPOLOGICAL_DIMENSION: Final[dict[int, int]] = {
    ELEMENT_TYPES["empty_cell"]: 0,
    ELEMENT_TYPES["vertex"]: 0,
    ELEMENT_TYPES["poly_vertex"]: 0,
    ELEMENT_TYPES["line"]: 1,
    ELEMENT_TYPES["poly_line"]: 1,
    ELEMENT_TYPES["triangle"]: 2,
    ELEMENT_TYPES["triangle_strip"]: 2,
    ELEMENT_TYPES["polygon"]: 2,
    ELEMENT_TYPES["pixel"]: 2,
    ELEMENT_TYPES["quad"]: 2,
    ELEMENT_TYPES["tetra"]: 3,
    ELEMENT_TYPES["voxel"]: 3,
    ELEMENT_TYPES["hexahedron"]: 3,
    ELEMENT_TYPES["wedge"]: 3,
    ELEMENT_TYPES["pyramid"]: 3,
    ELEMENT_TYPES["pentagonal_prism"]: 3,
    ELEMENT_TYPES["hexagonal_prism"]: 3,
    ELEMENT_TYPES["quadratic_edge"]: 1,
    ELEMENT_TYPES["quadratic_triangle"]: 2,
    ELEMENT_TYPES["quadratic_quad"]: 2,
    ELEMENT_TYPES["quadratic_polygon"]: 2,
    ELEMENT_TYPES["quadratic_tetra"]: 3,
    ELEMENT_TYPES["quadratic_hexahedron"]: 3,
    ELEMENT_TYPES["quadratic_wedge"]: 3,
    ELEMENT_TYPES["quadratic_pyramid"]: 3,
    ELEMENT_TYPES["biquadratic_quad"]: 2,
    ELEMENT_TYPES["triquadratic_hexahedron"]: 3,
    ELEMENT_TYPES["quadratic_linear_quad"]: 2,
    ELEMENT_TYPES["quadratic_linear_wedge"]: 3,
    ELEMENT_TYPES["biquadratic_quadratic_wedge"]: 3,
    ELEMENT_TYPES["biquadratic_quadratic_hexahedron"]: 3,
    ELEMENT_TYPES["biquadratic_triangle"]: 2,
    ELEMENT_TYPES["cubic_line"]: 1,
    ELEMENT_TYPES["convex_point_set"]: 3,
    ELEMENT_TYPES["polyhedron"]: 3,
    ELEMENT_TYPES["parametric_curve"]: 1,
    ELEMENT_TYPES["parametric_surface"]: 2,
    ELEMENT_TYPES["parametric_tri_surface"]: 2,
    ELEMENT_TYPES["parametric_quad_surface"]: 2,
    ELEMENT_TYPES["parametric_tetra_region"]: 3,
    ELEMENT_TYPES["parametric_hex_region"]: 3,
    ELEMENT_TYPES["higher_order_edge"]: 1,
    ELEMENT_TYPES["higher_order_triangle"]: 2,
    ELEMENT_TYPES["higher_order_quad"]: 2,
    ELEMENT_TYPES["higher_order_polygon"]: 2,
    ELEMENT_TYPES["higher_order_tetrahedron"]: 3,
    ELEMENT_TYPES["higher_order_wedge"]: 3,
    ELEMENT_TYPES["higher_order_pyramid"]: 3,
    ELEMENT_TYPES["higher_order_hexahedron"]: 3,
    ELEMENT_TYPES["lagrange_curve"]: 1,
    ELEMENT_TYPES["lagrange_triangle"]: 2,
    ELEMENT_TYPES["lagrange_quadrilateral"]: 2,
    ELEMENT_TYPES["lagrange_tetrahedron"]: 3,
    ELEMENT_TYPES["lagrange_hexahedron"]: 3,
    ELEMENT_TYPES["lagrange_wedge"]: 3,
    ELEMENT_TYPES["lagrange_pyramid"]: 3,
    ELEMENT_TYPES["bezier_curve"]: 1,
    ELEMENT_TYPES["bezier_triangle"]: 2,
}


# The faces of every volume element type, as local vertex indices into the
# element's own connectivity, in the order polyxios numbers them. It is one
# table because the two things that need it must agree: extracting a boundary
# surface, and reading a format that names a face by its number - an Abaqus
# ``*Surface`` says ``S3``, and which face that is depends on both numberings.
#
# Every ring is wound so its normal points out of the element, which is what
# lets a skin taken off a mesh be shaded: a face wound the other way is lit
# from inside, and one element's base among five outward sides is the shape
# that reads as a hole. Only the ring order carries this - the face numbering
# is what ``face_index`` and an Abaqus ``S<n>`` are spelled against, and it is
# the same either way.
ELEMENT_FACES: Final[dict[str, tuple[tuple[int, ...], ...]]] = {
    "tetra": (
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
        (0, 2, 1),
    ),
    "hexahedron": (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
    # VTK voxel: bit-encoded ordering differs from hex
    "voxel": (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    ),
    "wedge": (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    ),
    "pyramid": (
        (0, 3, 2, 1),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    ),
    "pentagonal_prism": (
        (0, 4, 3, 2, 1),
        (5, 6, 7, 8, 9),
        (0, 1, 6, 5),
        (1, 2, 7, 6),
        (2, 3, 8, 7),
        (3, 4, 9, 8),
        (4, 0, 5, 9),
    ),
    "hexagonal_prism": (
        (0, 5, 4, 3, 2, 1),
        (6, 7, 8, 9, 10, 11),
        (0, 1, 7, 6),
        (1, 2, 8, 7),
        (2, 3, 9, 8),
        (3, 4, 10, 9),
        (4, 5, 11, 10),
        (5, 0, 6, 11),
    ),
}
# Quadratic elements: reuse corner-node faces of their linear counterparts.
# The same object under both names, which tuples make safe to share.
ELEMENT_FACES["quadratic_tetra"] = ELEMENT_FACES["tetra"]
ELEMENT_FACES["triquadratic_hexahedron"] = ELEMENT_FACES["hexahedron"]
ELEMENT_FACES["biquadratic_quadratic_wedge"] = ELEMENT_FACES["wedge"]
ELEMENT_FACES["quadratic_hexahedron"] = ELEMENT_FACES["hexahedron"]
ELEMENT_FACES["quadratic_wedge"] = ELEMENT_FACES["wedge"]
ELEMENT_FACES["quadratic_pyramid"] = ELEMENT_FACES["pyramid"]
