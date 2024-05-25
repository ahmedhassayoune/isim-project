import heapq
from enum import Enum
from queue import Queue
from typing import Union
from uuid import uuid4

import bmesh
import bpy
import numpy as np
from mathutils import Vector, bvhtree

turbo_colormap = [
    (0.18995, 0.07176, 0.23217),
    (0.19483, 0.08339, 0.26149),
    (0.19956, 0.09498, 0.29024),
    (0.20415, 0.10652, 0.31844),
    (0.20860, 0.11802, 0.34607),
    (0.21291, 0.12947, 0.37314),
    (0.21708, 0.14087, 0.39964),
    (0.22111, 0.15223, 0.42558),
    (0.22500, 0.16354, 0.45096),
    (0.22875, 0.17481, 0.47578),
    (0.23236, 0.18603, 0.50004),
    (0.23582, 0.19720, 0.52373),
    (0.23915, 0.20833, 0.54686),
    (0.24234, 0.21941, 0.56942),
    (0.24539, 0.23044, 0.59142),
    (0.24830, 0.24143, 0.61286),
    (0.25107, 0.25237, 0.63374),
    (0.25369, 0.26327, 0.65406),
    (0.25618, 0.27412, 0.67381),
    (0.25853, 0.28492, 0.69300),
    (0.26074, 0.29568, 0.71162),
    (0.26280, 0.30639, 0.72968),
    (0.26473, 0.31706, 0.74718),
    (0.26652, 0.32768, 0.76412),
    (0.26816, 0.33825, 0.78050),
    (0.26967, 0.34878, 0.79631),
    (0.27103, 0.35926, 0.81156),
    (0.27226, 0.36970, 0.82624),
    (0.27334, 0.38008, 0.84037),
    (0.27429, 0.39043, 0.85393),
    (0.27509, 0.40072, 0.86692),
    (0.27576, 0.41097, 0.87936),
    (0.27628, 0.42118, 0.89123),
    (0.27667, 0.43134, 0.90254),
    (0.27691, 0.44145, 0.91328),
    (0.27701, 0.45152, 0.92347),
    (0.27698, 0.46153, 0.93309),
    (0.27680, 0.47151, 0.94214),
    (0.27648, 0.48144, 0.95064),
    (0.27603, 0.49132, 0.95857),
    (0.27543, 0.50115, 0.96594),
    (0.27469, 0.51094, 0.97275),
    (0.27381, 0.52069, 0.97899),
    (0.27273, 0.53040, 0.98461),
    (0.27106, 0.54015, 0.98930),
    (0.26878, 0.54995, 0.99303),
    (0.26592, 0.55979, 0.99583),
    (0.26252, 0.56967, 0.99773),
    (0.25862, 0.57958, 0.99876),
    (0.25425, 0.58950, 0.99896),
    (0.24946, 0.59943, 0.99835),
    (0.24427, 0.60937, 0.99697),
    (0.23874, 0.61931, 0.99485),
    (0.23288, 0.62923, 0.99202),
    (0.22676, 0.63913, 0.98851),
    (0.22039, 0.64901, 0.98436),
    (0.21382, 0.65886, 0.97959),
    (0.20708, 0.66866, 0.97423),
    (0.20021, 0.67842, 0.96833),
    (0.19326, 0.68812, 0.96190),
    (0.18625, 0.69775, 0.95498),
    (0.17923, 0.70732, 0.94761),
    (0.17223, 0.71680, 0.93981),
    (0.16529, 0.72620, 0.93161),
    (0.15844, 0.73551, 0.92305),
    (0.15173, 0.74472, 0.91416),
    (0.14519, 0.75381, 0.90496),
    (0.13886, 0.76279, 0.89550),
    (0.13278, 0.77165, 0.88580),
    (0.12698, 0.78037, 0.87590),
    (0.12151, 0.78896, 0.86581),
    (0.11639, 0.79740, 0.85559),
    (0.11167, 0.80569, 0.84525),
    (0.10738, 0.81381, 0.83484),
    (0.10357, 0.82177, 0.82437),
    (0.10026, 0.82955, 0.81389),
    (0.09750, 0.83714, 0.80342),
    (0.09532, 0.84455, 0.79299),
    (0.09377, 0.85175, 0.78264),
    (0.09287, 0.85875, 0.77240),
    (0.09267, 0.86554, 0.76230),
    (0.09320, 0.87211, 0.75237),
    (0.09451, 0.87844, 0.74265),
    (0.09662, 0.88454, 0.73316),
    (0.09958, 0.89040, 0.72393),
    (0.10342, 0.89600, 0.71500),
    (0.10815, 0.90142, 0.70599),
    (0.11374, 0.90673, 0.69651),
    (0.12014, 0.91193, 0.68660),
    (0.12733, 0.91701, 0.67627),
    (0.13526, 0.92197, 0.66556),
    (0.14391, 0.92680, 0.65448),
    (0.15323, 0.93151, 0.64308),
    (0.16319, 0.93609, 0.63137),
    (0.17377, 0.94053, 0.61938),
    (0.18491, 0.94484, 0.60713),
    (0.19659, 0.94901, 0.59466),
    (0.20877, 0.95304, 0.58199),
    (0.22142, 0.95692, 0.56914),
    (0.23449, 0.96065, 0.55614),
    (0.24797, 0.96423, 0.54303),
    (0.26180, 0.96765, 0.52981),
    (0.27597, 0.97092, 0.51653),
    (0.29042, 0.97403, 0.50321),
    (0.30513, 0.97697, 0.48987),
    (0.32006, 0.97974, 0.47654),
    (0.33517, 0.98234, 0.46325),
    (0.35043, 0.98477, 0.45002),
    (0.36581, 0.98702, 0.43688),
    (0.38127, 0.98909, 0.42386),
    (0.39678, 0.99098, 0.41098),
    (0.41229, 0.99268, 0.39826),
    (0.42778, 0.99419, 0.38575),
    (0.44321, 0.99551, 0.37345),
    (0.45854, 0.99663, 0.36140),
    (0.47375, 0.99755, 0.34963),
    (0.48879, 0.99828, 0.33816),
    (0.50362, 0.99879, 0.32701),
    (0.51822, 0.99910, 0.31622),
    (0.53255, 0.99919, 0.30581),
    (0.54658, 0.99907, 0.29581),
    (0.56026, 0.99873, 0.28623),
    (0.57357, 0.99817, 0.27712),
    (0.58646, 0.99739, 0.26849),
    (0.59891, 0.99638, 0.26038),
    (0.61088, 0.99514, 0.25280),
    (0.62233, 0.99366, 0.24579),
    (0.63323, 0.99195, 0.23937),
    (0.64362, 0.98999, 0.23356),
    (0.65394, 0.98775, 0.22835),
    (0.66428, 0.98524, 0.22370),
    (0.67462, 0.98246, 0.21960),
    (0.68494, 0.97941, 0.21602),
    (0.69525, 0.97610, 0.21294),
    (0.70553, 0.97255, 0.21032),
    (0.71577, 0.96875, 0.20815),
    (0.72596, 0.96470, 0.20640),
    (0.73610, 0.96043, 0.20504),
    (0.74617, 0.95593, 0.20406),
    (0.75617, 0.95121, 0.20343),
    (0.76608, 0.94627, 0.20311),
    (0.77591, 0.94113, 0.20310),
    (0.78563, 0.93579, 0.20336),
    (0.79524, 0.93025, 0.20386),
    (0.80473, 0.92452, 0.20459),
    (0.81410, 0.91861, 0.20552),
    (0.82333, 0.91253, 0.20663),
    (0.83241, 0.90627, 0.20788),
    (0.84133, 0.89986, 0.20926),
    (0.85010, 0.89328, 0.21074),
    (0.85868, 0.88655, 0.21230),
    (0.86709, 0.87968, 0.21391),
    (0.87530, 0.87267, 0.21555),
    (0.88331, 0.86553, 0.21719),
    (0.89112, 0.85826, 0.21880),
    (0.89870, 0.85087, 0.22038),
    (0.90605, 0.84337, 0.22188),
    (0.91317, 0.83576, 0.22328),
    (0.92004, 0.82806, 0.22456),
    (0.92666, 0.82025, 0.22570),
    (0.93301, 0.81236, 0.22667),
    (0.93909, 0.80439, 0.22744),
    (0.94489, 0.79634, 0.22800),
    (0.95039, 0.78823, 0.22831),
    (0.95560, 0.78005, 0.22836),
    (0.96049, 0.77181, 0.22811),
    (0.96507, 0.76352, 0.22754),
    (0.96931, 0.75519, 0.22663),
    (0.97323, 0.74682, 0.22536),
    (0.97679, 0.73842, 0.22369),
    (0.98000, 0.73000, 0.22161),
    (0.98289, 0.72140, 0.21918),
    (0.98549, 0.71250, 0.21650),
    (0.98781, 0.70330, 0.21358),
    (0.98986, 0.69382, 0.21043),
    (0.99163, 0.68408, 0.20706),
    (0.99314, 0.67408, 0.20348),
    (0.99438, 0.66386, 0.19971),
    (0.99535, 0.65341, 0.19577),
    (0.99607, 0.64277, 0.19165),
    (0.99654, 0.63193, 0.18738),
    (0.99675, 0.62093, 0.18297),
    (0.99672, 0.60977, 0.17842),
    (0.99644, 0.59846, 0.17376),
    (0.99593, 0.58703, 0.16899),
    (0.99517, 0.57549, 0.16412),
    (0.99419, 0.56386, 0.15918),
    (0.99297, 0.55214, 0.15417),
    (0.99153, 0.54036, 0.14910),
    (0.98987, 0.52854, 0.14398),
    (0.98799, 0.51667, 0.13883),
    (0.98590, 0.50479, 0.13367),
    (0.98360, 0.49291, 0.12849),
    (0.98108, 0.48104, 0.12332),
    (0.97837, 0.46920, 0.11817),
    (0.97545, 0.45740, 0.11305),
    (0.97234, 0.44565, 0.10797),
    (0.96904, 0.43399, 0.10294),
    (0.96555, 0.42241, 0.09798),
    (0.96187, 0.41093, 0.09310),
    (0.95801, 0.39958, 0.08831),
    (0.95398, 0.38836, 0.08362),
    (0.94977, 0.37729, 0.07905),
    (0.94538, 0.36638, 0.07461),
    (0.94084, 0.35566, 0.07031),
    (0.93612, 0.34513, 0.06616),
    (0.93125, 0.33482, 0.06218),
    (0.92623, 0.32473, 0.05837),
    (0.92105, 0.31489, 0.05475),
    (0.91572, 0.30530, 0.05134),
    (0.91024, 0.29599, 0.04814),
    (0.90463, 0.28696, 0.04516),
    (0.89888, 0.27824, 0.04243),
    (0.89298, 0.26981, 0.03993),
    (0.88691, 0.26152, 0.03753),
    (0.88066, 0.25334, 0.03521),
    (0.87422, 0.24526, 0.03297),
    (0.86760, 0.23730, 0.03082),
    (0.86079, 0.22945, 0.02875),
    (0.85380, 0.22170, 0.02677),
    (0.84662, 0.21407, 0.02487),
    (0.83926, 0.20654, 0.02305),
    (0.83172, 0.19912, 0.02131),
    (0.82399, 0.19182, 0.01966),
    (0.81608, 0.18462, 0.01809),
    (0.80799, 0.17753, 0.01660),
    (0.79971, 0.17055, 0.01520),
    (0.79125, 0.16368, 0.01387),
    (0.78260, 0.15693, 0.01264),
    (0.77377, 0.15028, 0.01148),
    (0.76476, 0.14374, 0.01041),
    (0.75556, 0.13731, 0.00942),
    (0.74617, 0.13098, 0.00851),
    (0.73661, 0.12477, 0.00769),
    (0.72686, 0.11867, 0.00695),
    (0.71692, 0.11268, 0.00629),
    (0.70680, 0.10680, 0.00571),
    (0.69650, 0.10102, 0.00522),
    (0.68602, 0.09536, 0.00481),
    (0.67535, 0.08980, 0.00449),
    (0.66449, 0.08436, 0.00424),
    (0.65345, 0.07902, 0.00408),
    (0.64223, 0.07380, 0.00401),
    (0.63082, 0.06868, 0.00401),
    (0.61923, 0.06367, 0.00410),
    (0.60746, 0.05878, 0.00427),
    (0.59550, 0.05399, 0.00453),
    (0.58336, 0.04931, 0.00486),
    (0.57103, 0.04474, 0.00529),
    (0.55852, 0.04028, 0.00579),
    (0.54583, 0.03593, 0.00638),
    (0.53295, 0.03169, 0.00705),
    (0.51989, 0.02756, 0.00780),
    (0.50664, 0.02354, 0.00863),
    (0.49321, 0.01963, 0.00955),
    (0.47960, 0.01583, 0.01055),
]


class Rotation(Enum):
    """Enum class for rotation."""

    NONE = 0
    CW = 1
    CCW = -1


class MyHeap(object):
    """Wrapper class for a heap with custom key function."""

    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        self.index = 0
        self._data = []
        if initial:
            for i, item in enumerate(initial):
                elem_uuid = uuid4().bytes
                heap_elem = (key(item), i, elem_uuid, item)
                item[UID_LAYER] = elem_uuid
                HEAP_ELEM_OCC[elem_uuid] = 1
                self._data.append(heap_elem)
            self.index = len(self._data)
            heapq.heapify(self._data)

    def push(self, item: bmesh.types.BMFace):
        # if len(item.verts) != 4:  # TODO: Should not happen
        #     return None
        if item[UID_LAYER] == 0:  # New item
            elem_uuid = uuid4().bytes
            item[UID_LAYER] = elem_uuid
            heap_elem = (self.key(item), self.index, elem_uuid, item)
            HEAP_ELEM_OCC[elem_uuid] = 1
        else:
            heap_elem = (self.key(item), self.index, item[UID_LAYER], item)
            HEAP_ELEM_OCC[item[UID_LAYER]] += 1

        heapq.heappush(self._data, heap_elem)
        self.index += 1

    def pop(self):
        elem = heapq.heappop(self._data)
        elem_uuid = elem[2]
        HEAP_ELEM_OCC[elem_uuid] -= 1
        return elem


def distance_vec(point1: Vector, point2: Vector) -> float:
    """Calculate distance between two points."""
    return (point2 - point1).length


def compute_min_diagonal_length(face: bmesh.types.BMFace) -> float:
    """Compute the minimum diagonal length of the given quad face."""
    if len(face.verts) != 4:
        if VERBOSE:
            print("Warning: Face is not a quad")
        return float("inf")
    v1, v2, v3, v4 = face.verts
    diag1_len = distance_vec(v1.co, v3.co)
    diag2_len = distance_vec(v2.co, v4.co)
    return min(diag1_len, diag2_len)


def compute_barycentric_coordinates(
    point: Vector, triangle: list[bmesh.types.BMVert]
) -> bool:
    """
    Check if a point is inside a triangle and return the Barycentric Coordinate System.
    Warning: The function assumes that the point is on the plane of the triangle.
        -> If the point is not on the plane, the function can return True.

    Parameters:
    - point: The point to check.
    - triangle: List of 3 vertices representing the triangle.

    Returns:
    - True if the point is inside the triangle, False otherwise.
    """
    triangle = np.array([v.co for v in triangle])
    point = np.array(point)

    # Compute the vectors from the first vertex of the triangle to the other vertices
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    dot20 = np.dot(v2, v0)
    dot21 = np.dot(v2, v1)

    # Compute barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    alpha = (dot11 * dot20 - dot01 * dot21) * inv_denom
    beta = (dot00 * dot21 - dot01 * dot20) * inv_denom
    gamma = 1 - alpha - beta

    inside = (alpha >= 0) and (beta >= 0) and (alpha + beta <= 1)

    return inside, gamma, alpha, beta


def interpolate_fitmap(face: bmesh.types.BMFace, point: Vector, fitmap_layer: str):
    """Interpolate the fitmap value of the given face at the given point."""
    # Fetch projected point and associated face on the source mesh M0 by casting 2 opposite rays
    projected, _, face_idx, dist = BVH.ray_cast(point, face.normal.normalized())
    projected_opp, _, face_idx_opp, dist_opp = (
        BVH.ray_cast(  # TODO: pas sur sur les directions
            point, -face.normal.normalized()
        )
    )

    # Handle case where no hit position is found
    if not projected and not projected_opp:
        raise Exception("Warning: No hit position or normal")

    if dist is None:
        dist = float("inf")
    if dist_opp is None:
        dist_opp = float("inf")
    # Get the closest hit position to the mid-vertex
    if dist_opp < dist:
        projected = projected_opp
        face_idx = face_idx_opp

    INITIAL_MESH.faces.ensure_lookup_table()
    target_face = INITIAL_MESH.faces[face_idx]
    if VERBOSE and len(target_face.verts) != 4:
        raise Exception("Warning: Face is not a quad")

    tri_ABC = [
        target_face.verts[0],
        target_face.verts[1],
        target_face.verts[2],
    ]
    tri_ACD = [
        target_face.verts[0],
        target_face.verts[2],
        target_face.verts[3],
    ]
    tri_ABD = [
        target_face.verts[0],
        target_face.verts[1],
        target_face.verts[3],
    ]
    tri_BCD = [
        target_face.verts[1],
        target_face.verts[2],
        target_face.verts[3],
    ]
    triangles = [tri_ABC, tri_ACD, tri_ABD, tri_BCD]

    for triangle in triangles:
        inside, *scalars = compute_barycentric_coordinates(projected, triangle)
        if inside:
            return (
                scalars[0] * triangle[0][fitmap_layer]
                + scalars[1] * triangle[1][fitmap_layer]
                + scalars[2] * triangle[2][fitmap_layer]
            )

    # Return last triangle in any case <-- Not reached normalement
    return (
        scalars[0] * triangles[-1][0][fitmap_layer]
        + scalars[1] * triangles[-1][1][fitmap_layer]
        + scalars[2] * triangles[-1][2][fitmap_layer]
    )


def compute_priority(face: bmesh.types.BMFace) -> float:
    """Compute the priority of the given face."""
    if len(face.verts) != 4:  # TODO: is this condition necessary in all code
        if VERBOSE:
            print("Warning: Face is not a quad")
        return float("inf")
    v1, v2, v3, v4 = face.verts
    diag1_len = distance_vec(v1.co, v3.co)
    diag2_len = distance_vec(v2.co, v4.co)

    # Compute the center point of the shortest diagonal
    if diag1_len < diag2_len:
        diag_len = diag1_len
        center = (v1.co + v3.co) / 2
    else:
        diag_len = diag2_len
        center = (v2.co + v4.co) / 2

    interpolated_sfitmap = interpolate_fitmap(face, center, SFITMAP_LAYER)

    return diag_len * interpolated_sfitmap


def build_bvh_tree():
    """Build a BVH tree for the given mesh."""
    global BVH
    BVH = bvhtree.BVHTree.FromBMesh(INITIAL_MESH)


def build_mesh_heap(mesh: bmesh.types.BMesh) -> list:
    """Build a heap with the faces of the given mesh. Assign a unique id to each face."""
    global UID_LAYER
    UID_LAYER = mesh.faces.layers.string.new("uid")

    global HEAP_ELEM_OCC
    HEAP_ELEM_OCC = dict()

    global HEAP
    faces = list(mesh.faces)
    HEAP = MyHeap(initial=faces, key=lambda face: compute_priority(face))


def get_neighbor_vert_from_pos(
    vert: bmesh.types.BMVert, pos: Vector
) -> bmesh.types.BMVert:
    """Get the neighbor vertex of the given vertex closest to the given position."""
    min_vert, min_dist = None, float("inf")
    for edge in vert.link_edges:
        other_vert = edge.other_vert(vert)
        if pos == other_vert.co:
            return other_vert
        if (dist := distance_vec(other_vert.co, pos)) < min_dist:
            min_dist, min_vert = dist, other_vert
    print("Warning: No exact neighbor found")
    return min_vert


def remove_doublets(
    bm: bmesh.types.BMesh,
    verts: list[bmesh.types.BMVert],
):
    """Remove doublets in given verts and update the heap."""
    clean_faces = []

    q = Queue()
    # Push all doublet vertices to the queue
    for vert in verts:
        if len(vert.link_edges) == 2:
            q.put(vert)

    while not q.empty():
        vert = q.get()

        # Fetch adjacent vertices
        av1, av2 = (
            vert.link_edges[0].other_vert(vert),
            vert.link_edges[1].other_vert(vert),
        )

        # Remove doublet
        region = bmesh.ops.dissolve_faces(bm, faces=vert.link_faces, use_verts=True)
        faces = region["region"]

        # Push adjacent vertices to the queue if they are doublets
        if av1.is_valid and len(av1.link_edges) == 2:
            q.put(av1)
        if av2.is_valid and len(av2.link_edges) == 2:
            q.put(av2)

        if faces:
            clean_faces.append(faces[0])

    # Keep only valid clean faces
    valid_clean_faces = []
    for face in clean_faces:
        if face.is_valid:
            valid_clean_faces.append(face)
            HEAP.push(face)

    return valid_clean_faces


def clean_local_zone(bm: bmesh.types.BMesh, verts: list[bmesh.types.BMVert]):
    # Remove doublets recursively
    clean_faces = remove_doublets(bm, verts)
    if not clean_faces:
        return clean_faces

    # Remove potiential generated singlets or other degenerates
    # e.g. edges w/o length, faces w/o area ...
    for cface in clean_faces:
        if cface.is_valid:
            bmesh.ops.dissolve_degenerate(bm, edges=cface.edges)

    # Make sure we keep only valid faces
    valid_clean_faces = [f for f in clean_faces if f.is_valid]

    return valid_clean_faces


def push_updated_faces(faces: list[bmesh.types.BMesh], out_index=None):
    """Push updated faces to the heap except the one with the given index."""
    for face in faces:
        if out_index and face.index == out_index:
            continue
        HEAP.push(face)


def allow_collapse(face: bmesh.types.BMFace) -> bool:
    """Check if the given face is allowed to collapse based on M-Fitmap on surrounding faces"""
    # Get surrounding faces
    visited = set()
    visited.add(face.index)
    surrounding_faces = []
    for vert in face.verts:
        for f in vert.link_faces:
            if f.index not in visited:
                visited.add(f.index)
                surrounding_faces.append(f)

    for sface in surrounding_faces:
        center = sface.calc_center_median()
        radius = max([distance_vec(center, v.co) for v in sface.verts])

        interpolated_mfitmap = interpolate_fitmap(face, center, MFITMAP_LAYER)

        if radius >= interpolated_mfitmap:
            return False
    return True


def collapse_diagonal(mesh: bmesh.types.BMesh, face: bmesh.types.BMFace):
    """Collapse the shortest diagonal of the given quad face."""
    v1, v2, v3, v4 = face.verts
    diag1_len, diag2_len = (
        distance_vec(v1.co, v3.co),
        distance_vec(v2.co, v4.co),
    )
    # Apply diagonal collapse on mid-point of the shortest diagonal
    if diag1_len < diag2_len:
        mid_position = (v1.co + v3.co) / 2
        diag_verts = [v1, v3]
        neighbor_vert = v2
    else:
        mid_position = (v2.co + v4.co) / 2
        diag_verts = [v2, v4]
        neighbor_vert = v1

    # Cast 2 opposite rays from the mid-vertex to find the hit positions on source mesh M0
    hit_pos, _, _, dist = BVH.ray_cast(mid_position, face.normal)
    hit_pos_opp, _, _, dist_opp = BVH.ray_cast(mid_position, -face.normal)

    # Handle case where no hit position is found
    if not hit_pos and not hit_pos_opp:
        raise Exception("Warning: No hit position or normal")

    if dist is None:
        dist = float("inf")
    if dist_opp is None:
        dist_opp = float("inf")
    # Get the closest hit position to the mid-vertex
    if dist_opp < dist:
        hit_pos = hit_pos_opp

    # Project mid-vertex on the hit position
    bmesh.ops.pointmerge(mesh, verts=diag_verts, merge_co=hit_pos)
    mesh.faces.ensure_lookup_table()

    merged_vert = get_neighbor_vert_from_pos(neighbor_vert, hit_pos)
    return merged_vert


def compute_energy(edge: bmesh.types.BMEdge, new_edge: list[bmesh.types.BMVert] = None):
    """Compute the energy of the given edge."""
    verts = get_unique_verts(edge.link_faces)

    # Transform to a list valence of each vertex
    valences = np.array([len(v.link_edges) for v in verts])

    if new_edge:
        # Update valence
        vA_edge, vB_edge = new_edge
        for i in range(valences.size):
            if verts[i].index == vA_edge.index or verts[i].index == vB_edge.index:
                # Increase valence of the new edge vertices
                valences[i] += 1
            elif (
                verts[i].index == edge.verts[0].index
                or verts[i].index == edge.verts[1].index
            ):
                # Decrease valence of the old edge vertices
                valences[i] -= 1

    # Compute the energy
    return np.sum(np.abs(valences - 4))


def is_valid_rotation(
    mid_vert: bmesh.types.BMVert,
    old_edge: bmesh.types.BMEdge,
    new_edge: list[bmesh.types.BMVert],
) -> bool:
    """Check if the rotation of the given edge is valid."""
    if not mid_vert.is_valid:
        return False
    if len(old_edge.link_faces) != 2:
        return False

    faceA, faceB = old_edge.link_faces
    if len(faceA.verts) != 4 or len(faceB.verts) != 4:
        return False

    iAmid_vert, iBmid_vert = 0, 0
    for i in range(4):
        vA, vB = faceA.verts[i], faceB.verts[i]
        if vA.index == mid_vert.index:
            iAmid_vert = i
        if vB.index == mid_vert.index:
            iBmid_vert = i

    target_vert = old_edge.other_vert(mid_vert)

    if faceA.verts[(iAmid_vert + 1) % 4].index == target_vert.index:
        # Swap faces
        faceA, faceB = faceB, faceA
        iAmid_vert, iBmid_vert = iBmid_vert, iAmid_vert

    """
    We have by now the following configuration:
        o---M---o
        | A | B |
        o---T---o
    """

    empty_left = (new_edge[0].index != faceA.verts[(iAmid_vert + 1) % 4].index) and (
        new_edge[1].index != faceA.verts[(iAmid_vert + 1) % 4].index
    )
    if empty_left:
        """
        We have by now the following configuration:
            v2---v1---v4
            |  A  |  B |
            v3----T----o
        """
        v1 = mid_vert
        v2 = faceA.verts[(iAmid_vert + 1) % 4]
        v3 = faceA.verts[(iAmid_vert + 2) % 4]
        v4 = new_edge[0] if v3.index == new_edge[1].index else new_edge[1]
    else:
        """
        We have by now the following configuration:
            v4---v1---v2
            |  A  |  B |
            v3----T----o
        """
        v1 = mid_vert
        v2 = faceB.verts[(iBmid_vert - 1) % 4]
        v3 = faceB.verts[(iBmid_vert - 2) % 4]
        v4 = new_edge[0] if v3.index == new_edge[1].index else new_edge[1]

    # Divide plane into 2 triangles
    tri_ABC = [v1, v2, v3]
    tri_ACD = [v1, v3, v4]

    # Determine if the point is inside the face
    inside_ABC, _, _, _ = compute_barycentric_coordinates(target_vert.co, tri_ABC)
    if inside_ABC:
        return False
    inside_ACD, _, _, _ = compute_barycentric_coordinates(target_vert.co, tri_ACD)
    if inside_ACD:
        return False

    return True


def best_rotation(edge: bmesh.types.BMEdge, mid_vert: bmesh.types.BMVert) -> Rotation:
    """Determine the best rotation for the given edge."""
    if not edge.is_valid:
        return Rotation.NONE

    if len(edge.link_faces) != 2:
        return Rotation.NONE

    face1, face2 = edge.link_faces
    if len(face1.verts) != 4 or len(face2.verts) != 4:
        return Rotation.NONE

    vA, vB = edge.verts
    iAface1, iAface2 = None, None
    iBface1, iBface2 = None, None

    for i, vert in enumerate(face1.verts):
        if vert.index == vA.index:
            iAface1 = i
        elif vert.index == vB.index:
            iBface1 = i

    for i, vert in enumerate(face2.verts):
        if vert.index == vA.index:
            iAface2 = i
        elif vert.index == vB.index:
            iBface2 = i

    if iAface1 is None or iAface2 is None or iBface1 is None or iBface2 is None:
        return Rotation.NONE

    if (iBface1 + 1) % 4 != iAface1:
        # Swap faces
        face1, face2 = face2, face1

    """
    we have by now the following configuration:
        o---A---o
        | 1 | 2 |
        o---B---o
    """

    base_energy = compute_energy(edge)

    # Compute the energy of CW rotation
    cw_vA = face2.verts[(iAface2 - 1) % 4]
    cw_vB = face1.verts[(iBface1 - 1) % 4]
    if not is_valid_rotation(mid_vert, edge, [cw_vA, cw_vB]):
        cw_energy = float("inf")
    else:
        cw_energy = compute_energy(edge, [cw_vA, cw_vB])

    # Compute the energy of CCW rotation
    ccw_vA = face1.verts[(iAface1 + 1) % 4]
    ccw_vB = face2.verts[(iBface2 + 1) % 4]
    if not is_valid_rotation(mid_vert, edge, [ccw_vA, ccw_vB]):
        ccw_energy = float("inf")
    else:
        ccw_energy = compute_energy(edge, [ccw_vA, ccw_vB])

    if base_energy <= min(cw_energy, ccw_energy):
        return Rotation.NONE
    elif cw_energy < ccw_energy:
        return Rotation.CW
    else:
        return Rotation.CCW


def rotate_edges(bm: bmesh.types.BMesh, mid_vert: bmesh.types.BMVert):
    """Rotate the edges of the given list to minimize the energy."""
    modified_faces = []
    if not mid_vert.is_valid:
        return modified_faces

    edges = list(mid_vert.link_edges)
    for i, edge in enumerate(edges):
        if len(edge.link_faces) != 2:
            continue

        rotation = best_rotation(edge, mid_vert)

        if rotation == Rotation.NONE:
            continue

        # Record edge vertices
        v1, v2 = edge.verts

        # Apply rotation
        if rotation == Rotation.CW:
            rotated_edge = bmesh.ops.rotate_edges(bm, edges=[edge], use_ccw=False)
            rotated_edge = rotated_edge["edges"]
        else:
            rotated_edge = bmesh.ops.rotate_edges(bm, edges=[edge], use_ccw=True)
            rotated_edge = rotated_edge["edges"]

        # Check if the rotation is valid
        if not rotated_edge:
            continue
        rotated_edge = rotated_edge[0]

        # Update faces and clean local zone
        for f in rotated_edge.link_faces:
            HEAP.push(f)
            modified_faces.append(f)
        clean_faces = clean_local_zone(bm, [v1, v2])
        modified_faces.extend(clean_faces)

    valid_modified_faces = [f for f in modified_faces if f.is_valid]
    return valid_modified_faces


def get_unique_verts(faces: list[bmesh.types.BMFace]) -> list[bmesh.types.BMVert]:
    """Get a list of unique vertices from the given list of faces."""
    verts = []
    visited = set()

    for face in faces:
        for vert in face.verts:
            if vert.index not in visited:
                visited.add(vert.index)
                verts.append(vert)
    return verts


def get_unique_faces(verts: list[bmesh.types.BMVert]) -> list[bmesh.types.BMFace]:
    """Get a list of unique faces from the given list of vertices."""
    faces = []
    set_visited = set()

    for vert in verts:
        for face in vert.link_faces:
            if face.index not in set_visited:
                faces.append(face)
                set_visited.add(face.index)
    return faces


def smooth_mesh(
    mesh: bmesh.types.BMesh, verts: list[bmesh.types.BMVert], relax_iter: int = 10
):
    """Smooth the local zone of the given vertices with `relax_iter` iterations."""
    euler_step = 0.1
    convergence_threshold = 0.001

    for i in range(relax_iter):
        average_length = np.zeros(len(verts), dtype=float)
        average_forces = np.array([Vector((0.0, 0.0, 0.0))] * len(verts))
        for i, vert in enumerate(verts):
            # Compute the average length of the edges of each vertex
            average_length[i] = np.mean(
                [edge.calc_length() for edge in vert.link_edges]
            )

            # Compute the average forces of each vertex based on the average length
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                force_vec = vert.co - other.co
                average_forces[i] += force_vec.normalized() * (
                    average_length[i] - force_vec.length
                )

        changed = False
        # Update the position of each vertex based on the average forces
        for i, vert in enumerate(verts):
            new_vert_pos = average_forces[i] * euler_step + vert.co
            closest_pos, _, _, dist = BVH.find_nearest(new_vert_pos)
            if closest_pos is None:
                if VERBOSE:
                    print("Warning: No closest position found")
                continue
            changed |= dist > (average_length[i] * convergence_threshold)
            vert.co = closest_pos

        if not changed:
            break
    # Update the normals of the mesh
    mesh.normal_update()


def simplify_mesh(mesh: bmesh.types.BMesh, nb_faces: int) -> bmesh.types.BMesh:
    """
    Apply quad mesh simplification to reduce the number of faces in the given BMesh object.

    This function simplifies the input quad mesh by reducing the number of faces while attempting
    to preserve the overall shape and structure of the mesh.
    To achieve simplification it applies the following local operations :
        - coarsening: diagonal collapse
        - optimizing: edge rotation
        - cleaning: Doublet/Singlet removal
        - smoothing: Tangent space smoothing

    Parameters:
    - mesh (bmesh.types.BMesh): The input quad BMesh object representing the mesh to be simplified.
    - nb_faces (int): The target number of faces for the simplified mesh. The function will
      attempt to reduce the number of faces in the input quad mesh to approximately this number.

    Returns:
    - bmesh.types.BMesh: The simplified quad BMesh object with the reduced number of faces.

    Note:
    This function modifies the input mesh in-place and returns the same object.
    """

    # Initialize BVH tree of input mesh for spatial queries
    build_bvh_tree()

    # Initialize the heap with the faces of the input mesh
    build_mesh_heap(mesh)

    initial_mesh_faces = len(mesh.faces)
    global MESH_ITERATION
    MESH_ITERATION = 0

    total_invalid_faces = 0
    total_outdated_faces = 0
    total_non_quad_faces = 0

    # Repeat simplification until the desired number of faces is reached
    while len(HEAP._data) > 0 and len(mesh.faces) > nb_faces:
        iteration_faces = len(mesh.faces)

        priority, _, uid, face = HEAP.pop()

        if not face.is_valid:
            total_invalid_faces += 1
            continue

        occ = HEAP_ELEM_OCC[uid]
        if occ is None:  # <-- Should not happen
            debug_here(mesh, [face])

        if priority != compute_priority(face):
            total_outdated_faces += 1
            continue

        if len(face.verts) != 4:
            total_non_quad_faces += 1
            continue

        if not allow_collapse(face):
            continue

        # -- Coarsening: Diagonal collapse --
        mid_vert = collapse_diagonal(mesh, face)
        mid_vert_faces = list(mid_vert.link_faces)
        push_updated_faces(mid_vert_faces)

        # --> Apply related cleaning operations
        neighbor_verts = [edge.other_vert(mid_vert) for edge in mid_vert.link_edges]
        clean_faces = clean_local_zone(mesh, [mid_vert] + neighbor_verts)

        # -- Optimizing: Edge rotation --
        modified_faces = rotate_edges(mesh, mid_vert)

        # -- Smoothing: Tangent space smoothing --
        all_modified_faces = mid_vert_faces + clean_faces + modified_faces
        all_modified_faces_valid = [f for f in all_modified_faces if f.is_valid]
        smooth_verts = get_unique_verts(all_modified_faces_valid)

        if smooth_verts:
            smooth_mesh(mesh, smooth_verts, relax_iter=10)
            push_updated_faces(all_modified_faces_valid)
        # if MESH_ITERATION == 100:
        #     debug_here(mesh, all_modified_faces_valid)

        if VERBOSE or MESH_ITERATION % 100 == 0:
            print(f"-- Iteration {MESH_ITERATION} done --")
            print(f"-> Total faces = {len(mesh.faces)}")
            print(f"-> Total removed faces = {initial_mesh_faces - len(mesh.faces)}")
            print(
                f"-> Total removed faces (in iter) = {iteration_faces - len(mesh.faces)}"
            )
            print(f"-> Heap size = {len(HEAP._data)}")

        MESH_ITERATION += 1

    print("\n--- # Simplification DONE # ---")
    print(f"Final number of faces: {len(mesh.faces)}")
    print(f"Heap size: {len(HEAP._data)}")
    print(f"Total iterations: {MESH_ITERATION}")
    print()
    print("--- # Stats # ---")
    print(f"Total removed faces: {initial_mesh_faces - len(mesh.faces)}")
    print(f"Total invalid faces: {total_invalid_faces}")
    print(f"Total outdated faces: {total_outdated_faces}")
    print(f"Total non-quad faces: {total_non_quad_faces}")

    return mesh


def get_neighbors_from_radius(vert: bmesh.types.BMVert, radius: float) -> list:
    """Get the neighbors of the given vertex within the given radius."""
    neighbors = []
    visited = set()
    visited.add(vert.index)

    q = Queue()
    for edge in vert.link_edges:
        other = edge.other_vert(vert)
        q.put(other)
    while not q.empty():
        current = q.get()
        if current.index in visited:
            continue
        visited.add(current.index)
        neighbors.append(current)
        for edge in current.link_edges:
            other = edge.other_vert(current)
            if distance_vec(vert.co, other.co) < radius:
                q.put(other)
    return neighbors


def get_faces_neighbors_from_verts(
    vert: bmesh.types.BMVert, verts_neighbors: list, radius: float
) -> list:
    """Get the faces neighbors of the given vertex within the given radius."""
    visited = set()
    neighbor_faces = []
    for nvert in verts_neighbors:
        faces = nvert.link_faces
        for face in faces:
            if face.index in visited:
                continue
            for fvert in face.verts:
                if distance_vec(vert.co, fvert.co) > radius:
                    visited.add(face.index)
                    break
            if face.index not in visited:
                neighbor_faces.append(face)
                visited.add(face.index)

    return neighbor_faces


def compute_radius_error(neighbors: list, vert: bmesh.types.BMVert) -> float:
    if len(neighbors) < 16:
        return 0.0

    yaxis = Vector((0.0, 1.0, 0.0))
    zaxis = Vector((0.0, 0.0, 1.0))
    normal = vert.normal.normalized()
    u = normal.cross(zaxis) if normal != zaxis else normal.cross(yaxis)
    u = u.normalized()
    v = normal.cross(u)
    v = v.normalized()

    # TODO: maybe they want plane fit
    A = np.zeros((len(neighbors), 16))
    b = np.zeros(len(neighbors))

    for i, neighbor in enumerate(neighbors):
        diff = neighbor.co - vert.co

        # Projection on local tangent frame
        x = diff.dot(u)
        y = diff.dot(v)
        z = diff.dot(normal)

        A[i] = np.array(
            [
                x**3 * y**3,
                x**3 * y**2,
                x**3 * y,
                x**3,
                x**2 * y**3,
                x**2 * y**2,
                x**2 * y,
                x**2,
                x * y**3,
                x * y**2,
                x * y,
                x,
                y**3,
                y**2,
                y,
                1,
            ]
        )
        b[i] = z

    # Fit a cubic polynomial on tangent frame
    coefs = np.linalg.lstsq(A, b, rcond=None)[0]

    # Compute RMS error
    residual = np.linalg.norm(b - A.dot(coefs)) / np.sqrt(len(neighbors))
    return residual


def compute_sfitmap(vert: bmesh.types.BMVert, radii: np.ndarray):
    """Compute the Scale fitmap for the given vertex."""
    radii_errors = []
    max_neighbors = get_neighbors_from_radius(vert, radii[-1])
    for radius in radii:
        neighbors_at_radius = [
            n for n in max_neighbors if distance_vec(vert.co, n.co) < radius
        ]
        radius_error = compute_radius_error(neighbors_at_radius, vert)
        radii_errors.append(radius_error)
    # fit ax^4
    A = np.power(radii, 4).reshape(-1, 1)
    b = np.array(radii_errors)

    coefs = np.linalg.lstsq(A, b, rcond=None)[0]
    a = coefs[0]

    # Add 1 to avoid 0 values that can cause issues to priority HEAP
    vert[SFITMAP_LAYER] = np.power(a, 0.25) + 1


def compute_mfitmap(vert: bmesh.types.BMVert, radii: np.ndarray, threshold=0.05):
    """Compute the Maximal radius fitmap for the given vertex."""
    max_neighbors = get_neighbors_from_radius(vert, radii[-1])
    for radius in radii:
        neighbors_at_radius = [
            n for n in max_neighbors if distance_vec(vert.co, n.co) < radius
        ]
        vert_normal = vert.normal.normalized()

        consistent_faces = 0
        inconsistent_faces = 0
        face_neighbors = get_faces_neighbors_from_verts(
            vert, neighbors_at_radius, radius
        )
        for fneighbor in face_neighbors:
            face_normal = fneighbor.normal.normalized()
            scalar = vert_normal.dot(face_normal)
            if scalar >= 0:
                consistent_faces += fneighbor.calc_area()
            else:
                inconsistent_faces += fneighbor.calc_area()

        if consistent_faces + inconsistent_faces == 0:
            continue
        proportion = inconsistent_faces / (consistent_faces + inconsistent_faces)
        if proportion >= threshold:
            break

        vert[MFITMAP_LAYER] = radius


def compute_fitmaps():
    """Compute the Scale and Maximal radius fitmaps for the given mesh."""
    global SFITMAP_LAYER, MFITMAP_LAYER
    SFITMAP_LAYER = INITIAL_MESH.verts.layers.float.new("sfitmap")
    MFITMAP_LAYER = INITIAL_MESH.verts.layers.float.new("mfitmap")

    avg_edges_length = np.mean([edge.calc_length() for edge in INITIAL_MESH.edges])
    max_radii = 5

    r0 = avg_edges_length
    radii = np.array([r0 * (1 + i) for i in range(max_radii)])

    len_verts = len(INITIAL_MESH.verts)
    for i, vert in enumerate(INITIAL_MESH.verts):
        compute_sfitmap(vert, radii)
        compute_mfitmap(vert, radii)

        if VERBOSE or i % 100 == 0:
            print(f"Computed fitmaps for {i}/{len_verts}")


def debug_here(
    bm: bmesh.types.BMesh,
    elements: list[
        Union[bmesh.types.BMVert, bmesh.types.BMEdge, bmesh.types.BMFace]
    ] = [],
):
    obj = bpy.context.object

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for face in bm.faces:
        face.select_set(False)
    for element in elements:
        element.select_set(True)

    bpy.ops.object.mode_set(mode="OBJECT")
    bm.to_mesh(obj.data)
    bm.free()

    # terminate the script
    raise Exception("Debug here")


def visualize_fitmap(sfitmap: bool = True):
    """Color the vertices of the given mesh based on the given fitmap layer."""
    # Get the active mesh
    mesh = bpy.context.object.data

    bpy.ops.object.mode_set(mode="EDIT")

    # Get a BMesh representation
    bm = bmesh.from_edit_mesh(mesh)
    global INITIAL_MESH
    INITIAL_MESH = bm

    compute_fitmaps()
    fitmap_layer = SFITMAP_LAYER if sfitmap else MFITMAP_LAYER

    # Get the min and max values of the fitmap
    min_val = min([v[fitmap_layer] for v in bm.verts])
    max_val = max([v[fitmap_layer] for v in bm.verts])

    point_color_attribute = mesh.color_attributes.get(
        "FitmapColors"
    ) or mesh.color_attributes.new(
        name="FitmapColors", type="BYTE_COLOR", domain="POINT"
    )
    point_color_layer = bm.verts.layers.color[point_color_attribute.name]
    # Compute the normalized fitmap values
    for vert in bm.verts:
        normalized_val = (vert[fitmap_layer] - min_val) / (max_val - min_val)
        color = turbo_colormap[int(normalized_val * 255)]
        vert[point_color_layer] = (
            color[0],
            color[1],
            color[2],
            1,
        )  # Set Point Colors
    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode="OBJECT")


if __name__ == "__main__":
    global VERBOSE
    VERBOSE = False

    # visualize_fitmap(sfitmap=True)

    # """
    # Get the active mesh
    me = bpy.context.object.data

    # Get a BMesh representation
    bm = bmesh.new()  # create an empty BMesh
    bm.from_mesh(me)  # fill it in from a Mesh

    # Preprocessing - Compute the Scale and Maximal radius fitmaps
    global INITIAL_MESH
    INITIAL_MESH = bm.copy()
    compute_fitmaps()

    bm = simplify_mesh(bm, 0)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  # free and prevent further access

    me.update()
    # """
