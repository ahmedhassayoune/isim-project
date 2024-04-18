import heapq
from typing import Union

import bmesh
import bpy
import numpy as np
from mathutils import Vector, bvhtree


class MyHeap(object):
    """Wrapper class for a heap with custom key function."""

    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        self.index = 0
        if initial:
            self._data = [(key(item), i, item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), self.index, item))
        self.index += 1

    def pop(self):
        return heapq.heappop(self._data)[2]


def distance_vec(point1: Vector, point2: Vector) -> float:
    """Calculate distance between two points."""
    return (point2 - point1).length


def compute_min_diagonal_length(face: bmesh.types.BMFace) -> float:
    """Compute the minimum diagonal length of the given quad face."""
    if len(face.verts) != 4:
        print("Warning: Face is not a quad")
        return 0
    v1, v2, v3, v4 = face.verts
    diag1_len = distance_vec(v1.co, v3.co)
    diag2_len = distance_vec(v2.co, v4.co)
    return min(diag1_len, diag2_len)


def build_bvh_tree(mesh: bmesh.types.BMesh):
    """Build a BVH tree for the given mesh."""
    global bvh
    bvh = bvhtree.BVHTree.FromBMesh(mesh.copy())


def build_mesh_heap(mesh: bmesh.types.BMesh) -> list:
    """Build a heap with the faces of the given mesh."""
    faces = list(mesh.faces)

    global heap
    heap = MyHeap(initial=faces, key=lambda face: compute_min_diagonal_length(face))


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


def remove_doublet(mesh: bmesh.types.BMesh, vert: bmesh.types.BMVert):
    """Remove a single doublet."""
    if len(vert.link_faces) != 2:
        return None

    # Dissolve linked faces
    region = bmesh.ops.dissolve_faces(mesh, faces=vert.link_faces, use_verts=True)
    face = region["region"][0]
    heap.push(face)

    return face


def remove_doublets(
    mesh: bmesh.types.BMesh,
    verts: list[bmesh.types.BMVert],
    visited: set = set(),
):
    """Remove doublets in verts recursively."""
    clean_face = None
    for vert in verts:
        if not vert.is_valid or vert.index in visited:
            continue
        visited.add(vert.index)
        other_verts = [edge.other_vert(vert) for edge in vert.link_edges]
        face = remove_doublet(mesh, vert)
        if not face:
            continue
        heap.push(face)
        clean_face = remove_doublets(mesh, other_verts, visited)
        clean_face = clean_face if clean_face else face

    return clean_face


def clean_local_zone(mesh: bmesh.types.BMesh, verts: list[bmesh.types.BMVert]):
    # Remove doublets recursively
    face = remove_doublets(mesh, verts, set())
    if not face:
        return None

    # Remove potiential generated singlets
    mesh.edges.ensure_lookup_table()
    bmesh.ops.dissolve_degenerate(mesh, edges=face.edges)

    return face


def tag_updated_faces(faces: list[bmesh.types.BMesh], out_index=None):
    """Tag updated faces and push them to the heap except the one with the given index."""
    for face in faces:
        if face.index == out_index:
            continue
        face.tag = True
        heap.push(face)


def collapse_diagonal(mesh: bmesh.types.BMesh, face: bmesh.types.BMFace):
    """Collapse the shortest diagonal of the given quad face."""
    v1, v2, v3, v4 = face.verts
    diag1_len, diag2_len = (
        distance_vec(v1.co, v3.co),
        distance_vec(v2.co, v4.co),
    )

    # Get mid-point of the shortest diagonal
    mid_position = (v1.co + v3.co) / 2

    # Apply diagonal collapse on mid-point of the shortest diagonal
    if diag1_len > diag2_len:
        bmesh.ops.pointmerge(mesh, verts=[v2, v4], merge_co=mid_position)
        mid_vert = get_neighbor_vert_from_pos(v1, mid_position)
    else:
        bmesh.ops.pointmerge(mesh, verts=[v1, v3], merge_co=mid_position)
        mid_vert = get_neighbor_vert_from_pos(v2, mid_position)

    mid_vert.normal_update()  # maybe useless

    # Cast 2 opposite rays from the mid-vertex to find the hit positions on source mesh M0
    hit_pos, _, _, dist = bvh.ray_cast(mid_vert.co, mid_vert.normal)
    hit_pos_opp, _, _, dist_opp = bvh.ray_cast(mid_vert.co, -mid_vert.normal)

    # Handle case where no hit position is found
    if not hit_pos and not hit_pos_opp:
        print("Warning: No hit position or normal")
        return mid_vert

    if dist is None:
        dist = float("inf")
    if dist_opp is None:
        dist_opp = float("inf")
    # Get the closest hit position to the mid-vertex
    if dist < dist_opp:
        proj_mid_vert = hit_pos
    else:
        proj_mid_vert = hit_pos_opp

    # Project mid-vertex on the hit position
    bmesh.ops.pointmerge(mesh, verts=[mid_vert], merge_co=proj_mid_vert)
    return mid_vert


def compute_energy(edge: bmesh.types.BMEdge) -> float:
    """Compute the energy of the given edge."""
    verts = get_unique_verts(edge.link_faces)

    # Transform to a list valence of each vertex
    verts = np.array([len(v.link_edges) for v in verts])

    # Compute the energy
    return np.sum(np.abs(verts - 4))


def rotate_edges(mesh: bmesh.types.BMesh, edges: list):
    """Rotate the edges of the given list to minimize the energy."""
    for edge in edges:
        if len(edge.link_faces) != 2:
            continue

        base_energy = compute_energy(edge)
        # Rotate edge in clockwise direction + compute energy
        cw_edge = bmesh.ops.rotate_edges(mesh, edges=[edge], use_ccw=False)
        cw_edge = cw_edge["edges"][0]
        cw_energy = compute_energy(cw_edge)

        # Revert rotation
        edge = bmesh.ops.rotate_edges(mesh, edges=[cw_edge], use_ccw=True)
        edge = edge["edges"][0]

        # Rotate edge in counter-clockwise direction + compute energy
        ccw_edge = bmesh.ops.rotate_edges(mesh, edges=[edge], use_ccw=True)
        ccw_edge = ccw_edge["edges"][0]
        ccw_energy = compute_energy(ccw_edge)

        if min(base_energy, cw_energy, ccw_energy) == base_energy:
            # Revert to initial state
            bmesh.ops.rotate_edges(mesh, edges=[ccw_edge], use_ccw=False)
            continue

        if min(cw_energy, ccw_energy) == cw_energy:
            # Revert to other rotation
            edge = bmesh.ops.rotate_edges(mesh, edges=[ccw_edge], use_ccw=False)
            edge = edge["edges"][0]

            cw_edge = bmesh.ops.rotate_edges(mesh, edges=[edge], use_ccw=False)
            cw_edge = cw_edge["edges"][0]

            return cw_edge

        return ccw_edge

    return None


def get_unique_verts(faces: list[bmesh.types.BMFace]) -> list[bmesh.types.BMVert]:
    """Get a list of unique vertices from the given list of faces."""
    verts = []
    set_visited = set()

    for face in faces:
        for vert in face.verts:
            if vert.index not in set_visited:
                verts.append(vert)
                set_visited.add(vert.index)
    return verts


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

    # Initialize bvh tree of input mesh for spatial queries
    build_bvh_tree(mesh)

    # Initialize the heap with the faces of the input mesh
    build_mesh_heap(mesh)

    initial_mesh_faces = len(mesh.faces)
    global mesh_iteration
    mesh_iteration = 0

    # Repeat simplification until the desired number of faces is reached
    while len(heap._data) > 0 and len(mesh.faces) > nb_faces:
        iteration_faces = len(mesh.faces)
        face = heap.pop()

        if not face.is_valid or len(face.verts) != 4:
            continue

        if face.tag:
            face.tag = False
            continue

        # -- Coarsening: Diagonal collapse --
        mid_vert = collapse_diagonal(mesh, face)

        # --> Apply related cleaning operations
        neighbor_verts = [edge.other_vert(mid_vert) for edge in mid_vert.link_edges]
        cface = clean_local_zone(mesh, [mid_vert] + neighbor_verts)

        # -- Optimizing: Edge rotation --
        cedges = mid_vert.link_edges if mid_vert.is_valid else cface.edges
        rotated_edge = rotate_edges(mesh, cedges)

        # --> Apply related cleaning operations + Tag updated faces
        if rotated_edge:
            tag_updated_faces(rotated_edge.link_faces)

            verts = get_unique_verts(rotated_edge.link_faces)
            clean_local_zone(mesh, verts)

        print(f"-- Iteration {mesh_iteration} done --")
        print(f"-> Total faces = {len(mesh.faces)}")
        print(f"-> Total removed faces = {initial_mesh_faces - len(mesh.faces)}")
        print(f"-> Total removed faces (in iter) = {iteration_faces - len(mesh.faces)}")
        print(f"-> Heap size = {len(heap._data)}")

        # -- Smoothing: Tangent space smoothing --

        # TO REMOVE
        # nb_faces = len(mesh.faces)
        mesh_iteration += 1

    return mesh


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


if __name__ == "__main__":
    # Get the active mesh
    me = bpy.context.object.data

    # Get a BMesh representation
    bm = bmesh.new()  # create an empty BMesh
    bm.from_mesh(me)  # fill it in from a Mesh

    bm = simplify_mesh(bm, 550)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  # free and prevent further access
