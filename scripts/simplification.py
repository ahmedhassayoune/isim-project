import heapq
from typing import Union
from uuid import uuid4

import bmesh
import bpy
import numpy as np
from mathutils import Vector, bvhtree


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
                item[uid_layer] = elem_uuid
                heap_elem_occ[elem_uuid] = 1
                self._data.append(heap_elem)
            self.index = len(self._data)
            heapq.heapify(self._data)

    def push(self, item: bmesh.types.BMFace):
        # if len(item.verts) != 4:  # FIXME: Should not happen
        #     return None
        if item[uid_layer] == 0:  # New item
            elem_uuid = uuid4().bytes
            item[uid_layer] = elem_uuid
            heap_elem = (self.key(item), self.index, elem_uuid, item)
            heap_elem_occ[elem_uuid] = 1
        else:
            heap_elem = (self.key(item), self.index, item[uid_layer], item)
            heap_elem_occ[item[uid_layer]] += 1

        heapq.heappush(self._data, heap_elem)
        self.index += 1

    def pop(self):
        elem = heapq.heappop(self._data)
        elem_uuid = elem[2]
        heap_elem_occ[elem_uuid] -= 1
        return elem


def distance_vec(point1: Vector, point2: Vector) -> float:
    """Calculate distance between two points."""
    return (point2 - point1).length


def compute_min_diagonal_length(face: bmesh.types.BMFace) -> float:
    """Compute the minimum diagonal length of the given quad face."""
    if len(face.verts) != 4:
        print("Warning: Face is not a quad")
        return float("inf")
    v1, v2, v3, v4 = face.verts
    diag1_len = distance_vec(v1.co, v3.co)
    diag2_len = distance_vec(v2.co, v4.co)
    return min(diag1_len, diag2_len)


def build_bvh_tree(mesh: bmesh.types.BMesh):
    """Build a BVH tree for the given mesh."""
    global bvh
    bvh = bvhtree.BVHTree.FromBMesh(mesh.copy())


def build_mesh_heap(mesh: bmesh.types.BMesh) -> list:
    """Build a heap with the faces of the given mesh. Assign a unique id to each face."""
    global uid_layer
    uid_layer = bm.faces.layers.string.new("uid")

    global heap_elem_occ
    heap_elem_occ = dict()

    global heap
    faces = list(mesh.faces)
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
    if len(vert.link_edges) != 2:
        return None

    # Dissolve linked faces
    region = bmesh.ops.dissolve_faces(mesh, faces=vert.link_faces, use_verts=True)
    face = region["region"]
    if not face:
        return None
    face = face[0]
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
        clean_face = clean_face if clean_face is not None else face

    return clean_face


def clean_local_zone(mesh: bmesh.types.BMesh, verts: list[bmesh.types.BMVert]):
    # Remove doublets recursively
    face = remove_doublets(mesh, verts, set())
    if not face:
        return None

    # Remove potiential generated singlets or other degenerates
    # e.g. edges w/o length, faces w/o area ...
    bmesh.ops.dissolve_degenerate(mesh, edges=face.edges)

    return face


def push_updated_faces(faces: list[bmesh.types.BMesh], out_index=None):
    """Push updated faces to the heap except the one with the given index."""
    for face in faces:
        if out_index and face.index == out_index:
            continue
        heap.push(face)


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
        bmesh.ops.pointmerge(mesh, verts=[v1, v3], merge_co=mid_position)
        mid_vert = get_neighbor_vert_from_pos(v2, mid_position)
    else:
        mid_position = (v2.co + v4.co) / 2
        bmesh.ops.pointmerge(mesh, verts=[v2, v4], merge_co=mid_position)
        mid_vert = get_neighbor_vert_from_pos(v1, mid_position)

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
    mid_vert.co = proj_mid_vert
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
    for i in range(len(edges)):
        edge = edges[i]
        if len(edge.link_faces) != 2:
            continue

        base_energy = compute_energy(edge)
        # Rotate edge in clockwise direction + compute energy
        cw_edge = bmesh.ops.rotate_edges(mesh, edges=[edge], use_ccw=False)
        cw_edge = cw_edge["edges"]
        cw_energy = compute_energy(cw_edge[0]) if cw_edge else float("inf")

        # Revert rotation to initial state
        if cw_edge:
            edge = bmesh.ops.rotate_edges(mesh, edges=[cw_edge[0]], use_ccw=True)
            edge = edge["edges"][0]

        # Rotate edge in counter-clockwise direction + compute energy
        ccw_edge = bmesh.ops.rotate_edges(mesh, edges=[edge], use_ccw=True)
        ccw_edge = ccw_edge["edges"]
        ccw_energy = compute_energy(ccw_edge[0]) if ccw_edge else float("inf")

        if min(base_energy, cw_energy, ccw_energy) == base_energy:
            # Revert rotation to initial state
            if ccw_edge:
                edge = bmesh.ops.rotate_edges(mesh, edges=[ccw_edge[0]], use_ccw=False)
                edge = edge["edges"][0]

            push_updated_faces(
                edge.link_faces
            )  # reupdate faces even if no rotation is done
            continue

        if min(cw_energy, ccw_energy) == cw_energy:
            # Revert rotation to initial state
            if ccw_edge:
                edge = bmesh.ops.rotate_edges(mesh, edges=[ccw_edge[0]], use_ccw=False)
                edge = edge["edges"][0]

            cw_edge = bmesh.ops.rotate_edges(mesh, edges=[edge], use_ccw=False)
            cw_edge = cw_edge["edges"][0]

            return cw_edge

        return ccw_edge[0]

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


def smooth_mesh(verts: list[bmesh.types.BMVert], relax_iter: int = 10):
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
            closest_pos, _, _, dist = bvh.find_nearest(new_vert_pos)
            if closest_pos is None:
                print("Warning: No closest position found")
                continue
            changed |= dist > average_length[i] * convergence_threshold
            vert.co = closest_pos

        if not changed:
            break


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

    total_invalid_faces = 0
    total_outdated_faces = 0
    total_non_quad_faces = 0

    # Repeat simplification until the desired number of faces is reached
    while len(heap._data) > 0 and len(mesh.faces) > nb_faces:
        iteration_faces = len(mesh.faces)

        min_diag, _, uid, face = heap.pop()

        if not face.is_valid:
            total_invalid_faces += 1
            continue

        occ = heap_elem_occ[uid]
        if occ is None:  # <-- Should not happen
            debug_here(mesh, [face])

        if occ > 1 and min_diag != compute_min_diagonal_length(face):
            total_outdated_faces += 1
            continue

        if len(face.verts) != 4:
            total_non_quad_faces += 1
            continue

        # -- Coarsening: Diagonal collapse --
        mid_vert = collapse_diagonal(mesh, face)
        push_updated_faces(mid_vert.link_faces)

        # --> Apply related cleaning operations
        neighbor_verts = [edge.other_vert(mid_vert) for edge in mid_vert.link_edges]
        cface = clean_local_zone(mesh, [mid_vert] + neighbor_verts)

        # -- Optimizing: Edge rotation --
        cedges = mid_vert.link_edges if mid_vert.is_valid else cface.edges
        rotated_edge = rotate_edges(mesh, cedges)

        # --> Apply related cleaning operations + push updated faces
        if rotated_edge:
            push_updated_faces(rotated_edge.link_faces)

            verts = get_unique_verts(rotated_edge.link_faces)
            cface = clean_local_zone(mesh, verts)

        # -- Smoothing: Tangent space smoothing --
        if rotated_edge:
            smooth_verts = (
                cface.verts
                if cface and cface.is_valid
                else get_unique_verts(rotated_edge.link_faces)
            )
        else:
            smooth_verts = (
                cface.verts
                if cface and cface.is_valid
                else get_unique_verts(mid_vert.link_faces)
            )

        smooth_mesh(smooth_verts, relax_iter=10)

        print(f"-- Iteration {mesh_iteration} done --")
        print(f"-> Total faces = {len(mesh.faces)}")
        print(f"-> Total removed faces = {initial_mesh_faces - len(mesh.faces)}")
        print(f"-> Total removed faces (in iter) = {iteration_faces - len(mesh.faces)}")
        print(f"-> Heap size = {len(heap._data)}")

        mesh_iteration += 1

    print("\n--- # Simplification DONE # ---")
    print(f"Final number of faces: {len(mesh.faces)}")
    print(f"Heap size: {len(heap._data)}")
    print(f"Total iterations: {mesh_iteration}")
    print()
    print("--- # Stats # ---")
    print(f"Total invalid faces: {total_invalid_faces}")
    print(f"Total outdated faces: {total_outdated_faces}")
    print(f"Total non-quad faces: {total_non_quad_faces}")

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

    global test_var
    test_var = bm
    bm = simplify_mesh(bm, 400)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  # free and prevent further access
    me.update()
