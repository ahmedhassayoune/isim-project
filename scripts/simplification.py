import heapq

import bmesh
import bpy
import numpy as np
from mathutils import Vector


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
        print("Face is not a quad")
        return 0
    v1, v2, v3, v4 = face.verts
    diag1_len = distance_vec(v1.co, v3.co)
    diag2_len = distance_vec(v2.co, v4.co)
    return min(diag1_len, diag2_len)


def build_mesh_heap(mesh: bmesh.types.BMesh) -> list:
    faces = list(mesh.faces)
    heap = MyHeap(initial=faces, key=lambda face: compute_min_diagonal_length(face))

    return heap


def get_neighbor_vert_from_pos(
    vert: bmesh.types.BMVert, pos: Vector
) -> bmesh.types.BMVert:
    """Get the neighbor vertex of the given vertex closest to the given position."""
    min_vert, min_dist = None, float("inf")
    for edge in vert.link_edges:
        other_vert = edge.other_vert(vert)
        if vert.co == other_vert.co:
            return other_vert
        if (dist := distance_vec(other_vert.co, pos)) < min_dist:
            min_dist, min_vert = dist, other_vert
    print("Warning: No exact neighbor found")
    return min_vert


def remove_doublets(mesh: bmesh.types.BMesh, vert: bmesh.types.BMVert):
    """Remove doublets recursively."""
    if len(vert.link_faces) != 2:
        return None

    # Dissolve linked faces
    mesh.faces.ensure_lookup_table()
    region = bmesh.ops.dissolve_faces(mesh, faces=vert.link_faces, use_verts=True)
    face = region["region"][0]

    # Iteratively remove doublets
    dissolvable = True
    while dissolvable:
        dissolvable = False
        for v in face.verts:
            if len(v.link_faces) != 2:
                continue
            # Dissolve linked faces
            mesh.faces.ensure_lookup_table()
            region = bmesh.ops.dissolve_faces(
                mesh, faces=vert.link_faces, use_verts=True
            )
            face = region["region"][0]
            dissolvable = True
            break
    return face


def clean_local_zone(mesh: bmesh.types.BMesh, vert: bmesh.types.BMVert):
    # Remove doublets iteratively
    face = remove_doublets(mesh, vert)
    if not face:
        return None

    # Remove potiential generated singlets
    mesh.edges.ensure_lookup_table()
    bmesh.ops.dissolve_degenerate(mesh, edges=face.edges)

    return face


def tag_updated_faces(faces, heap: MyHeap, out_index=None):
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

    # Apply diagonal collapse on mid-point of the shortest diagonal
    mid_position = (v1.co + v3.co) / 2
    if diag1_len > diag2_len:
        mesh.pointmerge(verts=[v2, v4], merge_co=mid_position)
        mid_vert = get_neighbor_vert_from_pos(v1, mid_position)
    else:
        mesh.pointmerge(verts=[v1, v3], merge_co=mid_position)
        mid_vert = get_neighbor_vert_from_pos(v2, mid_position)

    return mid_vert


def compute_energy(edge: bmesh.types.BMEdge) -> float:
    """Compute the energy of the given edge."""
    face1, face2 = edge.link_faces[0], edge.link_faces[1]
    v1, v2 = edge.verts[0], edge.verts[1]

    # Build the list of 6 vertices of the two faces
    verts = [v1, v2]
    all_verts = list(face1.verts) + list(face2.verts)
    for v in all_verts:
        if v.index != v1.index and v.index != v2.index:
            verts.append(v)

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
        cw_energy = compute_energy(cw_edge)

        # Revert the rotation
        edge = bmesh.ops.rotate_edges(mesh, edges=[cw_edge], use_ccw=True)

        # Rotate edge in counter-clockwise direction + compute energy
        ccw_edge = bmesh.ops.rotate_edges(mesh, edges=[edge], use_ccw=True)
        ccw_energy = compute_energy(ccw_edge)

        if min(base_energy, cw_energy, ccw_energy) == base_energy:
            # Revert to initial state
            bmesh.ops.rotate_edges(mesh, edges=[ccw_edge], use_ccw=False)
            continue

        if min(cw_energy, ccw_energy) == cw_energy:
            # Revert to other rotation
            edge = bmesh.ops.rotate_edges(mesh, edges=[ccw_edge], use_ccw=False)
            cw_edge = bmesh.ops.rotate_edges(mesh, edges=[edge], use_ccw=False)
            return cw_edge

        return ccw_edge

    return None


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

    heap = build_mesh_heap(mesh)

    # Repeat simplification until the desired number of faces is reached
    while len(mesh.faces) > nb_faces:
        face = heap.pop()

        if not face.is_valid or len(face.verts) != 4:
            continue

        if face.tag:
            face.tag = False
            continue

        # -- Coarsening: Diagonal collapse --
        mid_vert = collapse_diagonal(mesh, face)

        # --> Apply related cleaning operations
        cface = clean_local_zone(mesh, mid_vert)

        # --> Tag all updated faces and push them to the heap
        if cface:
            heap.push(cface)
            for cedge in cface.eges:
                tag_updated_faces(cedge.link_faces, heap, cface.index)
        else:
            tag_updated_faces(mid_vert.link_faces, heap)

        # -- Optimizing: Edge rotation --
        cedges = cface.edges if cface else mid_vert.link_edges
        rotated_edge = rotate_edges(mesh, cedges)

        # --> Apply related cleaning operations + Tag updated faces
        if rotated_edge:
            mesh.edges.ensure_lookup_table()
            cface1 = clean_local_zone(mesh, rotated_edge.verts[0])
            cface2 = clean_local_zone(mesh, rotated_edge.verts[1])

            if cface2:
                heap.push(cface2)
                for cedge in cface2.edges:
                    tag_updated_faces(cedge.link_faces, heap, cface2.index)
            if cface1 and (not cface2 or cface1.index != cface2.index):
                heap.push(cface1)
                for cedge in cface1.edges:
                    tag_updated_faces(cedge.link_faces, heap, cface1.index)

        # -- Smoothing: Tangent space smoothing --

        # TO REMOVE
        # nb_faces = len(mesh.faces)

    return mesh


if __name__ == "__main__":
    # Get the active mesh
    me = bpy.context.object.data

    # Get a BMesh representation
    bm = bmesh.new()  # create an empty BMesh
    bm.from_mesh(me)  # fill it in from a Mesh

    bm = simplify_mesh(bm, 10)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  # free and prevent further access
