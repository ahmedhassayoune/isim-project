import bpy
import bmesh

obj = bpy.context.scene.objects["Icosphere"]

# To be sure to have
# 0 --- 1
# |     |
# 3 --- 2
# With (1,3) the common edge of (0,1,3) and (1,2,3)
def getVertArrayFrom(triangle0, triangle1, commonEdge):
    vertices = []
    for vert in triangle0.verts:
        if vert not in commonEdge.verts:
            vertices.append(vert)
            break
    vertices.append(commonEdge.verts[0])
    for vert in triangle1.verts:
        if vert not in commonEdge.verts:
            vertices.append(vert)
            break
    vertices.append(commonEdge.verts[1])
    return vertices

def squareness(edge):
    triangles = edge.link_faces
    if len(triangles) != 2:
        return 3 # Bigger than the max dot_sum

    vertices = getVertArrayFrom(triangles[0], triangles[1], edge)

    v01 = (vertices[1].co - vertices[0].co).normalized()
    v03 = (vertices[3].co - vertices[0].co).normalized()
    v21 = (vertices[1].co - vertices[2].co).normalized()
    v23 = (vertices[3].co - vertices[2].co).normalized()

    dot_sum = 0 
    dot_sum += abs(v01.dot(v03))
    dot_sum += abs(v21.dot(v23))

    # dot_sum = 0 : rectangle
    # dot_sum >>> 0 : little squareness
    return dot_sum

def input_preparation(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    if len(bm.faces) % 2 == 1: # impossible on a closed mesh
        bm.edges.ensure_lookup_table()

        # Search of a border edge
        edge = None
        for edge in bm.edges:
            if edge.is_boundary:
                break
        triangle = edge.link_faces[0]
        # Search of the opposed vertex
        for opposed_vert in triangle.verts:
            if opposed_vert not in edge.verts:
                break

        # Mesh modification
        bm.faces.remove(triangle)
        middle = bm.verts.new((edge.verts[0].co + edge.verts[1].co) / 2)
        bm.faces.new([middle, edge.verts[0], opposed_vert])
        bm.faces.new([middle, edge.verts[1], opposed_vert])
        bm.to_mesh(obj.data)
        obj.data.update()

    bm.free()

# With this configuration
# 0 ----- 1
# |       |
# 3 ----- 2
def align(v0, v1, v2, v3):
    i = (v0.co + v2.co) / 2
    j = (v1.co + v3.co) / 2
    k = (i + j) / 2
    v0.co += k - i
    v2.co += k - i
    v1.co += k - j
    v3.co += k - j

def toQuadDominant(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # Squareness score
    scores = [0] * len(bm.edges)
    for edge in bm.edges:
        scores[edge.index] = squareness(edge)
    selected_edges = [0] * len(bm.faces)
    for face in bm.faces:
        selected_edges[face.index] = min(face.edges, key=lambda x: scores[x.index])

    # Quad creation
    faces_to_add = []
    edge_to_remove = []
    for i in range(len(selected_edges)):
        s = selected_edges[i]
        if s is None:
            continue # Face i already merged

        # Get the two faces
        t0 = bm.faces[i]
        t1 = s.link_faces[0]
        if t0 == t1:
            t1 = s.link_faces[1]

        if selected_edges[t1.index] is None:
            continue # Face t1 already merged

        if scores[selected_edges[t0.index].index] > scores[selected_edges[t1.index].index]:
            continue # Face t1 have a better edge squareness

        selected_edges[t1.index] = None # Flagged as merged

        faces_to_add.append(getVertArrayFrom(t0, t1, selected_edges[t0.index]))
        edge_to_remove.append(selected_edges[t0.index])

    # Mesh modification
    for edge in edge_to_remove:
        if edge is not None:
            bm.edges.remove(edge)

    for face in faces_to_add:
        f = bm.faces.new(face)
        f.select_set(True)
        align(f.verts[0], f.verts[1], f.verts[2], f.verts[3])

    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

"""
def toPureQuad(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # TODO
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
"""

input_preparation(obj)
toQuadDominant(obj)
# toPureQuad(obj)
