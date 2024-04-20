import bpy
import bmesh

def prepareTriangle(mesh: bmesh.types.BMesh) -> bmesh.types.BMesh:
    """
    Transform the mesh to a redefinition of it with an even number of
    triangles.
    
    This function search a border edge and split it in half.
    (Condition : Not closed)

    If the mesh is closed (= no border edge) and two-manifold, there is
    always an even number of triangles.
    
    Parameters:
    - mesh (bmesh.types.BMesh): The input triangle BMesh object representing
                                the mesh to be converted.

    Returns:
    - bmesh.types.BMesh: The triangle BMesh object with an even number of
                         triangles.

    Note:
    This function modifies the input mesh in-place and returns the same object.
    """  
    mesh.edges.ensure_lookup_table()
    
    # Search of a border edge
    borderEdge = None
    for edge in mesh.edges:
        if edge.is_boundary:
            borderEdge = edge
            break
    
    if borderEdge is None: # If this condition is reached, the mesh is invalid (not two-manifold)
        raise RuntimeError('Call to prepareTriangle with a closed mesh.') 
    
    faceToDivide = borderEdge.link_faces[0]
    
    # Search of the opposed vertex
    opposedVert = None
    for vert in faceToDivide.verts:
        if vert not in borderEdge.verts:
            opposedVert = vert
            break

    # Mesh modification
    mesh.edges.remove(borderEdge)
    middle = mesh.verts.new((borderEdge.verts[0].co + borderEdge.verts[1].co) / 2)
    mesh.faces.new([middle, borderEdge.verts[0], opposedVert])
    mesh.faces.new([middle, borderEdge.verts[1], opposedVert])
    
    return mesh

def merge(
    t0: bmesh.types.BMFace, t1: bmesh.types.BMFace,
) -> (bmesh.types.BMEdge, [bmesh.types.BMVert]):
    """
    Returns the edge to remove and the vertices of the theoric face that
    would result if the triangle parameter faces were merged.
    """
    Vert_t0 = None
    Vert_t1 = None
    CommonEdge = None

    for edge in t0.edges:
        if edge in t1.edges:
            CommonEdge = edge
            break
    
    for vert in t0.verts:
        if vert not in CommonEdge.verts:
            Vert_t0 = vert
            break

    for vert in t1.verts:
        if vert not in CommonEdge.verts:
            Vert_t1 = vert
            break

    # Vertices in the right order
    verts = [Vert_t0, CommonEdge.verts[0], Vert_t1, CommonEdge.verts[1]]

    return CommonEdge, verts

def squareness(edge: bmesh.types.BMEdge) -> float:
    """
    Calculate sum of pairwise dot products of the four normalized edges
    of the theoric face that would result if the parameter edge was dissolved.
    """
    if edge.is_boundary:
        return 3 # Bigger than the maximum possible squareness
    
    # Get the 4 vertices of the theoric face in the right order
    _, vertices = merge(edge.link_faces[0], edge.link_faces[1])
    
    # Compute the normalized edges
    v01 = (vertices[1].co - vertices[0].co).normalized()
    v03 = (vertices[3].co - vertices[0].co).normalized()
    v21 = (vertices[1].co - vertices[2].co).normalized()
    v23 = (vertices[3].co - vertices[2].co).normalized()
    
    # Compute the sum of pairwise dot products
    dot_sum = 0 
    dot_sum += abs(v01.dot(v03))
    dot_sum += abs(v21.dot(v23))

    return dot_sum

# def ...

def correctQuad(quad: bmesh.types.BMFace) -> bmesh.types.BMFace:
    """
    Transform a quad mesh to put all its vertices on the same plane, while
    attempting to preserve the structure.
    
    Note:
    This function modifies the input mesh in-place and returns the same object.
    """
    v0, v1, v2, v3 = quad.verts[0], quad.verts[1], quad.verts[2], quad.verts[3]

    # Compute the chosen plane
    middle_O2 = (v0.co + v2.co) / 2
    middle_13 = (v1.co + v3.co) / 2
    center = (middle_O2 + middle_13) / 2

    # Projection on the plane
    v0.co += center - middle_O2
    v2.co += center - middle_O2
    v1.co += center - middle_13
    v3.co += center - middle_13
    
    return quad

def triToQuad(mesh: bmesh.types.BMesh) -> bmesh.types.BMesh:
    """
    Apply tri-to-quad mesh conversion to prepare the application of the
    quad mesh simplification.

    This function convert the input triangle mesh to a quad mesh while
    attempting to preserve the overall shape and structure of the mesh.
    To achieve simplification it applies the following local operations :
        - quad-dominant mesh: edge collapse
        - pure quad-mesh: edge flip and collapse
    but it will be optimised doing :
        - quad-dominant mesh: theoric edge collapse
        - pure quad-mesh: theoric edge flip and collapse
        - application: effective edge collapse
    which will allow not to apply edge flip.

    Parameters:
    - mesh (bmesh.types.BMesh): The input triangle BMesh object representing
                                the mesh to be converted.

    Returns:
    - bmesh.types.BMesh: The quad BMesh object.

    Note:
    This function modifies the input mesh in-place and returns the same object.
    """
    l_faces = len(mesh.faces)
    if l_faces % 2 == 1:
        # Transform the mesh to a redefinition of it with an even number of triangles.
        prepareTriangle(mesh)
    
    # Tool used to preserve the overall shape and structure
    mesh.edges.ensure_lookup_table()
    squarenessList = [squareness(edge) for edge in mesh.edges]

    # Tool used to realise the mesh conversion
    mesh.faces.ensure_lookup_table()
    mergeList = {face.index: None for face in mesh.faces}
    
    # -- Quad-dominant mesh: theoric edge collapse --
    
    # Tool used to know best candidate edge to be dissolved for each triangle
    optimalEdges = [min(face.edges, key=lambda x: squarenessList[x.index]) for face in mesh.faces]
    
    for i in range(l_faces):
        if mergeList[i] is not None: # Face i is already linked
            continue

        optimalEdge = optimalEdges[i]
        
        # Get the other face using the optimal edge
        j = optimalEdge.link_faces[0].index
        if i == j:
            j = optimalEdge.link_faces[1].index

        if mergeList[j] is not None: # Face j is already linked
            continue

        i_score = squarenessList[optimalEdge.index]
        j_score = squarenessList[optimalEdges[j].index]
        if i_score > j_score: # Face j have a better (smaller) match
            continue

        mergeList[i] = j
        mergeList[j] = i
    
    # -- Pure quad-mesh: theoric edge flip and collapse --
    # toPureQuad(mesh, squarenessList, mergeList)
    
    # -- Application: effective edge flip and collapse --

    # Tool used to know which face is already merged
    merged = [False] * l_faces

    edgesToRemove = []
    facesToAdd = []

    for i in range(l_faces):
        if merged[i] or mergeList[i] is None: # Face i is already merged or unmergeable
            continue
        
        j = mergeList[i]
        merged[i] = True
        merged[j] = True

        edgeToRemove, faceToAdd = merge(mesh.faces[i], mesh.faces[j])
        
        edgesToRemove.append(edgeToRemove)
        facesToAdd.append(faceToAdd)

    for edge in edgesToRemove:
        mesh.edges.remove(edge)

    for face in facesToAdd:
        meshFace = mesh.faces.new(face)
        correctQuad(meshFace)

    return mesh

if __name__ == "__main__":
    # Get the active mesh
    me = bpy.context.object.data

    # Get a BMesh representation
    bm = bmesh.new()  # create an empty BMesh
    bm.from_mesh(me)  # fill it in from a Mesh

    bm = triToQuad(bm)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  # free and prevent further access
    
    me.update()
