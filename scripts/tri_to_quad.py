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
        raise RuntimeError('Call to prepareTriangle with a closed mesh (not two-manifold mesh ?).') 
    
    faceToDivide = borderEdge.link_faces[0]
    
    # Search of the opposed vertex
    opposedVert = None
    for vert in faceToDivide.verts:
        if vert not in borderEdge.verts:
            opposedVert = vert
            break

    # Mesh modification
    middle = mesh.verts.new((borderEdge.verts[0].co + borderEdge.verts[1].co) / 2)
    mesh.faces.new([middle, borderEdge.verts[0], opposedVert])
    mesh.faces.new([middle, borderEdge.verts[1], opposedVert])
    
    mesh.edges.remove(borderEdge)
    
    return mesh

def merge(
    t0: bmesh.types.BMFace, t1: bmesh.types.BMFace,
) -> (bmesh.types.BMEdge, [bmesh.types.BMVert]):
    """
    Returns the edge to remove and the vertices of the theoric face that
    would result if the triangle parameter faces were merged.
    """
    for edge in t0.edges:
        if edge in t1.edges:
            commonEdge = edge
            break
    
    for vert in t0.verts:
        if vert not in commonEdge.verts:
            vert_t0 = vert
            break

    for vert in t1.verts:
        if vert not in commonEdge.verts:
            vert_t1 = vert
            break

    # Vertices in the right order
    verts = [vert_t0, commonEdge.verts[0], vert_t1, commonEdge.verts[1]]

    return commonEdge, verts

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

def neighborsOf(
    triangle0: bmesh.types.BMFace,
    triangle1: bmesh.types.BMFace
) -> [bmesh.types.BMFace]:
    """
    Returns the list of neighbors faces from the theoric face that would
    result if t0 and t1 were merged.
    """
    triangles = [triangle0, triangle1]
    neighbors = []

    for triangle in triangles:
        for edge in triangle.edges:
            if not edge.is_boundary: # Else, the only face linked to it is triangle
                for i in range(2):
                    face = edge.link_faces[i]
                    if face not in triangles and face not in neighbors:
                        neighbors.append(face)
    
    return neighbors

def commonEdgeOf(
    face1: [bmesh.types.BMFace],
    face2: [bmesh.types.BMFace]
) -> (bmesh.types.BMEdge, bmesh.types.BMFace, bmesh.types.BMFace):
    """
    Returns a common edge between the two theoric faces, each composed of triangles, and the
    subfaces of the common edge.
    """
    for subface1 in face1:
        for subface2 in face2:
            for edge in subface1.edges:
                if edge in subface2.edges:
                    return edge, subface1, subface2

    raise RuntimeError('No common edge between the two theoric faces.')

def getShortestPathToTriangleFrom(
    mesh: bmesh.types.BMesh,
    triangle: bmesh.types.BMFace,
    mergeList,
    squarenessList
) -> [bmesh.types.BMFace]:
    """
    Find the shortest* path between the unlinked parameter triangle and the
    first other unlinked triangle found by a breadth first visit.
    
    * shortest : the path with the smaller number of faces visited. Ties are
                 broken in favor of the minimal squareness score.
    Returns:
    - [bmesh.types.BMFace]: The visited faces, including the given triangle.
                            Its length is always even (first + pairs + last).
    """
    # Tool used to know the shortest path from triangle to any face
    paths = {face: None for face in mesh.faces}

    # Initial queue
    queue = [triangle]

    # This will mask triangle to avoid to take it at the end of the path
    # WARNING : This will cause triangle to appear twice on paths (corrected here *)
    paths[triangle] = [triangle, triangle]

    # Breadth first visit of faces
    while queue:
        current = queue.pop(0)

        # A triangle is reached
        if len(paths[current]) % 2 == 1:
            # (*) Remove the twice first triangle occurence
            paths[current].pop(0)
            return paths[current]

        # Neighbors of the theoric rectangle
        neighbors = neighborsOf(current, paths[current][-1])
        
        # Visit on all neighbors
        for neighbor in neighbors:
            if paths[neighbor] is not None:
                # The actual path to neighbor is shorter 
                if len(paths[neighbor]) < len(paths[current]) + 1:
                    continue

                # Here, the actual path to neighbor have the same number of faces
                # ----> Squareness tie break <----   

                # The neighbor is an unlinked triangle -> Squareness of the common edge
                # between the triangle and the previous rectangle
                if len(paths[neighbor]) % 2 == 1:
                    destination = [paths[neighbor][-1]]
                    actualOrigin = [paths[neighbor][-2], paths[neighbor][-3]]
                    newOrigin = [paths[current][-1], paths[current][-2]]

                # The neighbor is a rectangle -> Squareness of the common edge
                # between the rectangle and the previous rectangle
                else:
                    destination = [paths[neighbor][-1], paths[neighbor][-2]]
                    actualOrigin = [paths[neighbor][-3], paths[neighbor][-4]]
                    newOrigin = [paths[current][-1], paths[current][-2]]

                actualEdge,_,_ = commonEdgeOf(actualOrigin, destination)
                newEdge,_,_ = commonEdgeOf(newOrigin, destination)

                actualSquareness = squarenessList[actualEdge]
                newSquareness = squarenessList[newEdge]
                
                # The actual path to neighbor is shorter
                if actualSquareness <= newSquareness:
                    continue

                # ----> End of the squareness tie break <----
            else:
                queue.append(neighbor)

            paths[neighbor] = paths[current] + [neighbor]
            
            # The linked face must be taken in the path
            neighborLink = mergeList[neighbor]
            if neighborLink is not None:
                paths[neighbor].append(neighborLink)
                paths[neighborLink] = paths[neighbor]

    print(paths)
    raise RuntimeError('Breadth first visit failed to find another reachable triangle (not connected mesh ?).')

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
        - pure quad-mesh: theoric edge flip and collapse (with some effective changes)
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
    if len(mesh.faces) % 2 == 1:
        # Transform the mesh to a redefinition of it with an even number of triangles.
        prepareTriangle(mesh)
    
    # Tool used to preserve the overall shape and structure
    mesh.edges.ensure_lookup_table()
    squarenessList = {edge: squareness(edge) for edge in mesh.edges}

    # Tool used to realise the quad mesh conversion
    mesh.faces.ensure_lookup_table()
    mergeList = {face: None for face in mesh.faces}
    
    # -- Quad-dominant mesh: theoric edge collapse --
    
    # Tool used to know best candidate edge to be dissolved for each triangle
    optimalEdges = {face: min(face.edges, key=lambda x: squarenessList[x]) for face in mesh.faces}
    
    for face in mesh.faces:
        if mergeList[face] is not None: # Current face is already linked
            continue

        optimalEdge = optimalEdges[face]
        
        if optimalEdge.is_boundary: # Current face is isolated
            continue
        
        # Get the other face using the optimal edge
        otherFace = optimalEdge.link_faces[0]
        if face == otherFace:
            otherFace = optimalEdge.link_faces[1]

        if mergeList[otherFace] is not None: # Optimal link of the current face is already linked
            continue

        face_score = squarenessList[optimalEdge]
        otherFace_score = squarenessList[optimalEdges[otherFace]]
        if face_score > otherFace_score: # Optimal link of the current face has a better link (= smaller score)
            continue

        mergeList[face] = otherFace
        mergeList[otherFace] = face
    
    # -- Pure quad-mesh: theoric edge flip and collapse (with some effective changes) --
    
    while True:
        # > Stop condition of the iteration
        existingTriangle = False
        for face in mesh.faces:
            if mergeList[face] is not None:  # Current face is already linked
                continue
            else:
                existingTriangle = True
                break
        if not existingTriangle:
            break

        # > Current face is an unlinked triangle (the iteration continue)

        # Breadth first visit to find path to the nearest unlinked triangle
        path = getShortestPathToTriangleFrom(mesh, face, mergeList, squarenessList)

        i = 0
        l_path = len(path)
        while i+3 < l_path:
            # From (i) (i+1 i+2) (i+3) to (i i+1) (i+2) (i+3)
            # From  A    B - C     D   to  A - B    C     D
            
            _,_,subfaceA_BC = commonEdgeOf([path[i]], [path[i+1], path[i+2]])
            _,_,subfaceBC_D = commonEdgeOf([path[i+3]], [path[i+1], path[i+2]])

            # For a same path A-BC-D there are two possibilities:
            
            # Case 0 -> subfaceA_BC != subfaceBC_D 
            #        -> A connected to B, D connected to C
            #        => Simple theoric edge flip between IJ and JK
            # x-------J-------x           x-------x-------x
            #  \  A  / .  C  / \           \  A  . \  C  / \
            #   \   /   .   /   \    ->     \   .   \   /   \
            #    \ /  B  . /  D  \           \ .  B  \ /  D  \
            #     I-------K-------x           x-------x-------x

            if subfaceA_BC == subfaceBC_D:
                # Case 1 -> subfaceA_BC == subfaceBC_D 
                #        -> A connected to B, D not connected to C
                #        => effective modification of triangles B and C
                #        ==> now, subfaceA_BC != subfaceBC_D (=> simple theoric edge flip between IJ and JK)
                # x-------2-------x           x-------J-------x           x-------J-------x
                #  \  A  / \  D  /             \  A  /.\  D  /             \  A  .|\  D  /
                #   \   /   \   /               \   / . \   /               \   . | \   /
                #    \ /  B  \ /                 \ /  .  \ /                 \ .  |  \ /
                #     1 . . . 1        ->         I B'.C' L        ->         I B'|C' L
                #      \  C  /                     \  .  /                     \  |  /
                #       \   /                       \ . /                       \ | /
                #        \ /                         \./                         \|/
                #         0                           K                           K
                
                # Get the other subface
                subfaceBC_D = path[i+1]
                if subfaceA_BC == subfaceBC_D:
                    subfaceBC_D = path[i+2]
                    
                # Analyse vertex usage (I,J,K and L are indicated by the schema)
                commonEdge,verts = merge(subfaceA_BC, subfaceBC_D)
                for vert in subfaceA_BC.verts:
                    if vert not in commonEdge.verts:
                        vertJ = vert
                        break
                for vert in subfaceBC_D.verts:
                    if vert not in commonEdge.verts:
                        vertK = vert
                        break
                vertI,vertL = commonEdge.verts[0], commonEdge.verts[1]
                if vertI in path[i+3].verts:
                    vertI,vertL = vertL,vertI

                # Tools update
                squarenessList.pop(commonEdge)
                mergeList.pop(subfaceA_BC)
                mergeList.pop(subfaceBC_D)
                
                # Mesh update
                mesh.edges.remove(commonEdge)
                newEdge = mesh.edges.new((vertJ, vertK))
                mesh.edges.ensure_lookup_table()
                
                newFaceB = mesh.faces.new([vertI, vertJ, vertK])
                newFaceC = mesh.faces.new([vertJ, vertK, vertL])
                mesh.faces.ensure_lookup_table()

                squarenessList[newEdge] = squareness(newEdge)

                # Conversion to a "case 0" A-BC-D path
                subfaceA_BC = newFaceB
                subfaceBC_D = newFaceC

            # Theoric edge flip
            mergeList[path[i]] = subfaceA_BC
            mergeList[subfaceA_BC] = path[i]
            
            path[i+2] = subfaceBC_D
            i += 2

        if i+1 < l_path:
            # From (i) (i+1) to (i i+1)
            mergeList[path[i]] = path[i+1]
            mergeList[path[i+1]] = path[i]        
    
    # -- Application: effective edge flip and collapse --

    # Tool used to know which face is already merged
    merged = {face: False for face in mesh.faces}

    edgesToRemove = []
    facesToAdd = []

    for face in mesh.faces:
        if merged[face] or mergeList[face] is None: # Face i is already merged or unmergeable
            continue
        
        otherFace = mergeList[face]
        merged[face] = True
        merged[otherFace] = True

        edgeToRemove, faceToAdd = merge(face, otherFace)
        
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
