import bpy
import bmesh
import mathutils
from math import acos, cos, sin, sqrt

def verticesInOrder(face: bmesh.types.BMFace) -> [bmesh.types.BMVert]:
    """
    Returns the list of vertices on the face so that two successive
    vertices are linked by an edge (including the first and the last one).
    """
    verticesSorted = []
    actualVertex = face.verts[0]

    size = len(face.verts)
    while True:
        # Add the vertex to the list
        verticesSorted.append(actualVertex)
        if len(verticesSorted) == size:
            break

        # Follow the edge to get the next vertex
        for edge in actualVertex.link_edges:
            if edge in face.edges and edge.other_vert(actualVertex) not in verticesSorted:
                actualVertex = edge.other_vert(actualVertex)
                break

    return verticesSorted

def inTriangle(
    point: mathutils.Vector,
    triangle: (mathutils.Vector, mathutils.Vector, mathutils.Vector)
) -> bool:
    """
    Returns if point is in the theoric triangle formed by the
    three points of triangle.

    Let's call ABC the triangle and P the point. Points are
    colinear so AP = k * AB + m * AC, and "P in ABC" means
    that k >= 0 and m >= 0 and k + m <= 1.
    """
    A, B, C = triangle
    AB = B - A
    AC = C - A
    AP = point - A

    if AB.x == 0 and AC.x == 0:
        # xAP = 0
        # yAP = k * yAB + m * yAC
        return AP.x == 0 and min(AB.y, AC.y) < AP.y and AP.y < max(AB.y, AC.y)

    if AB.y == 0 and AC.y == 0:
        # xAP = k * xAB + m * xAC
        # yAP = 0
        return AP.y == 0 and min(AB.x, AC.x) < AP.x and AP.x < max(AB.x, AC.x)

    if AB.x == 0:
        # xAP = m * xAC
        # yAP = k * yAB + m * yAC
        m = AP.x / AC.x
        k = (AP.y - m * AC.y) / AB.y


    # | xAP = k * xAB + m * xAC <=> | (xAP - m * xAC) / xAB = k
    # | yAP = k * yAB + m * yAC     | yAP = k * yAB + m * yAC
    
    # yAP = {(xAP - m * xAC) / xAB} * yAB + m * yAC <=>
    # yAP = xAP * yAB/xAB - m * xAC * yAB/xAB + m * yAC <=>
    # yAP - xAP * yAB/xAB =  m (yAC - xAC * yAB/xAB) <=>
    elif AB.x * AC.y == AC.x * AB.y:
        # AB and AC are colinear
        # | xAP = (k + m * k') * xAB
        # | yAP = (k + m * k') * yAB
        return AP.x * AB.y == AB.x * AP.y and min(AB.x, AC.x) < AP.x and AP.x < max(AB.x, AC.x)
    else:
        m = (AP.y - AP.x * AB.y / AB.x) / (AC.y - AC.x * AB.y / AB.x)
        k = (AP.x - m * AC.x) / AB.x
    
    return k >= 0 and m >= 0 and k + m <= 1

def recTriangulation(
    coords: [(mathutils.Vector, bmesh.types.BMVert)],
    trianglesToAdd: [[bmesh.types.BMVert]],
    biggestYUnknown: bool = True
):
    """
    Recursive implementation of the triangulation of the face represented by coords based on the two ears theorem.
    The result is added in the trianglesToAdd list.

    To achieve triangulation it applies the following local operations :
    - if the face has three vertices, the triangulation is over
    - else :
        - take the point with the biggest Y value
        - try to create a ear from it :
            - success : the ear is added to the list, recursion without the ear
            - failure : the point is theoretically linked with its nearest blocking point, recursion on the two
                        new faces.
    """
    size = len(coords)

    # -- If the face has three vertices, the triangulation is over --
    
    if size == 3:
        _, p0 = coords[0]
        _, p1 = coords[1]
        _, p2 = coords[-1]
        trianglesToAdd.append([p0, p1, p2])
        return

    if biggestYUnknown:
        # -- Take the point with the biggest Y value --
        
        iMax, yMax = None, None
        for i in range(size):
            coord, vert = coords[i]
            if yMax is None or coord.y > yMax:
                iMax = i
                yMax = coord.y
        coords = coords[iMax:] + coords[:iMax]

    c0, p0 = coords[0]
    c1, p1 = coords[1]
    c2, p2 = coords[-1]

    # -- Try to create a ear from it --

    intersection = None

    for i in range(2, len(coords) - 1):
        ci, vi = coords[i]

        if inTriangle(ci, (c0, c1, c2)):
            if intersection is None:
                intersection = i
            else:
                cIntersection, _ = coords[intersection]
                if (cIntersection - c0).length > (ci - c0).length:
                    intersection = i

    # -- The ear is added to the list, recursion without the ear --

    if intersection is None:
        trianglesToAdd.append([p0, p1, p2])
        recTriangulation(coords[1:], trianglesToAdd)
        return

    else:
        # The point with the biggest Y value is still in the list:
        # No need to recalculate it.
        recTriangulation(coords[:intersection+1], trianglesToAdd, False)
        recTriangulation([coords[0]] + coords[intersection:], trianglesToAdd, False)
        return

def triangulation(mesh: bmesh.types.BMesh) -> bmesh.types.BMesh:
    """
    Apply triangulation of the mesh to prepare the application of the
    triToQuad mesh conversion.

    This function convert the input polygonal mesh to a triangle mesh while
    preserving the shape and structure of the mesh. The algorithm used requires
    all the vertices of a face to be coplanar in the input polygonal mesh and
    that no vertices overlap.

    To achieve triangulation it applies the following local operations :
        - vertices projection on XY: translation and rotation
        - recursive implementation based on two ears theorem: theoric face collapse
        - application: face creation and removal

    Parameters:
    - mesh (bmesh.types.BMesh): The input BMesh object representing
                                the mesh to be transformed.

    Returns:
    - bmesh.types.BMesh: The triangle BMesh object.

    Note:
    This function modifies the input mesh in-place and returns the same object.
    """
    mesh.faces.ensure_lookup_table()

    # Lists used to realise the triangulation
    polygonsToRemove = []
    trianglesToAdd = []

    # Selection of non-triangle faces
    for face in mesh.faces:
        if len(face.verts) != 3:
            polygonsToRemove.append(face)

    for face in polygonsToRemove:
        face_verts = verticesInOrder(face)

        # -- Vertices projection on XY: translation and rotation --

        coords = []
        for vert in face_verts:
            coords.append((vert.co.copy(), vert))

        # Translation to put the first vertex as the origin
        origin, _ = coords[0]
        origin = origin.copy()
        if origin.x != 0 or origin.y != 0 or origin.z != 0 :
            for coord, _ in coords:
                coord -= origin

        # Rotation to put the second vertex on the X axis
        R_BX = mathutils.Matrix.Identity(3)

        toAxisX, _ = coords[1]
        if toAxisX.y != 0 or toAxisX.z != 0:
            normalBX = mathutils.Vector((0, toAxisX.z, -toAxisX.y)).normalized()
            angle = acos( toAxisX.x / toAxisX.length )
            Q_BX = mathutils.Matrix([[0, -normalBX.z, normalBX.y], [normalBX.z, 0, -normalBX.x], [-normalBX.y, normalBX.x, 0]])
            R_BX += sin(angle) * Q_BX + (1 - cos(angle)) * (Q_BX @ Q_BX)

        # Rotation to put the third vertex on the XY plane
        R_CX = mathutils.Matrix.Identity(3)

        toPlaneXY, _ = coords[2]
        new_C = R_BX @ toPlaneXY
        if new_C.z != 0:
            sqrtYZ = sqrt(new_C.y * new_C.y + new_C.z * new_C.z)
            Y_sqrt = new_C.y / sqrtYZ
            Z_sqrt = new_C.z / sqrtYZ
            R_CX = mathutils.Matrix([[1, 0, 0], [0, Y_sqrt, Z_sqrt], [0, Z_sqrt, -Y_sqrt]])

        R = R_CX @ R_BX

        # If all the vertices are coplanar, the rotation put them all on the XY plane
        for i in range(len(coords)):
            coord, vert = coords[i]
            coords[i] = (R @ coord, vert)

        # -- Recursive implementation based on two ears theorem: theoric face collapse --

        recTriangulation(coords, trianglesToAdd)

    # -- Application: face creation and removal --

    for face in polygonsToRemove:
        mesh.faces.remove(face)

    for face in trianglesToAdd:
        mesh.faces.new(face)

    return mesh

if __name__ == "__main__":
    # Get the active mesh
    me = bpy.context.object.data

    # Get a BMesh representation
    bm = bmesh.new()  # create an empty BMesh
    bm.from_mesh(me)  # fill it in from a Mesh

    bm = triangulation(bm)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  # free and prevent further access
    
    me.update()
