bl_info = {
    "name": "Adaptive Quad Mesh Simplification",
    "author": "Ahmed Hassayoune, Samuel Gonçalves",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "Object > Transform > AQMS",
    "description": "Apply the Adaptive Quad Mesh Simplification protocol",
    "category": "Object Transform",
}

import bpy
import bmesh

from .tri_to_quad import triToQuad

class OBJECT_OT_adaptive_quad_mesh_simplification(bpy.types.Operator):
    """Apply the Adaptive Quad Mesh Simplification protocol"""
    bl_idname = "transform.adaptive_quad_mesh_simplification"
    bl_label = "Adaptive Quad Mesh Simplification"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Get the active mesh
        obj = context.object
        if obj is None or obj.type != 'MESH':
            self.report({'WARNING'}, "No active mesh object")
            return {'CANCELLED'}
        
        me = obj.data

        # Get a BMesh representation
        bm = bmesh.new()  # create an empty BMesh
        bm.from_mesh(me)  # fill it in from a Mesh

        bm = triToQuad(bm) # adaptativeQuadMeshSimplificationProtocol(bm)

        # Finish up, write the bmesh back to the mesh
        bm.to_mesh(me)
        bm.free()  # free and prevent further access

        me.update()
        
        return {'FINISHED'}

# Registration

def menu_func(self, context):
    self.layout.operator(OBJECT_OT_adaptive_quad_mesh_simplification.bl_idname)

def register():
    bpy.utils.register_class(OBJECT_OT_adaptive_quad_mesh_simplification)
    bpy.types.VIEW3D_MT_transform_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_adaptive_quad_mesh_simplification)
    bpy.types.VIEW3D_MT_transform_object.remove(menu_func)

if __name__ == "__main__":
    register()
