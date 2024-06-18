bl_info = {
    "name": "Adaptive Quad Mesh Simplification",
    "author": "Ahmed Hassayoune, Samuel Gonçalves",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "Object > Transform > AQMS",
    "description": "Apply the Adaptive Quad Mesh Simplification protocol",
    "category": "Object Transform",
}

from importlib import reload  # needed to import classes from other files

import bmesh
import bpy

import scripts.simplification as simplification
from scripts.tri_to_quad import triToQuad
from scripts.triangulation import triangulation

reload(simplification)


class OBJECT_OT_adaptive_quad_mesh_simplification(bpy.types.Operator):
    """Apply the Adaptive Quad Mesh Simplification protocol"""

    bl_idname = "transform.adaptive_quad_mesh_simplification"
    bl_label = "Adaptive Quad Mesh Simplification"
    bl_options = {"REGISTER", "UNDO"}

    triangulation_bool: bpy.props.BoolProperty(name="Triangulation ?", default=False)

    tri_to_quad_bool: bpy.props.BoolProperty(name="Triangles to quads ?", default=False)

    simplification_bool: bpy.props.BoolProperty(name="Simplification ?", default=False)

    simplification_verbose_bool: bpy.props.BoolProperty(name="Verbose ?", default=True)

    simplification_factor: bpy.props.IntProperty(
        name="Face limit",
        description="Factor for the mesh simplification",
        default=0,
        min=0,
        max=40000,
    )

    def execute(self, context):
        # Get the active mesh
        obj = context.object
        if obj is None or obj.type != "MESH":
            self.report({"WARNING"}, "No active mesh object")
            return {"CANCELLED"}

        me = obj.data

        # Get a BMesh representation
        bm = bmesh.new()  # create an empty BMesh
        bm.from_mesh(me)  # fill it in from a Mesh

        if self.triangulation_bool:
            bm = triangulation(bm)
        if self.tri_to_quad_bool:
            bm = triToQuad(bm)
        if self.simplification_bool:
            bm.normal_update()
            bm = simplification.MeshSimplifier(
                bm, verbose=self.simplification_verbose_bool
            ).simplify_mesh(nb_faces=self.simplification_factor)

        # Finish up, write the bmesh back to the mesh
        bm.to_mesh(me)
        bm.free()  # free and prevent further access

        me.update()

        return {"FINISHED"}


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
