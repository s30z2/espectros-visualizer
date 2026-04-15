"""
Blender script: Render a 3D skull background with depth, lighting, and parallax layers.
Outputs multiple layers (far, mid, near) as RGBA PNGs.

Run: blender --background --python blender_render_bg.py
"""
import bpy
import bmesh
import math
import random
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/Users/sebastiantobiascordova/Documents/ESPECTROS CONTENT"
W, H = 1320, 2160  # slightly larger than 1080x1920 for parallax margin

# ── CLEANUP ──
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 64
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# ── WORLD ──
world = bpy.data.worlds.new("DarkWorld")
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
bg = nodes.new('ShaderNodeBackground')
bg.inputs['Color'].default_value = (0.003, 0.006, 0.01, 1)
bg.inputs['Strength'].default_value = 0.15
output = nodes.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

# ── SKULL MATERIAL (dark, bony, slight blue tint) ──
skull_mat = bpy.data.materials.new("SkullMat")
skull_mat.use_nodes = True
sn = skull_mat.node_tree.nodes
sl = skull_mat.node_tree.links
sn.clear()
principled = sn.new('ShaderNodeBsdfPrincipled')
principled.inputs['Base Color'].default_value = (0.06, 0.07, 0.09, 1)
principled.inputs['Roughness'].default_value = 0.7
principled.inputs['Specular IOR Level'].default_value = 0.3
sout = sn.new('ShaderNodeOutputMaterial')
sl.new(principled.outputs['BSDF'], sout.inputs['Surface'])


def create_skull_proxy(location, scale, rotation):
    """Create a simplified skull shape using basic meshes."""
    # Cranium (elongated sphere)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.5, segments=16, ring_count=12,
        location=location
    )
    cranium = bpy.context.active_object
    cranium.scale = (scale * 0.8, scale * 0.9, scale * 1.0)
    cranium.rotation_euler = rotation

    # Eye sockets (dark indentations)
    for side in [-0.15, 0.15]:
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.12 * scale, segments=8, ring_count=6,
            location=(location[0] + side * scale, location[1] - 0.35 * scale, location[2] + 0.1 * scale)
        )
        eye = bpy.context.active_object
        eye.scale = (1.0, 0.6, 0.8)
        eye.rotation_euler = rotation
        # Dark material for eye sockets
        dark_mat = bpy.data.materials.new(f"Dark_{id(eye)}")
        dark_mat.use_nodes = True
        dn = dark_mat.node_tree.nodes
        dn['Principled BSDF'].inputs['Base Color'].default_value = (0.01, 0.01, 0.015, 1)
        dn['Principled BSDF'].inputs['Roughness'].default_value = 0.9
        eye.data.materials.append(dark_mat)

    # Jaw area
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.3 * scale, segments=12, ring_count=8,
        location=(location[0], location[1] - 0.25 * scale, location[2] - 0.3 * scale)
    )
    jaw = bpy.context.active_object
    jaw.scale = (0.7, 0.5, 0.5)
    jaw.rotation_euler = rotation

    # Apply skull material to cranium and jaw
    cranium.data.materials.append(skull_mat)
    jaw.data.materials.append(skull_mat)

    return cranium


def render_layer(name, skulls_config, z_offset, blur_focus=None):
    """Render a layer of skulls."""
    # Delete previous skulls
    bpy.ops.object.select_all(action='SELECT')
    # Keep camera and lights
    for obj in bpy.data.objects:
        if obj.type in ('CAMERA', 'LIGHT'):
            obj.select_set(False)
    bpy.ops.object.delete()

    # Create skulls
    for cfg in skulls_config:
        loc = (cfg['x'], z_offset, cfg['y'])
        create_skull_proxy(loc, cfg['scale'], cfg['rot'])

    # Shade smooth all meshes
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()

    out = os.path.join(OUT_DIR, f"bg_3d_{name}.png")
    scene.render.filepath = out
    print(f"[Blender] Rendering {name} layer ({len(skulls_config)} skulls) ...")
    bpy.ops.render.render(write_still=True)
    print(f"[Blender] -> {out}")


# ── CAMERA (orthographic, looking down Y axis) ──
bpy.ops.object.camera_add(location=(0, -8, 0), rotation=(math.pi/2, 0, 0))
cam = bpy.context.active_object
scene.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 8  # controls visible area

# ── LIGHTING ──
# Key light (upper, cool blue)
bpy.ops.object.light_add(type='AREA', location=(0, -5, 4))
key = bpy.context.active_object
key.data.energy = 50
key.data.color = (0.5, 0.7, 1.0)
key.data.size = 6
key.rotation_euler = (math.radians(45), 0, 0)

# Rim light (lower, cyan)
bpy.ops.object.light_add(type='AREA', location=(2, -3, -3))
rim = bpy.context.active_object
rim.data.energy = 25
rim.data.color = (0.3, 0.8, 0.9)
rim.data.size = 4

# Subtle point light (central, for depth)
bpy.ops.object.light_add(type='POINT', location=(0, -4, 0))
center = bpy.context.active_object
center.data.energy = 15
center.data.color = (0.6, 0.85, 1.0)

# ── GENERATE SKULL CONFIGURATIONS ──
rng = random.Random(42)

# Aspect ratio of output
aspect = H / W  # ~1.636

def gen_skulls(n, spread_x, spread_y, scale_range):
    skulls = []
    for _ in range(n):
        skulls.append({
            'x': rng.uniform(-spread_x, spread_x),
            'y': rng.uniform(-spread_y, spread_y),
            'scale': rng.uniform(*scale_range),
            'rot': (rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4))
        })
    return skulls

# Far layer: small, many, deep
far_skulls = gen_skulls(35, 5.0, 5.0 * aspect, (0.25, 0.45))
# Mid layer: medium
mid_skulls = gen_skulls(20, 4.5, 4.5 * aspect, (0.4, 0.7))
# Near layer: large, few, close
near_skulls = gen_skulls(12, 4.0, 4.0 * aspect, (0.6, 1.0))

# ── RENDER LAYERS ──
render_layer("far", far_skulls, -2.0)
render_layer("mid", mid_skulls, -1.0)
render_layer("near", near_skulls, 0.0)

print("[Blender] All background layers rendered!")
