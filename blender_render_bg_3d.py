"""
Blender script: Render a 3D displacement-based skull background.

Takes the existing AI-generated skulls_bg_gemini.png and extrudes it into
real 3D geometry via a displacement modifier on a high-poly plane. Renders
both a color pass and a linear depth pass so audio_visualizer.py can do
true-3D parallax (per-pixel shift driven by actual Z) and depth-based DOF
blur at runtime.

Keeps the Gemini palette (which scores 8/10 consistently) while adding the
real Z-depth the reference video has.

Outputs (to the script directory):
  - bg_3d_scene.png  — color pass, 1320x2160 RGBA
  - bg_3d_depth.png  — linear depth, 16-bit grayscale PNG, near=white/far=black

Run: blender --background --python blender_render_bg_3d.py
"""
import bpy
import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
SRC_IMAGE = os.path.join(SCRIPT_DIR, "skulls_bg_gemini.png")
OUT_COLOR = os.path.join(SCRIPT_DIR, "bg_3d_scene.png")
OUT_DEPTH = os.path.join(SCRIPT_DIR, "bg_3d_depth.png")

W, H = 1320, 2160  # matches audio_visualizer BG_MARGIN-padded size

# Displacement strength — how far the bright pixels push out. Tuned for
# the skulls to pop ~1-2 "head radii" forward from the plane.
DISP_STRENGTH = 1.2

# Plane subdivision — more = finer displacement detail, slower render.
# 512 cuts on the long axis gives ~4px per cut at 2160 height.
PLANE_SUBDIV = 8  # 8 levels of subsurf from a 2x2 plane = 513x513 verts

# Camera distance — farther = less perspective parallax, closer = more.
CAM_Z = 6.0

# ── RESET ──
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ── RENDER SETTINGS ──
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 96
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.color_depth = '16'

# Enable depth pass on the view layer
view_layer = scene.view_layers[0]
view_layer.use_pass_z = True

# ── WORLD (very dark, slight cool ambient) ──
world = bpy.data.worlds.new("DarkWorld")
scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
wn.clear()
wbg = wn.new('ShaderNodeBackground')
wbg.inputs['Color'].default_value = (0.004, 0.008, 0.012, 1.0)
wbg.inputs['Strength'].default_value = 0.2
wout = wn.new('ShaderNodeOutputWorld')
wl.new(wbg.outputs['Background'], wout.inputs['Surface'])

# ── PLANE (portrait aspect, high poly via subsurf) ──
aspect = H / W  # ~1.636
plane_w = 4.0
plane_h = plane_w * aspect
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
plane = bpy.context.active_object
plane.scale = (plane_w, plane_h, 1)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Stand plane up facing +Y so camera at (0, -CAM_Z, 0) sees it
plane.rotation_euler = (math.radians(90), 0, 0)
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

# Subsurface modifier for displacement detail
subsurf = plane.modifiers.new("Subsurf", type='SUBSURF')
subsurf.subdivision_type = 'SIMPLE'
subsurf.levels = PLANE_SUBDIV
subsurf.render_levels = PLANE_SUBDIV

# ── LOAD IMAGE ──
if not os.path.isfile(SRC_IMAGE):
    raise RuntimeError(f"Source image not found: {SRC_IMAGE}")
img = bpy.data.images.load(SRC_IMAGE)

# ── DISPLACEMENT MODIFIER ──
# Use the image brightness as a height map. Bright areas (skulls) push out.
disp_tex = bpy.data.textures.new("SkullDisp", type='IMAGE')
disp_tex.image = img
disp_tex.use_interpolation = True
disp_tex.filter_type = 'EWA'

disp = plane.modifiers.new("Displace", type='DISPLACE')
disp.texture = disp_tex
disp.texture_coords = 'UV'
disp.strength = DISP_STRENGTH
disp.mid_level = 0.15  # below this luminance = pushed back; above = pushed forward
disp.direction = 'NORMAL'

# ── MATERIAL ──
mat = bpy.data.materials.new("SkullMat")
mat.use_nodes = True
mn = mat.node_tree.nodes
ml = mat.node_tree.links
mn.clear()

tex_coord = mn.new('ShaderNodeTexCoord')
mapping = mn.new('ShaderNodeMapping')
img_node = mn.new('ShaderNodeTexImage')
img_node.image = img
img_node.interpolation = 'Cubic'

principled = mn.new('ShaderNodeBsdfPrincipled')
principled.inputs['Roughness'].default_value = 0.65
principled.inputs['Specular IOR Level'].default_value = 0.35
# Slight subsurface scattering to give the "bony" quality
if 'Subsurface Weight' in principled.inputs:
    principled.inputs['Subsurface Weight'].default_value = 0.08
    principled.inputs['Subsurface Radius'].default_value = (0.3, 0.2, 0.15)

mat_out = mn.new('ShaderNodeOutputMaterial')

ml.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
ml.new(mapping.outputs['Vector'], img_node.inputs['Vector'])
ml.new(img_node.outputs['Color'], principled.inputs['Base Color'])
ml.new(principled.outputs['BSDF'], mat_out.inputs['Surface'])

plane.data.materials.append(mat)
bpy.context.view_layer.objects.active = plane
bpy.ops.object.shade_smooth()

# ── CAMERA ──
bpy.ops.object.camera_add(location=(0, -CAM_Z, 0), rotation=(math.pi/2, 0, 0))
cam = bpy.context.active_object
scene.camera = cam
cam.data.type = 'PERSP'
cam.data.lens = 50  # 50mm-equivalent, mild perspective for parallax
cam.data.sensor_width = 36

# Match the render aspect so no cropping
cam.data.sensor_fit = 'VERTICAL'

# ── LIGHTING (cool teal key + warm cyan rim) ──
# Key light: upper-left, cool
bpy.ops.object.light_add(type='AREA', location=(-3, -4, 3.5))
key = bpy.context.active_object
key.data.energy = 180
key.data.color = (0.55, 0.75, 1.0)
key.data.size = 5
key.rotation_euler = (math.radians(55), 0, math.radians(-20))

# Rim light: lower-right, cyan
bpy.ops.object.light_add(type='AREA', location=(3.5, -3, -2))
rim = bpy.context.active_object
rim.data.energy = 90
rim.data.color = (0.3, 0.85, 1.0)
rim.data.size = 4
rim.rotation_euler = (math.radians(-45), 0, math.radians(20))

# Fill: soft cyan center
bpy.ops.object.light_add(type='POINT', location=(0, -3, 0))
fill = bpy.context.active_object
fill.data.energy = 40
fill.data.color = (0.5, 0.8, 1.0)

# ── COMPOSITOR: output color + normalized depth ──
scene.use_nodes = True
tree = scene.node_tree
for node in tree.nodes:
    tree.nodes.remove(node)

rlayers = tree.nodes.new('CompositorNodeRLayers')

# Color output (composite + file output)
composite = tree.nodes.new('CompositorNodeComposite')
tree.links.new(rlayers.outputs['Image'], composite.inputs['Image'])

# File output for color pass
color_out = tree.nodes.new('CompositorNodeOutputFile')
color_out.base_path = SCRIPT_DIR
color_out.file_slots[0].path = "bg_3d_scene_"
color_out.format.file_format = 'PNG'
color_out.format.color_mode = 'RGBA'
color_out.format.color_depth = '16'
tree.links.new(rlayers.outputs['Image'], color_out.inputs[0])

# Depth pass: normalize so near=1, far=0, write as 16-bit grayscale PNG
# Z values are in world-space from camera. Map [CAM_Z - DISP_STRENGTH, CAM_Z + 1.0]
# to [1, 0] for a useful parallax range.
z_near = CAM_Z - DISP_STRENGTH * 1.1   # closest displaced vert
z_far = CAM_Z + 0.5                     # mid-level / back of plane

# Normalize: normalized = (z_far - z) / (z_far - z_near)
# Use Map Range node for clarity.
map_range = tree.nodes.new('CompositorNodeMapRange')
map_range.inputs['From Min'].default_value = z_near
map_range.inputs['From Max'].default_value = z_far
map_range.inputs['To Min'].default_value = 1.0   # near -> white
map_range.inputs['To Max'].default_value = 0.0   # far  -> black
map_range.use_clamp = True

tree.links.new(rlayers.outputs['Depth'], map_range.inputs['Value'])

depth_out = tree.nodes.new('CompositorNodeOutputFile')
depth_out.base_path = SCRIPT_DIR
depth_out.file_slots[0].path = "bg_3d_depth_"
depth_out.format.file_format = 'PNG'
depth_out.format.color_mode = 'BW'
depth_out.format.color_depth = '16'
tree.links.new(map_range.outputs['Value'], depth_out.inputs[0])

# ── RENDER ──
print(f"[Blender] Rendering 3D displacement scene ({W}x{H}) ...")
print(f"[Blender]   source:       {SRC_IMAGE}")
print(f"[Blender]   disp strength: {DISP_STRENGTH}")
print(f"[Blender]   subsurf levels: {PLANE_SUBDIV}  ({2**PLANE_SUBDIV + 1}x{2**PLANE_SUBDIV + 1} verts)")
print(f"[Blender]   samples:      {scene.cycles.samples}")
scene.render.filepath = os.path.join(SCRIPT_DIR, "_bg_3d_tmp.png")
bpy.ops.render.render(write_still=True)

# FileOutput nodes append the frame number (0001). Rename to stable names.
def _finalize(prefix, dest):
    # FileOutput writes e.g. bg_3d_scene_0001.png
    frame = scene.frame_current
    written = os.path.join(SCRIPT_DIR, f"{prefix}{frame:04d}.png")
    if os.path.isfile(written):
        if os.path.isfile(dest):
            os.remove(dest)
        os.rename(written, dest)
        print(f"[Blender] -> {dest}")
    else:
        print(f"[Blender] WARNING: expected {written} not found")

_finalize("bg_3d_scene_", OUT_COLOR)
_finalize("bg_3d_depth_", OUT_DEPTH)

# Clean up the intermediate Composite output
tmp = os.path.join(SCRIPT_DIR, "_bg_3d_tmp.png")
if os.path.isfile(tmp):
    os.remove(tmp)

print("[Blender] Done.")
