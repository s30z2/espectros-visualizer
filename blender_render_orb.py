"""
Blender script: Render a glossy glass orb with "DX" logo.
Outputs a single RGBA PNG sprite at high resolution.

Run: blender --background --python blender_render_orb.py
"""
import bpy
import math
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/Users/sebastiantobiascordova/Documents/ESPECTROS CONTENT"
OUT_PATH = os.path.join(OUT_DIR, "orb_3d.png")
ORB_SIZE = 512  # output image size

# ── CLEANUP ──
bpy.ops.wm.read_factory_settings(use_empty=True)

# ── RENDER SETTINGS ──
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 128
scene.render.resolution_x = ORB_SIZE
scene.render.resolution_y = ORB_SIZE
scene.render.film_transparent = True  # alpha background
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.filepath = OUT_PATH

# ── WORLD (dark with subtle blue environment) ──
world = bpy.data.worlds.new("DarkWorld")
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
bg = nodes.new('ShaderNodeBackground')
bg.inputs['Color'].default_value = (0.005, 0.01, 0.02, 1)
bg.inputs['Strength'].default_value = 0.3
output = nodes.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])

# ── GLASS ORB ──
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=64, ring_count=32, location=(0, 0, 0))
orb = bpy.context.active_object
orb.name = "GlassOrb"

# Subdivision for smoothness
mod = orb.modifiers.new("Subsurf", 'SUBSURF')
mod.levels = 2
mod.render_levels = 3
bpy.ops.object.shade_smooth()

# Glass material
mat = bpy.data.materials.new("GlassMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

output = nodes.new('ShaderNodeOutputMaterial')
# Mix: dark glossy base + glass for refraction
mix = nodes.new('ShaderNodeMixShader')
mix.inputs['Fac'].default_value = 0.4

# Dark glossy component
glossy = nodes.new('ShaderNodeBsdfGlossy')
glossy.inputs['Color'].default_value = (0.02, 0.04, 0.06, 1)
glossy.inputs['Roughness'].default_value = 0.08

# Glass component (slight teal tint)
glass = nodes.new('ShaderNodeBsdfGlass')
glass.inputs['Color'].default_value = (0.6, 0.85, 0.9, 1)
glass.inputs['Roughness'].default_value = 0.05
glass.inputs['IOR'].default_value = 1.45

links.new(glossy.outputs['BSDF'], mix.inputs[1])
links.new(glass.outputs['BSDF'], mix.inputs[2])
links.new(mix.outputs['Shader'], output.inputs['Surface'])

orb.data.materials.append(mat)

# ── "DX" TEXT (embossed on orb) ──
bpy.ops.object.text_add(location=(0, -1.05, 0), rotation=(math.pi/2, 0, 0))
text = bpy.context.active_object
text.data.body = "DX"
text.data.align_x = 'CENTER'
text.data.align_y = 'CENTER'
text.data.size = 0.65
text.data.extrude = 0.02
text.data.bevel_depth = 0.008

# Try to find a bold font
import glob
font_paths = glob.glob("/System/Library/Fonts/Supplemental/Arial Bold*") + \
             glob.glob("/System/Library/Fonts/Helvetica*")
if font_paths:
    try:
        text.data.font = bpy.data.fonts.load(font_paths[0])
    except:
        pass

# White emissive material for text
text_mat = bpy.data.materials.new("TextMat")
text_mat.use_nodes = True
tn = text_mat.node_tree.nodes
tl = text_mat.node_tree.links
tn.clear()
emission = tn.new('ShaderNodeEmission')
emission.inputs['Color'].default_value = (0.9, 0.95, 1.0, 1)
emission.inputs['Strength'].default_value = 3.0
tout = tn.new('ShaderNodeOutputMaterial')
tl.new(emission.outputs['Emission'], tout.inputs['Surface'])
text.data.materials.append(text_mat)

# Shrinkwrap text onto sphere surface
sw = text.modifiers.new("Shrinkwrap", 'SHRINKWRAP')
sw.target = orb
sw.wrap_method = 'PROJECT'
sw.use_project_z = True
sw.use_negative_direction = True
sw.offset = 0.03

# ── LIGHTING ──
# Key light (upper-left, cyan-white)
bpy.ops.object.light_add(type='AREA', location=(-2, -3, 3))
key = bpy.context.active_object
key.data.energy = 80
key.data.color = (0.8, 0.95, 1.0)
key.data.size = 3

# Rim light (right side, blue)
bpy.ops.object.light_add(type='AREA', location=(2.5, -1, 1))
rim = bpy.context.active_object
rim.data.energy = 40
rim.data.color = (0.4, 0.7, 1.0)
rim.data.size = 2

# Fill light (bottom, subtle warm)
bpy.ops.object.light_add(type='POINT', location=(0, -2, -2))
fill = bpy.context.active_object
fill.data.energy = 15
fill.data.color = (1.0, 0.9, 0.8)

# ── CAMERA ──
bpy.ops.object.camera_add(location=(0, -3.5, 0), rotation=(math.pi/2, 0, 0))
cam = bpy.context.active_object
scene.camera = cam
cam.data.lens = 85

# ── RENDER ──
print(f"[Blender] Rendering orb to {OUT_PATH} ...")
bpy.ops.render.render(write_still=True)
print(f"[Blender] Done! -> {OUT_PATH}")
