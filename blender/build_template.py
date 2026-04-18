"""
Programmatically build template_scene.blend — the base 3D scene for the phonk visualizer.
No manual UI work: camera, lights, orb, skull wall, waveform ring, materials, render settings,
all via the bpy Python API. Run once to create/refresh the template:

    blender --background --python blender/build_template.py

Output:
    blender/template_scene.blend

The scene per Grok's spec:
- Camera at (0, -8, 1.2), ~80° rotation → 9:16 vertical framing
- Skull-wall BG plane at Z=-6 (uses skulls_bg_gemini.png as texture)
- Central orb: icosphere + glass shader + emissive inner core
- Waveform ring: 360-vertex torus with Geometry Nodes audio-reactive displacement driver
- Rim lights (2) + world ambient (dark, palette-driven)
- Eevee-Next, 1080x1920 @ 60fps, Bloom + Motion Blur + DoF on
- Custom scene properties: palette_r/g/b (Python will drive these)
"""
from __future__ import annotations
import bpy
import math
import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BLEND_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BLEND_DIR.parent
BG_TEXTURE = PROJECT_ROOT / "skulls_bg_gemini.png"
TEMPLATE_OUT = BLEND_DIR / "template_scene.blend"
RENDER_OUT_DIR = BLEND_DIR / "renders"

# ------------------------------------------------------------------
# Reset scene
# ------------------------------------------------------------------
def clear_scene():
    """Nuke every object, material, texture, node tree. Start clean."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for block in list(bpy.data.materials):  bpy.data.materials.remove(block)
    for block in list(bpy.data.textures):   bpy.data.textures.remove(block)
    for block in list(bpy.data.meshes):     bpy.data.meshes.remove(block)
    for block in list(bpy.data.objects):    bpy.data.objects.remove(block)
    for block in list(bpy.data.cameras):    bpy.data.cameras.remove(block)
    for block in list(bpy.data.lights):     bpy.data.lights.remove(block)
    for block in list(bpy.data.images):     bpy.data.images.remove(block)
    for block in list(bpy.data.node_groups): bpy.data.node_groups.remove(block)

# ------------------------------------------------------------------
# Camera — vertical 9:16
# ------------------------------------------------------------------
def setup_camera():
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 2.8
    cam_data.dof.aperture_fstop = 1.2  # strong DoF blur on back skulls
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0, -8, 1.2)
    cam.rotation_euler = (math.radians(82), 0, 0)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam

# ------------------------------------------------------------------
# Skull-wall background — textured plane
# ------------------------------------------------------------------
def setup_skull_wall():
    bpy.ops.mesh.primitive_plane_add(size=18, location=(0, 2.5, 0.5))
    plane = bpy.context.active_object
    plane.name = "SkullWall"
    plane.rotation_euler = (math.radians(90), 0, 0)
    plane.scale = (1, 1.9, 1)  # taller for 9:16

    # Material: emission of skull texture (so BG is self-lit, not dependent on lamps)
    mat = bpy.data.materials.new("SkullWallMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-800, 0)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-600, 0)

    img_tex = nt.nodes.new("ShaderNodeTexImage")
    img_tex.location = (-400, 0)
    if BG_TEXTURE.is_file():
        try:
            img_tex.image = bpy.data.images.load(str(BG_TEXTURE))
            img_tex.image.colorspace_settings.name = "Non-Color"
        except Exception as e:
            print(f"[!] Could not load BG texture: {e}")

    emission = nt.nodes.new("ShaderNodeEmission")
    emission.location = (-200, 0)
    emission.inputs["Strength"].default_value = 0.8

    output = nt.nodes.new("ShaderNodeOutputMaterial")
    output.location = (0, 0)

    nt.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], img_tex.inputs["Vector"])
    nt.links.new(img_tex.outputs["Color"], emission.inputs["Color"])
    nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plane.data.materials.append(mat)

    return plane

# ------------------------------------------------------------------
# Central orb — icosphere with glass shader + emissive inner core
# ------------------------------------------------------------------
def setup_orb():
    # Outer glass shell
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.9, subdivisions=4, location=(0, 0, 0.8))
    orb = bpy.context.active_object
    orb.name = "Orb"
    # Subdivision surface for smoothness
    mod = orb.modifiers.new("Subsurf", "SUBSURF")
    mod.render_levels = 2
    mod.levels = 2

    mat = bpy.data.materials.new("OrbGlass")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    glass = nt.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = 1.45
    glass.inputs["Color"].default_value = (0.08, 0.10, 0.18, 1.0)
    glass.inputs["Roughness"].default_value = 0.05
    glass.location = (-300, 100)

    # Palette-tinted emission ring on the orb edge (driven per-frame)
    emis = nt.nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (0.3, 0.6, 1.0, 1.0)
    emis.inputs["Strength"].default_value = 2.0
    emis.location = (-300, -100)
    emis.name = "OrbEmission"

    # Fresnel to mix glass/emission (stronger emission at rim)
    fres = nt.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.45
    fres.location = (-500, 0)

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (-100, 0)
    nt.links.new(fres.outputs["Fac"], mix.inputs["Fac"])
    nt.links.new(glass.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emis.outputs["Emission"], mix.inputs[2])

    output = nt.nodes.new("ShaderNodeOutputMaterial")
    output.location = (100, 0)
    nt.links.new(mix.outputs["Shader"], output.inputs["Surface"])

    orb.data.materials.append(mat)

    # Inner emissive core (separate small sphere)
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.35, subdivisions=3, location=(0, 0, 0.8))
    core = bpy.context.active_object
    core.name = "OrbCore"
    core.parent = orb

    core_mat = bpy.data.materials.new("OrbCoreEmission")
    core_mat.use_nodes = True
    nt2 = core_mat.node_tree
    nt2.nodes.clear()
    core_emis = nt2.nodes.new("ShaderNodeEmission")
    core_emis.inputs["Color"].default_value = (0.5, 0.8, 1.0, 1.0)
    core_emis.inputs["Strength"].default_value = 4.0
    core_emis.location = (-200, 0)
    core_emis.name = "CoreEmission"
    core_out = nt2.nodes.new("ShaderNodeOutputMaterial")
    core_out.location = (0, 0)
    nt2.links.new(core_emis.outputs["Emission"], core_out.inputs["Surface"])
    core.data.materials.append(core_mat)

    return orb, core

# ------------------------------------------------------------------
# Energy aura — flat plane that morphs/scales on bass (replaces torus ring)
# ------------------------------------------------------------------
def setup_energy_aura(parent_orb):
    # Small baseline radius — Python scales it up on kicks
    bpy.ops.mesh.primitive_circle_add(
        vertices=48, radius=1.05, location=(0, 0, 0.8), fill_type='NGON'
    )
    aura = bpy.context.active_object
    aura.name = "EnergyAura"
    aura.rotation_euler = (math.radians(90), 0, 0)
    # Place behind orb so it doesn't hide it (in camera Z)
    aura.location = (0, 0.15, 0.8)
    aura.parent = parent_orb
    aura.scale = (0.001, 0.001, 1.0)  # invisible idle — Python scales it up on bass only

    mat = bpy.data.materials.new("AuraEmission")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    emis = nt.nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (0.9, 0.95, 1.0, 1.0)
    emis.inputs["Strength"].default_value = 8.0
    emis.location = (-400, 100)
    emis.name = "AuraEmission"

    # Gradient fade at edges via geometry pointiness / texture coordinates → transparency
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-800, -100)
    gradient = nt.nodes.new("ShaderNodeTexGradient")
    gradient.gradient_type = "SPHERICAL"
    gradient.location = (-600, -100)
    # Invert gradient so edge = more transparent
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-400, -100)
    # Set ramp: middle opaque, edges transparent
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
    ramp.color_ramp.elements[1].position = 0.5
    ramp.color_ramp.elements[1].color = (0, 0, 0, 0)

    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (-200, -200)

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (-100, 0)

    nt.links.new(tex_coord.outputs["Generated"], gradient.inputs["Vector"])
    nt.links.new(gradient.outputs["Fac"], ramp.inputs["Fac"])
    # Use ramp alpha as mix factor (higher ramp = more emission, lower = more transparent)
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[1])  # fac=0 path
    nt.links.new(emis.outputs["Emission"], mix.inputs[2])     # fac=1 path

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (100, 0)
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # Blender 5.x material blending
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"

    aura.data.materials.append(mat)
    return aura


# ------------------------------------------------------------------
# Flares — two long thin planes that sweep across on bass onsets
# ------------------------------------------------------------------
def setup_flare(name, offset_z=0.8):
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0.1, offset_z))
    flare = bpy.context.active_object
    flare.name = name
    # Thin horizontal bar
    flare.scale = (0.01, 0.01, 1.0)  # invisible baseline, Python will scale up on kicks

    mat = bpy.data.materials.new(f"{name}_Mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    emis = nt.nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)  # pure white
    emis.inputs["Strength"].default_value = 20.0
    emis.location = (-300, 0)
    emis.name = f"{name}_Emission"

    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (-300, -150)

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (-100, 0)
    mix.inputs["Fac"].default_value = 0.15

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (100, 0)
    nt.links.new(emis.outputs["Emission"], mix.inputs[1])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    flare.data.materials.append(mat)
    return flare

# ------------------------------------------------------------------
# Lights
# ------------------------------------------------------------------
def setup_lights():
    # Rim left (palette-tinted, driven per-frame)
    left_data = bpy.data.lights.new("RimLeft", "AREA")
    left_data.energy = 200
    left_data.color = (0.5, 0.7, 1.0)
    left_data.size = 3.0
    left = bpy.data.objects.new("RimLeft", left_data)
    left.location = (-4, -1, 2)
    bpy.context.collection.objects.link(left)

    right_data = bpy.data.lights.new("RimRight", "AREA")
    right_data.energy = 200
    right_data.color = (0.4, 0.6, 1.0)
    right_data.size = 3.0
    right = bpy.data.objects.new("RimRight", right_data)
    right.location = (4, -1, 2)
    bpy.context.collection.objects.link(right)

    # World: dark ambient, will be palette-driven
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    world_nt = world.node_tree
    world_nt.nodes.clear()
    bg = world_nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.01, 0.015, 0.03, 1.0)
    bg.inputs["Strength"].default_value = 0.3
    bg.location = (-200, 0)
    bg.name = "WorldBG"
    out = world_nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (0, 0)
    world_nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    return left, right

# ------------------------------------------------------------------
# Render settings — Eevee, 1080x1920 @ 60fps, bloom/motion blur/DoF
# ------------------------------------------------------------------
def setup_render_settings():
    scene = bpy.context.scene
    # Engine
    # Blender 5.x uses the unified 'BLENDER_EEVEE' (this IS Eevee-Next under the hood)
    scene.render.engine = "BLENDER_EEVEE"
    # Resolution
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 100
    scene.render.fps = 60
    # File output: PNG sequence
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    RENDER_OUT_DIR.mkdir(exist_ok=True)
    scene.render.filepath = str(RENDER_OUT_DIR / "frame_")
    # Eevee-Next settings (Blender 5.x)
    eevee = scene.eevee
    # Bloom is no longer a toggle in Eevee-Next; it's a compositor effect
    # Motion blur
    if hasattr(eevee, "use_motion_blur"):
        eevee.use_motion_blur = True
        eevee.motion_blur_shutter = 0.5
    # Raytracing (Eevee-Next)
    if hasattr(eevee, "use_raytracing"):
        eevee.use_raytracing = True
    # Taa samples — higher = less noise but slower
    if hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = 32
    # Screen-space effects
    if hasattr(eevee, "use_bloom"):
        eevee.use_bloom = True  # ignored on newer Blender, that's ok

# ------------------------------------------------------------------
# Scene-level custom properties for palette (Python sets per-render)
# ------------------------------------------------------------------
def setup_scene_properties():
    scene = bpy.context.scene
    scene["palette_r"] = 0.2
    scene["palette_g"] = 0.6
    scene["palette_b"] = 1.0

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def build():
    clear_scene()
    setup_camera()
    setup_skull_wall()
    orb, core = setup_orb()
    setup_energy_aura(orb)
    setup_flare("Flare1")
    setup_flare("Flare2")
    setup_lights()
    setup_render_settings()
    setup_scene_properties()
    # Save
    bpy.ops.wm.save_as_mainfile(filepath=str(TEMPLATE_OUT))
    print(f"[build_template] Saved → {TEMPLATE_OUT}")


if __name__ == "__main__":
    build()
