"""
Load template_scene.blend, read features.json, keyframe per-frame properties,
render PNG sequence via Eevee. Called headlessly:

    blender --background blender/template_scene.blend --python blender/render_scene.py \
            -- --features features.json --out-dir blender/renders [--start 0] [--end N]

Drives:
- Orb scale burst + rim emission strength from bass_energy + bass_onset
- OrbCore inner emission from bass_energy
- EnergyAura scale/rotation — explodes on drops
- Flare1/Flare2 — only visible on bass_onset, random rotation per kick
- Camera micro-shake on bass_onset
- Rim light colors from palette

All anchored to the template built by build_template.py.
"""
from __future__ import annotations
import bpy
import json
import math
import random
import sys
from pathlib import Path


def parse_args():
    try:
        idx = sys.argv.index("--")
        argv = sys.argv[idx + 1:]
    except ValueError:
        argv = []
    args = {
        "features": None,
        "out_dir": "blender/renders",
        "start": 0,
        "end": None,
        "width": 1080,
        "height": 1920,
        "fps": 60,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--features", "-f"):
            args["features"] = Path(argv[i+1]); i += 2
        elif a == "--out-dir":
            args["out_dir"] = argv[i+1]; i += 2
        elif a == "--start":
            args["start"] = int(argv[i+1]); i += 2
        elif a == "--end":
            args["end"] = int(argv[i+1]); i += 2
        elif a == "--width":
            args["width"] = int(argv[i+1]); i += 2
        elif a == "--height":
            args["height"] = int(argv[i+1]); i += 2
        elif a == "--fps":
            args["fps"] = int(argv[i+1]); i += 2
        else:
            i += 1
    if args["features"] is None:
        print("ERROR: --features PATH required")
        sys.exit(1)
    return args


def bgr_to_rgb_norm(bgr):
    """Palette accent stored as BGR 0-255 ints → RGB 0-1 float tuple."""
    if max(bgr) > 1.5:
        return (bgr[2]/255.0, bgr[1]/255.0, bgr[0]/255.0)
    return (bgr[2], bgr[1], bgr[0])


def animate_and_render(features: dict, args: dict):
    scene = bpy.context.scene
    scene.render.resolution_x = args["width"]
    scene.render.resolution_y = args["height"]
    scene.render.fps = args["fps"]

    frames = features["frames"]
    start = args["start"]
    end = args["end"] if args["end"] else len(frames)
    end = min(end, len(frames))
    scene.frame_start = start
    scene.frame_end = end - 1

    # Palette → shader colors + scene props
    palette_bgr = features.get("palette", [220, 70, 20])
    rgb = bgr_to_rgb_norm(palette_bgr)
    scene["palette_r"], scene["palette_g"], scene["palette_b"] = rgb

    # Resolve objects
    orb = bpy.data.objects.get("Orb")
    core = bpy.data.objects.get("OrbCore")
    dx_logo = bpy.data.objects.get("DXLogo")
    aura = bpy.data.objects.get("EnergyAura")
    flare1 = bpy.data.objects.get("Flare1")
    flare2 = bpy.data.objects.get("Flare2")
    cam = scene.camera
    rim_left = bpy.data.objects.get("RimLeft")
    rim_right = bpy.data.objects.get("RimRight")

    # Drive compositor's PaletteTint RGB node with the palette color
    comp_nt = getattr(scene, "compositing_node_group", None) or getattr(scene, "node_tree", None)
    if scene.use_nodes and comp_nt:
        for node in comp_nt.nodes:
            if node.type == "RGB" and node.name == "PaletteTint":
                tinted = (max(0.6, rgb[0]), max(0.45, rgb[1]), max(0.35, rgb[2]), 1.0)
                node.outputs["Color"].default_value = tinted
                break

    if orb is None:
        print("ERROR: Orb object missing from scene")
        sys.exit(1)

    # Tint rim lights with palette
    if rim_left:
        rim_left.data.color = (rgb[0], rgb[1]*0.9, rgb[2])
    if rim_right:
        rim_right.data.color = (rgb[0]*0.9, rgb[1], rgb[2])

    # Tint orb rim emission to palette accent
    if orb.data.materials:
        emis_node = orb.data.materials[0].node_tree.nodes.get("OrbEmission")
        if emis_node:
            emis_node.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)

    cam_base_loc = cam.location.copy()

    # Animation loop — keyframe per frame
    for fi in range(start, end):
        f = frames[fi]
        scene.frame_set(fi)

        bass_e = float(f.get("bass_energy", 0.0))
        bass_o_raw = f.get("bass_onset", 0.0)
        bass_o = float(bass_o_raw) if not isinstance(bass_o_raw, bool) else (1.0 if bass_o_raw else 0.0)
        beat_d = float(f.get("beat_decay", 0.0))

        # Gate onset — only trigger strong kicks
        kick = max(0.0, (bass_o - 0.55) / 0.45) if bass_o > 0.55 else 0.0

        # ── Orb pulse (bigger — reference glows hard on drops) ──
        pulse = 1.0 + 0.35 * bass_e + 0.30 * kick
        orb.scale = (pulse, pulse, pulse)
        orb.keyframe_insert("scale")

        # ── Orb rim emission strength (22× on peak bass per ref) ──
        if orb.data.materials:
            emis = orb.data.materials[0].node_tree.nodes.get("OrbEmission")
            if emis:
                emis.inputs["Strength"].default_value = 3.0 + bass_e * 16.0 + kick * 8.0
                emis.inputs["Strength"].keyframe_insert("default_value")

        # ── Core inner emission ──
        if core and core.data.materials:
            ce = core.data.materials[0].node_tree.nodes.get("CoreEmission")
            if ce:
                ce.inputs["Strength"].default_value = 2.5 + bass_e * 10.0 + kick * 6.0
                ce.inputs["Strength"].keyframe_insert("default_value")

        # ── DX logo emission strength driven by bass ──
        if dx_logo and dx_logo.data.materials:
            de = dx_logo.data.materials[0].node_tree.nodes.get("DXEmission")
            if de:
                de.inputs["Strength"].default_value = 5.0 + bass_e * 6.0 + kick * 8.0
                de.inputs["Strength"].keyframe_insert("default_value")

        # ── EnergyAura: visible always, expands+rotates on bass ──
        if aura:
            aura_s = 2.2 + bass_e * 2.8
            aura_s_z = 0.6 + bass_e * 1.2
            aura.scale = (aura_s, aura_s, aura_s_z)
            aura.rotation_euler = (math.radians(90), 0, bass_e * 3.0 + fi * 0.008)
            aura.keyframe_insert("scale")
            aura.keyframe_insert("rotation_euler")
            if aura.data.materials:
                ae = aura.data.materials[0].node_tree.nodes.get("AuraEmission")
                if ae:
                    ae.inputs["Strength"].default_value = 10.0 + bass_e * 18.0 + kick * 12.0
                    ae.inputs["Strength"].keyframe_insert("default_value")

        # ── Sweeping flares on bass kicks (bigger intensity per ref) ──
        rng = random.Random(fi * 31337 + 7)
        for idx, fl in enumerate((flare1, flare2)):
            if fl is None:
                continue
            if kick > 0.0:
                length = bass_e * 28.0 * (1.4 if idx == 0 else 1.0)
                fl.scale = (length, 0.08, 1.0)
                fl.rotation_euler = (0, 0, rng.uniform(-math.pi, math.pi))
            else:
                fl.scale = (0.01, 0.01, 1.0)
            fl.keyframe_insert("scale")
            fl.keyframe_insert("rotation_euler")

        # ── Camera shake on bass kicks + smooth return between ──
        if kick > 0.0:
            shake = bass_e * 0.15
            dx = (rng.random() - 0.5) * shake
            dz = (rng.random() - 0.5) * shake
            cam.location = (cam_base_loc.x + dx, cam_base_loc.y, cam_base_loc.z + dz)
        else:
            # Smooth exponential return toward base
            cx = cam.location.x * 0.85
            cz = cam_base_loc.z + (cam.location.z - cam_base_loc.z) * 0.85
            cam.location = (cx, cam_base_loc.y, cz)
        cam.keyframe_insert("location")

    # Render
    out_dir = Path(args["out_dir"]).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out_dir / "frame_")
    print(f"[render_scene] Rendering {end-start} frames @ {scene.render.resolution_x}x{scene.render.resolution_y} → {out_dir}")
    bpy.ops.render.render(animation=True, write_still=False)
    print("[render_scene] Done")


def main():
    args = parse_args()
    features = json.loads(Path(args["features"]).read_text())
    animate_and_render(features, args)


if __name__ == "__main__":
    main()
