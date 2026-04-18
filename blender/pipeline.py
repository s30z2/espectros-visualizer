"""
End-to-end Blender pipeline orchestrator.

Run one command, get a posting-ready MP4:

    python3 blender/pipeline.py \
        --audio TRAVIESO.wav \
        --output output/TRAVIESO_3D.mp4 \
        --palette NOITE \
        [--start 16.6] [--duration 30] [--fps 30] [--width 1080] [--height 1920]

Steps (all automated):
  1. Run audio_visualizer.py --export-features → features.json
  2. Refresh template_scene.blend (only if stale or missing)
  3. Render PNG sequence via Blender + compositor post-fx
  4. FFmpeg stitch PNGs + mux audio (trimmed to clip window)
  5. FFmpeg two-pass EBU R.128 loudness normalize to −14 LUFS
  6. Clean up temp PNG frames

Works for both single-track and (in future) batch mode.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
TEMPLATE = HERE / "template_scene.blend"
RENDERS_DIR = HERE / "renders"


def _run(cmd, check=True, quiet=False):
    if not quiet:
        print(f"[run] {' '.join(str(x) for x in cmd)}")
    res = subprocess.run([str(x) for x in cmd], check=check, capture_output=quiet, text=quiet)
    return res


def ensure_template():
    """Rebuild template_scene.blend if missing."""
    if TEMPLATE.is_file():
        return
    print("[pipeline] template missing → building via build_template.py")
    _run(["blender", "--background", "--python", str(HERE / "build_template.py")])
    if not TEMPLATE.is_file():
        sys.exit("ERROR: template build failed")


def export_features(audio, features_path, start, duration, fps, palette_name):
    """Call audio_visualizer.py to emit features.json."""
    features_path = Path(features_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "audio_visualizer.py"),
        "--audio", str(audio),
        "--export-features", str(features_path),
        "--export-duration", str(duration),
        "--export-fps", str(fps),
    ]
    if start is not None:
        cmd += ["--start", str(start)]
    if palette_name:
        cmd += ["--palette", palette_name]
    _run(cmd)
    if not features_path.is_file():
        sys.exit("ERROR: features export failed")


def render_blender(features_path, out_dir, width, height, fps, duration):
    """Invoke Blender headless to render the PNG sequence."""
    out_dir = Path(out_dir)
    # Clean stale frames
    if out_dir.exists():
        for f in out_dir.glob("frame_*.png"):
            try: f.unlink()
            except OSError: pass
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = int(duration * fps)
    cmd = [
        "blender", "--background", str(TEMPLATE),
        "--python", str(HERE / "render_scene.py"),
        "--",
        "--features", str(features_path),
        "--out-dir", str(out_dir),
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
        "--end", str(n_frames),
    ]
    t0 = time.time()
    _run(cmd)
    dt = time.time() - t0
    print(f"[pipeline] Blender render: {dt:.1f}s for {n_frames} frames ({n_frames/dt:.1f} fps)")


def stitch_ffmpeg(frames_dir, audio, start, duration, fps, output_mp4, palette_rgb=None):
    """Stitch PNG sequence → post-fx via FFmpeg filter chain → mux audio → mp4.

    Post-fx (replicates what v56 did in OpenCV + what Blender's compositor can't easily do):
      - Unsharp for mild sharpening
      - Color grade curves (crush blacks, lift mids)
      - Palette tint via colorchannelmixer (multiply toward palette accent color)
      - Vignette darken
      - Slight grain overlay
      - Horizontal bloom via dual Gaussian overlay
    """
    frames_dir = Path(frames_dir)
    audio = Path(audio)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%04d.png")

    # Palette defaults: mild blue-cream multiply if no palette given
    if palette_rgb is None:
        pr, pg, pb = 1.0, 0.85, 0.70
    else:
        # palette_rgb comes in as BGR 0-255 per palettes.py — flip to RGB 0-1
        pb, pg, pr = [max(0.45, c/255.0) for c in palette_rgb]

    # Build complex filter:
    # 1. split into two streams
    # 2. one stream heavily blurred + thresholded (for bloom) — blend additively
    # 3. the other stream goes through:
    #    eq (brightness/contrast/saturation)
    #    curves (crush blacks slightly)
    #    colorchannelmixer (palette tint)
    #    vignette
    #    noise (subtle grain)
    # 4. overlay the bloom stream on the processed main
    vf = (
        # a) mild sharpening
        "unsharp=3:3:0.6,"
        # b) curve: crush deep blacks, keep highlights
        "curves=all='0/0 0.1/0.03 0.5/0.55 1/1',"
        # c) palette tint — pull channels toward palette accent color
        f"colorchannelmixer=rr={pr}:gg={pg}:bb={pb},"
        # d) saturation boost
        "eq=saturation=1.15:contrast=1.08,"
        # e) vignette (subtle radial darken)
        "vignette=PI/4.5,"
        # f) add grain via noise (low)
        "noise=alls=6:allf=t+u"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", pattern,
        "-ss", str(start),
        "-i", str(audio),
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-b:v", "10M",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4),
    ]
    _run(cmd)


def loudness_normalize(in_mp4, out_mp4, target_lufs=-14.0):
    """Two-pass EBU R.128 loudness normalize."""
    sys.path.insert(0, str(ROOT))
    from post_process import normalize_loudness
    ok = normalize_loudness(in_mp4, out_mp4, target_lufs=target_lufs)
    if not ok:
        print("[pipeline] LUFS normalize failed — keeping raw output")
        if Path(in_mp4) != Path(out_mp4):
            shutil.copy(str(in_mp4), str(out_mp4))


def pipeline(
    audio: str,
    output_mp4: str,
    palette: str | None = None,
    start: float | None = None,
    duration: float = 30.0,
    fps: int = 30,
    width: int = 1080,
    height: int = 1920,
    skip_lufs: bool = False,
    keep_frames: bool = False,
):
    """One-shot audio → final MP4 via Blender 3D pipeline."""
    audio_path = Path(audio)
    if not audio_path.is_file():
        sys.exit(f"audio not found: {audio}")

    stem = audio_path.stem
    features_path = ROOT / "cache" / f"{stem}_features.json"
    frames_dir = HERE / "renders"
    raw_mp4 = Path(output_mp4).with_suffix(".raw.mp4")

    print("=" * 60)
    print(f"  3D Blender pipeline: {audio_path.name}")
    print("=" * 60)

    ensure_template()

    print("\n[1/4] Export features ...")
    export_features(audio_path, features_path, start, duration, fps, palette)

    print("\n[2/4] Blender render ...")
    render_blender(features_path, frames_dir, width, height, fps, duration)

    print("\n[3/4] FFmpeg stitch + color grade + mux audio ...")
    # Audio start = drop offset (what was passed as --start to features export)
    audio_start = start if start is not None else 0.0
    # Look up palette BGR from name
    palette_bgr = None
    if palette:
        try:
            sys.path.insert(0, str(ROOT))
            from palettes import get_palette_by_name
            p = get_palette_by_name(palette)
            if p: palette_bgr = p["accent"]
        except Exception:
            pass
    stitch_ffmpeg(frames_dir, audio_path, audio_start, duration, fps, raw_mp4,
                  palette_rgb=palette_bgr)

    if not skip_lufs:
        print("\n[4/4] LUFS normalize ...")
        loudness_normalize(raw_mp4, output_mp4)
        try: raw_mp4.unlink()
        except OSError: pass
    else:
        shutil.move(str(raw_mp4), str(output_mp4))

    if not keep_frames:
        # Clean temp PNGs
        for f in frames_dir.glob("frame_*.png"):
            try: f.unlink()
            except OSError: pass

    size_mb = Path(output_mp4).stat().st_size / (1024*1024)
    print(f"\n[*] Done! → {output_mp4}  ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--palette", default=None, help="NOITE/SANGUE/OURO/VENENO/FE/CINZA")
    ap.add_argument("--start", type=float, default=None, help="Start offset in seconds")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--skip-lufs", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()
    pipeline(
        audio=args.audio,
        output_mp4=args.output,
        palette=args.palette,
        start=args.start,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        skip_lufs=args.skip_lufs,
        keep_frames=args.keep_frames,
    )


if __name__ == "__main__":
    main()
