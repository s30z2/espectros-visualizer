"""Batch pipeline: hash → cache lookup → drop detection → palette → render → LUFS.

Usage (from the CLI):
    python audio_visualizer.py --batch --input-dir ./input --output-dir ./output
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Local imports
from palettes import get_palette_for_track


CACHE_DIR = Path(__file__).parent / "cache"
CLIP_LEN = 30.0          # seconds per output clip
PRE_ROLL = 2.0           # seconds of buildup before the drop


def _hash_audio(path: Path) -> str:
    """sha256 first 16 hex chars of the audio bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_cache(track_hash: str) -> dict | None:
    f = CACHE_DIR / f"{track_hash}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(data: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / f"{data['trackHash']}.json"
    f.write_text(json.dumps(data, indent=2))


def _format_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def process_single(
    src: Path,
    out: Path,
    palette_override: str | None = None,
    logo_text: str = "DX",
    bg_path: str | None = None,
    fps: int = 30,
    loop_fade: bool = True,
) -> dict:
    """Render one track. Returns result dict for summary reporting."""
    # Import here so batch.py doesn't force audio_visualizer import on palettes-only usage
    from audio_visualizer import AudioAnalyzer, generate_video
    from post_process import normalize_loudness

    t0 = time.time()
    src = Path(src)
    out = Path(out)

    # Hash for cache
    track_hash = _hash_audio(src)
    cached = _load_cache(track_hash)

    # Palette routing
    palette = get_palette_for_track(src, override=palette_override)

    # Drop detection — use cache if available and hash matches
    track_duration = None
    if cached and "dropTimestamp" in cached:
        drop_t = float(cached["dropTimestamp"])
        clip_start = float(cached.get("clipStart", max(0.0, drop_t - PRE_ROLL)))
        track_duration = cached.get("duration")
        print(f"[cache] {src.name}: drop={drop_t:.2f}s (from cache)")
    else:
        # Analyze audio to detect drop
        _aa = AudioAnalyzer(str(src))
        drop_t = _aa.detect_drop()
        clip_start = max(0.0, drop_t - PRE_ROLL)
        track_duration = float(_aa.duration)

    # Render to temp file, then normalize loudness onto final output
    tmp_out = out.with_suffix(".raw.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_video(
        str(src),
        str(tmp_out),
        logo_text=logo_text,
        duration=CLIP_LEN,
        bg_path=bg_path,
        start=clip_start,
        palette=palette,
        loop_fade=loop_fade,
        skip_intro=True,   # batch posts skip the "USE EARPHONES" intro
    )

    # LUFS normalize → final output
    print(f"[*] Normalizing loudness to -14 LUFS ...")
    ok = normalize_loudness(tmp_out, out, target_lufs=-14.0)
    if not ok:
        print("[!] Normalize failed, keeping raw output")
        tmp_out.rename(out)
    else:
        try:
            tmp_out.unlink()
        except OSError:
            pass

    # Write cache AFTER successful render
    _save_cache({
        "trackHash": track_hash,
        "srcFilename": src.name,
        "duration": track_duration,
        "dropTimestamp": drop_t,
        "clipStart": clip_start,
        "clipEnd": clip_start + CLIP_LEN,
        "palette": palette["name"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    elapsed = time.time() - t0
    return {
        "track": src.name,
        "palette": palette["name"],
        "drop": drop_t,
        "render_sec": elapsed,
        "size_mb": out.stat().st_size / (1024*1024) if out.exists() else 0,
    }


def _print_summary(results: list) -> None:
    if not results:
        print("\n[summary] no tracks processed")
        return
    print("\n" + "="*78)
    print(f"{'track':<28s} {'palette':<8s} {'drop':>7s} {'time':>7s} {'size':>8s}")
    print("-"*78)
    for r in results:
        print(f"{r['track'][:28]:<28s} {r['palette']:<8s} "
              f"{r['drop']:>6.1f}s {_format_duration(r['render_sec']):>7s} "
              f"{r['size_mb']:>6.1f}MB")
    print("="*78)


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    skip_existing: bool = True,
    palette_override: str | None = None,
    logo_text: str = "DX",
    bg_path: str | None = None,
    fps: int = 30,
    loop_fade: bool = True,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        print(f"[!] Input dir does not exist: {input_dir}")
        print("    Creating it and exiting. Put some .mp3/.wav files in there.")
        input_dir.mkdir(parents=True, exist_ok=True)
        return

    files = sorted([*input_dir.glob("*.mp3"), *input_dir.glob("*.wav")])
    if not files:
        print(f"[!] No .mp3/.wav files in {input_dir}")
        return

    print(f"[*] Batch processing {len(files)} tracks")
    print(f"[*] Output: {output_dir}")

    results: list = []
    for src in files:
        out = output_dir / f"{src.stem}.mp4"
        if skip_existing and out.exists():
            print(f"[skip] {src.name} (output exists)")
            continue
        try:
            r = process_single(
                src, out,
                palette_override=palette_override,
                logo_text=logo_text,
                bg_path=bg_path,
                fps=fps,
                loop_fade=loop_fade,
            )
            results.append(r)
        except Exception as e:
            print(f"[error] {src.name}: {e}")
            import traceback
            traceback.print_exc()

    _print_summary(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python audio_visualizer.py --batch [--input-dir X] [--output-dir Y]")
        sys.exit(1)
    # Allow direct invocation: python batch.py input_dir output_dir
    in_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    run_batch(in_dir, out_dir)
