"""Post-processing: FFmpeg loudness normalization to TikTok's −14 LUFS target.

Two-pass loudnorm for accuracy — single-pass overshoots on dynamic phonk
tracks. Keeps video codec copy; only re-encodes audio.
"""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path


def _run(cmd, capture=True):
    """Run subprocess, return CompletedProcess. Never raise on non-zero (caller handles)."""
    return subprocess.run(cmd, capture_output=capture, text=True)


def _measure_loudness(input_path: Path, target_lufs: float = -14.0) -> dict | None:
    """Pass 1: measure integrated LUFS + true peak on input. Returns parsed JSON block."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    res = _run(cmd)
    # ffmpeg's loudnorm JSON block is printed to stderr after the "[Parsed_loudnorm_"... header
    stderr = res.stderr or ""
    # Grab the last JSON block
    matches = re.findall(r"\{\s*\"input_i\".*?\}", stderr, flags=re.DOTALL)
    if not matches:
        print(f"[!] Could not parse loudnorm pass 1 for {input_path.name}")
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def normalize_loudness(input_path, output_path, target_lufs: float = -14.0) -> bool:
    """Two-pass EBU R.128 loudness normalize. Returns True on success.

    Pass 1: measure.  Pass 2: apply with measured values (linear mode = more accurate).
    Video stream is copied (no re-encode). Audio gets AAC at 192k.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        print(f"[!] Input not found: {input_path}")
        return False

    measured = _measure_loudness(input_path, target_lufs)
    if not measured:
        print(f"[!] Skipping normalize for {input_path.name} (pass 1 failed)")
        return False

    # Pass 2 — linear mode, requires measured values
    af = (
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
        f":measured_I={measured['input_i']}"
        f":measured_LRA={measured['input_lra']}"
        f":measured_TP={measured['input_tp']}"
        f":measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}"
        f":linear=true:print_format=summary"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-y",
        "-i", str(input_path),
        "-af", af,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]
    res = _run(cmd)
    if res.returncode != 0:
        print(f"[!] FFmpeg normalize failed: {res.stderr[-300:] if res.stderr else '?'}")
        return False
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python post_process.py input.mp4 output.mp4 [target_lufs]")
        sys.exit(1)
    tgt = float(sys.argv[3]) if len(sys.argv) > 3 else -14.0
    ok = normalize_loudness(sys.argv[1], sys.argv[2], tgt)
    sys.exit(0 if ok else 1)
