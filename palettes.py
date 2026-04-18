"""Palette system for phonk visualizer batch mode.

BGR tuples (OpenCV convention). 6 palettes.
Orb body and DX logo stay constant across all palettes — only the ring,
waveform stroke, BG grade tint, and vignette tint respond to palette.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional

PALETTES = [
    # v54 ENERGY BURST: brighter accents, richer tints
    {"name": "NOITE",  "accent": (255, 215, 160), "tint_dark": (30, 14, 5),   "tint_mid": (58, 34, 14),  "vignette": (20, 8, 0)},
    {"name": "SANGUE", "accent": (50, 50, 220),   "tint_dark": (6, 4, 18),    "tint_mid": (16, 8, 40),   "vignette": (6, 0, 14)},
    {"name": "OURO",   "accent": (150, 200, 240), "tint_dark": (8, 12, 14),   "tint_mid": (16, 26, 36),  "vignette": (6, 10, 14)},
    {"name": "VENENO", "accent": (50, 255, 100),  "tint_dark": (12, 18, 4),   "tint_mid": (30, 52, 12),  "vignette": (8, 14, 2)},
    {"name": "FE",     "accent": (230, 110, 185), "tint_dark": (26, 8, 16),   "tint_mid": (54, 18, 36),  "vignette": (22, 8, 14)},
    {"name": "CINZA",  "accent": (240, 240, 240), "tint_dark": (8, 8, 10),    "tint_mid": (22, 22, 26),  "vignette": (6, 6, 8)},
]

DEFAULT_PALETTE = PALETTES[0]  # NOITE — closest to v51


def get_palette_by_name(name: str) -> Optional[dict]:
    """Return palette dict by name (case-insensitive), or None if not found."""
    name_up = name.strip().upper()
    return next((p for p in PALETTES if p["name"] == name_up), None)


def get_palette_for_track(audio_path, override: Optional[str] = None) -> dict:
    """Route an audio file to a palette.

    Resolution order:
    1. Explicit `override` arg (e.g. from --palette CLI flag)
    2. Sidecar file `<audio>.palette.txt` containing a palette name
    3. Deterministic hash of the filename (first 8 hex chars of sha256,
       mod len(PALETTES))
    """
    path = Path(audio_path)

    # 1. Explicit override
    if override:
        match = get_palette_by_name(override)
        if match is not None:
            return match

    # 2. Sidecar file
    sidecar = path.with_suffix(".palette.txt")
    if sidecar.exists():
        try:
            name = sidecar.read_text().strip()
            match = get_palette_by_name(name)
            if match is not None:
                return match
        except OSError:
            pass

    # 3. Deterministic hash routing
    h = hashlib.sha256(path.name.encode()).hexdigest()[:8]
    return PALETTES[int(h, 16) % len(PALETTES)]


if __name__ == "__main__":
    # Quick manual verification
    import sys
    if len(sys.argv) > 1:
        p = get_palette_for_track(sys.argv[1])
        print(f"{sys.argv[1]} -> {p['name']}")
    else:
        print("Available palettes:")
        for p in PALETTES:
            print(f"  {p['name']:8s} accent BGR {p['accent']}")
