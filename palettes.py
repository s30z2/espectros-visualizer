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
    {"name": "NOITE",  "accent": (255, 201, 127), "tint_dark": (20, 8, 3),   "tint_mid": (41, 25, 10),  "vignette": (20, 8, 0)},
    {"name": "SANGUE", "accent": (0, 0, 179),     "tint_dark": (0, 0, 10),   "tint_mid": (3, 3, 28),    "vignette": (0, 0, 10)},
    {"name": "OURO",   "accent": (116, 165, 212), "tint_dark": (5, 8, 10),   "tint_mid": (10, 18, 26),  "vignette": (5, 8, 10)},
    {"name": "VENENO", "accent": (20, 255, 57),   "tint_dark": (10, 10, 2),  "tint_mid": (43, 43, 13),  "vignette": (10, 10, 2)},
    {"name": "FE",     "accent": (221, 78, 157),  "tint_dark": (21, 5, 10),  "tint_mid": (46, 11, 26),  "vignette": (21, 5, 10)},
    {"name": "CINZA",  "accent": (232, 232, 232), "tint_dark": (5, 5, 5),    "tint_mid": (16, 16, 16),  "vignette": (5, 5, 5)},
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
