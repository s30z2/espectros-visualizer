# ESPECTROS — Audio Visualizer

Python audio visualizer targeting a TikTok-style dark cyberpunk/gothic aesthetic. Generates 1080×1920 vertical videos with beat-reactive FFT waveform, refracting 3D orb, skull background, and cinematic post-processing.

**Read [`CONTEXT.md`](./CONTEXT.md)** for the full development history, iteration log, and technical architecture. That's the file to pick up from if you're resuming work.

## Quick Start

```bash
pip install numpy opencv-python pillow scipy librosa moviepy google-genai

# Fast 20s preview at half res:
python3 audio_visualizer.py --audio montagem_alquimia.wav --output preview.mp4 \
    --duration 20 --scale 0.5 --fps 20

# Full 115s render at 1080x1920:
python3 audio_visualizer.py --audio montagem_alquimia.wav --output full.mp4

# Rate similarity to reference via Gemini:
python3 rate_video.py demo_visualizer.mp4 full.mp4
```

Set `GEMINI_API_KEY` env var before rating, or use the one hardcoded in `rate_video.py`.

## Files

| File | Purpose |
|------|---------|
| `audio_visualizer.py` | Main visualizer — librosa FFT → cv2/numpy compositing → MoviePy encode |
| `rate_video.py` | Gemini API rater (compares generated vs reference) |
| `blender_render_orb.py` | Generated `orb_3d.png` (PBR glass orb with DX emission text) |
| `blender_render_bg.py` | Generated abstract skull layers |
| `CONTEXT.md` | **Full session history and architectural notes** |

## Not in git
- All `.mp4` outputs (regenerate)
- `demo_visualizer.mp4` (the reference TikTok — add manually)

## Current State (v43)
- Shakes now gated to bass hits only (no constant drift between beats)
- Best Gemini Pro rating: 4/10 (palette 8, bloom 6, orb 5 on v40)
- Plateau: 5-6/10 ceiling with current 2D-layered approach; pushing further likely needs a proper Blender 3D scene

See `CONTEXT.md` § "Open Paths to Push Past 6/10".
