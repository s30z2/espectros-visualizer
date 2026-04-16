# ESPECTROS Audio Visualizer — Session Context

## Goal
Generate a dark cyberpunk/gothic audio visualizer video (1080x1920, 9:16, 30fps) matching a TikTok reference video (`demo_visualizer.mp4`). Target: Gemini API rates the result **9/10** similarity to the reference.

Final audio to render against: `montagem_alquimia.wav` (113.5s).

## Reference Video Style
- Skull-dense background with parallax
- Central glossy 3D orb with "DX" logo
- Beat-reactive FFT waveform/aura
- Aggressive screen shakes + zoom on beats
- Particles, light flares, bloom
- Small orb (~15% frame width), minimal gothic aesthetic
- Diagonal light streak (lens flare)

## Current Best Deliverable
`montagem_alquimia_v41_full.mp4` — **NOT in repo** (114 MB, too big for GitHub).  
Regenerate locally:
```bash
python3 audio_visualizer.py --audio montagem_alquimia.wav --output montagem_alquimia_v41_full.mp4
```

## Architecture
- **`audio_visualizer.py`** — single-file Python visualizer. Reads wav → librosa FFT/beat detection → cv2/numpy compositing → MoviePy encode (libx264, 8 Mbps).
- **`rate_video.py`** — Gemini API rating script. Compares generated video to reference, returns per-aspect scores and feedback. Tries `gemini-2.5-pro` → `gemini-2.5-flash` → `gemini-3-flash-preview` → `gemini-2.5-flash-lite` in order.
- **`blender_render_orb.py`** — Blender script that generated `orb_3d.png` (glass orb with PBR + emission DX text).
- **`blender_render_bg.py`** — Blender script that generated `bg_3d_far/mid/near.png` (abstract skull layers, unused currently).
- **`blender_render_bg_3d.py`** — v44 Blender script. Loads `skulls_bg_gemini.png` as a displacement map on a high-poly plane, renders color + depth passes (`bg_3d_scene.png`, `bg_3d_depth.png`). Used at runtime by `audio_visualizer.py` for depth-driven parallax + DOF.
- **`orb_3d.png`** — Blender-rendered orb used at runtime.
- **`skulls_bg_gemini.png`** — AI-generated skull background used at runtime.
- **`demo_visualizer.mp4`** — reference TikTok video (excluded from git via .gitignore).

## Key Constants (in `audio_visualizer.py`)
```python
W, H = 1080, 1920
FPS = 30
INTRO_DUR = 1.5
ORB_R = 195          # big orb (~36% of frame width)
N_FFT_BINS = 128
SMOOTH_ALPHA = 0.55
```

## Iteration History & Ratings

Ratings were inconsistent (Gemini Flash gave ±2 point swings on identical settings). Pro was the most reliable but frequently 503'd.

| Version | Change | Flash | Pro | Notes |
|---------|--------|-------|-----|-------|
| v28 | Added refraction inside orb (cv2.remap spherical distort + cyan tint) | 4/10 | — | Orb got 7/10 |
| v29 | Thicker waveform (3-layer stroke) + wider bloom | 5/10 | — | |
| v30 | Sharper savgol peaks + inner white ring around orb | **6/10** | — | Particles 7/10, orb 7/10 |
| v31 | Added cyan atmospheric mist + wider bloom radius | 4/10 | — | Mist killed contrast — reverted |
| v32 | Bigger waveform extension (0.60→0.90 × W) | 4/10 | — | Too extended — reverted |
| v33 | Brighter bg skulls (0.55→0.72) + more particles | 4/10 | — | |
| v34 | Diagonal anamorphic flare (pivoted from horizontal) | 5/10 | — | Matches reference |
| v35 | Lower bloom threshold (110→95) + bigger zoom-punch (22→28%) | **6/10** | — | Vibe 7/10 |
| v36 | Beat-reactive BG brightness pulse | 5/10 | — | |
| v37 | Stronger CA (4-12 px) | 5/10 | — | |
| v38 | Even brighter skulls (0.72→0.88) | 4/10 | — | Hurt palette — reverted |
| v39 | Full 115s render with v37 config | 4/10 | 4/10 | Pro: palette 8, bloom 6 |
| v40 | Stronger film grain (±6→±12) + color noise | 4/10 | **4/10** | **Pro**: palette 8, bloom 6, orb 5 |
| v41 | Turbulent displacement on waveform path (Pro's explicit advice) | 5/10 | 3/10 (20s only) | Full-length: palette 8, orb 6 |
| v42 | Longer beat decay (350→500 ms) | — | — | Killed — didn't have bass-gated shakes |
| **v43** | **Bass-gated shakes** (removed constant drift, only fires on `bi > 0.25` or onset > 0.48) | — | — | User's explicit request |
| **v44** | **3D displacement bg** — `blender_render_bg_3d.py` extrudes `skulls_bg_gemini.png` into real 3D geometry via displacement modifier; outputs color + depth passes. Runtime: depth-driven per-pixel parallax (near pixels shift more on camera drift) + depth-based DOF blend (far skulls soft, near skulls sharp). Falls back to v43 2D parallax if 3D assets missing. | — | — | **Requires `blender --background --python blender_render_bg_3d.py` to generate `bg_3d_scene.png` + `bg_3d_depth.png` before render.** |
| **v45** | **Major composition overhaul** — ORB_R 195→80 (small orb like reference), killed giant FFT waveform arcs (now subtle thin pulsing ring), brighter bg skulls (0.88x), lighter vignette (18% vs 35%), synthetic depth parallax from luminance (no Blender needed), bloom thresh 95→170, CA 4-12px→1-4px, flash 32%→12%, removed orbit flare, sparse scattered particles, orb darkened to 0.12x for dark glass look, reduced refraction darken to 0.30. | — | **4/10** | **Pro**: palette 7, vibe 4, bg 3, orb 2, waveform 3, beat reactivity **1**, bloom 4, particles 3. Gemini says: bring back beat-reactive shakes (#1), spikier waveform (#2), 3D depth polish (#3). Composition matches reference but pulled back effects too far. |

## What's Currently in `audio_visualizer.py` (v45)

### Orb (`_build_orb`, `_render_orb`) — ORB_R = 80 (small, ~15% frame width)
- Blender-rendered glass orb, darkened to 0.12× for dark glass look
- Body alpha 30% (very transparent), edges/specular opaque
- Fresnel edge ring, 3D radial gradient shading, specular highlights
- **Refraction layer**: bg distort → darken 0.30× + cyan tint
- Subtle drop shadow, thin white ring (1.02×R, single-pass glow)
- Beat rim flash only on bi > 0.3

### Waveform (`_render_waveform`) — subtle ring, NOT arcs
- 128 FFT bins with savgol smoothing
- Thin pulsing ring at ORB_R + 6px, max ±8px displacement
- Single cyan polyline (2px) + one-pass soft glow
- No turbulent displacement, no ghost trail, no multi-pass bloom

### Background (`_build_bg`, `_render_bg`) — BG DOMINANT
- Synthetic depth from luminance (Gaussian blur → power curve) when no Blender
- Also supports Blender-rendered depth passes (v44 path still works)
- Brighter S-curve grade (0.88× vs 0.72×), teal tint preserved
- Depth-driven parallax: near pixels shift more on camera drift
- DOF: sharp near, blur far (depth^1.3 blend)
- Light vignette (0.82 clamp vs 0.70), subtle orb illumination

### Beat Reactivity — same as v43/v44
- Shakes gated at bi > 0.25 or onset > 0.48 (unchanged)
- Up to 28% zoom, ±110px shake, ±4.5° rotation
- Zero motion between beats

### Post Processing (v45 — all reduced)
1. Bloom: thresh=170, strength=0.35
2. Desaturate 15% + S-curve
3. WarpAffine for zoom/shake/rotation
4. Radial motion blur when beat > 0.15
5. Chromatic aberration (1-4 px, blend 0.15-0.5)
6. Energy ring on beats > 0.15 (smaller radius)
7. Beat flash max 12% (was 32%), threshold 0.5 (was 0.4)
8. Kick flash max 8% (was 20%), threshold 0.55
9. Light vignette (18% edge darkening, was 35%)
10. Film grain (±5 mono + ±2 per R/B)
11. Fade in over first 1.5s
12. No orbit flare (removed)

## Fast Preview Mode
Half-res 20fps mode — didn't help much because Gaussian blurs dominate and don't scale well:
```bash
# Half res, 20fps, 20s (~15 min anyway due to blur cost):
python3 audio_visualizer.py --audio montagem_alquimia.wav --output preview.mp4 --duration 20 --scale 0.5 --fps 20
```

## Render Times (on Sebastian's M-series Mac)
- 30s preview at 1080x1920: **~15-20 min** (~0.5-0.8 fps)
- Full 115s at 1080x1920: **~60-75 min**
- 20s preview at 540x960: **~10-15 min** (blurs bottleneck, not pixel count)

## Known Gemini Rating Noise
- Flash gives ±2 point swings on identical configs
- Flash frequently says "no beat reactivity" when video has 28% zoom + 110 px shake (model isn't reading motion)
- Pro is more consistent but 503'd ~70% of the time during this session
- Hitting a Flash rating ceiling around 5-6/10 after 14 iterations
- Consistent strengths: palette (7-8/10), orb (5-7/10), bloom (4-6/10)
- Consistent weaknesses: background (3-5/10), beat reactivity (2-4/10 by rating, but visually fine)

## Open Paths to Push Past 4/10 (from v45 Gemini Pro feedback)
1. **Bring back beat-reactive camera shakes** — Gemini's #1 ask. The code is there (bi>0.25 gate) but Gemini rated beat reactivity 1/10, likely because the half-res 20fps preview doesn't convey motion well. Try full-res render for rating.
2. **Spikier waveform aura** — Gemini wants more visible FFT ring, not the giant arcs from v41 but something between v45's invisible ring and v41's electric arcs. Moderate amplitude, sharper peaks.
3. **Brighten skulls** — bg grade 0.88× may still be too dark. Try 0.95-1.0. Also reduce vignette clamp from 0.82 to 0.90.
4. **Orb glass visibility** — darkened to 0.12× which makes it nearly invisible. Try 0.22-0.25× so you can see the glass refraction.
5. **Orb white ring** — ring_bri base 120 is too faint. Try 160-180.
6. ~~**True 3D scene in Blender**~~ — ✅ v44. Synthetic depth fallback works fine.
7. ~~**DOF blur on bg**~~ — ✅ v44/v45.
8. **Emissive skulls** — glowing eye sockets that pulse on beats.
9. **Camera-relative bg parallax** — couple beat shake into depth warp.

## API Key
`GEMINI_API_KEY` is loaded from `.env` file (gitignored). Old hardcoded key was revoked.

## File Inventory (what's in git)
- `audio_visualizer.py` — main visualizer
- `rate_video.py` — Gemini rating script
- `blender_render_orb.py`, `blender_render_bg.py` — Blender generators
- `orb_3d.png` — rendered orb sprite
- `skulls_bg_gemini.png` — bg texture
- `bg_3d_far/mid/near.png` — unused abstract skull layers (keep for reference)
- `montagem_alquimia.wav` — target audio (21 MB)
- `demo_beat.wav` — shorter test audio
- `ref_t2/4/5/7/10/12/15.png` — reference video frame extracts (for visual comparison)
- `v26-v41_frame200.png` — frame extracts from my renders per iteration (visual diff)
- `compare_v20.png` — side-by-side comparison
- `CONTEXT.md` — this file
- `.gitignore` — excludes all `.mp4`

## What NOT in git (regenerate)
- All `.mp4` files (preview and full renders)
- `TEMP_MPY_wvf_snd.mp4` intermediate files
- `demo_visualizer.mp4` (reference) — need to re-download from original TikTok source or copy manually

## To Continue on Another Machine
```bash
# Install Git LFS first (one-time per machine):
#   macOS:   brew install git-lfs
#   Windows: https://git-lfs.github.com/
#   Linux:   sudo apt install git-lfs
git lfs install

git clone https://github.com/s30z2/espectros-visualizer.git
cd espectros-visualizer
git lfs pull    # pulls ~330 MB of videos (reference + key iterations)

pip install numpy opencv-python pillow scipy librosa moviepy google-genai

# Run a fast preview:
python3 audio_visualizer.py --audio montagem_alquimia.wav --output out.mp4 --duration 20 --scale 0.5 --fps 20
# Or full 115s render:
python3 audio_visualizer.py --audio montagem_alquimia.wav --output full.mp4
# Rate it against the reference:
python3 rate_video.py demo_visualizer.mp4 out.mp4
```
