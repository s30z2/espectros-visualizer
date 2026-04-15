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

## What's Currently in `audio_visualizer.py` (v44)

### Orb (`_build_orb`, `_render_orb`)
- Blender-rendered glass orb with circular alpha mask
- Fresnel edge ring (bright at 0.92 × ORB_R)
- 3D radial gradient shading (1.4× top-left → 0.25× bottom-right)
- Top-left specular highlight + secondary reflection point
- **Refraction layer**: captures bg behind orb → spherical distort (r^1.7 pinch) → darken 0.55× + cyan tint → composite under orb sprite (orb body alpha ~140, edges/specular 255)
- Drop shadow beneath
- Inner white glowing ring (~1.015 × ORB_R) with 3-layer bloom
- Beat rim flash

### Waveform (`_render_waveform`)
- 128 FFT bins, 20-400Hz bass range
- Savitzky-Golay smooth (window=7)
- Peak amplification: values > mean get boosted 2.6×, top 15% get extra 1.35×
- **Turbulent displacement**: multi-frequency sine noise + random spikes drive per-vertex radial offset — gives "electric arc" look
- 3-layer stroke: 18-25 px wide cyan tube + 10 px mid + 3-5 px white-hot core
- Ghost trail 5 frames behind at 1.08× radius
- 4-pass bloom (15/45/101/251 px blurs, additive composite)

### Background (`_build_bg`, `_render_bg`)

**v44 path (`_build_bg_3d`, `_render_bg_3d`) — active when `bg_3d_scene.png` + `bg_3d_depth.png` exist:**
- Loads Blender-rendered color + depth passes
- Same S-curve + cold teal grade applied to color pass (preserves the 8/10 palette)
- Precomputes one 41-px Gaussian blur level for DOF blending
- Per frame: `cv2.remap` with per-pixel offset = `drift * depth` (near shifts more, far stays still) → true 3D parallax from 2D+depth
- DOF: per-pixel blend `sharp * depth^1.3 + blur * (1 - depth^1.3)` — near skulls crisp, far skulls soft
- Energy + beat brightness boost, shared vignette + orb-as-light illumination

**v43 path (2D, fallback):**
- `skulls_bg_gemini.png` cropped to 1320x2160
- S-curve contrast + cold teal grade (0.72 brightness, B+10%, G-15%, R-55%)
- 3-layer parallax: far (blur 35+0.35×), mid (15+0.55×), near (5+boost)
- Beat-reactive brightness boost on near layer (0.28× extra on beats)
- Soft radial vignette (keeps edges darker)

### Beat Reactivity (`generate_video` main loop)
```python
bi = self.audio.beat_decay(t)  # 500ms decay with ** 0.85 rolloff
# Shakes ONLY when bi > 0.25 OR onset > 0.48
if bi > 0.25:
    gate_snap = ((bi - 0.25) / 0.75) ** 0.5
    zoom = 1.0 + 0.28 * gate_snap           # up to 28% zoom
    shake_x = gate_snap * 110 * randn()      # up to ±110 px
    shake_y = gate_snap * 85 * randn()       # up to ±85 px
    rot = gate_snap * 4.5 * randn()          # up to ±4.5°
if onset_v > 0.48:  # strong bass transient
    kick = (onset_v - 0.48) / 0.52
    zoom += 0.18 * kick
    shake_x += kick * 95 * randn()
    shake_y += kick * 75 * randn()
```
**Between beats: zero motion** (clean static frame).

### Post Processing
1. Bloom: thresh=95, strength=0.88, 4-pass Gaussian 31/91/181/251
2. Desaturate 15% + S-curve crush blacks (×1.15 contrast)
3. WarpAffine for zoom/shake/rotation
4. Radial motion blur when beat > 0.15
5. Chromatic aberration (4-12 px R/B split, opposite directions)
6. Energy ring on beats > 0.1
7. Beat flash (white overlay up to 32% on bi > 0.4)
8. Kick flash (up to 20% on onset > 0.45)
9. Strong vignette (35% edge darkening)
10. Film grain (±12 mono + ±4 per R/B)
11. Fade in over first 1.5s after intro

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

## Open Paths to Push Past 6/10
1. ~~**True 3D scene in Blender**~~ — ✅ landed in v44 via displacement-based 3D (`blender_render_bg_3d.py`). Needs rating pass to confirm rating uplift.
2. ~~**DOF blur on bg**~~ — ✅ landed in v44 (depth-driven per-pixel DOF blend).
3. **Emissive skulls** — give some skulls glowing eye sockets that pulse on beats. (Could add an emission mask to the v44 Blender material driven by image luminance peaks.)
4. **Higher waveform amplitude multiplier** — 3× current would make peaks reach screen edges.
5. **Camera-relative bg parallax** — v44 has drift-driven parallax; could extend to couple the beat shake itself into the depth warp (currently beat shake is still 2D post-warpAffine).
6. **Better reference sampling** — shorter reference clip focused on peak action for Gemini rating (not full 30s).

## API Key
`GEMINI_API_KEY` is hardcoded in `rate_video.py` (Sebastian's personal key). Swap via env var.

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
