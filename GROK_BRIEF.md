# Brief for Grok

You're joining an in-progress project. Read this once, top to bottom, before answering anything. There's no time for Q&A — the owner wants you to come in hot with concrete answers.

---

## What this project is

**An automated TikTok phonk-music visualizer pipeline.** You drop an MP3/WAV in `input/`, it renders a 30-second 1080×1920 9:16 vertical MP4 to `output/` that's ready to upload. Each video has the same general look: a dark skull-wall background, a central orb with a "DX" logo, and an audio-reactive waveform ring around the orb. The color palette varies per track (hash-routed).

Repo: https://github.com/s30z2/espectros-visualizer

Current version: **v56**. We just finished a motion-blur + smooth-curves pass. The output looks OK but still feels 2D-layered. The user wants it to feel more cinematic and 3D. That's where you come in.

## Tech stack (current)

| Layer | Tool |
|---|---|
| Audio analysis | librosa (STFT, beat tracking, custom bass-onset detector) |
| Compositing | OpenCV + numpy in pure Python |
| Encoding | MoviePy + FFmpeg (libx264, 8 Mbps) |
| Loudness | FFmpeg EBU R.128 two-pass loudnorm to −14 LUFS |
| Eval | `rate_video.py` — Gemini 2.5 Pro paired A/B judge with ensemble voting |

Rendering speed ~3 fps thanks to a custom `fast_blur` pyramid helper (downsample → blur small → upsample for large kernels, 4–16× faster than naive cv2.GaussianBlur).

## Pipeline (current)

```
audio.mp3 ─► AudioAnalyzer (librosa)
          │     ├─ detect_drop()   → finds first dense bass-kick cluster (≥4 kicks/2s above 0.3)
          │     ├─ bass_onset(t)   → kick detector (<140 Hz STFT slice, per-frame derivative)
          │     ├─ bass_energy(t)  → bass RMS 0–1
          │     └─ beat_decay(t)   → 500 ms beat envelope
          │
          ├─ clip window = [drop − 2 s, drop + 28 s] = 30 s
          ├─ palette routing (hash(filename) mod 6 palettes, or override via CLI / sidecar file)
          │
          ▼
Visualizer.make_frame(t) — called by MoviePy per frame:
  _render_bg(frame, t, energy, beat_i)           ← 3-layer DoF-separated skull BG with parallax
  _render_orb(frame, t, energy, beat_i)          ← flat disc + palette-tinted ring + always-on glow
  _render_waveform(frame, t, energy, beat_i)     ← 16 FFT control points → interp 360 → savgol smooth,
                                                   capped inside frame, dark fill + white outline
  _render_particles(frame, t, energy)            ← 70 ambient dust + bass-triggered ember burst
  _render_anamorphic_flare(...)                  ← diagonal lens scratch, bass-pulsed
  bloom()                                        ← multi-pass pyramid Gaussian
  palette color wash                             ← lift blacks toward palette accent
  zoom punch + shake + rot on bass kicks
  chromatic aberration burst on kicks
  white flash overlay on strong kicks
  motion blur (multi-step radial zoom blur) on kicks ≥ 0.25
  vignette + grain + fade
          │
          ▼
loop-fade (last 15 frames crossfade back to opening) — seamless TikTok loop
          │
          ▼
FFmpeg 2-pass LUFS normalize → output/XXX.mp4
          │
          ▼
cache/<sha256>.json (drop time, clip window, palette)
```

## Where we want to go

The 2D layered look is the ceiling of what you can do with OpenCV. Gemini Pro just recommended **Blender + `bpy` Python API in headless mode** as the next step:

```bash
blender -b template_scene.blend --python render_scene.py -- --features features.json
```

- Python side: `librosa` emits `features.json` (per-frame bass energy, beat flags, FFT bins, drop timestamp)
- Blender side: `render_scene.py` loads `features.json`, programmatically keyframes materials/geometry/camera from it, renders image sequence via Eevee
- Back to Python: FFmpeg stitches image sequence + normalized audio into final MP4

Expected gain: volumetric lighting, real 3D parallax, true motion blur, geometry-node audio-reactive meshes, proper emission shaders, DoF. All while staying fully automated (no manual timeline work per track).

Render budget target: ≤10 min per 30 s 1080p60 clip. Gemini estimated 10–20 min for Eevee on M-series Mac.

## State of the repo you should know about

| File | What |
|---|---|
| `audio_visualizer.py` | Main file. `AudioAnalyzer`, `Visualizer`, `generate_video`, `generate_keyframes`, CLI. |
| `palettes.py` | 6 palettes (NOITE/SANGUE/OURO/VENENO/FE/CINZA) + hash/sidecar/CLI routing. |
| `batch.py` | Batch orchestrator + SHA-256 cache + summary table. |
| `post_process.py` | FFmpeg 2-pass LUFS normalize. |
| `rate_video.py` | **v2 structured Gemini judge**: paired A/B, ensemble 3 runs, pixel-property rubric (no vibe words), JSON schema enforced. Uses `google.generativeai`. |
| `CLAUDE_HANDOFF.md` | Older handoff doc for Claude instances. Partly outdated post-v56. |
| `CONTEXT.md` | Iteration history v28–v43 (pre-pivot). |
| `domixx_ref.mp4` | An older reference we matched at v51 and then abandoned. |
| `TRAVIESO.wav` | Default test song. Drop at 16.60 s (detected automatically). |
| `skulls_bg_gemini.png` | AI-generated skull wall BG. |
| `orb_3d.png` | A Blender-rendered glass orb from early iterations. Unused in v56. |
| `output/TRAVIESO.mp4` | Latest render. Watch this to see where we are. |
| `hooks/pre-commit` | Blocks commits containing `AIza…` API key patterns. |
| `.env` | Gitignored. Contains `GEMINI_API_KEY`. |

## What I/the owner actually need from you

We need you to **help execute or co-design the Blender migration**. Specifically:

1. **Scene template** (`template_scene.blend`) — propose a minimal 3D scene that could replace the current OpenCV composite: a camera, a central orb mesh, a skull wall (plane with texture or a particle system of low-poly skulls at varying Z depths), a waveform mesh (Torus that Geometry Nodes deforms based on FFT input), lights (emissive orb center + rim + global ambient), post fx (Eevee bloom + DoF + motion blur).
2. **`render_scene.py`** — the Blender Python script that: parses `features.json`, sets camera/material/geometry keyframes for every frame, triggers `bpy.ops.render.render(animation=True)`. Pay attention to performance (Eevee-Next on Metal).
3. **Feature schema** — what fields does `features.json` need to drive a compelling 3D reactive scene? We already have bass_onset, bass_energy, beat_decay, full-spectrum RMS, 128-bin FFT per frame. Tell us what else to pre-compute in Python and expose to Blender (e.g., spectral centroid for hue shift, tempo for camera dolly, drop flag for keyframed reveals).
4. **Geometry-Nodes audio-reactive ring** — how do you drive the 360 vertices of a torus from a JSON array of 128 FFT amps per frame in a way that looks fluid? Driver per-vertex? Vertex group from a texture baked per-frame? A mesh-deform-by-texture setup?
5. **Palette integration in Blender** — what's the cleanest way to pipe a BGR triple from Python into a shader's emission color that also feeds global lighting? (Equivalent of our current palette accent usage.)

## Practical constraints

- macOS (M-series). Blender 5.1.0 installed at `/opt/homebrew/bin/blender`. Metal backend supported.
- The pipeline must stay fully automated. We will not manually tweak `.blend` files per song. Everything through code.
- The user speaks Spanish and English interchangeably. Direct feedback, no wandering prose.
- No secret handling issues — `.env` + pre-commit already handle that.

## Iteration etiquette

- Use `--keyframes` mode for fast iteration (3–5 s per 4-frame preview). Don't do full 30 s renders to test parameter tweaks.
- The owner vetos with "feo" / "not nice" / "needs more X". When that happens, don't argue, adjust.
- Gemini Pro's aesthetic recommendations (toward "cyberpunk premium") have been unreliable. The structured judge (`rate_video.py` v2 paired A/B) is reliable. Use it.
- Never commit without explicit "commit" or "push" instruction.

---

If you accept, your first task is: propose concrete scaffolding for steps 1–5 above in a single message, ranked by which to do first. Go.
