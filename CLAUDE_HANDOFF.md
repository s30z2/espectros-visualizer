# Handoff — Claude Opus 4.7 continuation

You are taking over a Python audio visualizer project for TikTok-style posting. Read this whole file before touching code. Then read `CONTEXT.md` for older history (iterations v28–v43), `README.md` for install, and `audio_visualizer.py` for the code itself.

## The goal in one line

Generate a 9:16 vertical TikTok-ready music visualizer that matches the style of **`domixx_ref.mp4`** (this file lives in the project root — WATCH IT BEFORE ANYTHING ELSE). The user's song of choice is **`TRAVIESO.wav`** starting at 0:09.

## Most important thing to understand

**There are TWO "reference" videos in the repo and they are different styles. Do not confuse them.**

- `demo_visualizer.mp4` — the OLD reference we used through v28–v50. It's more aggressive/cyberpunk. Ignore it now.
- `domixx_ref.mp4` — the NEW reference (added v51). Simpler, darker, minimalist gothic. THIS IS THE TARGET.

The user watched my v50-era output, said it looked wrong, and sent `domixx_ref.mp4` as the *real* target. Everything from v51 onward targets `domixx_ref.mp4`.

## What domixx_ref.mp4 actually shows (do not rely on Gemini Pro's interpretation, read this)

The full visual spec is in the chat history under "Spec de referencia — @domixx007 video, análisis frame-a-frame". Summary:

- **Frame 0–3 s**: pure black + low-opacity "USE EARPHONES FOR THE BEST EXPERIENCE" text and small AirPods icon.
- **Then skulls wall**: dense 3D-rendered realistic skull wall filling the frame, monochrome blue-gray metallic palette. DoF blur on some skulls.
- **Small orb** (~18 % of frame width — NOT a big bowl): flat dark-blue disc with a thin white ring (≈ 3 px) on its edge. White "DX" text centered. Small glint top-left.
- **Waveform**: a single thin (~3–4 px) almost-white closed curve hugging the orb, gap ~5–10 px. Deforms like an elastic membrane with asymmetric bass-driven stretches. **NOT radial FFT bars. NOT pointy crown. NOT rays.** Think of a blob of jelly around the orb.
- **Camera**: mostly quiet. 3 % zoom punch on loud bass only. No persistent shake/drift.
- **Post**: desaturated palette, crushed blacks, light grain, small bloom only on orb ring. Minimal CA (≈ 0–1 px).
- **Particles**: ~15–20 static white twinkles, that's it. No embers, no streaks, no orbiting lights.
- **Diagonal flare**: dim static lens-scratch line. Not audio-reactive.

Gemini Pro keeps scoring this target's clone at 3–5/10 because Pro has a fixed mental image of "cyberpunk" (it asks for 15–20 px violent shakes, glossy 3D spheres, aggressive multi-layer bloom). **Ignore Pro's advice about adding violent motion or glossy 3D orbs — it's wrong for this reference.** The user explicitly does not want that.

## What the user is like (communication style)

- Bilingual (mostly Spanish, some English). Short messages.
- Vetos strongly when something feels wrong ("feo", "bastante feo", "no me gusta"). Do not argue, iterate.
- Does not want questions; wants action. Auto mode is often on.
- Values fast turnaround. Spends time watching output.
- Trusts visual judgment over numeric ratings.

## Current state (v51 — active as of this handoff)

- `audio_visualizer.py` targets the domixx_ref style.
- Latest preview: `v51_preview.mp4` (10 s, 1080p30). Open it. Also open `domixx_ref.mp4`. The v51 is close but not perfect.
- Gemini Pro rated v51 **3/10** — but **Pro is wrong for this target**. Palette got 8/10, waveform went 2→4 (actual improvement).

### What v51 does per subsystem

**Audio analysis (`AudioAnalyzer`)**
- 22 050 Hz mono load via librosa. `--start` offset supported.
- STFT n_fft=2048, hop 512.
- `bass_onset(t)`: per-frame derivative of <140 Hz STFT slice, normalized 0–1. This is the KICK detector. Use it for shake/zoom/flash gating.
- `bass_energy(t)`: RMS of <140 Hz band, normalized 0–1. Continuous bass loudness.
- `beat_decay(t)`: 500 ms decay with 0.85 exponent from librosa beat_track. Avoid for punchy effects — it rarely returns 0 at normal BPM (constant pulse problem).
- `energy(t)`: full-spectrum RMS.

**Orb (`_build_orb`)**
- Flat 2D disc, radius `ORB_R = 98`.
- Body fill `(16, 22, 30)` BGR (dark blue-black).
- White ring 3 px on circumference.
- Inner dark ring 1 px just inside (separates body from ring).
- Top-left glint: small soft white spot at (−0.45, −0.48) × ORB_R.
- "DX" text via `render_text_bgra`, color (245, 248, 252), size 0.95 × ORB_R.
- No refraction, no fresnel, no radial gradient shading. Intentionally minimal.

**Orb render (`_render_orb`)**
- Pulse factor: `1.0 + 0.04 * bass_pulse` (only up to 4 % zoom, bass_v>0.60 gated).
- Subtle rim glow only when kicking (dim, blurred circle under orb).
- Paste orb sprite + DX logo. Done. No shadows, halos, energy rings, CA, flash.

**Waveform (`_render_waveform`)**
- 12 control points (not 128 like before).
- Bass FFT bins bucketed into 12 amps. Power-compressed (^0.5) then mixed with 2-term sine drift (phase = t·0.55, evolves per-point).
- Interpolated to 360 points via periodic np.interp, then heavy Savitzky-Golay (win=31) for smooth C1 curve.
- Radial displacement `r_base + amps_interp * max_stretch * intensity`.
  - `r_base = ORB_R + 8` (close to orb, ~5–10 px gap)
  - `max_stretch = ORB_R * 0.60`
  - `intensity = 0.25 + bass_e * 1.2 + beat_i * 0.4`
- Single white stroke, 3–4 px thick, bri=235. Small halo blurs (9, 25 px) additive low-opacity.

**Particles (`_render_particles`)**
- Exactly 18 static white twinkles with per-point phase/freq. No embers, no streaks.

**Flares**
- `_render_anamorphic_flare`: static diagonal lens scratch, brightness 0.28 fixed. Not audio-reactive.
- `_render_orbit_flare`: disabled (early `return`).
- `_render_flares`: still fires on strong beats with 45° flare; consider disabling further if it's too much.

**Camera / post-pipeline (in `make_frame` main loop)**
- Bloom: `bloom(frame, thresh=130, strength=0.45)` — intentionally weak.
- Desat 40 % + blue bias + hard S-curve (`(f−0.5)*1.22+0.42`).
- Zoom: 1.0 + 0.03 × kick_gate (3 % max, only on loud bass). No translation, no rotation.
- CA: 1–2 px only when kick > 0.4, blend=0.18.
- NO energy rings, NO beat white flashes (ref doesn't have them).
- Vignette: cached radial mask, 35 % darken.
- Grain: ±7 mono noise (reduced from ±12).

**Background (`_build_bg` / `_render_bg`)**
- Loads `skulls_bg_gemini.png`, applies S-curve + cool grade (0.72 brightness).
- 3-layer DoF parallax: bg_far (blur 71 + 0.22×), bg_mid (blur 21 + 0.62×), bg_base (sharp).
- Kick-gated BG brightness bump (up to 20 % on loud bass only — no constant pulse).

## Fast iteration workflow (USE THIS, do not render full videos to tune)

Full 1080p60 renders take ~10–15 min each — too slow to iterate. I built a keyframe mode:

```bash
python3 audio_visualizer.py --audio TRAVIESO.wav --output kf.png --keyframes --start 9
```

- Auto-picks 4 timestamps: 1 quiet + 3 loudest bass hits within the first 20 s.
- Renders each as a PNG, concatenates horizontally, saves.
- Total time: **~2–5 seconds** at 1080p (thanks to `fast_blur`).
- You can override timestamps: `--keyframes-at "2,10,13,15"`.

Always preview via `--keyframes` before doing a full render.

For a fast partial video preview:
```bash
python3 audio_visualizer.py --audio TRAVIESO.wav --output v.mp4 --duration 10 --start 9 --fps 30 --scale 0.5
```

## Critical performance trick — `fast_blur`

Gaussian blurs were the bottleneck. I added `fast_blur(img, ksize, sigma)` that:
- Uses plain cv2.GaussianBlur for ksize ≤ 60.
- For 61–140: downsamples 2 ×, blurs, upsamples.
- For 141–250: 4 ×.
- For 251+: 8 ×.

Result: visually indistinguishable, 4–16 × faster. All big blurs in the codebase should use `fast_blur`, not `cv2.GaussianBlur` directly. If you see `cv2.GaussianBlur(..., (101,101),...)` or larger in new code, swap it.

## Safety / secrets

`rate_video.py` previously had a hardcoded Gemini key. It leaked when we pushed to a public GitHub repo and Google auto-revoked it. Now:
- Key lives in `.env` (gitignored).
- `rate_video.py` auto-loads `.env` on import.
- `.gitignore` blocks `.env`, `.env.*`, `*_api_key*`, `secrets.json`.
- `hooks/pre-commit` scans staged diffs for `AIza…` pattern and blocks commits. Enabled via `git config core.hooksPath hooks`.

If you need to rate a video, assume the key is already in `.env`. If not, ask the user to put it there. Never commit it to any file.

## Render times (M-series Mac, current codebase)

- Keyframes (4 frames, 1080×1920): ~2–5 s
- 10 s @ 1080p30: ~2 min
- 10 s @ 1080p60: ~4 min
- 20 s @ 1080p60: ~8 min
- Full 115 s @ 1080p30: ~35 min
- Full 115 s @ 1080p60: ~70 min

## What to iterate on next (prioritized)

The user said "segui puliendo" (keep polishing) after v51. Comparing `v51_preview.mp4` to `domixx_ref.mp4`, these are the gaps:

1. **Waveform is too quiet when bass hits** — the elastic blob barely deforms on the loudest kicks. Either boost `intensity` multiplier on `bass_e` or reduce `amps^0.5` compression so peaks come through stronger.
2. **Skulls bg is too uniformly bright** — ref has real deep shadows between skulls, ours is lit evenly. Crush contrast further or darken mid tones.
3. **Orb ring could be slightly brighter/thicker** — ref's white ring has more presence than 3 px.
4. **Line thickness of waveform** — ref line is ~3 px but looks brighter because of tight bloom. Try thresh 100 strength 0.7 just on the waveform layer before full-frame bloom.
5. **Asymmetric bass stretch is weak** — the 12-point blob deforms symmetrically. Try weighting 2–3 random control points extra on each bass hit so the stretch looks directional (as ref does).

Always iterate via `--keyframes` first, show the user the PNG, adjust, iterate. Only do a full-length render after the user approves the style.

## File structure summary

```
audio_visualizer.py     — main (class Visualizer, generate_video, generate_keyframes, main CLI)
rate_video.py           — Gemini rater, auto-loads .env
domixx_ref.mp4          — ACTUAL REFERENCE (use this with rate_video.py)
demo_visualizer.mp4     — OLD reference (ignore)
TRAVIESO.wav            — target song
montagem_alquimia.wav   — old song (v28–v50 tested with this)
orb_3d.png              — Blender orb (unused in v51, kept for history)
skulls_bg_gemini.png    — bg texture (used)
bg_3d_far/mid/near.png  — abstract skull layers (unused)
hooks/pre-commit        — blocks API-key commits
CONTEXT.md              — older history (v28–v43)
CLAUDE_HANDOFF.md       — THIS FILE
README.md               — quickstart
.env.example            — key template
.env                    — real key (gitignored)
*.mp4 (tracked LFS)     — reference + key iteration previews
kf_*.png                — keyframe grids from iterations
v*_frame200.png         — single-frame extracts from iterations
```

## User's emotional state / trust model

- The user has been iterating on this for 3+ days.
- They value the fast_blur optimization and keyframe mode I added last session — those were the right calls.
- They are skeptical of Gemini ratings (rightly so — Pro gives contradictory advice run-to-run).
- They trust their own eye. Always show them output and let them judge, don't push them with rating numbers.
- They want this **content-ready for posting automation**, not a math-optimized-against-metric project.

## Commit etiquette

- Do not commit unless asked. User prefers explicit "push/commit" instructions.
- When asked: use `fast_blur` as the co-author tag-line pattern already in git history.
- Never force push.
- Respect `hooks/pre-commit`.

Good luck. The user is kind but direct. Get to the point.
