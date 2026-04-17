# ESPECTROS — Phonk Audio Visualizer

Python audio visualizer targeting TikTok-style vertical 9:16 output. Generates 1080×1920 videos with a beat-reactive waveform around a DX orb on a skull-wall background.

**Current target:** `domixx_ref.mp4` (minimalist gothic style — see `CLAUDE_HANDOFF.md`). Current version is **v52**.

---

## Quick Start

```bash
# Install Git LFS once per machine
brew install git-lfs  # macOS  (Windows: https://git-lfs.github.com/  Linux: apt install git-lfs)
git lfs install

git clone https://github.com/s30z2/espectros-visualizer.git
cd espectros-visualizer
git lfs pull           # pulls tracked videos (~330 MB)
git config core.hooksPath hooks   # enables API-key pre-commit scanner

pip install numpy opencv-python pillow scipy librosa moviepy google-genai
```

## Single-song mode

```bash
# Fast 4-frame keyframe preview (~2–5 s per render)
python3 audio_visualizer.py --audio TRAVIESO.wav --output kf.png --keyframes --start 16.6

# 10 s 1080p30 preview
python3 audio_visualizer.py --audio TRAVIESO.wav --output out.mp4 --duration 10 --start 16.6

# Full-length render
python3 audio_visualizer.py --audio TRAVIESO.wav --output out.mp4 --start 16.6
```

## Batch mode (new in v52)

Drop audio files in `input/`, get TikTok-ready 30-second clips in `output/`, normalized to −14 LUFS with a seamless loop seam:

```bash
python3 audio_visualizer.py --batch
# or with explicit paths:
python3 audio_visualizer.py --batch --input-dir ./mixtapes --output-dir ./posts
```

What happens per track:

1. Hash the audio file → look in `cache/<hash>.json`
2. If not cached: detect the **first sustained bass-kick cluster** (first moment with ≥4 kicks in a 2s window above threshold)
3. Clip window = `[drop − 2 s, drop + 28 s]` (30 s total, 2 s buildup)
4. Route to a palette: deterministic hash of filename → one of 6 palettes. Palette colors the orb ring, waveform, BG tint, vignette. Orb body + DX text stay constant.
5. Render at 1080×1920@30fps, 8 Mbps
6. Crossfade last 15 visual frames back into the opening (audio untouched) for TikTok loop seam
7. Two-pass FFmpeg loudness normalize to −14 LUFS
8. Write cache after successful render

Rerun the same command — already-rendered outputs are skipped (`--skip-existing` default in batch mode).

## Palettes

6 palettes ship with the tool. Each has an accent color, BG tint_dark/tint_mid, and vignette color.

| Name | Accent (BGR) | Vibe |
|------|-------------|------|
| NOITE | `(255, 201, 127)` | Blue ice — the v51 default |
| SANGUE | `(0, 0, 179)` | Deep red |
| OURO | `(116, 165, 212)` | Warm amber/gold |
| VENENO | `(20, 255, 57)` | Acid green |
| FE | `(221, 78, 157)` | Violet pink |
| CINZA | `(232, 232, 232)` | Neutral grey/near-white |

### Selecting a palette

Three options, in priority order:

1. **`--palette FE`** — CLI flag, overrides everything
2. **Sidecar file `<audio>.palette.txt`** — e.g. drop a file `phonk1.palette.txt` next to `phonk1.mp3` containing `OURO`
3. **Default: hash of filename** — deterministic, same filename always gets the same palette, different filenames usually differ

## Files

| File | Purpose |
|------|---------|
| `audio_visualizer.py` | Main visualizer. Contains `AudioAnalyzer`, `Visualizer`, `generate_video`, `generate_keyframes`, CLI `main`. |
| `palettes.py` | Palette definitions + `get_palette_for_track` router |
| `batch.py` | Batch orchestrator + cache + per-track reporting |
| `post_process.py` | FFmpeg two-pass loudness normalization (−14 LUFS) |
| `rate_video.py` | Gemini video similarity rater (auto-loads `.env`) |
| `CLAUDE_HANDOFF.md` | Architectural handoff doc for Claude instances |
| `CONTEXT.md` | Older iteration history (v28–v43) |
| `domixx_ref.mp4` | The reference video (what we're matching) |
| `TRAVIESO.wav` | Default test song |

## API key setup

Four layers of protection against leaking a Gemini key:

1. **`.env` file** (gitignored) — copy `.env.example` to `.env` and paste the key.
2. **`rate_video.py` auto-loads `.env`** on startup. No hardcoding.
3. **`.gitignore`** blocks `.env`, `.env.*`, `*_api_key*`, `secrets.json`.
4. **Pre-commit hook** (`hooks/pre-commit`) scans staged diffs for `AIza…` and blocks the commit. Enabled via `git config core.hooksPath hooks` once per clone.

Get a fresh key: https://aistudio.google.com/app/apikey

## Render times (M-series Mac)

- Keyframes (4 frames, 1080×1920): **2–5 s**
- 10 s preview @ 1080p30: **~2 min**
- 30 s clip @ 1080p30 (batch default): **~5 min** (not counting LUFS pass)
- Full 115 s @ 1080p30: **~35 min**

The `fast_blur` pyramid-downsample trick keeps the big Gaussian kernels cheap — without it renders would be 4–16× slower.

## Iteration discipline

Use `--keyframes` for visual tuning. Iteration cost is ~3 seconds instead of 5 minutes. Do full video renders only to verify the final look.

Gemini Pro is unreliable for rating this style — it has a documented bias toward aggressive cyberpunk. Trust your eye + the reference video, not the numeric rating.
