#!/usr/bin/env python3
"""Rate video similarity using Gemini API (new google.genai SDK)."""
import sys, os, time
from google import genai

# Load from .env file if env var not set
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()

API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

def upload_video(path):
    print(f"[*] Uploading {os.path.basename(path)} ...")
    f = client.files.upload(file=path)
    while f.state.name == "PROCESSING":
        time.sleep(3)
        f = client.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"Upload failed: {f.state.name}")
    print(f"    -> {f.name} (ACTIVE)")
    return f

def rate(reference_path, generated_path):
    ref = upload_video(reference_path)
    gen = upload_video(generated_path)

    prompt = """You are an expert video effects / motion graphics judge.

I'm showing you TWO videos:
1. **REFERENCE** (first video) — the target style I want to replicate (a dark cyberpunk/gothic audio visualizer with skulls background, central orb, neon waveform/aura, beat-reactive shakes, bloom, particles).
2. **GENERATED** (second video) — my Python-generated attempt to replicate that style.

Rate the GENERATED video's visual similarity to the REFERENCE on a scale of 1-10 where:
- 1-3: Not similar at all
- 4-5: Some elements match but major differences
- 6-7: Good attempt, recognizable style but clear gaps
- 8-9: Very close, professional quality match
- 10: Identical

Evaluate these specific aspects (rate each 1-10):
1. **Overall vibe/mood** — dark cyberpunk gothic atmosphere
2. **Background** — skull density, depth, parallax movement
3. **Central orb** — size, glossiness, 3D look, logo
4. **Waveform/aura** — FFT reactivity, shape, color, exaggeration
5. **Beat reactivity** — shake intensity, zoom-punch, screen movement
6. **Bloom/glow** — neon glow quality, light bleeding
7. **Particles & flares** — quantity, behavior, look
8. **Color palette** — dark teal/cyan neon match

Then give an **OVERALL SCORE** (1-10).

Finally, list the **TOP 3 CHANGES** that would most improve the similarity score, in order of impact. Be very specific and technical (e.g. "increase waveform peak extension to 250px", "add 3px chromatic aberration offset").

Format your response as:
SCORES:
Overall vibe: X/10
Background: X/10
Central orb: X/10
Waveform: X/10
Beat reactivity: X/10
Bloom/glow: X/10
Particles: X/10
Color palette: X/10

OVERALL: X/10

TOP 3 CHANGES:
1. ...
2. ...
3. ..."""

    models_to_try = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]
    response = None
    for m in models_to_try:
        try:
            print(f"[*] Trying model: {m} ...")
            response = client.models.generate_content(
                model=m,
                contents=[prompt, ref, gen],
            )
            break
        except Exception as e:
            print(f"    {m} failed: {e}")
            continue
    if response is None:
        raise RuntimeError("All models failed")

    # Cleanup
    try:
        client.files.delete(name=ref.name)
        client.files.delete(name=gen.name)
    except:
        pass

    return response.text

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python rate_video.py <reference.mp4> <generated.mp4>")
        sys.exit(1)
    result = rate(sys.argv[1], sys.argv[2])
    print("\n" + "="*60)
    print("GEMINI RATING")
    print("="*60)
    print(result)
