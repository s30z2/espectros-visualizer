#!/usr/bin/env python3
"""
ESPECTROS Dark Cyberpunk Audio Visualizer v5
=============================================
FFT radial waveform + zoom-punch + AI skull background.
Based on frame-by-frame analysis from Gemini & Grok.

Usage:
    python audio_visualizer.py --audio music.mp3
    python audio_visualizer.py --audio music.wav --logo "DX" --bg skulls_bg_gemini.png
"""

import os, sys, math, random, argparse
import time as _time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter

# ═══════════════════════════════════════════════════════════════
# CONFIG (tuned from Gemini/Grok analysis)
# ═══════════════════════════════════════════════════════════════
W, H = 1080, 1920
FPS = 30
INTRO_DUR = 1.5
CX, CY = W // 2, H // 2
ORB_R = 98                 # v51: small orb like domixx ref (~18% frame width)
BG_MARGIN = 120
N_FFT_BINS = 128           # radial waveform resolution (higher = more detail)
SMOOTH_ALPHA = 0.55        # more reactive, clearly changes per frame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BG = os.path.join(SCRIPT_DIR, "skulls_bg_gemini.png")

# Colors RGB (from Gemini palette)
C_BG       = (2, 4, 6)
C_SKULL_MID = (26, 58, 74)
C_SKULL_HI  = (93, 182, 217)
C_GLOW     = (178, 235, 242)
C_WHITE    = (255, 255, 255)

# Convert to BGR for OpenCV
def rgb2bgr(c): return (c[2], c[1], c[0])
BG_BGR      = rgb2bgr(C_BG)
GLOW_BGR    = rgb2bgr(C_GLOW)
SKULL_HI_BGR = rgb2bgr(C_SKULL_HI)


# ═══════════════════════════════════════════════════════════════
# TEXT / FONT
# ═══════════════════════════════════════════════════════════════
def get_font(size, bold=False):
    for p in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/SFNSDisplay.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: continue
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()

def render_text_bgra(text, font_size, color_rgb=(255,255,255), bold=False):
    font = get_font(font_size, bold)
    dummy = Image.new("RGBA", (1, 1))
    dd = ImageDraw.Draw(dummy)
    lines = text.split("\n")
    bboxes = [dd.textbbox((0,0), ln, font=font) for ln in lines]
    max_w = max(b[2]-b[0] for b in bboxes) if bboxes else 1
    line_h = max(b[3]-b[1] for b in bboxes) if bboxes else font_size
    sp = 8
    total_h = line_h * len(lines) + sp * (len(lines)-1)
    pad = 10
    img = Image.new("RGBA", (max_w+pad*2, total_h+pad*2), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    y = pad
    for ln, bb in zip(lines, bboxes):
        x = pad + (max_w - (bb[2]-bb[0]))//2
        draw.text((x-bb[0], y-bb[1]), ln, fill=(*color_rgb, 255), font=font)
        y += line_h + sp
    arr = np.array(img)
    return arr[:, :, [2,1,0,3]]


# ═══════════════════════════════════════════════════════════════
# DRAWING UTILS
# ═══════════════════════════════════════════════════════════════
def paste_centered(canvas, sprite_bgra, cx, cy, scale=1.0, opacity=1.0):
    if sprite_bgra is None or sprite_bgra.size == 0: return
    sp = sprite_bgra
    if scale != 1.0:
        nh, nw = max(1, int(sp.shape[0]*scale)), max(1, int(sp.shape[1]*scale))
        sp = cv2.resize(sp, (nw, nh), interpolation=cv2.INTER_LINEAR)
    sh, sw = sp.shape[:2]
    x1, y1 = cx - sw//2, cy - sh//2
    sx1, sy1 = max(0,-x1), max(0,-y1)
    dx1, dy1 = max(0,x1), max(0,y1)
    dx2, dy2 = min(canvas.shape[1], x1+sw), min(canvas.shape[0], y1+sh)
    sx2, sy2 = sx1+(dx2-dx1), sy1+(dy2-dy1)
    if dx2<=dx1 or dy2<=dy1: return
    region = sp[sy1:sy2, sx1:sx2]
    a = region[:,:,3].astype(np.float32)/255.0 * opacity
    a3 = a[:,:,np.newaxis]
    bg = canvas[dy1:dy2, dx1:dx2].astype(np.float32)
    fg = region[:,:,:3].astype(np.float32)
    canvas[dy1:dy2, dx1:dx2] = np.clip(bg*(1-a3) + fg*a3, 0, 255).astype(np.uint8)

def additive(base, layer):
    return np.clip(base.astype(np.int16) + layer.astype(np.int16), 0, 255).astype(np.uint8)

def fast_blur(img, ksize, sigma=0):
    """Pyramid-accelerated Gaussian blur.

    For large kernels (>60px), downsample → blur at low res → upsample.
    Visually indistinguishable from full-res Gaussian for large sigmas,
    but 4-16x faster because the O(k^2) kernel ops run on a much smaller image.

    - ksize <= 60  : regular cv2.GaussianBlur (already fast enough)
    - 61-140       : downsample 2x (4x fewer pixels)
    - 141-250      : downsample 4x (16x fewer pixels)
    - 251+         : downsample 8x (64x fewer pixels)
    """
    if ksize <= 60:
        return cv2.GaussianBlur(img, (ksize|1, ksize|1), sigma)
    if ksize <= 140:
        factor = 2
    elif ksize <= 250:
        factor = 4
    else:
        factor = 8
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//factor, h//factor), interpolation=cv2.INTER_AREA)
    k_small = max(3, (ksize // factor) | 1)  # odd kernel size
    s_small = max(1.0, (sigma or ksize * 0.2) / factor)
    blurred = cv2.GaussianBlur(small, (k_small, k_small), s_small)
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)

def bloom(frame, thresh=85, ksize=101, strength=0.95):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Soft threshold — smooth rolloff instead of hard cutoff for natural blow-out
    bright_f = np.clip((gray.astype(np.float32) - thresh) / (255 - thresh), 0, 1)
    bright_f = bright_f ** 1.5  # concentrate on truly bright pixels
    bright_mask = (bright_f * 255).astype(np.uint8)
    m3 = cv2.merge([bright_mask]*3)
    brights = (frame.astype(np.float32) * (m3.astype(np.float32)/255.0)).astype(np.uint8)
    # Tint bloom toward desaturated blue (BGR)
    brights_f = brights.astype(np.float32)
    brights_f[:,:,0] *= 1.05   # blue slight
    brights_f[:,:,1] *= 0.90   # green down (shifts cyan→blue)
    brights_f[:,:,2] *= 0.55   # red down hard
    brights = np.clip(brights_f, 0, 255).astype(np.uint8)
    # Multi-pass bloom — pyramid-accelerated for large kernels
    g1 = fast_blur(brights, 31, 6)
    g2 = fast_blur(brights, 91, 20)
    g3 = fast_blur(brights, 181, 42)
    g4 = fast_blur(brights, 251, 70)
    combined = additive(additive(additive(g1, g2), g3), (g4.astype(np.float32)*0.6).clip(0,255).astype(np.uint8))
    return cv2.addWeighted(frame, 1.0, combined, strength, 0)


# ═══════════════════════════════════════════════════════════════
# AUDIO ANALYSIS (with FFT per-frame)
# ═══════════════════════════════════════════════════════════════
class AudioAnalyzer:
    def __init__(self, path, max_duration=None, start_offset=0.0):
        import librosa
        print(f"[*] Loading audio: {path} (offset={start_offset}s, duration={max_duration})")
        self.y, self.sr = librosa.load(path, sr=22050, mono=True,
                                       duration=max_duration, offset=start_offset)
        self.start_offset = start_offset
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        print("[*] Analyzing audio ...")

        # Beats
        self.tempo, bf = librosa.beat.beat_track(y=self.y, sr=self.sr)
        if hasattr(self.tempo, '__len__'):
            self.tempo = float(self.tempo[0]) if len(self.tempo) else 120.0
        self.beat_times = librosa.frames_to_time(bf, sr=self.sr)

        # Onsets (for kick detection) — full spectrum
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.onset_t = librosa.frames_to_time(np.arange(len(self.onset_env)), sr=self.sr)
        mx = self.onset_env.max()
        if mx > 0: self.onset_env /= mx

        # BASS-ONLY onset — only reacts to kick drums / sub-bass (<140 Hz)
        # Isolate low-freq band via STFT mel-slice
        S_full = np.abs(librosa.stft(self.y, n_fft=2048, hop_length=512))
        freqs_full = librosa.fft_frequencies(sr=self.sr, n_fft=2048)
        bass_mask = freqs_full < 140
        bass_slice = S_full[bass_mask].mean(axis=0)
        # Per-frame derivative (onset = rising edges)
        bass_onset = np.diff(bass_slice, prepend=bass_slice[0])
        bass_onset = np.clip(bass_onset, 0, None)
        if bass_onset.max() > 0:
            bass_onset /= bass_onset.max()
        self.bass_onset_env = bass_onset
        self.bass_onset_t = librosa.frames_to_time(np.arange(len(bass_onset)), sr=self.sr, hop_length=512)

        # RMS
        self.rms = librosa.feature.rms(y=self.y)[0]
        self.rms_t = librosa.frames_to_time(np.arange(len(self.rms)), sr=self.sr)
        mx = self.rms.max()
        if mx > 0: self.rms /= mx

        # Pre-compute STFT for FFT waveform (the key change)
        self.stft = np.abs(librosa.stft(self.y, n_fft=2048, hop_length=512))
        self.stft_t = librosa.frames_to_time(np.arange(self.stft.shape[1]), sr=self.sr)
        # Frequency bins in Hz
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        print(f"[*] {self.duration:.1f}s | {self.tempo:.0f} BPM | {len(self.beat_times)} beats")

    def energy(self, t):
        i = min(np.searchsorted(self.rms_t, t), len(self.rms)-1)
        return float(self.rms[i])

    def bass_onset(self, t):
        """Bass-only onset (kick detection), normalized 0-1."""
        if not hasattr(self, 'bass_onset_env'): return 0.0
        i = min(np.searchsorted(self.bass_onset_t, t), len(self.bass_onset_env)-1)
        return float(self.bass_onset_env[i])

    def beat_decay(self, t, window=0.50):
        """500ms decay with slower rolloff — beat reactions more visible."""
        if len(self.beat_times) == 0: return 0.0
        past = self.beat_times[self.beat_times <= t]
        if len(past) == 0: return 0.0
        since = t - past[-1]
        if since > window: return 0.0
        return max(0.0, (1.0 - since/window) ** 0.85)

    def get_fft_bins(self, t, n_bins=N_FFT_BINS, low_hz=20, high_hz=150):
        """Get n_bins FFT magnitudes in bass range [20-150 Hz] at time t.
        Uses GLOBAL peak normalization (not per-frame) so quiet moments
        stay visually quiet — prevents background noise from triggering
        the full visual response when bass is silent."""
        frame_idx = min(np.searchsorted(self.stft_t, t), self.stft.shape[1]-1)
        lo = np.searchsorted(self.freqs, low_hz)
        hi = np.searchsorted(self.freqs, high_hz)
        spectrum = self.stft[lo:hi, frame_idx]
        if len(spectrum) == 0:
            return np.zeros(n_bins)
        indices = np.linspace(0, len(spectrum)-1, n_bins).astype(int)
        bins = spectrum[indices]
        # GLOBAL normalization — cached peak across the entire bass band
        if not hasattr(self, '_bass_peak'):
            self._bass_peak = self.stft[lo:hi].max()
        if self._bass_peak > 0:
            bins = bins / self._bass_peak
        return bins

    def bass_energy(self, t):
        """RMS energy in the bass band (20-150 Hz) only, normalized to 0-1."""
        if not hasattr(self, '_bass_energy_env'):
            lo = np.searchsorted(self.freqs, 20)
            hi = np.searchsorted(self.freqs, 150)
            bass_rms = np.sqrt((self.stft[lo:hi] ** 2).mean(axis=0))
            mx = bass_rms.max()
            if mx > 0: bass_rms /= mx
            self._bass_energy_env = bass_rms
            # hop_length=512 @ sr=22050 → frame time = idx * 512 / 22050
            self._bass_energy_t = np.arange(len(bass_rms)) * 512 / self.sr
        i = min(np.searchsorted(self._bass_energy_t, t), len(self._bass_energy_env)-1)
        return float(self._bass_energy_env[i])


# ═══════════════════════════════════════════════════════════════
# VISUALIZER v3
# ═══════════════════════════════════════════════════════════════
class Visualizer:
    def __init__(self, audio_path, logo_text="DX", duration=None, bg_path=None, start_offset=0.0):
        self.audio = AudioAnalyzer(audio_path, duration, start_offset=start_offset)
        self.dur = min(duration, self.audio.duration) if duration else self.audio.duration
        self.total_dur = self.dur + INTRO_DUR
        self.logo_text = logo_text

        # Temporal smoothing state for FFT
        self.prev_bins = np.zeros(N_FFT_BINS)
        self.last_t = -1

        self._build_bg(bg_path or DEFAULT_BG)
        self._build_orb()
        self._build_intro()
        self.flare_beats = [bt for bt in self.audio.beat_times if self.audio.energy(bt) > 0.4]
        print(f"[*] Ready. {self.total_dur:.1f}s ({int(self.total_dur*FPS)} frames)")

    def _build_bg(self, bg_path):
        print(f"[*] Loading background: {bg_path}")
        if not os.path.isfile(bg_path):
            bh, bw = H+BG_MARGIN*2, W+BG_MARGIN*2
            self.bg_base = np.full((bh, bw, 3), BG_BGR, dtype=np.uint8)
            self.bg_far = self.bg_base.copy()
            return
        img = cv2.imread(bg_path)
        bh, bw = H+BG_MARGIN*2, W+BG_MARGIN*2
        ih, iw = img.shape[:2]
        scale = max(bh/ih, bw/iw)
        img = cv2.resize(img, (int(iw*scale), int(ih*scale)), interpolation=cv2.INTER_LANCZOS4)
        cy_i, cx_i = img.shape[0]//2, img.shape[1]//2
        img = img[cy_i-bh//2:cy_i-bh//2+bh, cx_i-bw//2:cx_i-bw//2+bw]
        # Strong contrast to bring out skull shapes + dark base for gothic
        f = img.astype(np.float32)
        f_n = f / 255.0
        # S-curve: crush blacks, lift skull highlights
        f_n = np.clip((f_n - 0.48) * 1.6 + 0.42, 0, 1)
        f = f_n * 255.0
        # Cold desaturated blue-teal grade — v37 sweet spot
        f *= 0.72
        f[:,:,0] *= 1.10
        f[:,:,1] *= 0.85
        f[:,:,2] *= 0.45
        self.bg_base = np.clip(f, 0, 255).astype(np.uint8)
        # v48: Stronger DoF separation for visible depth
        # Near: sharper, brighter
        self.bg_base = cv2.GaussianBlur(self.bg_base, (3,3), 0)  # minimal blur (sharper near)
        # Mid: moderate blur, moderate dim
        self.bg_mid = fast_blur(self.bg_base, 21, 6)
        self.bg_mid = (self.bg_mid.astype(np.float32)*0.62).clip(0,255).astype(np.uint8)
        # Far: heavy blur, very dim (deep focus falloff)
        self.bg_far = fast_blur(self.bg_base, 71, 22)
        self.bg_far = (self.bg_far.astype(np.float32)*0.22).clip(0,255).astype(np.uint8)
        print(f"[*] Background: {self.bg_base.shape[1]}x{self.bg_base.shape[0]}")

    def _build_orb(self):
        """v51: flat 2D disc + white ring + DX text + top-left glint.
        Matches domixx ref: NOT a 3D glass sphere, just a minimalist disc."""
        print("[*] Building flat disc orb (v51) ...")
        pad = 30
        size = ORB_R*2 + pad*2
        oc = size // 2
        orb = np.zeros((size, size, 4), dtype=np.uint8)

        # Dark blue-black fill body
        cv2.circle(orb, (oc, oc), ORB_R, (16, 22, 30, 255), -1, cv2.LINE_AA)
        # White-blue ring on border (delgado, 3px)
        ring_color = (235, 240, 250, 255)
        cv2.circle(orb, (oc, oc), ORB_R, ring_color, 3, cv2.LINE_AA)
        # Inner soft dark ring just inside to separate body from ring
        cv2.circle(orb, (oc, oc), ORB_R-5, (6, 10, 16, 255), 1, cv2.LINE_AA)

        # Top-left glint — small bright spot
        glint = np.zeros((size, size), dtype=np.float32)
        gx, gy = int(oc - ORB_R*0.45), int(oc - ORB_R*0.48)
        cv2.circle(glint, (gx, gy), int(ORB_R*0.10), 1.0, -1, cv2.LINE_AA)
        glint = cv2.GaussianBlur(glint, (0, 0), ORB_R*0.05)
        # Mask glint to inside orb
        yy, xx = np.ogrid[:size, :size]
        dd = np.sqrt((xx-oc)**2 + (yy-oc)**2)
        in_orb = (dd <= ORB_R).astype(np.float32)
        glint *= in_orb
        glint_rgba = np.dstack([glint*255]*3 + [glint*255]).astype(np.uint8)
        # Additive blend glint onto orb body
        ga = glint_rgba[:,:,3:4].astype(np.float32) / 255.0
        orb[:,:,:3] = np.clip(orb[:,:,:3].astype(np.float32) + glint_rgba[:,:,:3].astype(np.float32) * ga * 1.5, 0, 255).astype(np.uint8)
        orb[:,:,3] = np.maximum(orb[:,:,3], glint_rgba[:,:,3])

        self.orb_sprite = orb
        # DX logo: white sans-serif ~60% of diameter
        self.logo_sprite = render_text_bgra(self.logo_text, int(ORB_R*0.95),
                                            color_rgb=(245, 248, 252), bold=True)
        # No refraction for flat disc (safe cleanup)
        if hasattr(self, 'refr_map_x'):
            del self.refr_map_x

    def _build_intro(self):
        print("[*] Building intro ...")
        sz = 140
        iw, ih = int(sz*2.2), int(sz*1.6)
        icon = np.zeros((ih, iw, 4), dtype=np.uint8)
        col = (255,255,255,255)
        mid = iw//2
        # Left AirPod
        lx = mid - int(sz*0.22)
        cv2.ellipse(icon, (lx, int(ih*0.28)), (int(sz*0.13), int(sz*0.18)), 8, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(icon, (lx-int(sz*0.08), int(ih*0.22)), (int(sz*0.06), int(sz*0.12)), 20, -90, 90, col, max(2,int(sz*0.035)), cv2.LINE_AA)
        sw = max(3, int(sz*0.04))
        cv2.line(icon, (lx+int(sz*0.02), int(ih*0.42)), (lx-int(sz*0.01), int(ih*0.72)), col, sw, cv2.LINE_AA)
        cv2.ellipse(icon, (lx-int(sz*0.01), int(ih*0.73)), (sw+1, sw//2+1), 0, 0, 360, col, -1, cv2.LINE_AA)
        # Right AirPod
        rx = mid + int(sz*0.22)
        cv2.ellipse(icon, (rx, int(ih*0.28)), (int(sz*0.13), int(sz*0.18)), -8, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(icon, (rx+int(sz*0.08), int(ih*0.22)), (int(sz*0.06), int(sz*0.12)), -20, 90, 270, col, max(2,int(sz*0.035)), cv2.LINE_AA)
        cv2.line(icon, (rx-int(sz*0.02), int(ih*0.42)), (rx+int(sz*0.01), int(ih*0.72)), col, sw, cv2.LINE_AA)
        cv2.ellipse(icon, (rx+int(sz*0.01), int(ih*0.73)), (sw+1, sw//2+1), 0, 0, 360, col, -1, cv2.LINE_AA)
        self.intro_icon = icon
        self.intro_text = render_text_bgra("USE EARPHONES FOR THE BEST EXPERIENCE", 30, color_rgb=C_WHITE)

    # ════════════════════ RENDER ════════════════════

    def _render_intro(self, frame, t):
        if t < 0.5: a = t/0.5
        elif t < 2.5: a = 1.0
        elif t < INTRO_DUR: a = max(0, (INTRO_DUR-t)/1.0)
        else: a = 0.0
        if a <= 0: return
        paste_centered(frame, self.intro_icon, CX, CY-80, opacity=a)
        paste_centered(frame, self.intro_text, CX, CY+100, opacity=a)

    def _render_bg(self, frame, t, energy, beat_i=0.0):
        # 3-layer parallax — each layer moves at different speed for depth
        # Far layer (slowest, darkest, most blurred)
        dx1 = int(5*math.sin(t*0.12))
        dy1 = int(7*math.sin(t*0.09))
        y1, x1 = BG_MARGIN+dy1, BG_MARGIN+dx1
        far = self.bg_far[y1:y1+H, x1:x1+W]
        np.maximum(frame, far, out=frame)
        # Mid layer (medium speed)
        dx_m = int(12*math.sin(t*0.25+0.7))
        dy_m = int(16*math.sin(t*0.18+0.3))
        ym, xm = BG_MARGIN+dy_m, BG_MARGIN+dx_m
        mid = self.bg_mid[ym:ym+H, xm:xm+W]
        np.maximum(frame, mid, out=frame)
        # Near layer (fastest, sharpest, energy-reactive)
        dx2 = int(22*math.sin(t*0.35+1.5))
        dy2 = int(28*math.sin(t*0.28+0.8))
        y2, x2 = BG_MARGIN+dy2, BG_MARGIN+dx2
        near = self.bg_base[y2:y2+H, x2:x2+W]
        if energy > 0.15:
            boost = 1.0 + (energy-0.15) * 0.8
            near = np.clip(near.astype(np.float32)*boost, 0, 255).astype(np.uint8)
        # v51: BG kick bump only on strong bass onsets (not constant)
        bass_v = self.audio.bass_onset(t)
        kick_bg = max(0.0, (bass_v - 0.60) / 0.40) if bass_v > 0.60 else 0.0
        if kick_bg > 0.0:
            near = np.clip(near.astype(np.float32) * (1.0 + kick_bg * 0.20), 0, 255).astype(np.uint8)
        np.maximum(frame, near, out=frame)
        # Mild vignette darkening — keep skulls visible
        if not hasattr(self, '_bg_radial'):
            yg, xg = np.ogrid[0:H, 0:W]
            rd = np.sqrt((xg-CX)**2 + (yg-CY)**2).astype(np.float32)
            self._bg_radial = np.clip(1.0 - rd / (max(W,H)*1.2), 0.7, 1.0)[:,:,np.newaxis]
            # Precompute radial light mask — orb as light source illuminating skulls
            light_r = max(W, H) * 0.55
            self._bg_lightmask = np.clip(1.0 - rd / light_r, 0, 1) ** 1.8
            self._bg_lightmask = self._bg_lightmask[:,:,np.newaxis]
        frame[:] = np.clip(frame.astype(np.float32) * self._bg_radial, 0, 255).astype(np.uint8)
        # Orb-as-light-source: multiply skulls near center by cyan illumination (beat-reactive)
        light_strength = 0.4 + energy * 0.6
        light_tint = np.array([1.0 + 0.35*light_strength,  # B more
                               1.0 + 0.25*light_strength,  # G
                               1.0 + 0.10*light_strength], dtype=np.float32)  # R less
        illumination = 1.0 + self._bg_lightmask * (light_tint - 1.0)
        frame[:] = np.clip(frame.astype(np.float32) * illumination, 0, 255).astype(np.uint8)

    def _get_smoothed_bins(self, t):
        """FFT bins with minimal smoothing — keep sharp spikes."""
        raw = self.audio.get_fft_bins(t)
        raw = np.clip(raw, 0, None)
        # Light temporal smoothing only (no spatial — keep jagged)
        smoothed = raw * SMOOTH_ALPHA + self.prev_bins * (1 - SMOOTH_ALPHA)
        # Save ghost (lagging trail) — store last few frames
        if not hasattr(self, '_ghost_bins'):
            self._ghost_bins = []
        self._ghost_bins.append(smoothed.copy())
        if len(self._ghost_bins) > 6:
            self._ghost_bins.pop(0)
        self.prev_bins = smoothed
        return smoothed

    def _get_ghost_bins(self):
        """Return bins from ~5 frames ago for ghost trail effect."""
        if hasattr(self, '_ghost_bins') and len(self._ghost_bins) > 0:
            return self._ghost_bins[0]
        return np.zeros(N_FFT_BINS)

    def _render_waveform(self, frame, t, energy, beat_i):
        """v51 ELASTIC BLOB — closed curve of ~12 control points that deforms like
        a rubber membrane around the orb. Matches domixx ref: bumps are rounded,
        asymmetric, bass stretches in 1-2 directions (not a crown of rays)."""
        if energy < 0.02: return

        # Target ~12 bump points around orb (smooth closed curve)
        N_CTRL = 12
        bass_e = self.audio.bass_energy(t)

        # Pull bass FFT bins and bin them into N_CTRL buckets (averaged)
        bins = self._get_smoothed_bins(t)
        bucket = len(bins) // N_CTRL
        amps = np.array([bins[i*bucket:(i+1)*bucket].mean() for i in range(N_CTRL)])

        # Slow phase drift — per-point noise offset that evolves over time (like breathing blob)
        t_phase = t * 0.55
        drift = np.array([
            math.sin(t_phase * 1.3 + i*0.91) * 0.5 +
            math.sin(t_phase * 2.1 + i*1.71) * 0.25
            for i in range(N_CTRL)
        ])
        # Combine bass amplitude with drift — bass stretches 1-2 directions, drift adds lumpy idle motion
        amps = np.clip(amps ** 0.5, 0, None)            # compress peaks so stretches are organic
        amps = amps * 0.6 + np.abs(drift) * 0.4          # drift guarantees always a bit of wobble

        # Interpolate smoothly between N_CTRL points → 360 points on curve
        N_OUT = 360
        angles_ctrl = np.linspace(0, 2*math.pi, N_CTRL, endpoint=False)
        angles_out = np.linspace(0, 2*math.pi, N_OUT, endpoint=False)
        # Wrap: extend both ends for periodic interpolation
        amps_wrap = np.concatenate([amps[-3:], amps, amps[:3]])
        angles_wrap = np.concatenate([angles_ctrl[-3:] - 2*math.pi, angles_ctrl, angles_ctrl[:3] + 2*math.pi])
        amps_interp = np.interp(angles_out, angles_wrap, amps_wrap)
        # Heavy smooth after interp for continuous curve (Savitzky-Golay)
        from scipy.signal import savgol_filter as _savgol
        amps_interp = _savgol(np.concatenate([amps_interp[-25:], amps_interp, amps_interp[:25]]),
                              31, 3)[25:25+N_OUT]

        # Radial displacement: 5-10 px gap at rest, stretches up to ~60% ORB_R on bass hit
        r_gap = 8
        r_base = ORB_R + r_gap
        max_stretch = ORB_R * 0.60
        intensity = 0.25 + bass_e * 1.2 + beat_i * 0.4  # mostly idle, pops on bass

        r_disp = r_base + amps_interp * max_stretch * intensity
        pts = []
        for i in range(N_OUT + 1):
            idx = i % N_OUT
            theta = angles_out[idx]
            pts.append([int(CX + r_disp[idx] * math.cos(theta)),
                        int(CY + r_disp[idx] * math.sin(theta))])
        pts_np = np.array(pts, dtype=np.int32).reshape((-1,1,2))

        # Draw: single thick (3-4px) NEARLY WHITE line with subtle blue tint, tight halo only
        layer = np.zeros_like(frame)
        line_bri = 235
        line_thick = max(3, int(3 + bass_e * 1.5))
        cv2.polylines(layer, [pts_np], False,
                      (line_bri, line_bri, int(line_bri*0.92)),  # almost pure white, hint of blue
                      line_thick, cv2.LINE_AA)

        # Tight inner halo (12-18 px) — small bloom only around line
        halo_sm = fast_blur(layer, 9, 3)
        halo_md = fast_blur(layer, 25, 8)
        frame[:] = additive(frame, layer)
        frame[:] = additive(frame, (halo_sm.astype(np.float32)*0.70).clip(0,255).astype(np.uint8))
        frame[:] = additive(frame, (halo_md.astype(np.float32)*0.35).clip(0,255).astype(np.uint8))

    def _render_orb(self, frame, t, energy, beat_i):
        # v50: orb only pulses on STRONG bass hits, not continuously
        bass_v = self.audio.bass_onset(t)
        bass_pulse = max(0.0, (bass_v - 0.60) / 0.40) if bass_v > 0.60 else 0.0
        pulse = 1.0 + 0.04 * bass_pulse  # tiny 4% bump (was 12%) — ref has 3% max

        # v51: no drop shadow, no refraction, no big halo — just the disc + subtle rim glow

        # Subtle rim glow — very dim, same size as orb, activates only on bass
        if bass_pulse > 0.0:
            gl = np.zeros_like(frame)
            gl_bri = min(90, int(90 * bass_pulse))
            cv2.circle(gl, (CX, CY), int(ORB_R * 1.05),
                       (gl_bri, gl_bri, int(gl_bri*0.92)), -1, cv2.LINE_AA)
            gl = fast_blur(gl, 61, 18)
            frame[:] = additive(frame, (gl.astype(np.float32)*0.5).clip(0,255).astype(np.uint8))

        # Orb sprite (flat disc with ring + glint) + DX logo
        paste_centered(frame, self.orb_sprite, CX, CY, scale=pulse)
        if self.logo_sprite is not None:
            paste_centered(frame, self.logo_sprite, CX, CY, scale=pulse*0.85)

    def _render_particles(self, frame, t, energy):
        """v51: ONLY ~18 static twinkling white dots, no embers/streaks."""
        twinkle_layer = np.zeros_like(frame)
        for i in range(18):
            rng = random.Random(i * 7919 + 13)
            # Fixed positions (no drift)
            px = int(rng.uniform(60, W-60))
            py = int(rng.uniform(60, H-60))
            # Twinkle: slow opacity modulation with individual phase
            phase = rng.uniform(0, 2*math.pi)
            freq = rng.uniform(0.7, 1.6)
            brightness = 0.5 + 0.5 * math.sin(t * freq + phase)
            size = rng.uniform(1.0, 2.2)
            bri = int((80 + brightness * 120))
            cv2.circle(twinkle_layer, (px, py), max(1, int(size)),
                       (bri, bri, int(bri*0.92)), -1, cv2.LINE_AA)
        frame[:] = additive(frame, twinkle_layer)

    def _render_flares(self, frame, t):
        dur_f = 0.35
        layer = np.zeros_like(frame)
        active = False
        for idx, bt in enumerate(self.flare_beats):
            if bt > t: break
            age = t - bt
            if age > dur_f: continue
            active = True
            prog = age/dur_f
            opac = (1-prog)**1.5 * self.audio.energy(bt)
            rng = random.Random(idx*31+17)
            # 45-degree angle as Gemini specified
            angle = math.radians(45 + rng.uniform(-10, 10))
            # ~2000px/sec travel speed
            travel = prog * 2000 / FPS * dur_f
            sx = int(CX - 500 + travel*math.cos(angle))
            sy = int(CY - 500 + travel*math.sin(angle))
            length = int(H*1.2)
            x1 = sx - int(length*math.sin(angle)*0.5)
            y1 = sy - int(length*math.cos(angle)*0.5)
            x2 = sx + int(length*math.sin(angle)*0.5)
            y2 = sy + int(length*math.cos(angle)*0.5)
            thick = max(1, int(20*opac))
            val = min(255, int(100*opac))  # 40% max opacity as specified
            cv2.line(layer, (x1,y1), (x2,y2), (val,val,int(val*0.9)), thick, cv2.LINE_AA)
        if active:
            layer = cv2.GaussianBlur(layer, (41,41), 0)
            frame[:] = additive(frame, layer)

    def _render_anamorphic_flare(self, frame, t, energy, beat_i):
        """v51: static diagonal lens scratch (dim, non-reactive, matches ref)."""
        base_intensity = 0.28  # fixed low baseline, not audio-reactive
        bass = 0.0
        bass_pulse = 0.0
        # Diagonal angle ~55deg from top-left to bottom-right (like reference camera lens)
        angle_deg = 58.0
        ang_rad = math.radians(angle_deg)
        # Draw diagonal line through center
        length = int(max(W, H) * 1.6)
        dx = int(math.cos(ang_rad) * length/2)
        dy = int(math.sin(ang_rad) * length/2)
        layer = np.zeros_like(frame)
        flare_h = max(2, int(2 + beat_i * 4))
        bri = min(255, int(255 * base_intensity))
        cv2.line(layer, (CX - dx, CY - dy), (CX + dx, CY + dy),
                 (bri, bri, bri), flare_h, cv2.LINE_AA)  # PURE WHITE core
        # Rotate-then-blur approximation: blur along the diagonal direction
        # Use 2 passes: thin sharp core + wide bloom
        layer_sharp = cv2.GaussianBlur(layer, (5, 5), 1)
        layer_wide = fast_blur(layer, 61, 18)
        layer_glow = fast_blur(layer, 181, 55)
        frame[:] = additive(frame, layer_sharp)
        frame[:] = additive(frame, (layer_wide.astype(np.float32) * 0.75).clip(0,255).astype(np.uint8))
        frame[:] = additive(frame, (layer_glow.astype(np.float32) * 0.40).clip(0,255).astype(np.uint8))

    def _render_orbit_flare(self, frame, t, energy, beat_i):
        """v51: disabled — ref doesn't have orbiting light source."""
        return
        orbit_r = ORB_R * 1.3
        speed = 0.8  # orbits per second
        angle = 2 * math.pi * t * speed
        fx = int(CX + orbit_r * math.cos(angle))
        fy = int(CY + orbit_r * math.sin(angle))

        flare_layer = np.zeros_like(frame)
        bri = min(255, int(180 * (0.5 + energy * 0.5 + beat_i * 0.3)))

        # Draw trail (echoes)
        for echo in range(8):
            ea = angle - echo * 0.08
            er = orbit_r
            ex = int(CX + er * math.cos(ea))
            ey = int(CY + er * math.sin(ea))
            eb = max(10, int(bri * (1 - echo * 0.12)))
            esz = max(1, int(4 - echo * 0.3))
            cv2.circle(flare_layer, (ex, ey), esz,
                       (int(eb*0.9), int(eb*0.85), int(eb*0.5)), -1, cv2.LINE_AA)

        # Main flare point
        cv2.circle(flare_layer, (fx, fy), 5,
                   (int(bri*0.95), int(bri*0.9), int(bri*0.5)), -1, cv2.LINE_AA)
        cv2.circle(flare_layer, (fx, fy), 3, (bri, bri, int(bri*0.8)), -1, cv2.LINE_AA)

        flare_layer = cv2.GaussianBlur(flare_layer, (21, 21), 0)
        frame[:] = additive(frame, flare_layer)

    # ════════════════════ FRAME ════════════════════

    def make_frame(self, t):
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        if t < INTRO_DUR:
            self._render_intro(frame, t)
        else:
            e = self.audio.energy(t)
            bi = self.audio.beat_decay(t)

            self._render_bg(frame, t, e, bi)

            # v51: no constant light bleed from orb — ref doesn't have this

            # Glass sphere distortion behind orb (refraction)
            if not hasattr(self, '_distort_map'):
                ym, xm = np.mgrid[0:H, 0:W].astype(np.float32)
                dx = xm - CX; dy = ym - CY
                d = np.sqrt(dx**2 + dy**2)
                mask = d < ORB_R * 0.95
                strength_d = np.clip(1.0 - (d / (ORB_R*0.95))**2, 0, 1) * 25
                self._distort_mapx = xm.copy()
                self._distort_mapy = ym.copy()
                self._distort_mapx[mask] += (dx[mask] / np.maximum(d[mask], 1)) * strength_d[mask]
                self._distort_mapy[mask] += (dy[mask] / np.maximum(d[mask], 1)) * strength_d[mask]
            frame = cv2.remap(frame, self._distort_mapx, self._distort_mapy,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            self._render_waveform(frame, t, e, bi)
            self._render_orb(frame, t, e, bi)
            self._render_particles(frame, t, e)
            self._render_flares(frame, t)
            self._render_orbit_flare(frame, t, e, bi)
            self._render_anamorphic_flare(frame, t, e, bi)

            # v51 SUBTLE BLOOM — only on orb ring, not scene-wide haze
            frame = bloom(frame, thresh=130, strength=0.45)

            # Desaturate 40% + S-curve — colder metallic look (matches ref)
            frame_f = frame.astype(np.float32)
            gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[:,:,np.newaxis]
            frame_f = frame_f * 0.60 + gray_f * 0.40  # 40% desaturation
            # Cool tint — bias toward blue (ref is blue-gray)
            frame_f[:,:,0] = np.clip(frame_f[:,:,0] * 1.06, 0, 255)
            frame_f[:,:,2] = np.clip(frame_f[:,:,2] * 0.82, 0, 255)
            f_n = frame_f / 255.0
            f_n = np.clip((f_n - 0.5) * 1.22 + 0.42, 0, 1)  # crush blacks harder
            frame = (f_n * 255.0).clip(0, 255).astype(np.uint8)

            # ── QUIET CAMERA — only 3% zoom punch on bass, no shake ──
            onset_i = min(np.searchsorted(self.audio.onset_t, t), len(self.audio.onset_env)-1)
            onset_v = float(self.audio.onset_env[onset_i])
            bass_v = self.audio.bass_onset(t)
            kick = max(0.0, (bass_v - 0.60) / 0.40) ** 0.8

            zoom = 1.0
            shake_x, shake_y, rot = 0, 0, 0.0
            if kick > 0.0:
                zoom = 1.0 + 0.03 * kick   # 3% max zoom (ref spec)
                # no translational shake, no rotation — ref is quiet

            if zoom != 1.0:
                M = cv2.getRotationMatrix2D((W/2, H/2), rot, zoom)
                M[0,2] += shake_x
                M[1,2] += shake_y
                frame = cv2.warpAffine(frame, M, (W, H),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

            # Chromatic aberration — VERY subtle, only on peak kicks
            if kick > 0.4:
                ca_px = max(1, int(1 + 1 * kick))  # 1-2 px only
                h_f, w_f = frame.shape[:2]
                frame_ca = frame.copy()
                frame_ca[:, ca_px:, 2] = frame[:, :w_f-ca_px, 2]
                frame_ca[:, :w_f-ca_px, 0] = frame[:, ca_px:, 0]
                ca_blend = 0.18  # fixed low blend
                frame = cv2.addWeighted(frame, 1.0 - ca_blend, frame_ca, ca_blend, 0)

            # NO energy rings, NO beat flashes (ref doesn't have them)

            # ── STRONG VIGNETTE (35% edge darkening) ──
            if not hasattr(self, '_vignette'):
                vig = np.zeros((H, W), dtype=np.float32)
                cv2.circle(vig, (CX, CY), int(max(W,H)*0.55), 1.0, -1, cv2.LINE_AA)
                vig = cv2.GaussianBlur(vig, (351,351), 0)
                vig = np.clip(vig, 0.0, 1.0)
                self._vignette = (0.65 + 0.35 * vig)[:,:,np.newaxis]
            frame = np.clip(frame.astype(np.float32) * self._vignette, 0, 255).astype(np.uint8)

            # v51 SUBTLE MONO GRAIN (~3%)
            grain = np.random.randint(-7, 8, (H, W), dtype=np.int16)
            grain3 = np.stack([grain]*3, axis=-1)
            frame = np.clip(frame.astype(np.int16) + grain3, 0, 255).astype(np.uint8)

            # Fade in
            if t < INTRO_DUR + 1.5:
                alpha = (t - INTRO_DUR)/1.5
                frame = (frame.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════
def generate_keyframes(audio_path, output_path, logo_text="DX", bg_path=None,
                       times=None, start=0.0, window=20.0):
    """Render a few key moments (quiet + bass hits) as a single PNG grid.
    Use this for FAST iteration on visual parameters — ~5 sec instead of 15 min.

    Auto-picks 1 quiet + 3 strongest bass onsets within the first `window`
    seconds of the offset audio (default 20s = a typical postable clip).
    """
    viz = Visualizer(audio_path, logo_text, duration=None, bg_path=bg_path, start_offset=start)
    if times is None:
        # Restrict to the first `window` seconds so previews match a short clip
        bass_env = viz.audio.bass_onset_env
        bass_t = viz.audio.bass_onset_t
        upper = min(window, viz.audio.duration - 0.5)
        ranked = np.argsort(-bass_env)
        picked = []
        for idx in ranked:
            tc = bass_t[idx]
            if tc < INTRO_DUR + 0.5 or tc > upper:
                continue
            if all(abs(tc - p) > 1.0 for p in picked):
                picked.append(tc)
            if len(picked) >= 3:
                break
        quiet_t = INTRO_DUR + 1.0
        times = [quiet_t] + sorted(picked)
    print(f"[*] Rendering keyframes at t = {[f'{x:.2f}' for x in times]}")

    t0 = _time.time()
    frames = []
    for t in times:
        f = viz.make_frame(t)   # returns RGB
        # BGR for OpenCV imwrite
        frames.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    dt = _time.time() - t0

    # Stack horizontally with small labels
    labeled = []
    for i, (t, f) in enumerate(zip(times, frames)):
        # Write timestamp label on top-left
        out = f.copy()
        cv2.rectangle(out, (0, 0), (330, 60), (0, 0, 0), -1)
        cv2.putText(out, f"t={t:.2f}s", (15, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (180, 255, 255), 2, cv2.LINE_AA)
        labeled.append(out)
    grid = np.hstack(labeled)
    # Downsample grid so it fits on screen (max 1920 wide)
    gh, gw = grid.shape[:2]
    if gw > 2400:
        scale = 2400 / gw
        grid = cv2.resize(grid, (int(gw*scale), int(gh*scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, grid)
    print(f"[*] Keyframes done in {dt:.1f}s → {output_path}")

def generate_video(audio_path, output_path, logo_text="DX", duration=None, bg_path=None, start=0.0):
    from moviepy import VideoClip, AudioFileClip
    viz = Visualizer(audio_path, logo_text, duration, bg_path, start_offset=start)
    total = viz.total_dur
    fc = [0]
    tf = int(total*FPS)
    ts = [_time.time()]

    def mf(t):
        f = viz.make_frame(t)
        fc[0] += 1
        if fc[0] % 30 == 0:
            el = _time.time()-ts[0]
            pct = fc[0]/tf*100
            fps_a = fc[0]/max(el, 0.01)
            eta = (tf-fc[0])/max(fps_a, 0.01)
            print(f"\r  Rendering: {pct:5.1f}%  ({fc[0]}/{tf})  {fps_a:.1f}fps  ETA {eta:.0f}s", end="", flush=True)
        return f

    clip = VideoClip(mf, duration=total)
    audio = AudioFileClip(audio_path)
    # Apply start offset: start audio at `start` seconds into source
    if start > 0:
        audio_end = min(start + total, audio.duration)
        audio = audio.subclipped(start, audio_end)
    elif total < audio.duration:
        audio = audio.subclipped(0, total)
    clip = clip.with_audio(audio)
    print(f"\n[*] Encoding {output_path} ...")
    # Auto-bump bitrate at 60fps for clean TikTok-quality output
    _bitrate = "12000k" if FPS >= 50 else "8000k"
    clip.write_videofile(output_path, fps=FPS, codec="libx264", audio_codec="aac",
                         bitrate=_bitrate, preset="medium", logger=None)
    print(f"\n[*] Done! -> {output_path}")

def main():
    p = argparse.ArgumentParser(description="ESPECTROS Audio Visualizer v3")
    p.add_argument("--audio", required=True)
    p.add_argument("--output", default="visualizer_output.mp4")
    p.add_argument("--logo", default="DX")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--bg", default=None)
    p.add_argument("--scale", type=float, default=1.0, help="Resolution scale (0.5 = half res for fast preview)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--keyframes", action="store_true",
                   help="Render just 4 key moments as a PNG grid (~5 sec, for fast iteration)")
    p.add_argument("--keyframes-at", type=str, default=None,
                   help="Comma-separated timestamps in seconds for --keyframes (overrides auto-pick)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start offset in seconds — begins the video at this point in the audio")
    a = p.parse_args()
    if not os.path.isfile(a.audio):
        print(f"Error: {a.audio} not found"); sys.exit(1)
    # Apply scale and FPS overrides BEFORE class instantiation
    global W, H, CX, CY, ORB_R, BG_MARGIN, FPS
    if a.scale != 1.0:
        W = int(1080 * a.scale)
        H = int(1920 * a.scale)
        CX, CY = W // 2, H // 2
        ORB_R = int(195 * a.scale)
        BG_MARGIN = int(120 * a.scale)
        print(f"[*] FAST PREVIEW: {W}x{H}")
    if a.fps != 30:
        FPS = a.fps
        print(f"[*] FPS override: {FPS}")
    if a.keyframes:
        times = None
        if a.keyframes_at:
            times = [float(x.strip()) for x in a.keyframes_at.split(",")]
        # Ensure output is .png
        kf_out = a.output if a.output.endswith(".png") else a.output.rsplit(".", 1)[0] + "_keyframes.png"
        generate_keyframes(a.audio, kf_out, a.logo, a.bg, times, start=a.start)
    else:
        generate_video(a.audio, a.output, a.logo, a.duration, a.bg, start=a.start)

if __name__ == "__main__":
    main()
