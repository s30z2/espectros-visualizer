#!/usr/bin/env python3
"""
ESPECTROS Dark Cyberpunk Audio Visualizer v48
=============================================
v48: orb inner emission glow, denser particles + light streaks,
beat-pulsing skull highlights. Targeting orb/particles/bg scores.

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
ORB_R = 80                 # small orb (~15% of frame width, matching reference)
BG_MARGIN = 120
N_FFT_BINS = 128           # radial waveform resolution (higher = more detail)
SMOOTH_ALPHA = 0.55        # more reactive, clearly changes per frame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BG = os.path.join(SCRIPT_DIR, "skulls_bg_gemini.png")
BG_3D_SCENE = os.path.join(SCRIPT_DIR, "bg_3d_scene.png")
BG_3D_DEPTH = os.path.join(SCRIPT_DIR, "bg_3d_depth.png")

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
    # Multi-pass bloom — wider spread for bleeding
    g1 = cv2.GaussianBlur(brights, (31,31), 6)
    g2 = cv2.GaussianBlur(brights, (91,91), 20)
    g3 = cv2.GaussianBlur(brights, (181,181), 42)
    g4 = cv2.GaussianBlur(brights, (251,251), 70)
    combined = additive(additive(additive(g1, g2), g3), (g4.astype(np.float32)*0.6).clip(0,255).astype(np.uint8))
    return cv2.addWeighted(frame, 1.0, combined, strength, 0)


# ═══════════════════════════════════════════════════════════════
# AUDIO ANALYSIS (with FFT per-frame)
# ═══════════════════════════════════════════════════════════════
class AudioAnalyzer:
    def __init__(self, path, max_duration=None):
        import librosa
        print(f"[*] Loading audio: {path}")
        self.y, self.sr = librosa.load(path, sr=22050, mono=True, duration=max_duration)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        print("[*] Analyzing audio ...")

        # Beats
        self.tempo, bf = librosa.beat.beat_track(y=self.y, sr=self.sr)
        if hasattr(self.tempo, '__len__'):
            self.tempo = float(self.tempo[0]) if len(self.tempo) else 120.0
        self.beat_times = librosa.frames_to_time(bf, sr=self.sr)

        # Onsets (for kick detection)
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.onset_t = librosa.frames_to_time(np.arange(len(self.onset_env)), sr=self.sr)
        mx = self.onset_env.max()
        if mx > 0: self.onset_env /= mx

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

    def beat_decay(self, t, window=0.50):
        """500ms decay with slower rolloff — beat reactions more visible."""
        if len(self.beat_times) == 0: return 0.0
        past = self.beat_times[self.beat_times <= t]
        if len(past) == 0: return 0.0
        since = t - past[-1]
        if since > window: return 0.0
        return max(0.0, (1.0 - since/window) ** 0.85)

    def get_fft_bins(self, t, n_bins=N_FFT_BINS, low_hz=20, high_hz=400):
        """Get n_bins FFT magnitudes in [low_hz, high_hz] range at time t."""
        frame_idx = min(np.searchsorted(self.stft_t, t), self.stft.shape[1]-1)
        # Find freq range indices
        lo = np.searchsorted(self.freqs, low_hz)
        hi = np.searchsorted(self.freqs, high_hz)
        spectrum = self.stft[lo:hi, frame_idx]
        if len(spectrum) == 0:
            return np.zeros(n_bins)
        # Resample to n_bins
        indices = np.linspace(0, len(spectrum)-1, n_bins).astype(int)
        bins = spectrum[indices]
        # Normalize
        mx = bins.max()
        if mx > 0:
            bins = bins / mx
        return bins


# ═══════════════════════════════════════════════════════════════
# VISUALIZER v3
# ═══════════════════════════════════════════════════════════════
class Visualizer:
    def __init__(self, audio_path, logo_text="DX", duration=None, bg_path=None):
        self.audio = AudioAnalyzer(audio_path, duration)
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
        # v45: prefer Blender-rendered 3D passes, else generate synthetic depth
        self.has_3d_bg = False
        if os.path.isfile(BG_3D_SCENE) and os.path.isfile(BG_3D_DEPTH):
            try:
                self._build_bg_3d(BG_3D_SCENE, BG_3D_DEPTH)
                return
            except Exception as exc:
                print(f"[!] 3D bg load failed ({exc}); falling back to synthetic depth.")
                self.has_3d_bg = False

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

        # v45: brighter skulls — let bg DOMINATE the frame (reference style)
        f = img.astype(np.float32)
        f_n = f / 255.0
        # Gentler S-curve — preserve more midtone detail in skulls
        f_n = np.clip((f_n - 0.45) * 1.4 + 0.42, 0, 1)
        f = f_n * 255.0
        # Cold teal grade — bright base (v46: 0.88→0.95 for bg-dominant)
        f *= 0.95
        f[:,:,0] *= 1.08   # B slight
        f[:,:,1] *= 0.88   # G
        f[:,:,2] *= 0.50   # R down (keeps teal)
        graded = np.clip(f, 0, 255).astype(np.uint8)

        # v45: generate synthetic depth from luminance → enables 3D parallax
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Smooth heavily to get "near vs far skull" regions, not per-pixel noise
        depth = cv2.GaussianBlur(gray, (81, 81), 25)
        depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1.0)
        # Boost contrast: exaggerate near/far separation
        depth = np.clip(depth ** 0.7, 0, 1).astype(np.float32)

        self.bg_3d_sharp = graded
        self.bg_3d_blur = cv2.GaussianBlur(graded, (41, 41), 0)
        self.bg_3d_depth = depth
        self.bg_base = graded
        self.has_3d_bg = True
        print(f"[*] Synthetic 3D bg: {bw}x{bh}  depth [{depth.min():.3f},{depth.max():.3f}]")

    # ─── v44: 3D displacement-rendered background ───────────────────────
    def _build_bg_3d(self, scene_path, depth_path):
        """Load the Blender-rendered color + depth passes, color-grade, and
        precompute two blur levels for depth-of-field blending.
        """
        print(f"[*] Loading 3D bg: {scene_path}")
        scene_raw = cv2.imread(scene_path, cv2.IMREAD_UNCHANGED)
        if scene_raw is None:
            raise RuntimeError(f"cv2 failed to read {scene_path}")
        if scene_raw.ndim == 3 and scene_raw.shape[2] == 4:
            scene_raw = scene_raw[:, :, :3]

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"cv2 failed to read {depth_path}")
        if depth_raw.ndim == 3:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

        # Target size matches the padded bg canvas so post shake/warp stays valid
        bh, bw = H + BG_MARGIN * 2, W + BG_MARGIN * 2
        if scene_raw.shape[:2] != (bh, bw):
            scene_raw = cv2.resize(scene_raw, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
        if depth_raw.shape[:2] != (bh, bw):
            depth_raw = cv2.resize(depth_raw, (bw, bh), interpolation=cv2.INTER_LANCZOS4)

        # Normalize depth to float32 [0,1], near=1 (per Blender script Map Range)
        if depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) / 65535.0
        else:
            depth = depth_raw.astype(np.float32) / 255.0

        # Same color grade the 2D path applies — keeps the 8/10 palette score
        f = scene_raw.astype(np.float32)
        f_n = f / 255.0
        f_n = np.clip((f_n - 0.48) * 1.6 + 0.42, 0, 1)
        f = f_n * 255.0
        f *= 0.72
        f[:, :, 0] *= 1.10
        f[:, :, 1] *= 0.85
        f[:, :, 2] *= 0.45
        graded = np.clip(f, 0, 255).astype(np.uint8)

        self.bg_3d_sharp = graded
        self.bg_3d_blur = cv2.GaussianBlur(graded, (41, 41), 0)
        self.bg_3d_depth = depth  # float32 (bh, bw), 1=near, 0=far

        # Also expose bg_base so legacy code paths (intro, logging) don't break
        self.bg_base = graded
        self.has_3d_bg = True
        print(f"[*] 3D bg: {bw}x{bh}  depth range [{float(depth.min()):.3f},{float(depth.max()):.3f}]")

    def _render_bg_3d(self, frame, t, energy, beat_i):
        """Depth-driven parallax + DOF blend + v47 beat-reactive bg shift.
        Near pixels shift more than far on camera drift AND on beat shakes.
        """
        # Camera drift (stronger than the 2D version — depth attenuates it)
        cam_dx = 28.0 * math.sin(t * 0.25 + 0.7)
        cam_dy = 34.0 * math.sin(t * 0.18 + 0.3)
        # v47: couple beat shakes into bg parallax — bg moves WITH beats
        if beat_i > 0.25:
            gate = ((beat_i - 0.25) / 0.75) ** 0.5
            rng = random.Random(int(t * FPS * 1000))
            cam_dx += gate * 40.0 * rng.uniform(-1, 1)
            cam_dy += gate * 30.0 * rng.uniform(-1, 1)

        bh, bw = self.bg_3d_depth.shape[:2]
        ox = (bw - W) // 2
        oy = (bh - H) // 2

        # Cache the (y,x) grid for the output region
        if not hasattr(self, '_bg3d_xg'):
            yg, xg = np.mgrid[0:H, 0:W].astype(np.float32)
            self._bg3d_xg = xg
            self._bg3d_yg = yg
            self._bg3d_d_crop = self.bg_3d_depth[oy:oy + H, ox:ox + W].copy()
            # Pre-expand depth for 3-channel broadcasting
            self._bg3d_d3 = self._bg3d_d_crop[:, :, np.newaxis]

        d_crop = self._bg3d_d_crop
        # Near pixels (depth=1) get full drift; far pixels (depth=0) stay put
        map_x = (self._bg3d_xg + ox + np.float32(-cam_dx) * d_crop).astype(np.float32)
        map_y = (self._bg3d_yg + oy + np.float32(-cam_dy) * d_crop).astype(np.float32)

        sharp = cv2.remap(self.bg_3d_sharp, map_x, map_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        blur = cv2.remap(self.bg_3d_blur, map_x, map_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # DOF: sharp on near (depth=1), blur on far. Power curve = steeper falloff.
        dof_w = self._bg3d_d3 ** 1.3
        composed = sharp.astype(np.float32) * dof_w + blur.astype(np.float32) * (1.0 - dof_w)

        # Energy + beat brightness boost (mirrors the 2D near-layer behavior)
        if energy > 0.15:
            composed *= 1.0 + (energy - 0.15) * 0.8
        if beat_i > 0.1:
            composed *= 1.0 + beat_i * 0.28

        np.maximum(frame, np.clip(composed, 0, 255).astype(np.uint8), out=frame)

        # v45: lighter bg vignette — let skulls be visible everywhere
        if not hasattr(self, '_bg_radial'):
            yg, xg = np.ogrid[0:H, 0:W]
            rd = np.sqrt((xg - CX) ** 2 + (yg - CY) ** 2).astype(np.float32)
            self._bg_radial = np.clip(1.0 - rd / (max(W, H) * 1.4), 0.82, 1.0)[:, :, np.newaxis]
            light_r = max(W, H) * 0.65
            self._bg_lightmask = np.clip(1.0 - rd / light_r, 0, 1) ** 2.0
            self._bg_lightmask = self._bg_lightmask[:, :, np.newaxis]
        frame[:] = np.clip(frame.astype(np.float32) * self._bg_radial, 0, 255).astype(np.uint8)
        light_strength = 0.2 + energy * 0.4
        light_tint = np.array([1.0 + 0.20 * light_strength,
                               1.0 + 0.15 * light_strength,
                               1.0 + 0.05 * light_strength], dtype=np.float32)
        illumination = 1.0 + self._bg_lightmask * (light_tint - 1.0)
        frame[:] = np.clip(frame.astype(np.float32) * illumination, 0, 255).astype(np.uint8)

        # v48: beat-pulsing skull highlights — scattered bright cyan spots that glow on beats
        # Simulates emissive skull eye sockets / glowing skull features (bg scored 3/10)
        if beat_i > 0.1:
            if not hasattr(self, '_skull_highlights'):
                # Pre-generate fixed positions for highlight spots (seeded for consistency)
                hl_rng = random.Random(42)
                self._skull_highlights = []
                for _ in range(16):
                    hx = hl_rng.randint(60, W - 60)
                    hy = hl_rng.randint(80, H - 80)
                    hr = hl_rng.randint(8, 22)
                    self._skull_highlights.append((hx, hy, hr))
            hl_layer = np.zeros_like(frame)
            for hx, hy, hr in self._skull_highlights:
                hl_bri = min(140, int(60 * beat_i * (0.5 + energy * 0.5)))
                cv2.circle(hl_layer, (hx, hy), hr,
                           (int(hl_bri * 0.95), int(hl_bri * 0.80), int(hl_bri * 0.30)),
                           -1, cv2.LINE_AA)
            hl_layer = cv2.GaussianBlur(hl_layer, (31, 31), 8)
            frame[:] = additive(frame, hl_layer)

    def _build_orb(self):
        print("[*] Building orb ...")
        orb_3d_path = os.path.join(SCRIPT_DIR, "orb_3d.png")
        if os.path.isfile(orb_3d_path):
            # Use Blender-rendered 3D glass orb
            print(f"[*] Loading 3D orb: {orb_3d_path}")
            orb_raw = cv2.imread(orb_3d_path, cv2.IMREAD_UNCHANGED)
            target_size = ORB_R * 2 + 20
            orb = cv2.resize(orb_raw, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            # Apply circular alpha mask — Blender glass fills entire square
            oc = target_size // 2
            circ_mask = np.zeros((target_size, target_size), dtype=np.float32)
            cv2.circle(circ_mask, (oc, oc), ORB_R, 1.0, -1, cv2.LINE_AA)
            circ_mask = cv2.GaussianBlur(circ_mask, (9, 9), 2)
            if orb.shape[2] == 4:
                orb[:, :, 3] = (circ_mask * 255).astype(np.uint8)
            else:
                alpha = (circ_mask * 255).astype(np.uint8)
                orb = np.dstack([orb, alpha])

            # Add FRESNEL + SPECULAR HIGHLIGHT overlay to make it look like glass
            # Fresnel: bright ring at the edge
            fresnel = np.zeros((target_size, target_size, 4), dtype=np.float32)
            yy, xx = np.ogrid[:target_size, :target_size]
            d = np.sqrt((xx - oc)**2 + (yy - oc)**2)
            # Fresnel edge: bright at r=0.85-0.98, fades
            fres_strength = np.clip(1.0 - np.abs(d - ORB_R * 0.92) / (ORB_R * 0.08), 0, 1) ** 2
            fres_strength *= (d <= ORB_R).astype(np.float32)
            fresnel[:, :, 0] = fres_strength * 210  # B
            fresnel[:, :, 1] = fres_strength * 220  # G
            fresnel[:, :, 2] = fres_strength * 150  # R (cyan-ish white)
            fresnel[:, :, 3] = fres_strength * 180  # alpha
            fresnel = fresnel.astype(np.uint8)
            # Composite fresnel onto orb
            fa = fresnel[:, :, 3:4].astype(np.float32) / 255.0
            orb_rgb = orb[:, :, :3].astype(np.float32)
            fres_rgb = fresnel[:, :, :3].astype(np.float32)
            orb[:, :, :3] = np.clip(orb_rgb * (1 - fa) + fres_rgb * fa + orb_rgb * fa, 0, 255).astype(np.uint8)

            # 3D Radial Gradient: bright top-left → dark bottom-right (lambertian sphere shading)
            grad_cx, grad_cy = oc - ORB_R * 0.35, oc - ORB_R * 0.40
            dx = (xx - grad_cx)
            dy = (yy - grad_cy)
            grad_d = np.sqrt(dx**2 + dy**2)
            grad_n = np.clip(grad_d / (ORB_R * 1.9), 0, 1)
            # Stronger shading: 1.4x brightness top-left, 0.25x bottom-right
            shade = 1.4 - grad_n * 1.15
            shade = np.clip(shade, 0.25, 1.4)
            orb_mask = (d <= ORB_R).astype(np.float32)
            # Apply shading only inside orb
            orb_rgb_f = orb[:, :, :3].astype(np.float32)
            shade3 = shade[:,:,np.newaxis] * orb_mask[:,:,np.newaxis] + (1 - orb_mask[:,:,np.newaxis])
            orb_shaded = orb_rgb_f * shade3
            orb[:, :, :3] = np.clip(orb_shaded, 0, 255).astype(np.uint8)

            # Specular highlight: bright soft spot at top-left
            spec = np.zeros((target_size, target_size), dtype=np.float32)
            spec_cx, spec_cy = int(oc - ORB_R * 0.35), int(oc - ORB_R * 0.40)
            spec_r = int(ORB_R * 0.20)
            cv2.circle(spec, (spec_cx, spec_cy), spec_r, 1.0, -1, cv2.LINE_AA)
            spec = cv2.GaussianBlur(spec, (0, 0), spec_r * 0.6)
            spec *= orb_mask
            spec_mask = (spec[:, :, np.newaxis] * 250).astype(np.uint8)
            orb[:, :, :3] = np.clip(orb[:, :, :3].astype(np.int16) + spec_mask, 0, 255).astype(np.uint8)

            # Second small specular (reflection point)
            spec2 = np.zeros((target_size, target_size), dtype=np.float32)
            cv2.circle(spec2, (int(oc + ORB_R*0.25), int(oc + ORB_R*0.35)),
                       int(ORB_R*0.06), 1.0, -1, cv2.LINE_AA)
            spec2 = cv2.GaussianBlur(spec2, (0, 0), ORB_R*0.04)
            spec2 *= orb_mask
            spec2_mask = (spec2[:, :, np.newaxis] * 150).astype(np.uint8)
            orb[:, :, :3] = np.clip(orb[:, :, :3].astype(np.int16) + spec2_mask, 0, 255).astype(np.uint8)

            # v45: more transparent orb body — dark glass look like the reference
            edge_keep = np.clip((d - ORB_R*0.78) / (ORB_R*0.22), 0, 1) ** 1.2
            spec_keep = np.clip(spec * 1.5 + spec2 * 1.5, 0, 1)
            body_alpha = 1.0 - (1.0 - 0.45) * (1.0 - edge_keep) * (1.0 - spec_keep)  # v47: body=45% (was 30%)
            body_alpha *= circ_mask
            orb[:, :, 3] = (body_alpha * 255).astype(np.uint8)

            # v47: brighter orb glass for visibility (0.22 still too dark, orb scored 2/10)
            orb_f = orb[:, :, :3].astype(np.float32) * 0.35
            orb[:, :, :3] = np.clip(orb_f, 0, 255).astype(np.uint8)

            # Precompute spherical refraction remap (pinch+distort like glass ball)
            self.refr_size = target_size
            yy_f = np.arange(target_size, dtype=np.float32)[:, np.newaxis]
            xx_f = np.arange(target_size, dtype=np.float32)[np.newaxis, :]
            ddx = xx_f - oc
            ddy = yy_f - oc
            r = np.sqrt(ddx**2 + ddy**2) + 1e-6
            r_norm = np.clip(r / ORB_R, 0, 1)
            # Spherical refraction: pinch toward center using r^1.7 mapping
            r_refr = r_norm ** 1.7 * ORB_R * 0.85
            scale_factor = r_refr / r
            map_x = (oc + ddx * scale_factor).astype(np.float32)
            map_y = (oc + ddy * scale_factor).astype(np.float32)
            # Outside orb: just zero (masked out anyway)
            outside = (r > ORB_R)
            map_x[outside] = 0
            map_y[outside] = 0
            self.refr_map_x = map_x
            self.refr_map_y = map_y
            # Refraction alpha mask: full inside (excluding rim+specular), feathered at edge
            refr_alpha = np.clip(1.0 - r_norm, 0, 1) ** 0.4  # darker at center, fades to edge
            refr_alpha *= (1.0 - edge_keep * 0.7)  # fade out at rim where Fresnel is
            refr_alpha *= circ_mask
            self.refr_alpha = refr_alpha

            self.orb_sprite = orb
            self.logo_sprite = None
        else:
            # Fallback: simple procedural orb
            print("[*] orb_3d.png not found, using procedural orb")
            pad = 30
            size = ORB_R*2 + pad*2
            orb = np.zeros((size, size, 4), dtype=np.uint8)
            oc = size//2
            cv2.circle(orb, (oc,oc), ORB_R, (10,14,18,235), -1, cv2.LINE_AA)
            cv2.circle(orb, (oc,oc), ORB_R, (200,210,218,255), 5, cv2.LINE_AA)
            self.orb_sprite = orb
            self.logo_sprite = render_text_bgra(self.logo_text, 110, color_rgb=C_WHITE, bold=True)

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
        if getattr(self, 'has_3d_bg', False):
            self._render_bg_3d(frame, t, energy, beat_i)
            return
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
        # BEAT PULSE: subtle brighten on skull BG ("skulls glowing from within")
        if beat_i > 0.1:
            beat_boost = 1.0 + beat_i * 0.28
            near = np.clip(near.astype(np.float32) * beat_boost, 0, 255).astype(np.uint8)
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
        """v46: spikier FFT ring around orb — moderate peaks, visible but not giant arcs.
        Gemini TOP 2 feedback: v45 ring was nearly invisible. Add clear spikes.
        """
        if energy < 0.03: return

        bins = self._get_smoothed_bins(t)
        n = len(bins)

        # Light savgol smoothing — keep some jaggedness for spiky look
        win = 7 if n >= 9 else (n if n % 2 == 1 else n - 1)
        if win >= 5:
            ext = np.concatenate([bins[-win:], bins, bins[:win]])
            ext_smooth = savgol_filter(ext, win, 3)
            smooth = ext_smooth[win:win+n]
        else:
            smooth = bins.copy()
        smooth = np.clip(smooth, 0, None)

        # Peak amplification — sharper spikes for visible FFT response
        mean_b = np.mean(smooth) + 1e-6
        peak_boost = np.where(smooth > mean_b,
                              1.0 + (smooth - mean_b) * 1.8,
                              1.0)
        smooth = smooth * peak_boost
        # Extra boost for top 15% bins
        top_thresh = np.percentile(smooth, 85)
        smooth = np.where(smooth > top_thresh, smooth * 1.25, smooth)

        normalized = smooth / max(smooth.max(), 1e-6)

        r_base = ORB_R + 6 + beat_i * 6
        max_disp = 35 + beat_i * 25 + energy * 20  # v46: visible spikes (was 8)
        intensity = 0.6 + energy * 0.8 + beat_i * 0.6

        layer = np.zeros_like(frame)
        angles = np.linspace(0, 2*math.pi, n, endpoint=False)
        pts = []
        for i in range(n + 1):
            idx = i % n
            theta = angles[idx]
            r = r_base + normalized[idx] * max_disp * intensity
            pts.append([int(CX + r * math.cos(theta)),
                        int(CY + r * math.sin(theta))])
        pts_np = np.array(pts, dtype=np.int32).reshape((-1,1,2))

        # Base thick cyan stroke (6-10px)
        base_bri = min(170, int(110 * (0.3 + energy * 0.5 + beat_i * 0.5)))
        base_thick = max(6, int(6 + beat_i * 4))
        cv2.polylines(layer, [pts_np], True,
                       (int(base_bri*0.90), int(base_bri*0.80), int(base_bri*0.28)),
                       base_thick, cv2.LINE_AA)

        # Hot white-cyan core (2-4px)
        core_bri = min(220, int(170 * (0.4 + energy * 0.4 + beat_i * 0.4)))
        core_thick = max(2, int(2 + beat_i * 2))
        cv2.polylines(layer, [pts_np], True,
                       (core_bri, core_bri, int(core_bri*0.85)),
                       core_thick, cv2.LINE_AA)

        # Multi-pass glow (2 passes — moderate, not 4-pass monster)
        g1 = cv2.GaussianBlur(layer, (15, 15), 4)
        g2 = cv2.GaussianBlur(layer, (51, 51), 14)
        frame[:] = additive(frame, layer)
        frame[:] = additive(frame, g1)
        frame[:] = additive(frame, (g2.astype(np.float32)*0.55).clip(0,255).astype(np.uint8))

    def _render_orb(self, frame, t, energy, beat_i):
        pulse = 1.0 + 0.08 * beat_i + 0.03 * energy

        # v45: subtle drop shadow for small orb
        shadow = np.zeros_like(frame)
        cv2.ellipse(shadow, (CX, CY + int(ORB_R * 0.12)),
                    (int(ORB_R * 1.2), int(ORB_R * 1.2)),
                    0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
        shadow = cv2.GaussianBlur(shadow, (41, 41), 10)
        frame[:] = np.clip(frame.astype(np.int16) - (shadow * 0.25).astype(np.int16), 0, 255).astype(np.uint8)

        # v45: very subtle rim halo — only on strong beats
        if beat_i > 0.3:
            gl = np.zeros_like(frame)
            gr_in = int(ORB_R * pulse * 1.1)
            gb_in = min(50, int(30 * beat_i))
            cv2.circle(gl, (CX,CY), gr_in, (gb_in, int(gb_in*0.85), int(gb_in*0.4)), -1, cv2.LINE_AA)
            gl = cv2.GaussianBlur(gl, (31,31), 0)
            frame[:] = additive(frame, gl)

        # REFRACTION LAYER: capture bg behind orb, distort spherically, darken+tint cyan
        if hasattr(self, 'refr_map_x'):
            rs = self.refr_size
            half = rs // 2
            x1, y1 = CX - half, CY - half
            x2, y2 = x1 + rs, y1 + rs
            if x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H:
                bg_patch = frame[y1:y2, x1:x2].copy()
                # Apply spherical refraction remap
                refracted = cv2.remap(bg_patch, self.refr_map_x, self.refr_map_y,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # v45: darker glass — reference orb absorbs more light
                refr_f = refracted.astype(np.float32)
                refr_f *= 0.30  # much darker
                refr_f[:,:,0] = np.clip(refr_f[:,:,0] * 1.15 + 4, 0, 255)
                refr_f[:,:,1] = np.clip(refr_f[:,:,1] * 1.02, 0, 255)
                refr_f[:,:,2] = np.clip(refr_f[:,:,2] * 0.55, 0, 255)
                refracted = refr_f.astype(np.uint8)
                # Composite over frame using refr_alpha
                a = self.refr_alpha[:, :, np.newaxis]
                mixed = frame[y1:y2, x1:x2].astype(np.float32) * (1 - a) + refracted.astype(np.float32) * a
                frame[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)

        # Orb sprite (3D Blender render or procedural)
        paste_centered(frame, self.orb_sprite, CX, CY, scale=pulse)
        # Logo (only if not baked into 3D orb)
        if self.logo_sprite is not None:
            paste_centered(frame, self.logo_sprite, CX, CY, scale=pulse*0.85)

        # v46: brighter ring + wider glow for orb visibility
        ring = np.zeros_like(frame)
        ring_r = int(ORB_R * pulse * 1.02)
        ring_bri = min(230, int(160 + beat_i * 60))
        cv2.circle(ring, (CX, CY), ring_r, (ring_bri, ring_bri, int(ring_bri*0.88)),
                   max(2, int(2 + beat_i*2)), cv2.LINE_AA)
        ring_glow = cv2.GaussianBlur(ring, (21, 21), 5)
        ring_wide = cv2.GaussianBlur(ring, (61, 61), 16)
        frame[:] = additive(frame, ring)
        frame[:] = additive(frame, (ring_glow.astype(np.float32)*0.5).clip(0,255).astype(np.uint8))
        frame[:] = additive(frame, (ring_wide.astype(np.float32)*0.25).clip(0,255).astype(np.uint8))

        # v48: inner emission glow — soft cyan light from inside the orb (orb scored 2/10)
        inner_glow = np.zeros_like(frame)
        ig_r = int(ORB_R * pulse * 0.65)
        ig_bri = min(180, int(80 * (0.4 + energy * 0.5 + beat_i * 0.7)))
        cv2.circle(inner_glow, (CX, CY), ig_r,
                   (int(ig_bri * 1.0), int(ig_bri * 0.85), int(ig_bri * 0.35)),
                   -1, cv2.LINE_AA)
        inner_glow = cv2.GaussianBlur(inner_glow, (31, 31), 8)
        frame[:] = additive(frame, (inner_glow.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8))

        # Beat rim flash — subtle
        if beat_i > 0.3:
            rl = np.zeros_like(frame)
            rr = int(ORB_R * pulse)
            rb = min(140, int(140*beat_i))
            cv2.circle(rl, (CX,CY), rr, (rb,rb,rb), max(1, int(1+beat_i*2)), cv2.LINE_AA)
            rl = cv2.GaussianBlur(rl, (9,9), 0)
            frame[:] = additive(frame, rl)

    def _render_particles(self, frame, t, energy):
        """v48: denser particles + horizontal light streaks (particles scored 3/10).
        More visible sparkles scattered across frame + ember-like streaks.
        """
        # Floating dust particles — denser (60 instead of 30)
        dust_layer = np.zeros_like(frame)
        for di in range(60):
            rng = random.Random(di*3571+7)
            dx = (rng.uniform(0, W) + t * rng.uniform(-5, 5)) % W
            dy = (rng.uniform(0, H) + t * rng.uniform(-3, 3)) % H
            dsz = rng.uniform(0.6, 2.2)
            dbri = int(rng.uniform(30, 75))
            cv2.circle(dust_layer, (int(dx), int(dy)), max(1, int(dsz)),
                       (dbri, dbri, dbri), -1, cv2.LINE_AA)
        frame[:] = additive(frame, dust_layer)

        # Beat-triggered sparkles — more particles, scattered wide
        if energy > 0.15:
            spark_layer = np.zeros_like(frame)
            interval = 0.08  # more frequent spawns (was 0.15)
            lifetime = 2.2   # longer life (was 1.8)
            t0 = max(0, t - lifetime)
            i_s = int(t0 / interval)
            i_e = int(t / interval)
            for si in range(i_s, i_e + 1):
                st = si * interval
                if st > t or st < 0: continue
                age = t - st
                if age > lifetime: continue
                rng = random.Random(si * 7919 + 13)
                e_at = self.audio.energy(st)
                n = max(2, int(e_at * 6))  # more particles per spawn (was 3)
                for _ in range(n):
                    # Mix: some near orb, some scattered across frame
                    if rng.random() < 0.4:
                        # Near orb
                        angle = rng.uniform(0, 2*math.pi)
                        r0 = ORB_R * 0.8 + rng.uniform(0, ORB_R * 1.5)
                        x = CX + r0 * math.cos(angle)
                        y = CY + r0 * math.sin(angle)
                        x += math.cos(angle) * rng.uniform(8, 25) * age
                        y += math.sin(angle) * rng.uniform(8, 25) * age - 5 * age
                    else:
                        # Scattered across frame
                        x = rng.uniform(50, W - 50)
                        y = rng.uniform(50, H - 50)
                        x += rng.uniform(-3, 3) * age
                        y -= rng.uniform(5, 15) * age
                    life = 1.0 - age / lifetime
                    if life <= 0: continue
                    sz = rng.uniform(1.2, 3.5) * life * (0.5 + e_at * 0.5)
                    bri = int(160 * life * life * (0.4 + e_at * 0.5))
                    ix, iy = int(x) % W, int(y) % H
                    cv2.circle(spark_layer, (ix, iy), max(1, int(sz)),
                               (bri, int(bri * 0.92), int(bri * 0.75)), -1, cv2.LINE_AA)
            if np.any(spark_layer):
                sg = cv2.GaussianBlur(spark_layer, (9, 9), 2)
                frame[:] = additive(frame, sg)
                frame[:] = additive(frame, spark_layer)

        # v48: short horizontal light streaks — anamorphic sparkle effect
        streak_layer = np.zeros_like(frame)
        for si_k in range(10):
            rng = random.Random(si_k*9173 + int(t*10))
            sx = int(rng.uniform(80, W-80))
            sy = int(rng.uniform(80, H-80))
            slen = int(rng.uniform(30, 100) * (0.4 + energy * 0.6))
            sbri = int(rng.uniform(50, 120) * (0.3 + energy * 0.5 + self.audio.beat_decay(t) * 0.3))
            cv2.line(streak_layer, (sx - int(slen)//2, sy), (sx + int(slen)//2, sy),
                     (sbri, int(sbri*0.92), int(sbri*0.70)), 1, cv2.LINE_AA)
        streak_layer = cv2.GaussianBlur(streak_layer, (21, 3), 0)
        frame[:] = additive(frame, streak_layer)

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
        """v45: subtle diagonal flare — only on beats, thinner."""
        base_intensity = 0.10 + energy * 0.3 + beat_i * 0.5
        if base_intensity < 0.15: return
        angle_deg = 58.0
        ang_rad = math.radians(angle_deg)
        length = int(max(W, H) * 1.4)
        dx = int(math.cos(ang_rad) * length/2)
        dy = int(math.sin(ang_rad) * length/2)
        layer = np.zeros_like(frame)
        flare_h = max(1, int(1 + beat_i * 2))
        bri = min(180, int(180 * base_intensity))
        cv2.line(layer, (CX - dx, CY - dy), (CX + dx, CY + dy),
                 (bri, bri, bri), flare_h, cv2.LINE_AA)
        layer_sharp = cv2.GaussianBlur(layer, (5, 5), 1)
        layer_wide = cv2.GaussianBlur(layer, (41, 41), 12)
        frame[:] = additive(frame, layer_sharp)
        frame[:] = additive(frame, (layer_wide.astype(np.float32) * 0.45).clip(0,255).astype(np.uint8))

    def _render_orbit_flare(self, frame, t, energy, beat_i):
        """Subtle light source orbiting the orb."""
        orbit_r = ORB_R * 1.5
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

            # v45: subtler light bleed scaled to smaller orb
            light_layer = np.zeros_like(frame)
            lb = min(100, int(50 * (0.2 + e * 0.5 + bi * 0.4)))
            cv2.circle(light_layer, (CX, CY), int(ORB_R * 3.0),
                       (int(lb*0.85), int(lb*0.75), int(lb*0.35)), -1, cv2.LINE_AA)
            light_layer = cv2.GaussianBlur(light_layer, (151, 151), 0)
            frame[:] = additive(frame, (light_layer.astype(np.float32)*0.12).clip(0,255).astype(np.uint8))

            # Glass sphere distortion behind orb (refraction)
            if not hasattr(self, '_distort_map'):
                ym, xm = np.mgrid[0:H, 0:W].astype(np.float32)
                dx = xm - CX; dy = ym - CY
                d = np.sqrt(dx**2 + dy**2)
                mask = d < ORB_R * 0.95
                strength_d = np.clip(1.0 - (d / (ORB_R*0.95))**2, 0, 1) * 12  # v45: lighter distortion
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
            # v45: orbit flare removed — reference has no orbiting light
            self._render_anamorphic_flare(frame, t, e, bi)

            frame = bloom(frame, thresh=110, strength=0.70)  # v46: restored bloom for beat reactivity

            # Desaturate 15% + S-curve crush blacks
            frame_f = frame.astype(np.float32)
            gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)[:,:,np.newaxis]
            frame_f = frame_f * 0.85 + gray_f * 0.15
            f_n = frame_f / 255.0
            f_n = np.clip((f_n - 0.5) * 1.15 + 0.47, 0, 1)
            frame = (f_n * 255.0).clip(0, 255).astype(np.uint8)

            # ── ZOOM-PUNCH + SHAKE (ONLY on strong bass hits, not constant) ──
            onset_i = min(np.searchsorted(self.audio.onset_t, t), len(self.audio.onset_env)-1)
            onset_v = float(self.audio.onset_env[onset_i])
            bi_snap = bi ** 0.5
            # Zoom only triggers on significant beats (bi > 0.25) — clean otherwise
            zoom = 1.0
            shake_x, shake_y, rot = 0, 0, 0.0
            if bi > 0.25:
                # Gate activation smoothly above threshold
                gate = (bi - 0.25) / 0.75
                gate_snap = gate ** 0.5
                zoom = 1.0 + 0.28 * gate_snap
                rng = random.Random(int(t*FPS*1000))
                shake_x = int(gate_snap * 110 * rng.uniform(-1, 1))
                shake_y = int(gate_snap * 85 * rng.uniform(-1, 1))
                rot = gate_snap * 4.5 * rng.uniform(-1, 1)
            # Extra slam on STRONG onsets only (raised threshold from 0.30 to 0.48)
            if onset_v > 0.48:
                kick = (onset_v - 0.48) / 0.52
                zoom += 0.18 * kick
                rng2 = random.Random(int(t*FPS*997))
                shake_x += int(kick * 95 * rng2.uniform(-1, 1))
                shake_y += int(kick * 75 * rng2.uniform(-1, 1))

            M = cv2.getRotationMatrix2D((W/2, H/2), rot, zoom)
            M[0,2] += shake_x
            M[1,2] += shake_y
            frame = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

            # Radial motion blur on beats (zoom-blur toward center)
            if bi > 0.15 or e > 0.4:
                amt = max(bi * 0.04, (e-0.4)/0.6 * 0.02 if e > 0.4 else 0)
                Ms = cv2.getRotationMatrix2D((W/2,H/2), 0, 1.0+amt)
                zoomed = cv2.warpAffine(frame, Ms, (W,H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                blend = min(0.5, amt * 8)
                frame = cv2.addWeighted(frame, 1.0-blend, zoomed, blend, 0)

            # v46: moderate CA — visible on beats for reactivity
            ca_px = max(3, int(3 + 6 * bi))
            h_f, w_f = frame.shape[:2]
            frame_ca = frame.copy()
            frame_ca[:, ca_px:, 2] = frame[:, :w_f-ca_px, 2]
            frame_ca[:, :w_f-ca_px, 0] = frame[:, ca_px:, 0]
            ca_blend = min(0.70, 0.25 + bi * 0.45)
            frame = cv2.addWeighted(frame, 1.0 - ca_blend, frame_ca, ca_blend, 0)

            # v45: energy ring scaled to smaller orb, subtler
            if bi > 0.15:
                ring_layer = np.zeros_like(frame)
                ring_r = int(ORB_R + 15 + (1.0 - bi) * 80)
                ring_a = int(120 * bi)
                cv2.circle(ring_layer, (CX, CY), ring_r, (ring_a, ring_a, int(ring_a*0.9)), max(1, int(2*bi)), cv2.LINE_AA)
                ring_layer = cv2.GaussianBlur(ring_layer, (15,15), 0)
                frame = additive(frame, ring_layer)

            # v46: stronger beat flash for visible reactivity (Gemini TOP 1)
            if bi > 0.4:
                flash_a = (bi - 0.4) / 0.6 * 0.25  # max 25% (was 12%)
                flash = np.full_like(frame, 255)
                frame = cv2.addWeighted(frame, 1.0, flash, flash_a, 0)
            if onset_v > 0.45:
                kick_f = (onset_v - 0.45) / 0.55 * 0.15  # max 15% (was 8%)
                flash2 = np.full_like(frame, 255)
                frame = cv2.addWeighted(frame, 1.0, flash2, kick_f, 0)

            # v45: lighter vignette — let skulls remain visible at edges
            if not hasattr(self, '_vignette'):
                vig = np.zeros((H, W), dtype=np.float32)
                cv2.circle(vig, (CX, CY), int(max(W,H)*0.60), 1.0, -1, cv2.LINE_AA)
                vig = cv2.GaussianBlur(vig, (351,351), 0)
                vig = np.clip(vig, 0.0, 1.0)
                self._vignette = (0.72 + 0.28 * vig)[:,:,np.newaxis]  # v47: 28% edge darkening (was 18%)
            frame = np.clip(frame.astype(np.float32) * self._vignette, 0, 255).astype(np.uint8)

            # v45: subtle film grain
            grain = np.random.randint(-5, 6, (H, W), dtype=np.int16)
            grain3 = np.stack([grain]*3, axis=-1)
            grain3[:,:,0] += np.random.randint(-2, 3, (H, W), dtype=np.int16)
            grain3[:,:,2] += np.random.randint(-2, 3, (H, W), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + grain3, 0, 255).astype(np.uint8)

            # Fade in
            if t < INTRO_DUR + 1.5:
                alpha = (t - INTRO_DUR)/1.5
                frame = (frame.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════
def generate_video(audio_path, output_path, logo_text="DX", duration=None, bg_path=None):
    from moviepy import VideoClip, AudioFileClip
    viz = Visualizer(audio_path, logo_text, duration, bg_path)
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
    if total < audio.duration:
        audio = audio.subclipped(0, total)
    clip = clip.with_audio(audio)
    print(f"\n[*] Encoding {output_path} ...")
    clip.write_videofile(output_path, fps=FPS, codec="libx264", audio_codec="aac",
                         bitrate="8000k", preset="medium", logger=None)
    print(f"\n[*] Done! -> {output_path}")

def main():
    p = argparse.ArgumentParser(description="ESPECTROS Audio Visualizer v48")
    p.add_argument("--audio", required=True)
    p.add_argument("--output", default="visualizer_output.mp4")
    p.add_argument("--logo", default="DX")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--bg", default=None)
    p.add_argument("--scale", type=float, default=1.0, help="Resolution scale (0.5 = half res for fast preview)")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()
    if not os.path.isfile(a.audio):
        print(f"Error: {a.audio} not found"); sys.exit(1)
    # Apply scale and FPS overrides BEFORE class instantiation
    global W, H, CX, CY, ORB_R, BG_MARGIN, FPS
    if a.scale != 1.0:
        W = int(1080 * a.scale)
        H = int(1920 * a.scale)
        CX, CY = W // 2, H // 2
        ORB_R = int(80 * a.scale)
        BG_MARGIN = int(120 * a.scale)
        print(f"[*] FAST PREVIEW: {W}x{H}")
    if a.fps != 30:
        FPS = a.fps
        print(f"[*] FPS override: {FPS}")
    generate_video(a.audio, a.output, a.logo, a.duration, a.bg)

if __name__ == "__main__":
    main()
