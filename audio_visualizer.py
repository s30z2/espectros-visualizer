#!/usr/bin/env python3
"""
ESPECTROS Dark Cyberpunk Audio Visualizer v44
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
ORB_R = 195                # much bigger orb (~36% of frame width)
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
        # v44: prefer 3D displacement-rendered bg if both passes exist
        self.has_3d_bg = False
        if os.path.isfile(BG_3D_SCENE) and os.path.isfile(BG_3D_DEPTH):
            try:
                self._build_bg_3d(BG_3D_SCENE, BG_3D_DEPTH)
                return
            except Exception as exc:
                print(f"[!] 3D bg load failed ({exc}); falling back to 2D parallax.")
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
        # Apply gentle DoF blur on the "near" layer too — keep some softness
        self.bg_base = cv2.GaussianBlur(self.bg_base, (5,5), 0)
        # 3-layer parallax: far (heavily blurred+dark), mid (medium blur), near (slight blur)
        self.bg_far = cv2.GaussianBlur(self.bg_base, (35,35), 0)
        self.bg_far = (self.bg_far.astype(np.float32)*0.35).clip(0,255).astype(np.uint8)
        self.bg_mid = cv2.GaussianBlur(self.bg_base, (15,15), 0)
        self.bg_mid = (self.bg_mid.astype(np.float32)*0.55).clip(0,255).astype(np.uint8)
        print(f"[*] Background: {self.bg_base.shape[1]}x{self.bg_base.shape[0]}")

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
        """Depth-driven parallax + DOF blend. Slow camera drift shifts near
        pixels more than far pixels via cv2.remap; two blur levels are
        blended per-pixel by depth for soft focus on the far skulls.
        """
        # Camera drift (stronger than the 2D version — depth attenuates it)
        cam_dx = 28.0 * math.sin(t * 0.25 + 0.7)
        cam_dy = 34.0 * math.sin(t * 0.18 + 0.3)

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
        map_x = self._bg3d_xg + ox + (-cam_dx) * d_crop
        map_y = self._bg3d_yg + oy + (-cam_dy) * d_crop

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

        # Shared vignette + orb-as-light-source illumination
        if not hasattr(self, '_bg_radial'):
            yg, xg = np.ogrid[0:H, 0:W]
            rd = np.sqrt((xg - CX) ** 2 + (yg - CY) ** 2).astype(np.float32)
            self._bg_radial = np.clip(1.0 - rd / (max(W, H) * 1.2), 0.7, 1.0)[:, :, np.newaxis]
            light_r = max(W, H) * 0.55
            self._bg_lightmask = np.clip(1.0 - rd / light_r, 0, 1) ** 1.8
            self._bg_lightmask = self._bg_lightmask[:, :, np.newaxis]
        frame[:] = np.clip(frame.astype(np.float32) * self._bg_radial, 0, 255).astype(np.uint8)
        light_strength = 0.4 + energy * 0.6
        light_tint = np.array([1.0 + 0.35 * light_strength,
                               1.0 + 0.25 * light_strength,
                               1.0 + 0.10 * light_strength], dtype=np.float32)
        illumination = 1.0 + self._bg_lightmask * (light_tint - 1.0)
        frame[:] = np.clip(frame.astype(np.float32) * illumination, 0, 255).astype(np.uint8)

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

            # REFRACTION: make orb body partially transparent so distorted bg shows through
            # Keep edges (Fresnel) and specular spots fully opaque; reduce body alpha
            edge_keep = np.clip((d - ORB_R*0.78) / (ORB_R*0.22), 0, 1) ** 1.2  # 0 inside, 1 at rim
            spec_keep = np.clip(spec * 1.5 + spec2 * 1.5, 0, 1)  # specular regions fully opaque
            body_alpha = 1.0 - (1.0 - 0.55) * (1.0 - edge_keep) * (1.0 - spec_keep)  # body=55%, edges/spec=100%
            body_alpha *= circ_mask  # respect circular mask
            orb[:, :, 3] = (body_alpha * 255).astype(np.uint8)

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
        """Smooth organic aura waveform with thick blurred base + sharp top line."""
        if energy < 0.02: return

        bins = self._get_smoothed_bins(t)
        ghost_bins = self._get_ghost_bins()
        layer_thin = np.zeros_like(frame)
        layer_aura = np.zeros_like(frame)
        layer_ghost = np.zeros_like(frame)

        max_r = int(W * 0.60 / 2)  # peak extension (v30 sweet spot)
        n = len(bins)

        bins_ex = np.power(np.clip(bins, 0, None), 0.4)
        rs = np.random.RandomState(int(t*FPS*7) % (2**32 - 1))
        jitter = rs.uniform(-0.06, 0.06, n) * (0.2 + energy * 0.6 + beat_i * 0.5)
        bins_ex = np.clip(bins_ex + jitter, 0, None)
        intensity = 1.0 + energy * 2.5 + beat_i * 2.2

        # Lighter Savitzky-Golay smoothing — keep sharper/jagged peaks (v30)
        win = 7 if n >= 9 else (n if n % 2 == 1 else n - 1)
        if win >= 5:
            ext = np.concatenate([bins_ex[-win:], bins_ex, bins_ex[:win]])
            ext_smooth = savgol_filter(ext, win, 3)
            smooth_bins = ext_smooth[win:win+n]
        else:
            smooth_bins = bins_ex
        smooth_bins = np.clip(smooth_bins, 0, None)

        # MUCH more aggressive peak amplification for sharp spiky look
        mean_b = np.mean(smooth_bins)
        peak_boost = np.where(smooth_bins > mean_b,
                              1.0 + (smooth_bins - mean_b) * 2.6,  # was 1.5
                              1.0)
        smooth_bins = smooth_bins * peak_boost
        # Extra sharp-spike bonus for top decile bins
        top_thresh = np.percentile(smooth_bins, 85)
        smooth_bins = np.where(smooth_bins > top_thresh,
                                smooth_bins * 1.35,
                                smooth_bins)

        # Build displaced circumference path + TURBULENT DISPLACEMENT for electric look
        # Perlin-ish noise: use multiple sine layers modulated by time
        turb_rng = np.random.RandomState(int(t*FPS*127) % (2**32 - 1))
        # Per-vertex turbulent offset that evolves with time (creates "lightning" feel)
        turb_phase = t * 3.7
        angles_turb = np.linspace(0, 2*math.pi, n, endpoint=False)
        turb_amp = max_r * 0.12 * (0.5 + energy * 0.6 + beat_i * 0.8)  # bigger on beats
        turb1 = np.sin(angles_turb * 7.0 + turb_phase) * turb_amp
        turb2 = np.sin(angles_turb * 13.0 - turb_phase * 1.3) * turb_amp * 0.55
        turb3 = turb_rng.uniform(-1, 1, n) * turb_amp * 0.35  # random noise spikes
        turbulence = turb1 + turb2 + turb3

        r_base = ORB_R + 4
        pts = []
        angles = np.linspace(0, 2*math.pi, n, endpoint=False)
        for i in range(n + 1):
            idx = i % n
            theta = angles[idx]
            # FFT amplitude + turbulent displacement (creates jagged electric look)
            r_disp = r_base + smooth_bins[idx] * max_r * intensity + turbulence[idx]
            pts.append([int(CX + r_disp * math.cos(theta)),
                        int(CY + r_disp * math.sin(theta))])
        pts_np = np.array(pts, dtype=np.int32).reshape((-1,1,2))

        # LAYER 0 (base): WIDE cyan tube — thick neon glow base (18-25px)
        base_bri = min(180, int(140 * (0.4 + energy * 0.5 + beat_i * 0.5)))
        base_thick = max(18, int(18 + beat_i * 7))
        cv2.polylines(layer_thin, [pts_np], False,
                       (int(base_bri*0.88), int(base_bri*0.76), int(base_bri*0.25)),
                       base_thick, cv2.LINE_AA)

        # LAYER 1 (middle): brighter neon tube (10px)
        glow_bri = min(220, int(185 * (0.4 + energy * 0.5 + beat_i * 0.4)))
        cv2.polylines(layer_thin, [pts_np], False,
                       (int(glow_bri*0.92), int(glow_bri*0.82), int(glow_bri*0.35)),
                       max(10, int(10 + beat_i * 4)), cv2.LINE_AA)

        # LAYER 2 (top): white-hot core (3-5px)
        stroke_bri = min(255, int(250 * (0.5 + energy * 0.4 + beat_i * 0.3)))
        thick = max(3, int(3 + beat_i * 2))
        cv2.polylines(layer_thin, [pts_np], False,
                       (stroke_bri, stroke_bri, int(stroke_bri*0.92)),
                       thick, cv2.LINE_AA)

        # GHOST: lagging trail (5 frames behind, larger radius, low opacity)
        if len(ghost_bins) == n:
            ghost_smooth = np.power(np.clip(ghost_bins, 0, None), 0.4)
            if win >= 5:
                ext = np.concatenate([ghost_smooth[-win:], ghost_smooth, ghost_smooth[:win]])
                ext_smooth = savgol_filter(ext, win, 3)
                ghost_smooth = ext_smooth[win:win+n]
            ghost_pts = []
            for i in range(n + 1):
                idx = i % n
                theta = angles[idx]
                # Slightly larger radius for "ghost" effect (waveform expanding outward)
                r_g = r_base + ghost_smooth[idx] * max_r * intensity * 1.08
                ghost_pts.append([int(CX + r_g * math.cos(theta)),
                                  int(CY + r_g * math.sin(theta))])
            ghost_np = np.array(ghost_pts, dtype=np.int32).reshape((-1,1,2))
            ghost_bri = min(120, int(80 * (0.4 + energy * 0.5 + beat_i * 0.4)))
            cv2.polylines(layer_ghost, [ghost_np], False,
                           (int(ghost_bri*0.85), int(ghost_bri*0.75), int(ghost_bri*0.30)),
                           max(4, int(5 + beat_i * 3)), cv2.LINE_AA)
            layer_ghost = cv2.GaussianBlur(layer_ghost, (21, 21), 6)
            frame[:] = additive(frame, (layer_ghost.astype(np.float32) * 0.45).clip(0,255).astype(np.uint8))

        # Composite: layered waveform + AGGRESSIVE multi-pass bloom
        frame[:] = additive(frame, layer_thin)
        g1 = cv2.GaussianBlur(layer_thin, (15,15), 4)
        g2 = cv2.GaussianBlur(layer_thin, (45,45), 12)
        g3 = cv2.GaussianBlur(layer_thin, (101,101), 25)
        # Very wide atmospheric aura — heavier for neon bleed
        g4 = cv2.GaussianBlur(layer_thin, (251,251), 70)
        frame[:] = additive(frame, g1)
        frame[:] = additive(frame, (g2.astype(np.float32)*0.85).clip(0,255).astype(np.uint8))
        frame[:] = additive(frame, (g3.astype(np.float32)*0.60).clip(0,255).astype(np.uint8))
        frame[:] = additive(frame, (g4.astype(np.float32)*0.35).clip(0,255).astype(np.uint8))

    def _render_orb(self, frame, t, energy, beat_i):
        pulse = 1.0 + 0.15 * beat_i + 0.05 * energy

        # Drop shadow — dark oval offset below to separate orb from background
        shadow = np.zeros_like(frame)
        cv2.ellipse(shadow, (CX, CY + int(ORB_R * 0.15)),
                    (int(ORB_R * 1.25), int(ORB_R * 1.25)),
                    0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
        shadow = cv2.GaussianBlur(shadow, (81, 81), 22)
        # Subtract (darken)
        frame[:] = np.clip(frame.astype(np.int16) - (shadow * 0.35).astype(np.int16), 0, 255).astype(np.uint8)

        # Subtle rim halo — only visible on beats, much smaller + darker
        gl = np.zeros_like(frame)
        gr_in = int(ORB_R * pulse * 1.08)
        gb_in = min(120, int(70 * (0.2 + energy*0.6 + beat_i*0.7)))
        cv2.circle(gl, (CX,CY), gr_in, (gb_in, int(gb_in*0.90), int(gb_in*0.55)), -1, cv2.LINE_AA)
        gl = cv2.GaussianBlur(gl, (51,51), 0)
        frame[:] = additive(frame, gl)
        # Outer haze (only on strong beats)
        if beat_i > 0.2:
            gl2 = np.zeros_like(frame)
            gr_out = int(ORB_R * pulse * 1.5)
            gb_out = min(100, int(45 * beat_i))
            cv2.circle(gl2, (CX,CY), gr_out, (gb_out, int(gb_out*0.85), int(gb_out*0.50)), -1, cv2.LINE_AA)
            gl2 = cv2.GaussianBlur(gl2, (121,121), 0)
            frame[:] = additive(frame, gl2)

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
                # Darken + cyan tint (glass absorption)
                refr_f = refracted.astype(np.float32)
                refr_f *= 0.55  # darken
                refr_f[:,:,0] = np.clip(refr_f[:,:,0] * 1.20 + 8, 0, 255)  # boost B
                refr_f[:,:,1] = np.clip(refr_f[:,:,1] * 1.05, 0, 255)      # slight G
                refr_f[:,:,2] = np.clip(refr_f[:,:,2] * 0.65, 0, 255)      # cut R
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

        # INNER WHITE GLOWING RING (just outside orb rim) — separate from waveform
        ring = np.zeros_like(frame)
        ring_r = int(ORB_R * pulse * 1.015)
        ring_bri = min(255, int(200 + beat_i * 55))
        cv2.circle(ring, (CX, CY), ring_r, (ring_bri, ring_bri, int(ring_bri*0.9)),
                   max(3, int(3 + beat_i*3)), cv2.LINE_AA)
        ring_blur_sm = cv2.GaussianBlur(ring, (7, 7), 2)
        ring_blur_md = cv2.GaussianBlur(ring, (35, 35), 10)
        ring_blur_lg = cv2.GaussianBlur(ring, (101, 101), 28)
        frame[:] = additive(frame, ring)
        frame[:] = additive(frame, (ring_blur_sm.astype(np.float32)*0.8).clip(0,255).astype(np.uint8))
        frame[:] = additive(frame, (ring_blur_md.astype(np.float32)*0.55).clip(0,255).astype(np.uint8))
        frame[:] = additive(frame, (ring_blur_lg.astype(np.float32)*0.30).clip(0,255).astype(np.uint8))

        # Beat rim flash
        if beat_i > 0.05:
            rl = np.zeros_like(frame)
            rr = int(ORB_R * pulse)
            rb = min(255, int(255*beat_i))
            cv2.circle(rl, (CX,CY), rr, (rb,rb,rb), max(2, int(3+beat_i*4)), cv2.LINE_AA)
            rl = cv2.GaussianBlur(rl, (15,15), 0)
            frame[:] = additive(frame, rl)

    def _render_particles(self, frame, t, energy):
        # Ember particles (beat-triggered, short streaks for "light dust")
        interval = 0.05
        lifetime = 2.4
        t0 = max(0, t-lifetime)
        i_s = int(t0/interval)
        i_e = int(t/interval)
        particle_layer = np.zeros_like(frame)
        for si in range(i_s, i_e+1):
            st = si*interval
            if st > t or st < 0: continue
            age = t - st
            if age > lifetime: continue
            rng = random.Random(si*7919+13)
            e_at = self.audio.energy(st)
            n = int(e_at * 9) + 4  # MORE particles per spawn (was 6+2)
            for _ in range(n):
                angle = rng.uniform(0, 2*math.pi)
                spd = rng.uniform(15, 50) * (0.5 + e_at)
                r0 = ORB_R * 0.6 + rng.uniform(0, ORB_R * 0.9)
                x0 = CX + r0 * math.cos(angle)
                y0 = CY + r0 * math.sin(angle)
                x = x0 + math.cos(angle) * spd * age * 0.3 + rng.uniform(-4,4)*age
                y = y0 - spd * age + math.sin(angle) * spd * age * 0.2
                life = 1.0 - age/lifetime
                if life <= 0: continue
                sz = rng.uniform(2.5, 6.5) * life * (0.5 + e_at)
                bri = int(255 * life * life * (0.6 + e_at*0.5))
                ix, iy = int(x), int(y)
                if 0 <= ix < W and 0 <= iy < H:
                    cv2.circle(particle_layer, (ix, iy), max(1, int(sz)),
                               (bri, int(bri*0.92), int(bri*0.82)), -1, cv2.LINE_AA)
        if np.any(particle_layer):
            pg = cv2.GaussianBlur(particle_layer, (15,15), 0)
            frame[:] = additive(frame, pg)
            frame[:] = additive(frame, particle_layer)

        # More dense floating dust (50 particles instead of 25)
        dust_layer = np.zeros_like(frame)
        for di in range(50):
            rng = random.Random(di*3571+7)
            dx = (rng.uniform(0, W) + t * rng.uniform(-6, 6)) % W
            dy = (rng.uniform(0, H) + t * rng.uniform(-3, 3)) % H
            dsz = rng.uniform(0.8, 2.2)
            dbri = int(rng.uniform(30, 75))
            cv2.circle(dust_layer, (int(dx), int(dy)), max(1, int(dsz)), (dbri, dbri, dbri), -1, cv2.LINE_AA)
        frame[:] = additive(frame, dust_layer)

        # Short horizontal light streaks — subtle anamorphic-like sparkles
        streak_layer = np.zeros_like(frame)
        for si_k in range(8):
            rng = random.Random(si_k*9173 + int(t*10))
            sx = int(rng.uniform(100, W-100))
            sy = int(rng.uniform(100, H-100))
            slen = int(rng.uniform(40, 120)) * (0.5 + energy * 0.5)
            sbri = int(rng.uniform(60, 140) * (0.4 + energy * 0.6))
            cv2.line(streak_layer, (sx - int(slen)//2, sy), (sx + int(slen)//2, sy),
                     (sbri, int(sbri*0.95), int(sbri*0.75)), 1, cv2.LINE_AA)
        streak_layer = cv2.GaussianBlur(streak_layer, (31, 3), 0)
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
        """Bright DIAGONAL anamorphic light streak across orb (matches reference style)."""
        base_intensity = 0.25 + energy * 0.5 + beat_i * 0.8
        if base_intensity < 0.05: return
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
        layer_wide = cv2.GaussianBlur(layer, (61, 61), 18)
        layer_glow = cv2.GaussianBlur(layer, (181, 181), 55)
        frame[:] = additive(frame, layer_sharp)
        frame[:] = additive(frame, (layer_wide.astype(np.float32) * 0.75).clip(0,255).astype(np.uint8))
        frame[:] = additive(frame, (layer_glow.astype(np.float32) * 0.40).clip(0,255).astype(np.uint8))

    def _render_orbit_flare(self, frame, t, energy, beat_i):
        """Bright light source orbiting the orb with a motion trail."""
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

            # Light bleed: subtle skull illumination from orb (much softer)
            light_layer = np.zeros_like(frame)
            lb = min(140, int(70 * (0.2 + e * 0.7 + bi * 0.6)))
            cv2.circle(light_layer, (CX, CY), int(ORB_R * 2.5),
                       (int(lb*0.85), int(lb*0.75), int(lb*0.35)), -1, cv2.LINE_AA)
            light_layer = cv2.GaussianBlur(light_layer, (251, 251), 0)
            frame[:] = additive(frame, (light_layer.astype(np.float32)*0.18).clip(0,255).astype(np.uint8))

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

            frame = bloom(frame, thresh=95, strength=0.88)  # lower thresh = more blown-out whites

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

            # Chromatic aberration — ALWAYS ON (stronger baseline + radial feel)
            ca_px = max(4, int(4 + 8 * bi))  # was 2+6, now 4+8 for stronger split
            h_f, w_f = frame.shape[:2]
            frame_ca = frame.copy()
            frame_ca[:, ca_px:, 2] = frame[:, :w_f-ca_px, 2]       # red → right
            frame_ca[:, :w_f-ca_px, 0] = frame[:, ca_px:, 0]       # blue → left
            ca_blend = min(0.85, 0.35 + bi * 0.55)  # higher baseline, higher max
            frame = cv2.addWeighted(frame, 1.0 - ca_blend, frame_ca, ca_blend, 0)

            # ── ENERGY RING on beats ──
            if bi > 0.1:
                ring_layer = np.zeros_like(frame)
                ring_r = int(ORB_R + 30 + (1.0 - bi) * 280)
                ring_a = int(180 * bi)
                cv2.circle(ring_layer, (CX, CY), ring_r, (ring_a, ring_a, int(ring_a*0.9)), max(2, int(4*bi)), cv2.LINE_AA)
                ring_layer = cv2.GaussianBlur(ring_layer, (21,21), 0)
                frame = additive(frame, ring_layer)

            # ── BEAT FLASH (white punch on strong beats — stronger + lower threshold) ──
            if bi > 0.4:
                flash_a = (bi - 0.4) / 0.6 * 0.32  # max 32% white overlay
                flash = np.full_like(frame, 255)
                frame = cv2.addWeighted(frame, 1.0, flash, flash_a, 0)
            # Kick flash on strong onsets
            if onset_v > 0.45:
                kick_f = (onset_v - 0.45) / 0.55 * 0.20
                flash2 = np.full_like(frame, 255)
                frame = cv2.addWeighted(frame, 1.0, flash2, kick_f, 0)

            # ── STRONG VIGNETTE (35% edge darkening) ──
            if not hasattr(self, '_vignette'):
                vig = np.zeros((H, W), dtype=np.float32)
                cv2.circle(vig, (CX, CY), int(max(W,H)*0.55), 1.0, -1, cv2.LINE_AA)
                vig = cv2.GaussianBlur(vig, (351,351), 0)
                vig = np.clip(vig, 0.0, 1.0)
                self._vignette = (0.65 + 0.35 * vig)[:,:,np.newaxis]
            frame = np.clip(frame.astype(np.float32) * self._vignette, 0, 255).astype(np.uint8)

            # ── STRONGER FILM GRAIN (1.5% mono noise + subtle color noise) ──
            grain = np.random.randint(-12, 13, (H, W), dtype=np.int16)
            grain3 = np.stack([grain]*3, axis=-1)
            # Add small color-channel variation
            grain3[:,:,0] += np.random.randint(-4, 5, (H, W), dtype=np.int16)
            grain3[:,:,2] += np.random.randint(-4, 5, (H, W), dtype=np.int16)
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
    p = argparse.ArgumentParser(description="ESPECTROS Audio Visualizer v3")
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
        ORB_R = int(195 * a.scale)
        BG_MARGIN = int(120 * a.scale)
        print(f"[*] FAST PREVIEW: {W}x{H}")
    if a.fps != 30:
        FPS = a.fps
        print(f"[*] FPS override: {FPS}")
    generate_video(a.audio, a.output, a.logo, a.duration, a.bg)

if __name__ == "__main__":
    main()
