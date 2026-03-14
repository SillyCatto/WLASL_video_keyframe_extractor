"""
ASL Keyframe Extraction Pipeline
=================================
Extracts 8–15 semantically meaningful keyframes from a word-level ASL video
using a three-signal hybrid approach:

    Signal 1 — Farneback Optical Flow (hand-region masked, CPU-friendly)
    Signal 2 — MediaPipe Wrist Velocity
    Signal 3 — MediaPipe Handshape Change (finger joint delta)
    Signal 4 — Flow Direction Reversal (catches circular/reversing signs)

Keyframe selection detects:
    - Motion transitions  (peaks of fused motion signal)
    - Held poses          (valleys of motion signal)
    - Sandwiched holds    (short holds flanked by two motion bursts —
                           fixes missed keyframes like "GO" index-up and
                           "WHY" forehead-touch)

Outputs:
    - Console-printed table of keyframe indices + timestamps + scores
    - Stitched side-by-side PNG for visual verification

Requirements:
    python 3.11 (recommended)
    pip install mediapipe opencv-python scipy numpy Pillow
"""

import os
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic as MpHolistic
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Keyframe count bounds
    min_keyframes: int = 8
    max_keyframes: int = 15

    # Signal fusion weights (must sum to 1.0)
    weight_flow_magnitude: float = 0.30
    weight_flow_direction: float = 0.15
    weight_wrist_velocity: float = 0.25
    weight_handshape:      float = 0.30

    # Smoothing sigma applied to the transition signal.
    # Hold signal uses smooth_sigma * 0.5 to preserve short holds.
    smooth_sigma: float = 2.0

    # Peak detection minimum prominence
    peak_prominence: float = 0.04

    # Minimum gap between keyframes (fraction of total frames)
    min_gap_fraction: float = 0.04

    # Sandwiched hold detector sensitivity.
    # How much the valley must drop below the average of its flanking peaks.
    # Lower = more sensitive (catches subtle holds). Higher = stricter.
    sandwiched_hold_threshold: float = 0.12

    # MediaPipe model complexity (0=fast, 1=balanced, 2=accurate)
    mediapipe_complexity: int = 1

    # Hand bounding-box padding (fraction of frame size)
    hand_bbox_padding: float = 0.08

    # Output image settings
    thumb_height: int = 256
    thumb_border: int = 4


# ─────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────

def load_video(video_path: str) -> tuple[list[np.ndarray], float]:
    """
    Load all frames from a video file.

    Returns
    -------
    frames : list of H×W×3 uint8 BGR numpy arrays
    fps    : float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) < 4:
        raise ValueError(f"Video too short: only {len(frames)} frames found.")

    print(f"[load]  {len(frames)} frames  |  {fps:.1f} fps  |  "
          f"{frames[0].shape[1]}×{frames[0].shape[0]}")
    return frames, fps


# ─────────────────────────────────────────────
# Signal 1 & 4 — Farneback Optical Flow
# ─────────────────────────────────────────────

class FlowSignalExtractor:
    """
    CPU-friendly dense optical flow via OpenCV Farneback.

    Produces two per-frame signals:
        flow_magnitude : mean flow magnitude inside hand bbox (motion energy)
        flow_direction : angular velocity between frames (catches reversals)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _hand_mask(self, frame_shape: tuple, bbox: Optional[tuple]) -> np.ndarray:
        H, W = frame_shape[:2]
        if bbox is None:
            return np.ones((H, W), dtype=bool)

        pad = self.config.hand_bbox_padding
        x0 = max(0.0, bbox[0] - pad);  y0 = max(0.0, bbox[1] - pad)
        x1 = min(1.0, bbox[2] + pad);  y1 = min(1.0, bbox[3] + pad)

        mask = np.zeros((H, W), dtype=bool)
        mask[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
        return mask

    def extract(
        self,
        frames: list[np.ndarray],
        hand_bboxes: list[Optional[tuple]],
    ) -> tuple[np.ndarray, np.ndarray]:
        N = len(frames)
        flow_magnitude = np.zeros(N)
        flow_direction = np.zeros(N)
        prev_angle = None

        print(f"[flow]  Computing Farneback flow for {N - 1} frame pairs …")

        for i in range(1, N):
            g1 = self._to_gray(frames[i - 1])
            g2 = self._to_gray(frames[i])

            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )  # → (H, W, 2)

            u, v = flow[..., 0], flow[..., 1]
            bbox = hand_bboxes[i] or hand_bboxes[i - 1]
            mask = self._hand_mask(frames[i].shape, bbox)

            # Signal 1 — magnitude
            mag = np.sqrt(u ** 2 + v ** 2)
            flow_magnitude[i] = mag[mask].mean() if mask.any() else mag.mean()

            # Signal 4 — direction reversal
            mu = u[mask].mean() if mask.any() else u.mean()
            mv = v[mask].mean() if mask.any() else v.mean()
            angle = float(np.arctan2(mv, mu))

            if prev_angle is not None:
                delta = abs(angle - prev_angle)
                if delta > np.pi:
                    delta = 2 * np.pi - delta
                flow_direction[i] = delta
            prev_angle = angle

        print("[flow]  Done.")
        return flow_magnitude, flow_direction


# ─────────────────────────────────────────────
# Signal 2 & 3 — MediaPipe Landmark Signals
# ─────────────────────────────────────────────

class LandmarkSignalExtractor:
    """
    Runs MediaPipe Holistic on every frame and extracts:
        wrist_velocity  : L2 displacement of wrist landmarks between frames
        handshape_score : L2 displacement of finger-joint landmarks
        hand_bboxes     : normalized (x0, y0, x1, y1) per frame (or None)
        landmarks_seq   : (N, 126) raw landmark array
    """

    FINGER_ALL = list(range(1, 21))  # all non-wrist hand landmarks (1–20)

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.holistic = MpHolistic(
            static_image_mode=False,
            model_complexity=config.mediapipe_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _hand_to_vec(self, hand_lms) -> np.ndarray:
        if hand_lms is None:
            return np.zeros(63)
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
        ).flatten()

    def _hand_bbox(self, hand_lms) -> Optional[tuple]:
        if hand_lms is None:
            return None
        xs = [lm.x for lm in hand_lms.landmark]
        ys = [lm.y for lm in hand_lms.landmark]
        return (min(xs), min(ys), max(xs), max(ys))

    def _merge_bboxes(self, b1, b2) -> Optional[tuple]:
        boxes = [b for b in (b1, b2) if b is not None]
        if not boxes:
            return None
        xs = [b[0] for b in boxes] + [b[2] for b in boxes]
        ys = [b[1] for b in boxes] + [b[3] for b in boxes]
        return (min(xs), min(ys), max(xs), max(ys))

    def extract(
        self, frames: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        N = len(frames)
        landmarks_seq = np.zeros((N, 126))
        hand_bboxes: list[Optional[tuple]] = []

        print(f"[mediapipe]  Processing {N} frames …")

        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.holistic.process(rgb)

            lh = self._hand_to_vec(res.left_hand_landmarks)
            rh = self._hand_to_vec(res.right_hand_landmarks)
            landmarks_seq[i] = np.concatenate([lh, rh])

            bbox = self._merge_bboxes(
                self._hand_bbox(res.left_hand_landmarks),
                self._hand_bbox(res.right_hand_landmarks),
            )
            hand_bboxes.append(bbox)

            if i % max(1, N // 10) == 0:
                print(f"        {int(i / N * 100):3d}%  frame {i}/{N - 1}", end="\r")

        print()

        # Signal 2 — wrist velocity
        wrists = landmarks_seq[:, [0, 1, 2, 63, 64, 65]]
        wrist_velocity = np.concatenate(
            [[0.0], np.linalg.norm(np.diff(wrists, axis=0), axis=1)]
        )

        # Signal 3 — handshape change (finger joints, no wrist)
        def finger_indices(offset):
            return [offset + j * 3 + k for j in self.FINGER_ALL for k in range(3)]

        fingers = landmarks_seq[:, finger_indices(0) + finger_indices(63)]
        handshape_score = np.concatenate(
            [[0.0], np.linalg.norm(np.diff(fingers, axis=0), axis=1)]
        )

        return wrist_velocity, handshape_score, hand_bboxes, landmarks_seq

    def close(self):
        self.holistic.close()


# ─────────────────────────────────────────────
# Signal Utilities
# ─────────────────────────────────────────────

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mn, mx = signal.min(), signal.max()
    return (signal - mn) / (mx - mn + 1e-8)


def smooth_signal(signal: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(signal.astype(float), sigma=sigma)


# ─────────────────────────────────────────────
# Signal Fusion
# ─────────────────────────────────────────────

def fuse_signals(
    flow_mag:  np.ndarray,
    flow_dir:  np.ndarray,
    wrist_vel: np.ndarray,
    handshape: np.ndarray,
    config:    PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    fused_transition : high = movement / handshape change
    fused_hold       : high = stillness / held pose

    The hold signal uses a tighter smoothing sigma (0.5×) to preserve
    short holds that get blurred out at full sigma — this is the primary
    reason sandwiched holds like GO's index-up or WHY's forehead-touch
    were previously missed.
    """
    fm = smooth_signal(normalize_signal(flow_mag),  config.smooth_sigma)
    fd = smooth_signal(normalize_signal(flow_dir),  config.smooth_sigma)
    wv = smooth_signal(normalize_signal(wrist_vel), config.smooth_sigma)
    hs = smooth_signal(normalize_signal(handshape), config.smooth_sigma)

    w = config
    fused_transition = (
        w.weight_flow_magnitude * fm +
        w.weight_flow_direction * fd +
        w.weight_wrist_velocity * wv +
        w.weight_handshape      * hs
    )

    # Tighter sigma for hold signal so short holds aren't smoothed away
    hold_sigma = max(1.0, config.smooth_sigma * 0.5)
    wv_tight   = smooth_signal(normalize_signal(wrist_vel), hold_sigma)
    hs_tight   = smooth_signal(normalize_signal(handshape), hold_sigma)
    fused_hold = 1.0 - smooth_signal(
        normalize_signal(0.5 * wv_tight + 0.5 * hs_tight), hold_sigma
    )

    return normalize_signal(fused_transition), normalize_signal(fused_hold)


# ─────────────────────────────────────────────
# Sandwiched Hold Detector
# ─────────────────────────────────────────────

def detect_sandwiched_holds(
    motion_signal:  np.ndarray,
    min_gap:        int,
    drop_threshold: float = 0.12,
) -> list[int]:
    """
    Finds holds that are sandwiched between two motion peaks.

    Standard valley detection misses these because their prominence score
    gets suppressed by flanking high-motion peaks — the valley looks
    statistically shallow even though it is a full kinematic stop.

    Strategy: find all motion peaks, then for every consecutive peak pair,
    locate the deepest point between them. If that point drops enough below
    the average of its two flanking peaks, it qualifies as a real hold.

    Parameters
    ----------
    motion_signal   : 1-D normalized array (higher = more motion)
    min_gap         : minimum frames between peaks
    drop_threshold  : required dip from flanking peak average to valley

    Returns
    -------
    List of frame indices identified as sandwiched holds.

    Signs this fixes
    ----------------
    "GO"  — index fingers pointing straight up, sandwiched between the
             arm-raise peak and the forward-swing peak.
    "WHY" — hand touching forehead, sandwiched between the raise peak
             and the telephone-shape transition peak.
    """
    peaks, _ = find_peaks(motion_signal, distance=min_gap, prominence=0.05)

    sandwiched = []
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]

        if p2 - p1 < 3:
            continue

        segment           = motion_signal[p1: p2 + 1]
        local_min_offset  = int(np.argmin(segment))
        local_min_idx     = p1 + local_min_offset

        min_val  = motion_signal[local_min_idx]
        peak_avg = (motion_signal[p1] + motion_signal[p2]) / 2.0

        if peak_avg - min_val >= drop_threshold:
            sandwiched.append(local_min_idx)

    return sandwiched


# ─────────────────────────────────────────────
# Keyframe Selection
# ─────────────────────────────────────────────

def select_keyframes(
    fused_transition: np.ndarray,
    fused_hold:       np.ndarray,
    config:           PipelineConfig,
    wrist_velocity:   Optional[np.ndarray] = None,
    handshape_score:  Optional[np.ndarray] = None,
) -> list[int]:
    """
    Select keyframe indices via four complementary strategies:

        1. Peaks of fused_transition  → motion onsets / handshape transitions
        2. Peaks of fused_hold        → prominent held poses
        3. Sandwiched hold detection  → short holds between motion bursts
                                        (the fix for the systematic miss)
        4. Trim / pad to [min_keyframes, max_keyframes]

    Parameters
    ----------
    wrist_velocity, handshape_score : raw (unsmoothed) signals required by
        the sandwiched hold detector. If None, step 3 is skipped.
    """
    N       = len(fused_transition)
    min_gap = max(2, int(N * config.min_gap_fraction))
    target  = (config.min_keyframes + config.max_keyframes) // 2

    selected = {0, N - 1}

    # 1. Transition peaks
    trans_peaks, trans_props = find_peaks(
        fused_transition, distance=min_gap, prominence=config.peak_prominence
    )
    if len(trans_peaks):
        order = np.argsort(trans_props["prominences"])[::-1]
        selected.update(trans_peaks[order][: target // 2])

    # 2. Hold peaks
    hold_peaks, hold_props = find_peaks(
        fused_hold, distance=min_gap, prominence=config.peak_prominence
    )
    if len(hold_peaks):
        order = np.argsort(hold_props["prominences"])[::-1]
        selected.update(hold_peaks[order][: target // 2])

    # 3. Sandwiched holds
    # Uses raw (unsmoothed) signals so flanking peaks stay sharp and the
    # detector can accurately locate the dip between them.
    if wrist_velocity is not None and handshape_score is not None:
        raw_motion = normalize_signal(
            0.5 * normalize_signal(wrist_velocity) +
            0.5 * normalize_signal(handshape_score)
        )
        sandwiched = detect_sandwiched_holds(
            raw_motion,
            min_gap=min_gap,
            drop_threshold=config.sandwiched_hold_threshold,
        )
        if sandwiched:
            selected.update(sandwiched)
            print(f"[select]  +{len(sandwiched)} sandwiched hold(s) at "
                  f"frames: {sandwiched}")

    sorted_idx = sorted(selected)

    # 4a. Trim if above max — drop lowest-scoring middle frames first
    while len(sorted_idx) > config.max_keyframes:
        middle = sorted_idx[1:-1]
        scores = [fused_transition[i] + fused_hold[i] for i in middle]
        worst  = middle[int(np.argmin(scores))]
        sorted_idx.remove(worst)

    # 4b. Pad if below min — insert highest-scoring unselected frame
    while len(sorted_idx) < config.min_keyframes:
        combined  = fused_transition + fused_hold
        remaining = [i for i in range(N) if i not in set(sorted_idx)]
        if not remaining:
            break
        best       = remaining[int(np.argmax([combined[i] for i in remaining]))]
        sorted_idx = sorted(sorted_idx + [best])

    return sorted_idx


# ─────────────────────────────────────────────
# Visualization — Individual Keyframe Images
# ─────────────────────────────────────────────

def save_individual_keyframes(
    frames:    list[np.ndarray],
    indices:   list[int],
    output_dir: str,
) -> None:
    """
    Saves each selected keyframe as an individual image file.
    Files are named frame_0.png, frame_1.png, … in selection order.
    """
    for rank, idx in enumerate(indices):
        bgr = frames[idx]
        out_path = os.path.join(output_dir, f"frame_{rank}.png")
        cv2.imwrite(out_path, bgr)

    print(f"[output]  {len(indices)} individual keyframe images → {output_dir}")


# ─────────────────────────────────────────────
# Visualization — Stitched Keyframe Strip
# ─────────────────────────────────────────────

def draw_keyframe_strip(
    frames:           list[np.ndarray],
    indices:          list[int],
    fps:              float,
    fused_transition: np.ndarray,
    fused_hold:       np.ndarray,
    output_path:      str,
    config:           PipelineConfig,
) -> None:
    """
    Saves a horizontal strip of annotated keyframe thumbnails.
    Each thumbnail shows: badge number, frame index, timestamp, T and H scores.
    """
    H_target = config.thumb_height
    border   = config.thumb_border
    header_h = 52

    try:
        font_badge = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_info  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        font_badge = font_info = font_title = ImageFont.load_default()

    thumbs = []
    for rank, idx in enumerate(indices):
        bgr     = frames[idx]
        orig_h, orig_w = bgr.shape[:2]
        new_w   = int(orig_w * H_target / orig_h)
        resized = cv2.resize(bgr, (new_w, H_target), interpolation=cv2.INTER_AREA)

        pil    = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        canvas = Image.new("RGB", (new_w, H_target + header_h), color=(18, 18, 28))
        canvas.paste(pil, (0, 0))
        draw   = ImageDraw.Draw(canvas)

        ts      = idx / fps
        t_score = fused_transition[idx]
        h_score = fused_hold[idx]

        # Yellow badge (top-left)
        r = 14
        draw.ellipse([(4, 4), (4 + r * 2, 4 + r * 2)], fill=(255, 200, 0))
        draw.text((9, 7), str(rank + 1), fill=(20, 20, 20), font=font_badge)

        # Annotation rows
        draw.text((4, H_target + 4),
                  f"Frame {idx}  |  {ts:.2f}s",
                  fill=(200, 200, 220), font=font_info)
        draw.text((4, H_target + 22),
                  f"T={t_score:.2f}  H={h_score:.2f}",
                  fill=(140, 220, 140), font=font_info)

        thumbs.append(np.array(canvas))

    # Stitch horizontally
    total_h = thumbs[0].shape[0]
    total_w = sum(t.shape[1] for t in thumbs) + border * (len(thumbs) + 1)
    strip   = np.full((total_h + border * 2, total_w, 3), 10, dtype=np.uint8)

    x = border
    for thumb in thumbs:
        th, tw = thumb.shape[:2]
        strip[border: border + th, x: x + tw] = thumb
        x += tw + border

    # Title bar
    title_h   = 36
    title_bar = np.full((title_h, total_w, 3), 8, dtype=np.uint8)
    final     = Image.fromarray(np.vstack([title_bar, strip]))
    draw_f    = ImageDraw.Draw(final)
    draw_f.text(
        (border + 4, 10),
        (f"ASL Keyframe Extraction  ·  {len(indices)} keyframes  "
         f"·  {len(frames)} total frames  ·  {fps:.1f} fps"),
        fill=(255, 220, 80),
        font=font_title,
    )

    final.save(output_path, quality=95)
    print(f"[output]  Stitched keyframe strip → {output_path}")


# ─────────────────────────────────────────────
# Results Reporting
# ─────────────────────────────────────────────

def print_results(
    indices:          list[int],
    fps:              float,
    fused_transition: np.ndarray,
    fused_hold:       np.ndarray,
) -> None:
    BAR = 8

    def bar(val: float) -> str:
        n = int(val * BAR)
        return "█" * n + "░" * (BAR - n)

    # Exact inner width of each column (content only, no ║)
    # c1=4  c2=9  c3=11  c4=18  c5=18  → total inner = 64, full row = 66
    W = [4, 9, 11, 18, 18]
    INNER = sum(W) + len(W) - 1   # 64  (separators between cols, not outer)
    FULL  = INNER + 2             # 66  (+ outer ║ × 2)

    def hline(lc, mc, rc):
        segs = [mc.join("═" * w for w in W)]
        return lc + segs[0] + rc

    # Correctly-spaced separator rows
    top    = "╔" + "═" * W[0] + "╦" + "═" * W[1] + "╦" + "═" * W[2] + "╦" + "═" * W[3] + "╦" + "═" * W[4] + "╗"
    mid    = "╠" + "═" * W[0] + "╬" + "═" * W[1] + "╬" + "═" * W[2] + "╬" + "═" * W[3] + "╬" + "═" * W[4] + "╣"
    div    = "╠" + "═" * W[0] + "╦" + "═" * W[1] + "╦" + "═" * W[2] + "╦" + "═" * W[3] + "╦" + "═" * W[4] + "╣"
    bot    = "╚" + "═" * W[0] + "╩" + "═" * W[1] + "╩" + "═" * W[2] + "╩" + "═" * W[3] + "╩" + "═" * W[4] + "╝"
    header = "╔" + "═" * (FULL - 2) + "╗"
    header_mid = "╠" + "═" * (FULL - 2) + "╣"

    def full_row(text: str) -> str:
        """Single-column full-width row."""
        return "║" + text.center(FULL - 2) + "║"

    def data_row(rank: int, idx: int, ts: float, t: float, h: float) -> str:
        c1 = f" {rank:2d} "                     # 4  chars
        c2 = f"  {idx:5d}  "                    # 9  chars
        c3 = f"  {ts:7.3f}  "                   # 11 chars
        c4 = f"  {t:.3f} {bar(t)}  "            # 18 chars  (2+5+1+8+2)
        c5 = f"  {h:.3f} {bar(h)}  "            # 18 chars
        return f"║{c1}║{c2}║{c3}║{c4}║{c5}║"

    print()
    print(header)
    print(full_row("ASL KEYFRAME EXTRACTION RESULTS"))
    print(header_mid)
    print(full_row(f"  Total keyframes selected : {len(indices)}"))
    print(div)
    print(f"║{'  #':>{W[0]}}║{'  Frame':<{W[1]}}║{'  Time(s)':<{W[2]}}║{'  Trans Score':<{W[3]}}║{'  Hold Score':<{W[4]}}║")
    print(mid)

    for rank, idx in enumerate(indices):
        ts = idx / fps
        t  = fused_transition[idx]
        h  = fused_hold[idx]
        print(data_row(rank + 1, idx, ts, t, h))

    print(bot)
    print()
    print("  Keyframe index list  :", [int(i) for i in indices])
    print("  Timestamps (seconds) :", [round(float(i) / fps, 3) for i in indices])
    print()


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    output_dir: str = ".",
    config: Optional[PipelineConfig] = None,
    save_individual: bool = False,
    save_stitched: bool = False,
) -> dict:
    """
    Full ASL keyframe extraction pipeline.

    Returns
    -------
    dict:
        keyframe_indices  — list[int]
        keyframe_times    — list[float]  (seconds)
        keyframe_images   — list[np.ndarray] BGR
        output_image_path — str
    """
    if config is None:
        config = PipelineConfig()

    os.makedirs(output_dir, exist_ok=True)
    stem              = Path(video_path).stem
    output_image_path = os.path.join(output_dir, f"{stem}_keyframes.png")

    # Stage 1 — Load
    print("\n━━━  STAGE 1/4 — Loading Video  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    frames, fps = load_video(video_path)
    N = len(frames)

    # Stage 2 — MediaPipe
    print("\n━━━  STAGE 2/4 — MediaPipe Landmark Signals  ━━━━━━━━━━━━━━━━━")
    lm_extractor = LandmarkSignalExtractor(config)
    wrist_velocity, handshape_score, hand_bboxes, landmarks_seq = \
        lm_extractor.extract(frames)
    lm_extractor.close()

    detected = sum(1 for b in hand_bboxes if b is not None)
    print(f"[mediapipe]  Hands detected in {detected}/{N} frames "
          f"({100 * detected / N:.0f}%)")

    # Stage 3 — Optical flow
    print("\n━━━  STAGE 3/4 — Optical Flow Signals  ━━━━━━━━━━━━━━━━━━━━━━━")
    flow_extractor = FlowSignalExtractor(config)
    flow_magnitude, flow_direction = flow_extractor.extract(frames, hand_bboxes)

    # Stage 4 — Fuse + select
    print("\n━━━  STAGE 4/4 — Signal Fusion + Keyframe Selection  ━━━━━━━━━")
    fused_transition, fused_hold = fuse_signals(
        flow_magnitude, flow_direction,
        wrist_velocity, handshape_score,
        config,
    )

    keyframe_indices = select_keyframes(
        fused_transition, fused_hold, config,
        wrist_velocity=wrist_velocity,
        handshape_score=handshape_score,
    )

    print(f"[select]  {len(keyframe_indices)} keyframes selected "
          f"(min={config.min_keyframes}, max={config.max_keyframes})")

    print_results(keyframe_indices, fps, fused_transition, fused_hold)

    keyframe_images = [frames[i] for i in keyframe_indices]

    if save_individual:
        save_individual_keyframes(frames, keyframe_indices, output_dir)

    if save_stitched:
        draw_keyframe_strip(
            frames, keyframe_indices, fps,
            fused_transition, fused_hold,
            output_image_path, config,
        )

    return {
        "keyframe_indices":  keyframe_indices,
        "keyframe_times":    [round(i / fps, 3) for i in keyframe_indices],
        "keyframe_images":   keyframe_images,
        "output_image_path": output_image_path if save_stitched else None,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ASL Keyframe Extractor — Farneback + MediaPipe hybrid pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video",                      help="Path to input ASL video")
    p.add_argument("--output-dir",  default=".", help="Output directory for PNG")
    p.add_argument("-i", "--individual", action="store_true",
                   help="Save each keyframe as an individual image (frame_0.png, frame_1.png, …)")
    p.add_argument("-s", "--stitched",   action="store_true",
                   help="Save stitched keyframe strip image")
    p.add_argument("--min-frames",  type=int,   default=8,    help="Minimum keyframes")
    p.add_argument("--max-frames",  type=int,   default=15,   help="Maximum keyframes")
    p.add_argument("--thumb-height",type=int,   default=256,  help="Thumbnail height (px)")
    p.add_argument("--mp-complexity", type=int, default=1,
                   choices=[0, 1, 2], help="MediaPipe model complexity")
    p.add_argument("--hold-threshold", type=float, default=0.12,
                   help="Sandwiched hold sensitivity — lower catches subtler holds")
    p.add_argument("--smooth-sigma",   type=float, default=2.0,
                   help="Gaussian smoothing sigma for transition signal")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = PipelineConfig(
        min_keyframes=args.min_frames,
        max_keyframes=args.max_frames,
        thumb_height=args.thumb_height,
        mediapipe_complexity=args.mp_complexity,
        sandwiched_hold_threshold=args.hold_threshold,
        smooth_sigma=args.smooth_sigma,
    )

    result = run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        config=cfg,
        save_individual=args.individual,
        save_stitched=args.stitched,
    )

    print("Done.")
    print(f"  Keyframe indices : {[int(i) for i in result['keyframe_indices']]}")
    print(f"  Timestamps (s)   : {[float(t) for t in result['keyframe_times']]}")
    if result['output_image_path']:
        print(f"  Output image     : {result['output_image_path']}")

