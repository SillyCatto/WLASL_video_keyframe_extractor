"""
ASL Landmark Extraction Pipeline
=================================
Extracts and processes MediaPipe landmarks from pre-extracted keyframe
images for word-level ASL sign language recognition.

Pipeline (based on Relative Quantization approach from BdSL research):

    Step 1 — Landmark Selection       (56 landmarks → 168 features per frame)
    Step 2 — Hand Dominance Correction (flip left-dominant → right-dominant)
    Step 3 — Shoulder Midpoint Calib.  (body-relative coordinates)
    Step 4 — Relative Quantization     (local centers + discretization)
    Step 5 — Feature Scaling ×100      (gradient-friendly magnitudes)
    Step 6 — Sequence Padding          (pad/truncate to fixed length)

Input:
    Directory of keyframe images (frame_0.png, frame_1.png, …)
    produced by asl_keyframe_extractor.py with the -i flag.

Output:
    NumPy .npy file of shape (target_len, 168)

Requirements:
    python 3.11 (recommended)
    pip install mediapipe opencv-python numpy
"""

import os
import re
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic as MpHolistic

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class LandmarkConfig:
    """Pipeline configuration."""

    # Sequence
    target_sequence_length: int = 15
    feature_dim: int = 168          # 56 landmarks × 3 (x, y, z)

    # MediaPipe
    mediapipe_complexity: int = 1   # 0 = fast, 1 = balanced, 2 = accurate

    # RQ quantization levels (x, y, z)
    hand_quant_levels: tuple = (10, 10, 5)
    face_quant_levels: tuple = (5, 5, 3)
    pose_quant_levels: tuple = (10, 10, 5)

    # Post-RQ feature scaling
    scale_factor: float = 100.0


# ─────────────────────────────────────────────
# Landmark Index Maps
# ─────────────────────────────────────────────

# MediaPipe Holistic pose landmark indices (8 selected)
POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
# Order: L shoulder, R shoulder, L elbow, R elbow,
#        L wrist, R wrist, L hip, R hip

# MediaPipe face mesh indices for 6 anchor points
FACE_INDICES = [1, 10, 234, 454, 152, 13]
# Order: nose tip, forehead, L cheek, R cheek, chin, upper lip

# Feature vector layout (56 landmarks × 3 = 168):
#   [0:63]    left hand  (21 × 3)
#   [63:126]  right hand (21 × 3)
#   [126:150] pose       (8 × 3)    — shoulders, elbows, wrists, hips
#   [150:168] face       (6 × 3)    — nose, forehead, cheeks, chin, lip

LH_START   = 0       # left hand start
LH_SIZE    = 63      # 21 landmarks × 3
RH_START   = 63      # right hand start
RH_SIZE    = 63
POSE_START = 126
POSE_SIZE  = 24      # 8 landmarks × 3
FACE_START = 150
FACE_SIZE  = 18      # 6 landmarks × 3

# Pose sub-block offsets (relative to POSE_START):
#   [0:6]   shoulders  (2 × 3) — calibration anchor, NOT quantized
#   [6:12]  elbows     (2 × 3)
#   [12:18] wrists     (2 × 3)
#   [18:24] hips       (2 × 3)
POSE_SHOULDERS_SIZE = 6     # first 6 features = 2 shoulders
POSE_LIMB_OFFSET    = 6     # elbows start at POSE_START + 6
POSE_LIMB_SIZE      = 18    # elbows + wrists + hips = 6 × 3


# ─────────────────────────────────────────────
# Frame Loading
# ─────────────────────────────────────────────

def _extract_frame_number(filename: str) -> int:
    """Pull the trailing integer from filenames like frame_0.png or frame_12.png."""
    match = re.search(r'(\d+)', Path(filename).stem.split("frame")[-1])
    if match:
        return int(match.group(1))
    # Fallback: last integer in filename
    nums = re.findall(r'\d+', Path(filename).stem)
    return int(nums[-1]) if nums else 0


def load_keyframe_images(frames_dir: str) -> list[np.ndarray]:
    """
    Load keyframe images from a directory.
    Expects files named frame_0.png, frame_1.png, … (produced by -i flag).
    Sorted by frame number in ascending order.
    """
    frames_path = Path(frames_dir)

    # Collect image files matching frame_*.png or frame_*.jpg
    image_files = sorted(
        list(frames_path.glob("frame_*.png")) +
        list(frames_path.glob("frame_*.jpg")),
        key=lambda p: _extract_frame_number(p.name),
    )

    if not image_files:
        raise FileNotFoundError(
            f"No frame_*.png/jpg images found in: {frames_dir}"
        )

    frames = []
    for img_path in image_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise ValueError(f"Cannot read image: {img_path}")
        frames.append(bgr)

    print(f"[load]  {len(frames)} keyframe images from {frames_dir}")
    return frames


# ─────────────────────────────────────────────
# Step 1 — MediaPipe Extraction + Selection
# ─────────────────────────────────────────────

def extract_landmarks(
    frames: list[np.ndarray],
    config: LandmarkConfig,
) -> np.ndarray:
    """
    Run MediaPipe Holistic on each frame and extract the selected
    56 landmarks into a (K, 168) array.

    static_image_mode=True because keyframes are independent images,
    not a continuous video stream.
    """
    K = len(frames)
    landmarks = np.zeros((K, config.feature_dim))

    holistic = MpHolistic(
        static_image_mode=True,
        model_complexity=config.mediapipe_complexity,
        min_detection_confidence=0.5,
    )

    print(f"[mediapipe]  Extracting landmarks from {K} keyframes …")

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        vec = np.zeros(config.feature_dim)

        # Left hand (21 landmarks → [0:63])
        if result.left_hand_landmarks:
            for j, lm in enumerate(result.left_hand_landmarks.landmark):
                vec[LH_START + j * 3: LH_START + j * 3 + 3] = [lm.x, lm.y, lm.z]

        # Right hand (21 landmarks → [63:126])
        if result.right_hand_landmarks:
            for j, lm in enumerate(result.right_hand_landmarks.landmark):
                vec[RH_START + j * 3: RH_START + j * 3 + 3] = [lm.x, lm.y, lm.z]

        # Pose (8 selected → [126:150])
        if result.pose_landmarks:
            for j, pose_idx in enumerate(POSE_INDICES):
                lm = result.pose_landmarks.landmark[pose_idx]
                vec[POSE_START + j * 3: POSE_START + j * 3 + 3] = [lm.x, lm.y, lm.z]

        # Face anchors (6 selected → [150:168])
        if result.face_landmarks:
            for j, face_idx in enumerate(FACE_INDICES):
                lm = result.face_landmarks.landmark[face_idx]
                vec[FACE_START + j * 3: FACE_START + j * 3 + 3] = [lm.x, lm.y, lm.z]

        landmarks[i] = vec

        # Progress
        hand_status = []
        if result.left_hand_landmarks:
            hand_status.append("L")
        if result.right_hand_landmarks:
            hand_status.append("R")
        hands = "+".join(hand_status) if hand_status else "none"

        if i % max(1, K // 5) == 0 or i == K - 1:
            print(f"        frame {i}/{K - 1}  hands: {hands}")

    holistic.close()

    # Detection summary
    detected_lh = sum(1 for k in range(K) if np.any(landmarks[k, LH_START:LH_START + LH_SIZE]))
    detected_rh = sum(1 for k in range(K) if np.any(landmarks[k, RH_START:RH_START + RH_SIZE]))
    detected_face = sum(1 for k in range(K) if np.any(landmarks[k, FACE_START:FACE_START + FACE_SIZE]))
    print(f"[mediapipe]  Left hand: {detected_lh}/{K}  |  "
          f"Right hand: {detected_rh}/{K}  |  Face: {detected_face}/{K}")

    return landmarks


# ─────────────────────────────────────────────
# Step 2 — Hand Dominance Correction
# ─────────────────────────────────────────────

def detect_dominant_hand(landmarks: np.ndarray) -> str:
    """
    Detect dominant hand by comparing non-zero landmark activity
    across all keyframes. Returns 'right' or 'left'.
    """
    left_activity  = np.count_nonzero(landmarks[:, LH_START:LH_START + LH_SIZE])
    right_activity = np.count_nonzero(landmarks[:, RH_START:RH_START + RH_SIZE])
    return "left" if left_activity > right_activity else "right"


def correct_hand_dominance(landmarks: np.ndarray) -> np.ndarray:
    """
    If the signer is left-dominant, mirror horizontally so that the
    dominant hand always occupies the right-hand slot [63:126].
    This halves the positional variance the model needs to learn.

    Operations per frame:
      1. Swap left / right hand data
      2. Flip x-coordinates (x_new = 1 − x) for hands, pose, face
      3. Swap left ↔ right pose pairs (shoulders, elbows, wrists, hips)
      4. Swap left ↔ right face cheeks

    Zero-guard: only flips non-zero landmark blocks to avoid injecting
    false 1.0 values where no landmark was detected.
    """
    dominant = detect_dominant_hand(landmarks)
    if dominant == "right":
        print("[dominance]  Right-dominant — no correction needed")
        return landmarks

    print("[dominance]  Left-dominant — flipping to right-dominant")
    out = landmarks.copy()

    for f in range(len(out)):
        # ── Hands: swap slots + flip x ────────────────────────────
        lh = out[f, LH_START:LH_START + LH_SIZE].copy()
        rh = out[f, RH_START:RH_START + RH_SIZE].copy()

        # Flip x only if that hand was actually detected
        if np.any(lh):
            lh[0::3] = 1.0 - lh[0::3]
        if np.any(rh):
            rh[0::3] = 1.0 - rh[0::3]

        # Swap: old left → new right slot, old right → new left slot
        out[f, LH_START:LH_START + LH_SIZE] = rh
        out[f, RH_START:RH_START + RH_SIZE] = lh

        # ── Pose: swap L↔R pairs + flip x ────────────────────────
        pose = out[f, POSE_START:POSE_START + POSE_SIZE].copy()
        if np.any(pose):
            # Swap each left/right pair (stored as consecutive 3-vectors)
            # Shoulders [0:3] ↔ [3:6]
            pose[0:3], pose[3:6] = pose[3:6].copy(), pose[0:3].copy()
            # Elbows [6:9] ↔ [9:12]
            pose[6:9], pose[9:12] = pose[9:12].copy(), pose[6:9].copy()
            # Wrists [12:15] ↔ [15:18]
            pose[12:15], pose[15:18] = pose[15:18].copy(), pose[12:15].copy()
            # Hips [18:21] ↔ [21:24]
            pose[18:21], pose[21:24] = pose[21:24].copy(), pose[18:21].copy()
            # Flip all pose x-coords
            pose[0::3] = 1.0 - pose[0::3]
            out[f, POSE_START:POSE_START + POSE_SIZE] = pose

        # ── Face: swap L↔R cheeks + flip x ───────────────────────
        face = out[f, FACE_START:FACE_START + FACE_SIZE].copy()
        if np.any(face):
            # L cheek (anchor 2 → offset 6:9) ↔ R cheek (anchor 3 → offset 9:12)
            face[6:9], face[9:12] = face[9:12].copy(), face[6:9].copy()
            # Flip all face x-coords
            face[0::3] = 1.0 - face[0::3]
            out[f, FACE_START:FACE_START + FACE_SIZE] = face

    return out


# ─────────────────────────────────────────────
# Step 3 — Shoulder Midpoint Calibration
# ─────────────────────────────────────────────

def calibrate_to_shoulders(landmarks: np.ndarray) -> np.ndarray:
    """
    Translate all coordinates so that the first frame's shoulder
    midpoint becomes the origin (0, 0, 0). This makes all coordinates
    body-relative rather than camera-relative, correcting for camera
    distance, off-center positioning, and depth variation.

    Uses the FIRST frame's midpoint as the fixed reference for the
    entire sequence, as per the research paper.
    """
    out = landmarks.copy()

    # First frame's shoulder midpoint
    left_shoulder  = out[0, POSE_START:POSE_START + 3]
    right_shoulder = out[0, POSE_START + 3:POSE_START + 6]
    midpoint = (left_shoulder + right_shoulder) / 2.0

    if np.allclose(midpoint, 0):
        print("[calibrate]  Warning: no shoulder landmarks in first frame — skipping")
        return out

    print(f"[calibrate]  Shoulder midpoint: "
          f"({midpoint[0]:.3f}, {midpoint[1]:.3f}, {midpoint[2]:.3f})")

    # Subtract midpoint from every (x, y, z) triplet in every frame
    n_landmarks = (FACE_START + FACE_SIZE) // 3   # 56
    tile = np.tile(midpoint, n_landmarks)
    for f in range(len(out)):
        out[f] -= tile

    return out


# ─────────────────────────────────────────────
# Step 4 & 5 — Relative Quantization (RQ)
# ─────────────────────────────────────────────

def _quantize_block(
    frame: np.ndarray,
    start: int,
    size: int,
    levels: tuple,
    center: np.ndarray,
) -> None:
    """
    In-place: translate a landmark block to a local center,
    then quantize x/y/z to discrete levels.

    Parameters
    ----------
    frame   : full 168-feature vector (modified in-place)
    start   : start index of the block within the frame
    size    : number of features in the block (must be divisible by 3)
    levels  : (x_levels, y_levels, z_levels)
    center  : (x, y, z) local center to subtract
    """
    n_landmarks = size // 3
    block = frame[start:start + size]

    # Translate to local center
    block -= np.tile(center, n_landmarks)

    # Quantize each axis independently
    for k, n_levels in enumerate(levels):
        coords = block[k::3]
        if len(coords) == 0:
            continue
        mn, mx = coords.min(), coords.max()
        span = mx - mn
        if span < 1e-8:
            coords[:] = 0     # degenerate — all same value
        else:
            normed = (coords - mn) / span
            coords[:] = np.floor(normed * n_levels).clip(0, n_levels - 1)


def relative_quantize(
    landmarks: np.ndarray,
    config: LandmarkConfig,
) -> np.ndarray:
    """
    Relative Quantization (RQ) — the paper's most impactful technique.

    Each landmark group is first translated to its local physiological
    center, then quantized to discrete levels. This achieves both
    pose-invariance and scale-invariance without learned parameters.

    Local centers:
        Hands       → own wrist (landmark 0 of each hand)
        Pose limbs  → shoulder midpoint (elbows, wrists, hips)
        Face        → nose tip
        Shoulders   → kept as calibrated global coords (NOT quantized)
    """
    out = landmarks.copy()
    K = len(out)

    print(f"[rq]  Applying Relative Quantization to {K} frames …")

    for f in range(K):
        frame = out[f]

        # Skip empty frames (zero-padded)
        if np.allclose(frame, 0):
            continue

        # ── Left hand → relative to left wrist ──────────────────
        lh_wrist = frame[LH_START:LH_START + 3].copy()
        if np.any(lh_wrist):
            _quantize_block(
                frame, LH_START, LH_SIZE,
                config.hand_quant_levels, lh_wrist,
            )

        # ── Right hand → relative to right wrist ────────────────
        rh_wrist = frame[RH_START:RH_START + 3].copy()
        if np.any(rh_wrist):
            _quantize_block(
                frame, RH_START, RH_SIZE,
                config.hand_quant_levels, rh_wrist,
            )

        # ── Pose limbs → relative to shoulder midpoint ──────────
        # (shoulders themselves are NOT quantized — they stay as
        #  calibrated global reference coordinates)
        l_sh = frame[POSE_START:POSE_START + 3]
        r_sh = frame[POSE_START + 3:POSE_START + 6]
        shoulder_mid = (l_sh + r_sh) / 2.0

        if np.any(shoulder_mid):
            limb_start = POSE_START + POSE_LIMB_OFFSET
            _quantize_block(
                frame, limb_start, POSE_LIMB_SIZE,
                config.pose_quant_levels, shoulder_mid,
            )

        # ── Face → relative to nose tip ─────────────────────────
        nose = frame[FACE_START:FACE_START + 3].copy()
        if np.any(nose):
            _quantize_block(
                frame, FACE_START, FACE_SIZE,
                config.face_quant_levels, nose,
            )

        out[f] = frame

    return out


# ─────────────────────────────────────────────
# Step 5 — Feature Scaling
# ─────────────────────────────────────────────

def scale_features(
    landmarks: np.ndarray,
    config: LandmarkConfig,
) -> np.ndarray:
    """
    Scale features by a constant factor (default ×100).

    After RQ, values are small integers (0–10 range). Scaling prevents
    vanishing gradients during BiLSTM training. The paper found this
    step essential for convergence.
    """
    return landmarks * config.scale_factor


# ─────────────────────────────────────────────
# Step 6 — Sequence Padding
# ─────────────────────────────────────────────

def pad_sequence(
    landmarks: np.ndarray,
    config: LandmarkConfig,
) -> np.ndarray:
    """
    Pad or truncate to fixed target_sequence_length.

    Shorter sequences are zero-padded at the end.
    The BiLSTM's Masking layer will learn to ignore the zero-padded
    timesteps during training.
    """
    K = len(landmarks)
    target = config.target_sequence_length

    if K >= target:
        return landmarks[:target]

    pad = np.zeros((target - K, config.feature_dim))
    return np.vstack([landmarks, pad])


# ─────────────────────────────────────────────
# Results Reporting
# ─────────────────────────────────────────────

def print_summary(
    landmarks: np.ndarray,
    output_path: str,
    original_len: int,
    config: LandmarkConfig,
) -> None:
    """Print a compact summary of the processed landmark output."""
    non_zero = np.count_nonzero(landmarks.sum(axis=1))
    val_min  = landmarks[landmarks != 0].min() if np.any(landmarks) else 0
    val_max  = landmarks.max()
    has_nan  = np.isnan(landmarks).any()
    has_inf  = np.isinf(landmarks).any()

    print()
    print("┌──────────────────────────────────────────────────┐")
    print("│            LANDMARK EXTRACTION SUMMARY           │")
    print("├──────────────────────────────────────────────────┤")
    print(f"│  Input keyframes : {original_len:>5}                         │")
    print(f"│  Output shape    : ({landmarks.shape[0]}, {landmarks.shape[1]}){'':>21}│")
    print(f"│  Non-zero frames : {non_zero:>5} / {config.target_sequence_length:<5}{'':>16}│")
    print(f"│  Value range     : [{val_min:>7.1f}, {val_max:>7.1f}]{'':>14}│")
    print(f"│  NaN / Inf       : {'YES ⚠' if has_nan or has_inf else 'None ✓':>10}{'':>18}│")
    print(f"│  Output file     : {Path(output_path).name:<28} │")
    print("└──────────────────────────────────────────────────┘")
    print()


# ─────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    frames_dir: str,
    output_path: str,
    config: Optional[LandmarkConfig] = None,
) -> np.ndarray:
    """
    Full landmark extraction + processing pipeline.

    Returns processed landmarks of shape (target_len, 168).
    """
    if config is None:
        config = LandmarkConfig()

    # Stage 1 — Load keyframes
    print("\n━━━  STAGE 1/6 — Loading Keyframes  ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    frames = load_keyframe_images(frames_dir)
    original_len = len(frames)

    # Stage 2 — MediaPipe extraction
    print("\n━━━  STAGE 2/6 — MediaPipe Landmark Extraction  ━━━━━━━━━━━━━━")
    landmarks = extract_landmarks(frames, config)

    # Stage 3 — Hand dominance correction
    print("\n━━━  STAGE 3/6 — Hand Dominance Correction  ━━━━━━━━━━━━━━━━━━")
    landmarks = correct_hand_dominance(landmarks)

    # Stage 4 — Shoulder midpoint calibration
    print("\n━━━  STAGE 4/6 — Shoulder Midpoint Calibration  ━━━━━━━━━━━━━━")
    landmarks = calibrate_to_shoulders(landmarks)

    # Stage 5 — Relative Quantization
    print("\n━━━  STAGE 5/6 — Relative Quantization  ━━━━━━━━━━━━━━━━━━━━━━")
    landmarks = relative_quantize(landmarks, config)

    # Scale
    landmarks = scale_features(landmarks, config)
    print(f"[scale]  Features scaled ×{config.scale_factor}")

    # Stage 6 — Sequence padding
    print("\n━━━  STAGE 6/6 — Sequence Padding  ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    landmarks = pad_sequence(landmarks, config)
    print(f"[pad]  {original_len} frames → {config.target_sequence_length} "
          f"(feature dim: {config.feature_dim})")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, landmarks)
    print(f"\n[output]  Saved → {output_path}")

    print_summary(landmarks, output_path, original_len, config)

    return landmarks


# ─────────────────────────────────────────────
# Batch Pipeline
# ─────────────────────────────────────────────

def run_batch_pipeline(
    root_dir: str,
    output_dir: str,
    config: Optional[LandmarkConfig] = None,
) -> None:
    """
    Process all subdirectories in root_dir, each containing keyframe
    images for one video. Saves one .npy per subdirectory.
    """
    root = Path(root_dir)
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in: {root_dir}")

    print(f"[batch]  Found {len(subdirs)} subdirectories to process\n")
    os.makedirs(output_dir, exist_ok=True)

    success = 0
    failed  = []

    for i, subdir in enumerate(subdirs, 1):
        output_path = os.path.join(output_dir, f"{subdir.name}.npy")

        print(f"\n{'=' * 60}")
        print(f"  [{i}/{len(subdirs)}]  {subdir.name}")
        print(f"{'=' * 60}")

        try:
            run_pipeline(str(subdir), output_path, config)
            success += 1
        except Exception as e:
            print(f"[ERROR]  {subdir.name}: {e}")
            failed.append(subdir.name)

    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE: {success}/{len(subdirs)} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ASL Landmark Extractor — MediaPipe → RQ processing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
                   help="Directory of keyframe images (or root dir with --batch)")
    p.add_argument("-o", "--output",
                   help="Output path: .npy file (single mode) or directory (batch mode)")
    p.add_argument("--batch", action="store_true",
                   help="Batch mode: process all subdirectories in input dir")
    p.add_argument("--target-len", type=int, default=15,
                   help="Target sequence length for padding/truncation")
    p.add_argument("--mp-complexity", type=int, default=1,
                   choices=[0, 1, 2],
                   help="MediaPipe model complexity (0=fast, 1=balanced, 2=accurate)")
    p.add_argument("--scale-factor", type=float, default=100.0,
                   help="Feature scaling factor applied after RQ")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = LandmarkConfig(
        target_sequence_length=args.target_len,
        mediapipe_complexity=args.mp_complexity,
        scale_factor=args.scale_factor,
    )

    if args.batch:
        output_dir = args.output or "./landmarks"
        run_batch_pipeline(args.input, output_dir, config)
    else:
        output_path = args.output or str(Path(args.input).with_suffix(".npy"))
        result = run_pipeline(args.input, output_path, config)

        print("Done.")
        print(f"  Output shape : {result.shape}")
        print(f"  Output file  : {output_path}")
