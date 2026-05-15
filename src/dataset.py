"""Dataset loading, normalization, augmentation for the sign-language model.

Input: per-video landmark .npy files of shape (T, 225) from extract_dataset.py
       (33 pose + 21 L hand + 21 R hand, each (x,y,z), in MediaPipe normalized
       image coords).

Pipeline:
  1) raw .npy  -> normalize_pose_relative (per-frame, center+scale)
  2) -> resample_to_fixed_length (uniform temporal sampling to SEQ_LEN frames)
  3) -> augment (training only: flip, jitter, time-warp)

The same normalize_pose_relative + resample_to_fixed_length is used at
inference time (predict.py) so train/test domain stays consistent.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = PROJECT_ROOT / "data" / "landmarks"
TARGET_CSV = PROJECT_ROOT / "data" / "target_words.csv"

NUM_POSE = 33
NUM_HAND = 21
POSE_DIMS = NUM_POSE * 3      # 99
HAND_DIMS = NUM_HAND * 3      # 63
VECTOR_SIZE = POSE_DIMS + 2 * HAND_DIMS  # 225

SEQ_LEN = 32                   # frames per sample after resampling
L_SHOULDER = 11                # pose landmark indices (MediaPipe BlazePose)
R_SHOULDER = 12


def load_target_words() -> tuple[list[str], dict[str, int], dict[str, str]]:
    word_ids: list[str] = []
    class_id: dict[str, int] = {}
    korean: dict[str, str] = {}
    with TARGET_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            wid = row["word_id"]
            word_ids.append(wid)
            class_id[wid] = int(row["class_id"])
            korean[wid] = row["korean_word"]
    return word_ids, class_id, korean


def normalize_pose_relative(arr: np.ndarray) -> np.ndarray:
    """Center on shoulder midpoint, scale by shoulder width.

    Makes the representation invariant to camera distance and horizontal
    framing. Frames where pose is undetected (all zero) are left as zeros.
    """
    pose = arr[:, :POSE_DIMS].reshape(-1, NUM_POSE, 3)
    lh = arr[:, POSE_DIMS:POSE_DIMS + HAND_DIMS].reshape(-1, NUM_HAND, 3)
    rh = arr[:, POSE_DIMS + HAND_DIMS:].reshape(-1, NUM_HAND, 3)

    has_pose = pose.any(axis=(1, 2))                  # (T,)
    center = (pose[:, L_SHOULDER] + pose[:, R_SHOULDER]) / 2  # (T, 3)
    shoulder_vec = pose[:, L_SHOULDER, :2] - pose[:, R_SHOULDER, :2]
    scale = np.linalg.norm(shoulder_vec, axis=1)       # (T,)
    scale = np.where(scale < 1e-3, 1.0, scale)
    scale = scale[:, None, None]
    center = center[:, None, :]

    pose_n = np.where(has_pose[:, None, None], (pose - center) / scale, pose)
    lh_n = np.where(lh.any(axis=(1, 2), keepdims=True), (lh - center) / scale, lh)
    rh_n = np.where(rh.any(axis=(1, 2), keepdims=True), (rh - center) / scale, rh)

    return np.concatenate([
        pose_n.reshape(-1, POSE_DIMS),
        lh_n.reshape(-1, HAND_DIMS),
        rh_n.reshape(-1, HAND_DIMS),
    ], axis=1).astype(np.float32)


def resample_to_fixed_length(arr: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Uniformly subsample/upsample along time axis to seq_len frames."""
    T = arr.shape[0]
    if T == seq_len:
        return arr
    if T < 2:
        return np.tile(arr, (seq_len, 1)).reshape(seq_len, -1)[:seq_len]
    idx = np.linspace(0, T - 1, seq_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (idx - lo)[:, None]
    return ((1 - w) * arr[lo] + w * arr[hi]).astype(np.float32)


def _swap_hands(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    lh = out[:, POSE_DIMS:POSE_DIMS + HAND_DIMS].copy()
    rh = out[:, POSE_DIMS + HAND_DIMS:].copy()
    out[:, POSE_DIMS:POSE_DIMS + HAND_DIMS] = rh
    out[:, POSE_DIMS + HAND_DIMS:] = lh
    return out


def _flip_x(arr: np.ndarray) -> np.ndarray:
    """Mirror across the vertical axis (negate x component of every landmark)."""
    out = arr.copy()
    # After normalize_pose_relative, x is centered at 0, so negate.
    out[:, 0::3] = -out[:, 0::3]
    return out


def augment(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentations to a (T, 225) array."""
    out = arr
    # Horizontal flip 50% (mirror + swap L/R hand to keep semantics)
    if rng.random() < 0.5:
        out = _flip_x(out)
        out = _swap_hands(out)
    # Small isotropic scale jitter
    if rng.random() < 0.7:
        s = 1.0 + rng.uniform(-0.1, 0.1)
        out = out * s
    # Small per-frame translation
    if rng.random() < 0.7:
        tx = rng.uniform(-0.05, 0.05)
        ty = rng.uniform(-0.05, 0.05)
        out = out.copy()
        out[:, 0::3] += tx
        out[:, 1::3] += ty
    # Gaussian coordinate noise
    if rng.random() < 0.7:
        out = out + rng.normal(0, 0.01, size=out.shape).astype(np.float32)
    # Temporal warping: drop or duplicate up to 3 frames
    if rng.random() < 0.5:
        T = out.shape[0]
        n_drop = rng.integers(0, 4)
        if n_drop > 0:
            keep = np.sort(rng.choice(T, T - n_drop, replace=False))
            out = out[keep]
            out = resample_to_fixed_length(out, T)
    return out.astype(np.float32)


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all .npy files; return X (N, T, 225), y (N,), filenames."""
    _, class_id, _ = load_target_words()
    Xs, ys, names = [], [], []
    for p in sorted(LANDMARKS_DIR.glob("*.npy")):
        word_id = p.stem.split("_")[0]
        if word_id not in class_id:
            continue
        arr = np.load(p)
        arr = normalize_pose_relative(arr)
        arr = resample_to_fixed_length(arr, SEQ_LEN)
        Xs.append(arr)
        ys.append(class_id[word_id])
        names.append(p.name)
    X = np.stack(Xs)
    y = np.array(ys, dtype=np.int64)
    return X, y, names


def make_augmented_set(
    X: np.ndarray, y: np.ndarray, multiplier: int = 30, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate `multiplier` random augmentations of every sample."""
    rng = np.random.default_rng(seed)
    Xa, ya = [], []
    # Always include the originals once (no aug) for an anchor.
    Xa.append(X)
    ya.append(y)
    for _ in range(multiplier):
        Xi = np.stack([augment(x, rng) for x in X])
        Xa.append(Xi)
        ya.append(y)
    return np.concatenate(Xa, axis=0), np.concatenate(ya, axis=0)
