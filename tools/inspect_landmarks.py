"""Inspect extracted landmark .npy files."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "landmarks"
TARGET_CSV = PROJECT_ROOT / "data" / "target_words.csv"

with TARGET_CSV.open(encoding="utf-8-sig", newline="") as f:
    targets = {row["word_id"]: row["korean_word"] for row in csv.DictReader(f)}

print(f"=== Landmark files in {DATA_DIR} ===\n")
files = sorted(DATA_DIR.glob("*.npy"))
print(f"{'file':<32} {'word':<8} {'shape':<14} {'pose%':<7} {'L%':<6} {'R%':<6}")
print("-" * 80)
for p in files:
    word_id = p.stem.split("_")[0]
    korean = targets.get(word_id, "?")
    arr = np.load(p)
    T = arr.shape[0]
    pose_pct = 100 * (arr[:, :99].any(axis=1)).sum() / T
    l_pct = 100 * (arr[:, 99:99+63].any(axis=1)).sum() / T
    r_pct = 100 * (arr[:, 99+63:].any(axis=1)).sum() / T
    print(f"{p.name:<32} {korean:<8} {str(arr.shape):<14} {pose_pct:>5.0f}% {l_pct:>5.0f}% {r_pct:>5.0f}%")

print(f"\nTotal: {len(files)} files")
