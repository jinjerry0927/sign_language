"""Sanity check: run the trained model on the 10 source .npy files.

These are the training anchors, so a healthy model should predict each
class with very high probability.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (  # noqa: E402
    SEQ_LEN,
    LANDMARKS_DIR,
    load_target_words,
    normalize_pose_relative,
    resample_to_fixed_length,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "best.keras"

model = tf.keras.models.load_model(MODEL_PATH)
word_ids, class_id, korean = load_target_words()

print(f"{'file':<32} {'true':<8} {'pred':<8} {'conf':>6}  result")
print("-" * 70)
correct = 0
total = 0
for p in sorted(LANDMARKS_DIR.glob("*.npy")):
    wid_true = p.stem.split("_")[0]
    if wid_true not in class_id:
        continue
    arr = np.load(p)
    arr = normalize_pose_relative(arr)
    arr = resample_to_fixed_length(arr, SEQ_LEN)
    probs = model.predict(arr[None, ...], verbose=0)[0]
    ci = int(probs.argmax())
    wid_pred = word_ids[ci]
    conf = float(probs[ci])
    ok = "OK" if wid_pred == wid_true else "WRONG"
    if ok == "OK":
        correct += 1
    total += 1
    print(f"{p.name:<32} {korean[wid_true]:<8} {korean[wid_pred]:<8} "
          f"{conf:>5.1%}  {ok}")
print("-" * 70)
print(f"Accuracy: {correct}/{total}")
