"""Train the sign-language LSTM on extracted landmark .npy files.

Usage:
    python train.py
    python train.py --epochs 80 --aug-mult 50

Output:
    checkpoints/best.keras      - best validation accuracy model
    checkpoints/history.json    - training curves
    checkpoints/last_eval.txt   - final val report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (  # noqa: E402
    SEQ_LEN,
    VECTOR_SIZE,
    load_dataset,
    load_target_words,
    make_augmented_set,
)
from model import build_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--aug-mult", type=int, default=40,
                        help="Augmented copies per training sample")
    parser.add_argument("--val-aug-mult", type=int, default=10,
                        help="Augmented copies for the val set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm-units", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    CKPT_DIR.mkdir(exist_ok=True)

    word_ids, _, korean = load_target_words()
    n_classes = len(word_ids)
    print(f"Classes ({n_classes}):")
    for wid in word_ids:
        print(f"  {wid}  {korean[wid]}")

    X, y, names = load_dataset()
    print(f"\nLoaded {len(X)} base samples, shape {X.shape}")
    if len(X) < n_classes:
        print(f"WARNING: fewer samples than classes ({len(X)} < {n_classes})")

    # Split: same single sample acts as anchor in both train and val
    # because we cannot truly hold out with N=1 per class. The val set is
    # a *different set of augmentations* on the same anchors. This measures
    # whether the model is robust to augmentation variability, not whether
    # it generalizes to new signers.
    X_train_aug, y_train_aug = make_augmented_set(
        X, y, multiplier=args.aug_mult, seed=args.seed
    )
    X_val_aug, y_val_aug = make_augmented_set(
        X, y, multiplier=args.val_aug_mult, seed=args.seed + 1
    )
    print(f"Train samples (with aug): {len(X_train_aug)}")
    print(f"Val   samples (with aug): {len(X_val_aug)}")

    # Shuffle train
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X_train_aug))
    X_train_aug, y_train_aug = X_train_aug[idx], y_train_aug[idx]

    model = build_model(
        seq_len=SEQ_LEN,
        n_features=VECTOR_SIZE,
        n_classes=n_classes,
        lstm_units=args.lstm_units,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_path = CKPT_DIR / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_path), monitor="val_accuracy",
            save_best_only=True, save_weights_only=False, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=8, factor=0.5,
            min_lr=1e-5, verbose=1,
        ),
    ]

    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val_aug, y_val_aug),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    (CKPT_DIR / "history.json").write_text(
        json.dumps({k: [float(v) for v in vs] for k, vs in history.history.items()},
                   indent=2),
        encoding="utf-8",
    )

    # Final per-class report on validation set
    probs = model.predict(X_val_aug, verbose=0)
    preds = probs.argmax(axis=1)
    correct = (preds == y_val_aug).sum()
    acc = correct / len(preds)
    print(f"\nFinal val accuracy: {acc:.3f} ({correct}/{len(preds)})")

    per_class = {}
    for i, wid in enumerate(word_ids):
        mask = y_val_aug == i
        if mask.any():
            per_class[wid] = float((preds[mask] == i).mean())
    print("Per-class val accuracy:")
    for wid, p in per_class.items():
        print(f"  {wid}  {korean[wid]:<6}  {p:.2%}")

    (CKPT_DIR / "last_eval.txt").write_text(
        "\n".join([f"val_acc={acc:.4f}"] +
                  [f"{wid}\t{korean[wid]}\t{p:.4f}" for wid, p in per_class.items()]),
        encoding="utf-8",
    )
    print(f"\nSaved: {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
