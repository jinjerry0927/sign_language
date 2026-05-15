"""Phase 0 install verification.

Imports each package, prints versions, runs MediaPipe HolisticLandmarker
(Tasks API) on a synthetic image to confirm the native runtime loads.
No webcam needed.
"""
import sys
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "holistic_landmarker.task"


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")

    import cv2
    print(f"opencv-python: {cv2.__version__}")

    import mediapipe as mp
    print(f"mediapipe: {mp.__version__}")

    import tensorflow as tf
    print(f"tensorflow: {tf.__version__}")

    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")

    print(f"numpy: {np.__version__}")

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        return 1

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
    )
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
        result = landmarker.detect(mp_image)

    print(
        "MediaPipe HolisticLandmarker OK "
        f"(pose={bool(result.pose_landmarks)}, "
        f"left_hand={bool(result.left_hand_landmarks)}, "
        f"right_hand={bool(result.right_hand_landmarks)})"
    )
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
