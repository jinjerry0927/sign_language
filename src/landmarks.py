"""MediaPipe Holistic landmark extraction utilities.

Used by both the live demo and the offline video to .npy converter.
Output vector layout (225-d float32):
    [pose 33 * (x,y,z)] + [left_hand 21 * (x,y,z)] + [right_hand 21 * (x,y,z)]
Missing parts are filled with zeros.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

NUM_POSE = 33
NUM_HAND = 21
VECTOR_SIZE = (NUM_POSE + 2 * NUM_HAND) * 3

DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "holistic_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_UPPER_IDX = [11, 12, 13, 14, 15, 16, 23, 24]
POSE_UPPER_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]


@contextmanager
def create_landmarker(
    model_path: Path = DEFAULT_MODEL,
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
):
    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    try:
        yield landmarker
    finally:
        landmarker.close()


def _to_array(lms, expected: int) -> np.ndarray:
    if not lms:
        return np.zeros(expected * 3, dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()


def result_to_vector(result) -> np.ndarray:
    return np.concatenate([
        _to_array(result.pose_landmarks, NUM_POSE),
        _to_array(result.left_hand_landmarks, NUM_HAND),
        _to_array(result.right_hand_landmarks, NUM_HAND),
    ])


def bgr_to_mp_image(bgr_frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_landmarks(frame: np.ndarray, result) -> None:
    h, w = frame.shape[:2]

    def px(lm):
        return int(lm.x * w), int(lm.y * h)

    pose = result.pose_landmarks
    if pose:
        for a, b in POSE_UPPER_CONNECTIONS:
            if a < len(pose) and b < len(pose):
                cv2.line(frame, px(pose[a]), px(pose[b]), (255, 200, 0), 2)
        for i in POSE_UPPER_IDX:
            if i < len(pose):
                cv2.circle(frame, px(pose[i]), 4, (0, 255, 255), -1)

    for hand_lms, color in [
        (result.left_hand_landmarks, (0, 255, 0)),
        (result.right_hand_landmarks, (0, 128, 255)),
    ]:
        if not hand_lms:
            continue
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, px(hand_lms[a]), px(hand_lms[b]), color, 2)
        for lm in hand_lms:
            cv2.circle(frame, px(lm), 3, color, -1)
