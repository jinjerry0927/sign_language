"""Phase 1 live demo.

Open webcam or a video file, run MediaPipe HolisticLandmarker on each frame,
and overlay pose + hand skeletons. Confirms the extraction pipeline works
end-to-end before we move on to dataset preprocessing.

Usage:
    python webcam_demo.py                   # webcam, device 0 (mirrored)
    python webcam_demo.py --camera 1        # webcam, device 1
    python webcam_demo.py --video clip.mp4  # video file

Press 'q' to quit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from landmarks import (  # noqa: E402
    bgr_to_mp_image,
    create_landmarker,
    draw_landmarks,
    result_to_vector,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    args = parser.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: failed to open source", file=sys.stderr)
        return 1

    if args.video is None:
        for _ in range(10):
            ok, _ = cap.read()
            if ok:
                break

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_dur_ms = max(1, int(1000 / fps))
    ts_ms = 0
    perf = []
    miss_streak = 0

    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video is not None:
                    print("End of stream.")
                    break
                miss_streak += 1
                if miss_streak > 30:
                    print("Webcam disconnected.", file=sys.stderr)
                    break
                continue
            miss_streak = 0
            if args.video is None:
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(bgr_to_mp_image(frame), ts_ms)
            ts_ms += frame_dur_ms

            draw_landmarks(frame, result)
            vec = result_to_vector(result)

            dt = (time.perf_counter() - t0) * 1000
            perf.append(dt)
            if len(perf) > 30:
                perf.pop(0)
            avg = sum(perf) / len(perf)

            status = " | ".join([
                f"pose: {'Y' if result.pose_landmarks else '-'}",
                f"L: {'Y' if result.left_hand_landmarks else '-'}",
                f"R: {'Y' if result.right_hand_landmarks else '-'}",
                f"vec: {vec.shape[0]}d",
                f"{avg:.1f}ms",
            ])
            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            cv2.imshow("MediaPipe Holistic - Phase 1 (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
