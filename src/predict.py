"""Phase 5: real-time sign-language word recognition.

Open webcam or a video file, run MediaPipe Holistic, accumulate a sliding
window of normalized landmarks, and periodically classify with the trained
LSTM. Overlays the predicted Korean word + confidence on the video.

Usage:
    python predict.py                          # webcam, device 0
    python predict.py --camera 1
    python predict.py --video clip.mp4
    python predict.py --threshold 0.6          # min prob to display
    python predict.py --infer-every 5          # inference every N frames

Press 'q' to quit.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (  # noqa: E402
    SEQ_LEN,
    load_target_words,
    normalize_pose_relative,
    resample_to_fixed_length,
)
from landmarks import (  # noqa: E402
    bgr_to_mp_image,
    create_landmarker,
    draw_landmarks,
    result_to_vector,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best.keras"

KOREAN_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
]


def load_korean_font(size: int) -> ImageFont.ImageFont:
    for path in KOREAN_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text_pil(frame: np.ndarray, text: str, pos: tuple[int, int],
                  font: ImageFont.ImageFont, fill=(255, 255, 255)) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    x, y = pos
    # outline
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min softmax probability to display a prediction")
    parser.add_argument("--infer-every", type=int, default=4,
                        help="Run model every N frames (default 4)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Average last K predictions for stability")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}", file=sys.stderr)
        return 1
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)

    word_ids, _, korean = load_target_words()
    print(f"Loaded model. Classes: {[korean[w] for w in word_ids]}")

    # Open input
    tmp_video: Path | None = None
    if args.video:
        src_path = Path(args.video)
        if not src_path.exists():
            print(f"ERROR: file not found: {src_path}", file=sys.stderr)
            return 1
        # OpenCV on Windows can't handle non-ASCII paths reliably -> copy to temp
        if not str(src_path).isascii():
            fd, tmp_name = tempfile.mkstemp(suffix=src_path.suffix)
            import os as _os
            _os.close(fd)
            tmp_video = Path(tmp_name)
            print(f"Copying to ASCII temp path (Windows + Korean path workaround)")
            shutil.copy(src_path, tmp_video)
            cap = cv2.VideoCapture(str(tmp_video))
        else:
            cap = cv2.VideoCapture(str(src_path))
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: failed to open input source", file=sys.stderr)
        if tmp_video and tmp_video.exists():
            tmp_video.unlink()
        return 1
    if args.video is None:
        for _ in range(10):
            cap.read()  # warm up

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_dur_ms = max(1, int(1000 / fps))
    ts_ms = 0
    miss_streak = 0

    window: deque[np.ndarray] = deque(maxlen=SEQ_LEN)
    recent_probs: deque[np.ndarray] = deque(maxlen=args.smooth)
    last_pred_text = ""
    last_pred_conf = 0.0
    frame_idx = 0
    font_big = load_korean_font(48)
    font_small = load_korean_font(22)

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
            window.append(vec)

            if len(window) == SEQ_LEN and frame_idx % args.infer_every == 0:
                arr = np.stack(window)
                arr = normalize_pose_relative(arr)
                arr = resample_to_fixed_length(arr, SEQ_LEN)
                probs = model.predict(arr[None, ...], verbose=0)[0]
                recent_probs.append(probs)
                avg = np.mean(recent_probs, axis=0)
                ci = int(avg.argmax())
                conf = float(avg[ci])
                if conf >= args.threshold:
                    last_pred_text = korean[word_ids[ci]]
                    last_pred_conf = conf
                else:
                    last_pred_text = ""
                    last_pred_conf = conf

            # HUD
            dt_ms = (time.perf_counter() - t0) * 1000
            status = " | ".join([
                f"pose:{'Y' if result.pose_landmarks else '-'}",
                f"L:{'Y' if result.left_hand_landmarks else '-'}",
                f"R:{'Y' if result.right_hand_landmarks else '-'}",
                f"buf:{len(window)}/{SEQ_LEN}",
                f"{dt_ms:.0f}ms",
            ])
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if last_pred_text:
                frame = draw_text_pil(
                    frame,
                    f"{last_pred_text}  ({last_pred_conf:.0%})",
                    (10, 50), font_big, fill=(0, 255, 0),
                )
            elif len(window) == SEQ_LEN:
                frame = draw_text_pil(
                    frame, "...", (10, 50), font_big, fill=(180, 180, 180),
                )

            cv2.imshow("KSL Predictor (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if tmp_video and tmp_video.exists():
        try:
            tmp_video.unlink()
        except PermissionError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
