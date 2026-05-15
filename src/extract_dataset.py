"""Bulk-extract MediaPipe landmark sequences from AI Hub videos.

Supports two input modes:
  --zip <path>      read .mp4 files directly from a zip (streaming, no extract)
  --videos <dir>    walk a directory tree of .mp4 files

Filters videos to a chosen camera angle (default F) and to the target words
in data/target_words.csv, runs MediaPipe Holistic on every frame, and saves
each video's landmarks as a (T, 225) float32 .npy. Resumable: skips outputs
that already exist.

Usage:
    python extract_dataset.py --zip "C:\\path\\to\\01_real_word_video.zip"
    python extract_dataset.py --videos "C:\\path\\to\\unzipped" --angle F
    python extract_dataset.py --zip ... --any-word --limit 1  # debug
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def safe_unlink(path: Path, attempts: int = 10, delay: float = 0.1) -> bool:
    """Windows-safe unlink. cv2 may hold the handle briefly after release()."""
    for _ in range(attempts):
        try:
            path.unlink()
            return True
        except PermissionError:
            gc.collect()
            time.sleep(delay)
        except FileNotFoundError:
            return True
    return False

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import (  # noqa: E402
    VECTOR_SIZE,
    bgr_to_mp_image,
    create_landmarker,
    result_to_vector,
)

NAME_RE = re.compile(r"NIA_SL_(WORD\d+)_REAL(\d+)_([DFLRU])\.mp4$", re.IGNORECASE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_WORDS_CSV = PROJECT_ROOT / "data" / "target_words.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "landmarks"


def load_targets() -> set[str]:
    with TARGET_WORDS_CSV.open(encoding="utf-8-sig", newline="") as f:
        return {row["word_id"] for row in csv.DictReader(f)}


class TimestampClock:
    """Global monotonic timestamp counter for MediaPipe VIDEO mode sessions."""

    def __init__(self):
        self.now_ms = 0

    def advance(self, frame_dur_ms: int) -> int:
        ts = self.now_ms
        self.now_ms += frame_dur_ms
        return ts

    def gap(self, ms: int = 10_000) -> None:
        self.now_ms += ms


def extract_video(video_path: Path, landmarker, clock: TimestampClock) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_dur_ms = max(1, int(1000 / fps))
    vectors: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = clock.advance(frame_dur_ms)
        result = landmarker.detect_for_video(bgr_to_mp_image(frame), ts)
        vectors.append(result_to_vector(result))
    cap.release()
    clock.gap()
    if not vectors:
        return None
    return np.stack(vectors).astype(np.float32)


def make_out_path(out_dir: Path, word_id: str, signer: str, angle: str) -> Path:
    return out_dir / f"{word_id}_REAL{signer}_{angle}.npy"


def collect_zip_candidates(zip_path: Path, targets: set[str], angle: str,
                           any_word: bool, out_dir: Path):
    candidates = []
    skipped_existing = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            m = NAME_RE.search(info.filename)
            if not m:
                continue
            wid, signer, a = m.group(1), m.group(2), m.group(3).upper()
            if a != angle:
                continue
            if not any_word and wid not in targets:
                continue
            out_path = make_out_path(out_dir, wid, signer, a)
            if out_path.exists():
                skipped_existing += 1
                continue
            candidates.append((info.filename, out_path, wid))
    return candidates, skipped_existing


def collect_dir_candidates(videos_root: Path, targets: set[str], angle: str,
                           any_word: bool, out_dir: Path):
    candidates = []
    skipped_existing = 0
    for p in videos_root.rglob("*.mp4"):
        m = NAME_RE.search(p.name)
        if not m:
            continue
        wid, signer, a = m.group(1), m.group(2), m.group(3).upper()
        if a != angle:
            continue
        if not any_word and wid not in targets:
            continue
        out_path = make_out_path(out_dir, wid, signer, a)
        if out_path.exists():
            skipped_existing += 1
            continue
        candidates.append((p, out_path, wid))
    return candidates, skipped_existing


def main() -> int:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", dest="zip_path", help="Read .mp4 files from this zip")
    src.add_argument("--videos", help="Walk this directory for .mp4 files")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--angle", default="F", help="Camera angle: D/F/L/R/U (default F)")
    parser.add_argument("--any-word", action="store_true",
                        help="Skip target-word filter (debug)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap number of videos processed (debug)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    target_angle = args.angle.upper()
    print(f"Target words ({len(targets)}): {sorted(targets)}")
    print(f"Angle filter: {target_angle}")
    print(f"Output dir: {out_dir}")

    using_zip = bool(args.zip_path)
    if using_zip:
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            print(f"ERROR: --zip not found: {zip_path}", file=sys.stderr)
            return 1
        candidates, skipped = collect_zip_candidates(
            zip_path, targets, target_angle, args.any_word, out_dir
        )
    else:
        videos_root = Path(args.videos)
        if not videos_root.exists():
            print(f"ERROR: --videos not found: {videos_root}", file=sys.stderr)
            return 1
        candidates, skipped = collect_dir_candidates(
            videos_root, targets, target_angle, args.any_word, out_dir
        )

    print(f"\n{len(candidates)} new videos to process "
          f"({skipped} already extracted)")
    if args.limit > 0:
        candidates = candidates[: args.limit]
        print(f"Limited to {len(candidates)} (debug)")
    if not candidates:
        print("Nothing to do.")
        return 0

    failures: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    if using_zip:
        z = zipfile.ZipFile(args.zip_path)
    try:
        clock = TimestampClock()
        with create_landmarker() as landmarker:
            for source, out_path, _wid in tqdm(candidates, desc="Extracting"):
                if using_zip:
                    tmp_path = None
                    try:
                        fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
                        os.close(fd)
                        tmp_path = Path(tmp_name)
                        with z.open(source) as f_in, tmp_path.open("wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        video_path = tmp_path
                        display = source
                    except Exception as e:
                        failures.append((source, f"extract-from-zip: {e}"))
                        if tmp_path and tmp_path.exists():
                            tmp_path.unlink()
                        continue
                else:
                    video_path = source
                    display = str(source.name)

                try:
                    arr = extract_video(video_path, landmarker, clock)
                except Exception as e:
                    failures.append((display, f"exception: {e}"))
                    arr = None
                finally:
                    if using_zip and video_path.exists():
                        if not safe_unlink(video_path):
                            print(f"  WARN: could not delete temp {video_path}",
                                  file=sys.stderr)

                if arr is None or arr.shape[0] < 5:
                    failures.append((display, "no/too-few frames"))
                    continue
                if arr.shape[1] != VECTOR_SIZE:
                    failures.append((display, f"unexpected vec size {arr.shape}"))
                    continue
                np.save(out_path, arr)
    finally:
        if using_zip:
            z.close()

    dt = time.perf_counter() - t0
    succeeded = len(candidates) - len(failures)
    print(f"\nProcessed {succeeded} videos in {dt:.1f}s "
          f"({len(failures)} failures)")
    for name, reason in failures[:20]:
        print(f"  FAIL  {name}: {reason}")
    if len(failures) > 20:
        print(f"  ... +{len(failures) - 20} more failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
