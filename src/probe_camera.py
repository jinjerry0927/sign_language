"""Probe available camera indices + backends on Windows.

Run this when webcam_demo.py says "End of stream." with no window.
Reports which combinations actually deliver frames.
"""
import cv2

BACKENDS = [
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF",  cv2.CAP_MSMF),
    ("CAP_ANY",   cv2.CAP_ANY),
]

for backend_name, backend in BACKENDS:
    for idx in range(4):
        cap = cv2.VideoCapture(idx, backend)
        opened = cap.isOpened()
        ok, frame = (False, None)
        if opened:
            for _ in range(5):
                ok, frame = cap.read()
                if ok:
                    break
        shape = frame.shape if ok else None
        cap.release()
        status = "OK" if ok else ("opened-but-no-frame" if opened else "fail")
        print(f"[{backend_name}] index={idx} -> {status}  shape={shape}")

print("\nDone. Use the first 'OK' combination in webcam_demo.py.")
