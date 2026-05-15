"""Extract a few F-angle target-word videos from zip 01 to test_videos/.

Names them with ASCII filenames so cv2 + Windows can open them reliably.
"""
import csv
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP = PROJECT_ROOT / "data" / "raw" / "01_real_word_video.zip"
OUT = PROJECT_ROOT / "test_videos"
OUT.mkdir(exist_ok=True)

# romaji-ish ASCII names per target word
NICE_NAMES = {
    "WORD1528": "eomma_mom",
    "WORD1516": "nuna_sister",
    "WORD1574": "hyeong_brother",
    "WORD1534": "bap_rice",
    "WORD1514": "norae_song",
    "WORD1510": "kkum_dream",
    "WORD1544": "jada_sleep",
    "WORD1515": "nolda_play",
    "WORD2485": "moreuda_dontknow",
    "WORD1637": "eopsda_nothave",
}

with zipfile.ZipFile(ZIP) as z:
    for wid, nice in NICE_NAMES.items():
        target_name = f"01/NIA_SL_{wid}_REAL01_F.mp4"
        try:
            info = z.getinfo(target_name)
        except KeyError:
            print(f"  MISS  {wid}  ({nice}) - not in zip")
            continue
        out_path = OUT / f"{nice}.mp4"
        with z.open(info) as src, out_path.open("wb") as dst:
            dst.write(src.read())
        print(f"  OK    {wid}  ->  {out_path.name}  ({info.file_size/1e6:.1f} MB)")
print(f"\nDone. Files in: {OUT}")
