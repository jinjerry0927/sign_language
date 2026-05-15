"""Print samples from word_mapping.csv."""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
p = PROJECT_ROOT / "data" / "word_mapping.csv"
sys.stdout.reconfigure(encoding="utf-8")

with p.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

print(f"Total rows (incl header): {len(rows)}")
print(f"Header: {rows[0]}")

print("\n=== First 30 words ===")
for r in rows[1:31]:
    print(f"  {r[0]:>10}  {r[1]:<25}  count={r[2]}")

print("\n=== WORD1501-1520 (matches our sample data) ===")
ids = {r[0]: r for r in rows[1:]}
for n in range(1501, 1521):
    key = f"WORD{n:04d}"
    if key in ids:
        r = ids[key]
        print(f"  {r[0]:>10}  {r[1]:<25}  count={r[2]}")

print("\n=== sample_count distribution ===")
from collections import Counter
counts = Counter(int(r[2]) for r in rows[1:])
for cnt, n in sorted(counts.items()):
    print(f"  {n} words have count={cnt}")

basic_words = [
    "안녕", "안녕하세요", "감사", "감사합니다", "사랑", "사랑해", "사랑해요",
    "미안", "미안해요", "죄송", "죄송합니다", "네", "아니", "아니요",
    "좋다", "좋아", "싫다", "싫어", "괜찮다", "괜찮아요",
    "이름", "나", "너", "우리", "엄마", "아빠", "친구",
    "도와주다", "도와주세요", "모르다", "알다", "있다", "없다",
    "물", "밥", "집", "학교", "병원", "화장실", "아프다",
]
print("\n=== Checking common useful words ===")
name_to_id = {r[1]: r[0] for r in rows[1:]}
for w in basic_words:
    if w in name_to_id:
        wid = name_to_id[w]
        cnt = ids[wid][2]
        print(f"  ✓ {w}  -> {wid}  (samples={cnt})")
    else:
        print(f"  ✗ {w}  (not found)")
