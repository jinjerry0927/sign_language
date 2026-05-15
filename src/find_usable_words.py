"""Find useful everyday words that exist in zip 01 (WORD1501-3000 range).

Prints candidates we could pick for the MVP given zip 01's word coverage.
"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
p = PROJECT_ROOT / "data" / "word_mapping.csv"
with p.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))

# Build mapping restricted to WORD1501..WORD3000
in_zip = {}
for r in rows[1:]:
    wid, korean = r[0], r[1]
    num = int(wid[4:])
    if 1501 <= num <= 3000:
        in_zip[korean] = wid

# Common useful words to check
candidates = [
    # pronouns / family
    "나", "너", "우리", "엄마", "아빠", "형", "오빠", "누나", "언니", "동생",
    "여동생", "남동생", "친구", "선생님", "학생", "사람",
    # emotions
    "좋다", "싫다", "사랑", "미워하다", "행복", "슬프다", "기쁘다", "화나다",
    "그립다", "외롭다", "재미있다", "지루하다",
    # social
    "감사", "고맙다", "죄송", "미안", "안녕", "잘", "네", "아니",
    # verbs everyday
    "먹다", "마시다", "자다", "일하다", "공부하다", "놀다", "쉬다", "가다",
    "오다", "보다", "듣다", "말하다", "읽다", "쓰다", "만나다", "사다",
    # cognition
    "알다", "모르다", "생각하다", "기억하다", "잊다", "이해하다",
    # state
    "있다", "없다", "괜찮다", "아프다", "피곤하다", "배고프다",
    # objects/places
    "밥", "물", "집", "학교", "회사", "병원", "친구", "책", "전화",
    "음식", "차", "음악", "노래", "꿈", "시간",
    # daily
    "오늘", "내일", "어제", "지금", "다음",
]

print("=== Common everyday words AVAILABLE in zip 01 ===")
found = []
for w in candidates:
    if w in in_zip:
        found.append((in_zip[w], w))
        print(f"  {in_zip[w]:>10}  {w}")
print(f"\nTotal found: {len(found)}")
