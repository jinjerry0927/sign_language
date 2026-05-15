"""Inspect the video zip - count files, list signers, check coverage of target words."""
from __future__ import annotations

import csv
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

NAME_RE = re.compile(r"NIA_SL_(WORD\d+)_REAL(\d+)_([DFLRU])\.mp4$", re.IGNORECASE)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    zip_path = Path(sys.argv[1])
    target_csv = Path(r"C:\Users\James\sign_language\data\target_words.csv")
    with target_csv.open(encoding="utf-8-sig", newline="") as f:
        targets = {row["word_id"]: row["korean_word"] for row in csv.DictReader(f)}

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    print(f"Zip total entries: {len(names)}")

    files = [n for n in names if not n.endswith("/")]
    print(f"Files: {len(files)}")
    print(f"\nFirst 5 entry paths:")
    for n in files[:5]:
        print(f"  {n}")

    angles = Counter()
    signers = Counter()
    word_signer_angle: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for n in files:
        m = NAME_RE.search(n)
        if not m:
            continue
        wid, signer, angle = m.group(1), m.group(2), m.group(3).upper()
        angles[angle] += 1
        signers[signer] += 1
        word_signer_angle[wid].add((signer, angle))

    print(f"\n=== Camera angles ===")
    for a, c in angles.most_common():
        print(f"  {a}: {c}")
    print(f"\n=== Signers (unique: {len(signers)}) ===")
    for s, c in sorted(signers.items())[:10]:
        print(f"  REAL{s}: {c} videos")
    if len(signers) > 10:
        print(f"  ... +{len(signers) - 10} more")

    print(f"\n=== Target word coverage (F-angle only) ===")
    total_f = 0
    for wid, korean in targets.items():
        f_signers = sorted(s for (s, a) in word_signer_angle.get(wid, set()) if a == "F")
        total_f += len(f_signers)
        print(f"  {wid:>10}  {korean:<8}  F-angle signers: {len(f_signers):>3}  "
              f"-> {f_signers[:5]}{'...' if len(f_signers) > 5 else ''}")
    print(f"\nTotal F-angle target videos: {total_f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
