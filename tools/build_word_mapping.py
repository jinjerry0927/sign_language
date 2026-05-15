"""Scan AI Hub morpheme zip and build WORD_ID -> Korean word mapping CSV.

Reads every *_morpheme.json inside the zip directly (no extraction needed),
extracts the first attribute.name (the Korean word), and writes a CSV with
columns: word_id, korean_word, sample_count.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

WORD_RE = re.compile(r"WORD(\d+)")


def main():
    if len(sys.argv) < 3:
        print("usage: build_word_mapping.py <morpheme.zip> <output.csv>")
        return 1
    zip_path = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    counts = defaultdict(int)
    name_by_id: dict[str, str] = {}
    skipped = 0

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("_morpheme.json")]
        print(f"Found {len(names)} morpheme JSONs")
        for i, n in enumerate(names, 1):
            m = WORD_RE.search(n)
            if not m:
                skipped += 1
                continue
            word_id = f"WORD{m.group(1)}"
            with z.open(n) as f:
                try:
                    obj = json.loads(f.read().decode("utf-8"))
                except Exception:
                    skipped += 1
                    continue
            data = obj.get("data", [])
            label = None
            for entry in data:
                for attr in entry.get("attributes", []):
                    if attr.get("name"):
                        label = attr["name"].strip()
                        break
                if label:
                    break
            if not label:
                skipped += 1
                continue
            counts[word_id] += 1
            existing = name_by_id.get(word_id)
            if existing is None:
                name_by_id[word_id] = label
            elif existing != label:
                pass
            if i % 5000 == 0:
                print(f"  ... processed {i}/{len(names)}")

    print(f"\nUnique word IDs: {len(name_by_id)}, skipped: {skipped}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word_id", "korean_word", "sample_count"])
        for wid in sorted(name_by_id, key=lambda x: int(x[4:])):
            writer.writerow([wid, name_by_id[wid], counts[wid]])
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
