"""Concatenate AI Hub split download parts back into a single zip.

The downloader splits one file into chunks named <name>.part<byte_offset>.
We sort by numeric offset and stream-concatenate.

Usage:
    python join_parts.py <parts_dir> <output_zip>
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("usage: join_parts.py <parts_dir> <output_zip>")
        return 1
    parts_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    pat = re.compile(r"^(?P<stem>.+\.zip)\.part(?P<offset>\d+)$")
    parts: list[tuple[int, Path]] = []
    stems = set()
    for p in parts_dir.iterdir():
        m = pat.match(p.name)
        if not m:
            continue
        stems.add(m.group("stem"))
        parts.append((int(m.group("offset")), p))
    if len(stems) > 1:
        print(f"ERROR: multiple stems found: {stems}", file=sys.stderr)
        return 1
    if not parts:
        print(f"ERROR: no .partN files found in {parts_dir}", file=sys.stderr)
        return 1
    parts.sort(key=lambda x: x[0])

    total = sum(p.stat().st_size for _, p in parts)
    print(f"Found {len(parts)} parts, total {total / 1e9:.2f} GB")
    print(f"Output: {out_path}")

    expected_offset = 0
    for offset, p in parts:
        if offset != expected_offset:
            print(f"ERROR: gap detected. Expected offset {expected_offset}, "
                  f"got {offset} for {p.name}", file=sys.stderr)
            return 1
        expected_offset += p.stat().st_size

    written = 0
    chunk = 16 * 1024 * 1024
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        for i, (offset, p) in enumerate(parts, 1):
            with p.open("rb") as f:
                shutil.copyfileobj(f, out, length=chunk)
            written += p.stat().st_size
            pct = 100 * written / total
            print(f"  [{i:>2}/{len(parts)}] +{p.stat().st_size / 1e9:.2f} GB  "
                  f"({pct:.1f}% done)")
    print(f"\nDone. Output size: {out_path.stat().st_size / 1e9:.2f} GB")
    if out_path.stat().st_size != total:
        print("WARNING: output size mismatch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
