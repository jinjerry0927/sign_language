"""Quick zip inspector - list contents + sample one file."""
import sys
import zipfile
from collections import Counter
from pathlib import Path


def main():
    p = Path(sys.argv[1])
    if not zipfile.is_zipfile(p):
        print(f"NOT a valid zip: {p}")
        return 1
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        print(f"Valid zip: {len(names)} entries, {p.stat().st_size / 1e6:.2f} MB on disk")
        print("\n=== First 15 entries ===")
        for n in names[:15]:
            info = z.getinfo(n)
            print(f"  {info.file_size:>10,} bytes   {n}")
        print("\n=== File extensions ===")
        exts = Counter()
        for n in names:
            if n.endswith("/"):
                exts["(dir)"] += 1
            else:
                ext = "." + n.rsplit(".", 1)[-1] if "." in n else "(none)"
                exts[ext] += 1
        for ext, cnt in exts.most_common():
            print(f"  {ext}: {cnt}")
        first_data = next((n for n in names if not n.endswith("/")), None)
        if first_data:
            print(f"\n=== Sample content from: {first_data} ===")
            with z.open(first_data) as f:
                head = f.read(2000)
            try:
                print(head.decode("utf-8"))
            except UnicodeDecodeError:
                print("(binary; first 64 bytes hex)")
                print(head[:64].hex())
    return 0


if __name__ == "__main__":
    sys.exit(main())
