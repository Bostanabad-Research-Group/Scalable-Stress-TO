#!/usr/bin/env python3

from pathlib import Path

root = Path(__file__).resolve().parent
count = 0

for path in root.rglob("._*"):
    if path.is_file() or path.is_symlink():
        try:
            path.unlink()
            print(f"Deleted: {path}")
            count += 1
        except Exception as e:
            print(f"Failed: {path} -> {e}")

print(f"\nDone. Deleted {count} ghost file(s).")