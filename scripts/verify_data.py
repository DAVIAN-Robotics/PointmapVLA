#!/usr/bin/env python
"""Check a downloaded dataset directory against its MANIFEST.json.

Every PointmapVLA dataset repository ships a MANIFEST.json listing each file's
size and SHA-256 as recorded by Hugging Face at upload time. By default only
sizes are compared, which catches truncated or missing files in seconds;
--sha256 also hashes every file, which takes about as long as reading the
dataset once.

Usage:
    python scripts/verify_data.py playground/Datasets/robocasa-MG_300-pointmap
    python scripts/verify_data.py playground/Datasets/libero_r6 --sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="dataset directory containing MANIFEST.json")
    parser.add_argument("--sha256", action="store_true", help="also verify content hashes")
    args = parser.parse_args()

    manifest_path = args.root / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"no MANIFEST.json in {args.root}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    problems = 0
    for entry in manifest["files"]:
        path = args.root / entry["path"]
        if not path.exists():
            print(f"MISSING  {entry['path']}")
            problems += 1
            continue
        size = path.stat().st_size
        if size != entry["size"]:
            print(f"SIZE     {entry['path']}: {size} != {entry['size']}")
            problems += 1
            continue
        if args.sha256 and entry.get("sha256"):
            digest = sha256_of(path)
            if digest != entry["sha256"]:
                print(f"SHA256   {entry['path']}")
                problems += 1

    total = len(manifest["files"])
    if problems:
        print(f"FAIL: {problems} of {total} files differ")
        return 1
    print(f"OK: {total} files verified ({'size + sha256' if args.sha256 else 'size only'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
