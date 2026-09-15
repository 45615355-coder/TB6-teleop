#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a dataset against a SHA256 manifest.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()

    failures = []
    checked = 0

    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = root / rel
        if not path.is_file():
            failures.append(f"MISSING: {rel}")
            continue
        actual = sha256_file(path)
        checked += 1
        if actual != digest:
            failures.append(f"MISMATCH: {rel}")

    print(f"Files checked: {checked}")
    if failures:
        print("RESULT: FAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
