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
    parser = argparse.ArgumentParser(description="Create a SHA256 manifest for a dataset tree.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    out = args.output.expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")

    # Keep the manifest outside the dataset so the frozen dataset tree stays untouched.
    try:
        out.relative_to(root)
    except ValueError:
        pass
    else:
        raise SystemExit("Manifest output must be outside the dataset root.")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for p in files:
            rel = p.relative_to(root).as_posix()
            digest = sha256_file(p)
            f.write(f"{digest}  {rel}\n")

    print(f"Dataset: {root}")
    print(f"Files hashed: {len(files)}")
    print(f"Manifest: {out}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
