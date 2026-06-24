#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lab5.spectral_library import rebuild_class_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild class summary spectra from raw spectral-library samples."
    )
    parser.add_argument(
        "--base-dir",
        default="data/spectral_library",
        help="Spectral library base directory. Default: data/spectral_library",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir)
    written_paths = rebuild_class_summaries(base_dir)

    if not written_paths:
        print(f"No class summaries rebuilt from {base_dir}")
        return 0

    print(f"Rebuilt {len(written_paths)} class summaries in {base_dir}")
    for class_name, path in sorted(written_paths.items()):
        print(f"- {class_name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
