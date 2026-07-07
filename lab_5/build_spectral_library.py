#!/usr/bin/env python3
"""Extract spectral library classes from the Odra airborne cube."""

import csv
from pathlib import Path

import numpy as np

from hs_utils import mask_invalid, open_odra, sample_patch

DATA_DIR = Path(__file__).parent / "data" / "images"
OUT_DIR = Path(__file__).parent / "spectral_library"

# Representative pixels (Odra Blok A 008); adjust via viewer.py if needed
SAMPLE_PIXELS = {
    "water": (1940, 1260),
    "vegetation": (1200, 1000),
    "forest": (1680, 1000),
    "bare_soil": (2900, 960),
}


def main():
    img, wl, ignore = open_odra(DATA_DIR)
    OUT_DIR.mkdir(exist_ok=True)

    with (OUT_DIR / "spectral_library.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + [f"{x:.1f}" for x in wl])
        for label, (row, col) in SAMPLE_PIXELS.items():
            patch = sample_patch(img, row, col)
            med = np.nanmedian(mask_invalid(patch, ignore), axis=0)
            w.writerow([label] + [("" if np.isnan(v) else float(v)) for v in med])
            with (OUT_DIR / f"{label}.csv").open("w", newline="") as cf:
                cw = csv.writer(cf)
                cw.writerow(["wavelength_nm", "reflectance"])
                for a, b in zip(wl, med):
                    cw.writerow([a, "" if np.isnan(b) else float(b)])
            print(f"{label}: ({row}, {col})")


if __name__ == "__main__":
    main()
