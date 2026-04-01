#!/usr/bin/env python3
"""
generate_test_cube.py
---------------------
Creates a small synthetic ENVI BSQ hyperspectral cube in data/images/
so the browser can be tested without real airborne data.

Usage:
    python generate_test_cube.py
"""

import os
import struct
import numpy as np
from pathlib import Path

# ── parameters ────────────────────────────────────────────────────────────────
SAMPLES   = 320
LINES     = 240
BANDS     = 128
WL_START  = 400.0   # nm
WL_END    = 2500.0  # nm
OUT_DIR   = "data/images"
STEM      = "synthetic_cube"
DATA_TYPE = 4        # float32
IGNORE    = -9999.0

def make_cube():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    bsq_path = os.path.join(OUT_DIR, STEM + ".bsq")
    hdr_path = os.path.join(OUT_DIR, STEM + ".hdr")

    wavelengths = np.linspace(WL_START, WL_END, BANDS)

    print(f"Generating {SAMPLES}×{LINES}×{BANDS} cube … ", end="", flush=True)

    rng = np.random.default_rng(42)
    cube = np.zeros((BANDS, LINES, SAMPLES), dtype=np.float32)

    # Create spatial "scenes": vegetation, soil, water blobs
    Y, X = np.mgrid[0:LINES, 0:SAMPLES]

    for b, wl in enumerate(wavelengths):
        # vegetation signature (high NIR, low vis)
        veg = (np.sin(np.pi * (wl - 400) / 2100) * 0.6 + 0.1).clip(0, 1)
        # soil signature (gradually rising)
        soil = (0.1 + 0.3 * (wl - 400) / 2100)
        # water (absorbs everything beyond ~750 nm)
        water = np.where(wl < 750, 0.05 + 0.02 * (wl - 400) / 350, 0.01)

        # spatial masks (simple blobs)
        veg_mask  = (((X - 80)  ** 2 + (Y - 60)  ** 2) < 2500).astype(np.float32)
        water_mask= (((X - 240) ** 2 + (Y - 160) ** 2) < 1500).astype(np.float32)

        band_data = (
            (1 - veg_mask - water_mask).clip(0, 1) * soil
            + veg_mask   * float(veg)
            + water_mask * float(water)
            + rng.normal(0, 0.005, (LINES, SAMPLES)).astype(np.float32)
        ).clip(0, 1).astype(np.float32)

        # small no-data border
        band_data[:5,  :] = IGNORE
        band_data[-5:, :] = IGNORE
        band_data[:, :5 ] = IGNORE
        band_data[:, -5:] = IGNORE

        cube[b] = band_data

    cube.tofile(bsq_path)
    print("done.")

    # default RGB bands near 660, 550, 480 nm → indices
    def wl_idx(target):
        return int(np.argmin(np.abs(wavelengths - target)))

    r_idx = wl_idx(660) + 1   # 1-based
    g_idx = wl_idx(550) + 1
    b_idx = wl_idx(480) + 1

    wl_list = ", ".join(f"{w:.2f}" for w in wavelengths)

    hdr = f"""ENVI
samples = {SAMPLES}
lines   = {LINES}
bands   = {BANDS}
header offset = 0
file type = ENVI Standard
data type = {DATA_TYPE}
interleave = bsq
byte order = 0
wavelength units = Nanometers
default bands = {{{r_idx}, {g_idx}, {b_idx}}}
data ignore value = {IGNORE}
reflectance scale factor = 10000
wavelength = {{{wl_list}}}
"""
    with open(hdr_path, "w") as f:
        f.write(hdr)

    print(f"Written: {bsq_path}")
    print(f"Written: {hdr_path}")
    print(f"\nLaunch the browser with:  python hyperspectral_browser.py")


if __name__ == "__main__":
    make_cube()
