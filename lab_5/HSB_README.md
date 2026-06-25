# Hyperspectral Cube Browser

A lightweight desktop tool for browsing ENVI BSQ hyperspectral data cubes.

## Features

- **Auto-discovery** – scans `data/images/` for `.hdr` files on startup; opens immediately if only one is found, otherwise shows a picker dialog
- **RGB preview** – reads three bands via memory-mapped I/O (fast even for multi-GB cubes)
- **Click-to-inspect** – click any pixel to display its full spectral signature on the right panel
- **CSV export** – export the current spectrum (wavelength + DN + validity flag) to a CSV file
- **No-data masking** – respects `data ignore value` from the header in both preview and spectrum
- **Graceful degradation** – works without Pillow (no image preview) or without matplotlib (canvas fallback for spectrum)

## Requirements

```
python >= 3.9
numpy         # required
Pillow        # optional – enables RGB image preview
matplotlib    # optional – enables rich spectrum plot
```

Install all optional dependencies:
```bash
pip install numpy Pillow matplotlib
```

## Quick Start

```bash
# 1. Generate a synthetic test cube
python generate_test_cube.py

# 2. Launch the browser
python hyperspectral_browser.py
```

## Supported ENVI Header Keys

| Key                      | Purpose                                               |
|--------------------------|-------------------------------------------------------|
| `samples`, `lines`, `bands` | Image dimensions                                 |
| `data type`              | Pixel dtype (1=uint8, 2=int16, 4=float32, 5=float64…)|
| `byte order`             | 0 = little-endian, 1 = big-endian                    |
| `interleave`             | Must be `bsq`                                        |
| `default bands`          | 1-based RGB band indices for the preview             |
| `wavelength`             | Per-band wavelengths in nm (used as X axis)          |
| `data ignore value`      | No-data threshold; masked in preview and spectrum    |
| `reflectance scale factor` | Recorded in CSV metadata; not applied automatically|

## CSV Output Format

```
# Hyperspectral spectrum export
# Source: data/images/my_scene.hdr
# Pixel: col=142 row=87
# data ignore value: -9999.0
# reflectance scale factor: 10000
wavelength_nm,dn_value,valid
400.00,0.132456,1
…
```

## File Layout

```
project/
├── hyperspectral_browser.py   ← main tool
├── generate_test_cube.py      ← synthetic data generator
├── README.md
└── data/
    └── images/
        ├── scene.hdr
        └── scene.bsq
```
