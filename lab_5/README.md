# Lab 5 — Hyperspectral Water Quality Analysis

Tools for fusing airborne hyperspectral and Sentinel-2 multispectral data to
monitor water quality and detect algal blooms. The project includes an
interactive desktop viewer for spectral-library collection, reusable processing
modules, and an analysis notebook that covers false-color composites,
water-quality indices, algae detection indices, cross-sensor comparison, and SAM
classification.

## Project Layout

```
lab_5/
├── viewer.py                  # Interactive desktop viewer (Tkinter + Matplotlib)
├── src/lab5/                  # Reusable processing modules
│   ├── envi_utils.py          # ENVI/BSQ I/O, metadata, ROI helpers
│   ├── indices.py             # Water-quality and algae detection indices
│   ├── spectral_library.py    # Spectral library data model and catalog
│   ├── sentinel2.py           # STAC search, download, reprojection, comparison
│   └── sam.py                 # SAM, calibration, classification
├── scripts/
│   └── rebuild_spectral_library.py   # Rebuild class summaries from raw samples
├── notebooks/
│   └── water_quality_analysis.ipynb  # Main analysis notebook
├── data/
│   ├── images/                # Input ENVI/BSQ scenes (.hdr + .bsq)
│   ├── spectral_library/      # Exported ROI samples and class summaries
│   ├── sentinel2/             # Cached Sentinel-2 GeoTIFFs
│   └── outputs/               # Generated plots, tables, comparison results
├── requirements.txt
└── README.md
```

## Requirements

Python 3.10+ and the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the Viewer

```bash
# auto-detect dataset in data/images/
python3 viewer.py

# open a specific file directly
python3 viewer.py data/images/221000_Odra_HS_Blok_A_008_VS_join_atm.hdr
```

On startup the viewer will:

1. Scan `data/images/` for `.hdr` files.
2. If only one is found it opens immediately; if multiple are found a picker
   dialog appears.
3. The RGB preview loads (reads only 3 bands — fast even for large cubes).
4. **Left-click** any pixel to see its full spectral signature on the right panel.
5. **Click-drag** a rectangle to inspect the ROI mean spectrum with standard
   deviation shading.

### Collecting ROI Samples for the Spectral Library

The viewer is the primary tool for building the spectral reference library used
by the SAM classifier.

**Step-by-step:**

1. Open a scene in the viewer.
2. Drag a rectangle over a representative land-cover patch (e.g. a stretch of
   river for `water`, a tree canopy area for `forest`).
3. The status bar shows the ROI bounds and valid-pixel count.
4. Click **Export ROI sample…**.
5. Enter a class label (e.g. `water`, `forest`, `green vegetation`).
6. Optionally add notes describing the sample location or conditions.
7. Repeat for each class and, ideally, multiple samples per class.

Each export creates:

- `data/spectral_library/raw/<class_name>/<sample_id>.csv` — per-ROI mean and
  std spectrum.
- A row in `data/spectral_library/catalog.csv` — metadata including scene ID,
  ROI bounds, map coordinates, pixel count, and timestamp.

You can also click **Export pixel spectrum to CSV…** to save a single pixel's
spectrum to an arbitrary location.

## Rebuilding the Spectral Library

After collecting ROI samples, rebuild the per-class summary files:

```bash
python3 scripts/rebuild_spectral_library.py --base-dir data/spectral_library
```

This script:

1. Reads `data/spectral_library/catalog.csv` to discover all exported samples.
2. Groups raw sample CSVs by class name.
3. Aligns samples by wavelength and computes per-wavelength mean and standard
   deviation across all samples in each class.
4. Writes `data/spectral_library/summary/<class_name>.csv` for each class.

The notebook's SAM section loads these summary files as reference endmembers. If
you add new ROI samples, re-run the rebuild script before re-executing the
notebook.

## Sentinel-2 Scene Selection

The notebook automatically finds the closest usable Sentinel-2 L2A scene to the
airborne acquisition date using the Microsoft Planetary Computer STAC catalog.

**How the selection works:**

1. The airborne scene's footprint is converted from its native UTM CRS to
   WGS-84.
2. The STAC catalog is queried for `sentinel-2-l2a` scenes within a configurable
   date window (default: ±5 days around `2025-06-17`).
3. Scenes with cloud cover above the limit (default: 20%) are excluded.
4. Each remaining candidate is scored as `|day_offset| + 0.01 * cloud_cover`,
   preferring scenes closest in time with low cloud cover.
5. The top-scoring item is selected.

**Required bands:** B03 (560 nm), B04 (665 nm), B05 (705 nm), B06 (740 nm),
B08 (842 nm), B11 (1610 nm).

Each band is downloaded as a Cloud-Optimised GeoTIFF, reprojected to the
airborne scene grid, and cached locally under `data/sentinel2/`. On subsequent
runs, the cached files are reused without re-downloading.

**Offline use:** If the live STAC query is unavailable, the notebook falls back
to the most recent complete local cache in `data/sentinel2/`.

## Running the Notebook

```bash
cd lab_5
jupyter lab notebooks/water_quality_analysis.ipynb
```

The notebook executes the full pipeline end-to-end:

1. Load and inspect the airborne scene.
2. Build natural RGB and false-color composites.
3. Compute airborne water-quality and algae detection index maps.
4. Apply the NDWI water mask and analyse algae indices over water pixels.
5. Search and load the closest Sentinel-2 scene.
6. Align both sources to a shared grid and compare products.
7. Load the spectral library and run SAM on both sensors.

All outputs are saved under `data/outputs/comparison/<scene_id>/`.

## Known Limitations

- **DOC is a proxy, not a calibrated concentration.** The `doc_proxy` index
  (`R560/R665`) correlates with dissolved organic carbon / CDOM but is not
  calibrated to physical units. The same applies to the other indices — they are
  dimensionless ratios, not absolute measurements.
- **Rectangle ROIs only.** The viewer supports rectangular region selection.
  Polygon or freehand ROI drawing is not implemented in v1.
- **Single-scene airborne workflow.** The notebook processes one ENVI scene at a
  time, parameterised by `SCENE_PATH`. Multi-scene mosaicking or batch
  processing is not included.
- **No atmospheric correction.** The airborne data is assumed to be
  atmospherically corrected (L2A-equivalent). The Sentinel-2 data uses the
  standard L2A product from Planetary Computer.
- **Calibration scope.** The per-band linear calibration is fitted over the
  single overlap footprint. It may not generalise to other scenes or dates
  without refitting.
- **FLH and MCI are airborne only.** These indices require narrow-band data
  near 681 nm that Sentinel-2 does not provide. They cannot be included in
  the cross-sensor comparison.
- **NDWI water mask threshold.** The default threshold (NDWI > 0) is standard
  but may need site-specific tuning for mixed shoreline pixels.
- **FAI detects floating algae only.** The Floating Algae Index is designed for
  surface scums and algal mats. Submerged or dissolved blooms may not produce
  a positive FAI signal.
- **Turbid water can mimic algae.** High sediment concentrations can produce
  elevated red-edge reflectance similar to chlorophyll. The spectral library
  should include a `sediment` class to help SAM distinguish the two.

## Data Format Assumptions

The tool reads standard ENVI headers (`.hdr` + `.bsq` pairs). The following
header keys are used when present:

| Key | Purpose |
|---|---|
| `samples`, `lines`, `bands` | Image dimensions |
| `data type` | Pixel data type (e.g. `2` = int16) |
| `byte order` | Endianness |
| `interleave` | Must be `BSQ` |
| `default bands` | 1-based RGB band indices for preview |
| `wavelength` | Per-band wavelengths (nm) for the X axis |
| `data ignore value` | No-data threshold; masked in both RGB and spectrum |
| `map info` | Geo-referencing for pixel-to-map coordinate conversion |

## Notes

- The `.bsq` files are excluded from the repository via `.gitignore` due to
  their size (7–18 GB each). Download the data from OneDrive and place it in
  `data/images/`.
- The tool never loads the full cube into RAM. Band reads are memory-mapped by
  the `spectral` library; reading one pixel's spectrum requires only 456 × 2
  bytes of I/O.
