# Lab 5 Implementation Plan

## Goal

Deliver the four assignment outputs in a way that reuses the current codebase,
keeps the workflow reproducible, and avoids building the same logic twice across
the viewer, notebook, and SAM pipeline.

## Current State

- `viewer.py` already satisfies the core of task 1:
  - open ENVI/BSQ data from `data/images/`
  - render an RGB preview
  - inspect a clicked pixel spectrum
  - export the selected spectrum to CSV
- `data/images/` contains airborne ENVI scenes with wavelength and map metadata.
- `../lab_3/deforestation.ipynb` already includes a working SAM pattern that can
  be adapted instead of rewritten from scratch.
- `lab_1` and `lectures_demo/notebooks/create_stac.ipynb` already show local
  STAC and Sentinel-oriented examples that can be reused for the satellite part.
- Missing from `lab_5`:
  - a defined spectral-library data model
  - a notebook for false-color composites and water-quality analysis
  - a Sentinel-2 retrieval and alignment workflow
  - a reusable SAM module for airborne and Sentinel-2 comparison
- The local metadata does not clearly expose the acquisition date, but the
  project requirement is now fixed to use 17 June 2025 for Sentinel-2 matching.

## Locked Decisions

These points are now fixed for planning:

- We should reuse the acquisition-search pattern from `../lab_2/aral_lab_2.ipynb`
  as the template for satellite data access.
- Water-quality work should use standard index-style implementations rather than
  site-specific regression models unless later calibration data is provided.
- The spectral library must support ROI averages from the start, not only
  single-pixel exports.
- The provided airborne acquisition date is 17 June 2025.

## Implementation Strategy

Treat the viewer as an existing baseline, then build the rest of the project
around reusable processing utilities instead of pushing all logic into one large
notebook.

Recommended order:

1. Stabilize the sampling workflow and spectral-library format.
2. Build reusable utilities for metadata, indices, and satellite loading.
3. Assemble the main notebook on top of those utilities.
4. Add calibration and SAM once the library and cross-sensor alignment are in
   place.

## Deliverables

### 1. Viewer and Sampling Workflow

Target outcome:

- Keep `viewer.py` as the entry point for interactive exploration.
- Extend it only where it directly helps tasks 2 and 4.

Planned scope:

- Preserve the current RGB preview, pixel click, and CSV export behavior.
- Add ROI selection and a structured export workflow for labeled samples.
- Show enough metadata during export to make each sample traceable to a scene
  and pixel location.
- Keep single-pixel export as a convenience path, but treat ROI-average export as
  the main library-building workflow.

ROI workflow to target:

- click-drag rectangular ROI on the RGB panel
- compute mean spectrum, standard deviation, and valid-pixel count
- optionally retain the center pixel and bounding box
- export ROI samples with class label and free-text notes

Why rectangle first:

- it fits the current matplotlib/Tkinter viewer with minimal UI complexity
- it is fast to implement and sufficient for the assignment
- it avoids committing early to a more complex polygon editor

Acceptance criteria:

- A user can open any available ENVI scene and inspect spectra.
- A user can export samples in a repeatable format suitable for later notebook
  and SAM use.
- Exported samples can be linked back to scene name, row, column, and map
  coordinates.
- ROI exports include bounding box, pixel count, and aggregate statistics.

### 2. Spectral Library

Target outcome:

- A saved library of reference spectra for at least:
  - water
  - green vegetation
  - forest
  - optionally bare soil, built-up surfaces, shadow, and mixed shoreline pixels

Planned storage model:

- `data/spectral_library/raw/<class_name>/<sample_id>.csv`
  - ROI-averaged spectrum with one row per wavelength
- `data/spectral_library/catalog.csv`
  - one row per sample with metadata such as:
    - `sample_id`
    - `class_name`
    - `scene_id`
    - `geometry_type`
    - `row_min`
    - `row_max`
    - `col_min`
    - `col_max`
    - `pixel_count`
    - `row`
    - `col`
    - `map_x`
    - `map_y`
    - `source_file`
    - `notes`
- `data/spectral_library/summary/<class_name>.csv`
  - aggregated mean and standard deviation by wavelength for each class
- `data/spectral_library/raw_pixels/<class_name>/<sample_id>.csv`
  - optional follow-up output containing per-pixel spectra inside the ROI if we
    decide we need within-ROI variability later

Why this structure:

- raw samples remain reproducible
- the catalog is simple to inspect in a spreadsheet
- summary spectra are directly usable for visualization and SAM
- ROI metadata is preserved without needing shapefile machinery on day one

Acceptance criteria:

- Samples are labeled consistently.
- The catalog can rebuild the library without manual guessing.
- Mean class spectra can be plotted directly from saved outputs.
- The same library format works for both single pixels and ROIs.

### 3. Water-Quality Notebook

Target outcome:

- One notebook that covers airborne visualization, index calculation,
  Sentinel-2 retrieval, and cross-sensor comparison.

Planned notebook sections:

1. Scene discovery and metadata summary.
2. False-color composites from the airborne cube.
3. Spectral-library overview and quick quality check.
4. Airborne water-quality index computation.
5. Sentinel-2 scene search and download.
6. Reprojection, resampling, and overlap masking.
7. Sentinel-2 water-quality index computation.
8. Comparison plots, maps, and summary statistics.

Notebook outputs should include:

- false-color composites
- index maps for airborne data
- index maps for Sentinel-2
- comparison tables
- scatter plots or histograms over the overlapping area
- short written conclusions inside the notebook

Important design choice:

- Implement the index formulas in reusable Python functions, then call those
  functions from the notebook. This keeps the notebook readable and makes the
  same formulas reusable for both airborne and Sentinel-2 data.

Planned standard formulas:

- Chl-a proxy:
  - primary: red-edge peak height at 705 nm against the 665-740 nm baseline
  - secondary diagnostic: `NDCI = (R705 - R665) / (R705 + R665)`
  - airborne: use nearest wavelengths around 665 nm, 705 nm, and 740 nm
  - Sentinel-2: use `B04`, `B05`, and `B06`
- DOC proxy:
  - implement as a CDOM/DOC optical proxy, not absolute DOC concentration
  - primary formula: green-red ratio using wavelengths nearest 560 nm and 665 nm
  - Sentinel-2: use `B03 / B04`
- Turbidity proxy:
  - primary notebook index: `NDTI = (Red - Green) / (Red + Green)`
  - Sentinel-2: use `B04` and `B03`
  - optional extension: Dogliotti single-band turbidity algorithm if the
    reflectance and atmospheric-correction assumptions hold well enough

Planned algae detection indices (cross-sensor — airborne + Sentinel-2):

- Normalized Difference Red Edge (NDRE):
  - `ndre = (R842 - R705) / (R842 + R705)`
  - Sentinel-2: `(B08 - B05) / (B08 + B05)`
  - sensitive to chlorophyll in dense blooms where NDCI saturates
- Floating Algae Index (FAI):
  - `fai = R842 - (R665 + ((842 - 665) / (1610 - 665)) * (R1610 - R665))`
  - slope constant: `(842 - 665) / (1610 - 665) ≈ 0.1873`
  - Sentinel-2: `B08 - (B04 + 0.1873 * (B11 - B04))`
  - detects floating algae mats and surface scums via the NIR reflectance peak
    above the RED–SWIR baseline; positive values indicate floating vegetation
    or algae on the water surface
- Normalized Difference Water Index (NDWI, McFeeters 1996):
  - `ndwi = (R560 - R842) / (R560 + R842)`
  - Sentinel-2: `(B03 - B08) / (B03 + B08)`
  - positive values → water; negative values → land
  - used as a water mask before algae index analysis

Planned algae detection indices (airborne only — narrow-band requirement):

- Fluorescence Line Height (FLH):
  - `flh = R681 - (R665 + ((681 - 665) / (709 - 665)) * (R709 - R665))`
  - slope constant: `(681 - 665) / (709 - 665) ≈ 0.3636`
  - measures chlorophyll fluorescence emission near 681 nm above the 665–709 nm
    baseline
  - requires narrow bands near 665, 681, and 709 nm; NOT computable from
    Sentinel-2 because Sentinel-2 lacks a 681 nm band
- Maximum Chlorophyll Index (MCI):
  - `mci = R709 - (R681 + ((709 - 681) / (753 - 681)) * (R753 - R681))`
  - slope constant: `(709 - 681) / (753 - 681) ≈ 0.3889`
  - measures the 709 nm chlorophyll absorption peak above the 681–753 nm
    baseline; sensitive to high chlorophyll concentrations in intense blooms
  - requires narrow bands near 681, 709, and 753 nm; NOT computable from
    Sentinel-2

Planned algae detection workflow in the notebook:

1. Compute all 9 indices (4 existing + 5 new) from the airborne cube.
2. Apply the NDWI water mask (NDWI > 0) to isolate water pixels.
3. Analyse NDCI, NDRE, FAI, FLH, and MCI over water pixels only.
4. Compute Sentinel-2 versions of NDRE, FAI, and NDWI (FLH and MCI are
   airborne-only) and add them to the cross-sensor comparison.
5. Collect spectral-library ROIs for new classes (`algae`, `sediment`) and
   rebuild the library; SAM will automatically pick up the new classes.

New spectral library target classes:

- `algae` — ROIs over visibly green or turbid water patches with elevated
  chlorophyll signatures (the target detection class)
- `sediment` — ROIs over water with high suspended sediment but no algal bloom
  signatures (helps SAM distinguish turbidity from actual algae)

Important terminology note:

- Chl-a and turbidity are straightforward to treat as spectral indices or
  proxies.
- DOC is not directly observable from reflectance in a universal way.
- The defensible plan is to compute a CDOM-style optical proxy and label it
  clearly in code and notebook text, while still matching the assignment wording
  in the section titles.
- FLH and MCI are airborne-only diagnostics because they exploit narrow-band
  chlorophyll fluorescence features that Sentinel-2's broad bands cannot
  resolve.

### 4. Sentinel-2 Retrieval and Alignment

Target outcome:

- Retrieve the closest suitable Sentinel-2 Level-2A scene for the airborne area
  and align it to the airborne data footprint.

Planned workflow:

1. Derive the airborne spatial footprint and CRS from the ENVI metadata.
2. Use the confirmed airborne acquisition date of 17 June 2025.
3. Query Sentinel-2 Level-2A scenes near that date.
4. Prefer low-cloud scenes and keep the scene-selection logic explicit.
5. Download or load only the bands required for indices and comparison.
6. Reproject everything to a shared CRS and common extent.
7. Compare on a common grid after masking invalid pixels.

Recommended implementation direction:

- Use STAC-based search/loading for Sentinel-2 if course constraints allow it.
- Keep the satellite-loading logic separate from the notebook so it can be
  rerun and cached cleanly.

Pattern to reuse from `Lab 2`:

- `pystac_client.Client.open(...)`
- a band-alias helper so the same loader can map logical names to assets
- cached per-band downloads to local GeoTIFFs
- reprojection to a shared analysis grid
- explicit cloud-based sorting and scene selection

Sentinel-2-specific adjustments:

- likely required bands:
  - `B02`, `B03`, `B04`, `B05`, `B08`
  - optionally `B8A` and `B11` for extra composites or later experiments
- date window:
  - search around 17 June 2025
  - start with a narrow window around that exact day, then widen only if needed
- selection logic:
  - prefer low-cloud Level-2A scenes
  - inspect whether one tile fully covers the footprint before adding any
    multi-scene mosaic logic

Acceptance criteria:

- The chosen Sentinel-2 scene and selection rationale are documented.
- Data loading is reproducible.
- The comparison uses a clearly defined overlap mask and common grid.

### 5. SAM and Cross-Sensor Calibration

Target outcome:

- A reusable SAM workflow derived from `Lab_3`, applied to this airborne and
  Sentinel-2 context.

Planned approach:

1. Extract or reimplement the SAM computation from `../lab_3/deforestation.ipynb`
   into a reusable function.
2. Use the saved spectral library as the reference input.
3. Prepare separate SAM inputs for:
   - airborne hyperspectral data
   - Sentinel-2 multispectral data
4. Before running SAM on Sentinel-2, harmonize the airborne reference spectra to
   Sentinel-2 spectral and spatial characteristics.

Calibration and harmonization plan:

1. Remove unusable hyperspectral bands and no-data regions.
2. Resample airborne spectra to Sentinel-2-like bands.
3. Aggregate airborne data spatially to Sentinel-2 resolution.
4. Align the overlapping footprint.
5. Fit a simple per-band calibration model on the overlap:
   - first choice: robust linear gain/offset per band
   - fallback: gain-only scaling if offsets prove unstable
6. Apply the calibration to the Sentinel-2 stack before SAM.
7. Run SAM and compare angle maps or class assignments against airborne-derived
   results.

Why this is the right order:

- SAM is sensitive to spectral shape.
- Sentinel-2 has far fewer and broader bands than the airborne cube.
- Calibration should happen after spectral and spatial harmonization, not before.

Acceptance criteria:

- The SAM implementation is reusable and not notebook-only.
- Calibration assumptions are documented.
- Outputs make it possible to compare airborne and Sentinel-2 behavior.

## Proposed Code Layout

This is the structure to target during implementation, not a statement that all
files already exist:

- `viewer.py`
- `notebooks/water_quality_analysis.ipynb`
- `src/lab5/envi_utils.py`
- `src/lab5/spectral_library.py`
- `src/lab5/indices.py`
- `src/lab5/sentinel2.py`
- `src/lab5/sam.py`
- `data/spectral_library/`
- `data/sentinel2/`
- `data/outputs/`

Reasoning:

- utility code becomes testable and reusable
- the notebook stays focused on analysis
- interactive viewer work remains separate from batch processing

## Dependency Plan

Current dependencies are enough for the existing viewer, but not for the full
assignment.

Likely additions:

- `pandas`
- `rasterio`
- `pyproj`
- `jupyterlab`
- `notebook`
- `xarray`
- `pystac-client`
- `odc-stac`
- `planetary-computer` or another provider-specific client only if we decide to
  follow a Planetary Computer workflow instead of a generic STAC source

Possible additions depending on implementation details:

- `shapely`
- `geopandas`
- `scipy`

## Risks and Open Questions

The main unknowns still worth resolving are:

1. Should the DOC output be presented explicitly as `DOC proxy` / `CDOM proxy`
   in the notebook, or does the course expect the shorter `DOC` label?
2. Should SAM output only angle rasters, or also a hard classification map?
3. Is there a course-preferred source for Sentinel-2 access and download?
4. Should the first ROI implementation support rectangles only, or also polygon
   ROIs?

## Practical Milestones

### Milestone 1

- Finalize the ROI-first spectral-library file format.
- Choose the Sentinel-2 data source and scene-selection rule.

### Milestone 2

- Upgrade the viewer for rectangle ROI selection and labeled export.
- Collect initial reference ROIs for the main classes.
- Produce the first version of the spectral-library catalog and summary files.

### Milestone 3

- Build reusable utilities for ENVI metadata, water-quality indices, and
  Sentinel-2 loading.
- Start the notebook and validate the airborne-only sections first.

### Milestone 4

- Add the Sentinel-2 comparison pipeline.
- Implement calibration and SAM.

## Immediate Next Step

Before writing more code, we should lock down three requirement-level decisions:

1. whether the first viewer ROI tool should be rectangle-only
2. which Sentinel-2 source we want to standardize on for implementation
3. whether the notebook should label the DOC output as `DOC` or `DOC proxy`
