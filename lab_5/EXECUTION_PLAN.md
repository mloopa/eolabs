# Lab 5 Execution Plan

This is the working execution checklist for Codex and for manual tracking.
If this file conflicts with `IMPLEMENTATION_PLAN.md`, this file wins.

## Execution Rules

- Implement phases in order.
- Do not start a later phase until the current phase passes its verification.
- Keep changes scoped to the current phase.
- After each phase:
  - update the checkbox status in this file
  - record what changed
  - run the listed verification steps
  - stop and review before moving on if the user wants strict phase-by-phase work

Status keys:

- `[ ]` not started
- `[-]` in progress
- `[x]` complete

## Locked Defaults

These are the execution defaults unless the user changes them later:

- Airborne acquisition date for satellite matching: `2025-06-17`
- ROI support in v1: rectangular ROIs only
- Primary Sentinel-2 source: Microsoft Planetary Computer STAC, following the
  same client pattern as `../lab_2/aral_lab_2.ipynb`
- Airborne workflow scope in v1: one scene at a time, parameterized by
  `SCENE_PATH`
- Default reference scene for development and testing:
  `data/images/221000_Odra_HS_Blok_A_008_VS_join_atm.hdr`
- DOC output label in code: `doc_proxy`
- DOC output label in notebook headings: `DOC / CDOM proxy`

## Tracking Board

- [x] P0. Project scaffold and dependencies
- [x] P1. Shared ENVI utility layer
- [x] P2. Spectral-library data model and rebuild logic
- [x] P3. Viewer ROI workflow
- [x] P4. Airborne index pipeline
- [x] P5. Sentinel-2 acquisition pipeline
- [x] P6. Cross-sensor alignment and comparison
- [x] P7. SAM and calibration
- [x] P8. Final notebook and documentation polish
- [-] P9. Algae detection extension

## Exact Outputs To Deliver

By the end of implementation, the repo should contain:

- `viewer.py` upgraded with ROI selection and labeled export
- `src/lab5/envi_utils.py`
- `src/lab5/indices.py`
- `src/lab5/spectral_library.py`
- `src/lab5/sentinel2.py`
- `src/lab5/sam.py`
- `scripts/rebuild_spectral_library.py`
- `notebooks/water_quality_analysis.ipynb`
- `data/spectral_library/`
- `data/sentinel2/`
- `data/outputs/`
- updated `README.md`
- updated `requirements.txt`

## Product Definitions

These are the exact index products to implement in code:

- Chlorophyll-a primary proxy:
  - `chl_red_edge_peak = R705 - (R665 + ((705 - 665) / (740 - 665)) * (R740 - R665))`
- Chlorophyll-a secondary diagnostic:
  - `ndci = (R705 - R665) / (R705 + R665)`
- DOC / CDOM proxy:
  - `doc_proxy = R560 / R665`
- Turbidity proxy:
  - `ndti = (R665 - R560) / (R665 + R560)`

Algae detection indices (cross-sensor — airborne + Sentinel-2):

- Normalized Difference Red Edge:
  - `ndre = (R842 - R705) / (R842 + R705)`
  - Sentinel-2: `(B08 - B05) / (B08 + B05)`
  - sensitive to chlorophyll concentration in dense blooms
- Floating Algae Index:
  - `fai = R842 - (R665 + 0.1873 * (R1610 - R665))`
  - Sentinel-2: `B08 - (B04 + 0.1873 * (B11 - B04))`
  - detects surface scums and floating algal mats; NIR peak above RED–SWIR
    baseline
- Normalized Difference Water Index:
  - `ndwi = (R560 - R842) / (R560 + R842)`
  - Sentinel-2: `(B03 - B08) / (B03 + B08)`
  - water mask: positive values → water, negative → land

Algae detection indices (airborne only — narrow-band requirement):

- Fluorescence Line Height:
  - `flh = R681 - (R665 + 0.3636 * (R709 - R665))`
  - measures chlorophyll fluorescence emission near 681 nm above the 665–709 nm
    baseline
  - NOT computable from Sentinel-2 (no 681 nm band)
- Maximum Chlorophyll Index:
  - `mci = R709 - (R681 + 0.3889 * (R753 - R681))`
  - measures the 709 nm chlorophyll absorption peak above the 681–753 nm
    baseline
  - NOT computable from Sentinel-2 (no 681 nm band)

Band mapping rules:

- Airborne:
  - pick nearest valid wavelengths to `560`, `665`, `681`, `705`, `709`, `740`,
    `753`, `842`, `1610`
- Sentinel-2:
  - `B03` for 560 nm
  - `B04` for 665 nm
  - `B05` for 705 nm
  - `B06` for 740 nm
  - `B08` for 842 nm
  - `B11` for 1610 nm

False-color composites to implement in the notebook:

- Natural RGB: default ENVI RGB bands
- Vegetation composite: `842 / 665 / 560`
- Moisture composite: `1610 / 842 / 665`
- Water-focused composite: `705 / 560 / 490` if a valid 490 nm airborne band is
  available, otherwise skip this composite

## Phase P0. Project Scaffold And Dependencies

Goal:

- Create the file and directory structure needed for the rest of the work.

Files to create or edit:

- `requirements.txt`
- `README.md`
- `src/lab5/__init__.py`
- `notebooks/`
- `scripts/`

Tasks:

- [x] P0.1 Create `src/lab5/` package structure.
- [x] P0.2 Create `scripts/` directory.
- [x] P0.3 Create `notebooks/` directory.
- [x] P0.4 Ensure these output directories exist or are created lazily by code:
  - `data/spectral_library/raw/`
  - `data/spectral_library/summary/`
  - `data/spectral_library/raw_pixels/`
  - `data/sentinel2/`
  - `data/outputs/`
- [x] P0.5 Update `requirements.txt` with the minimum execution dependencies:
  - `numpy`
  - `matplotlib`
  - `spectral`
  - `pandas`
  - `rasterio`
  - `pyproj`
  - `pystac-client`
  - `planetary-computer`
  - `jupyterlab`
  - `notebook`
- [x] P0.6 Add a short `README.md` section describing the new project layout.

Done when:

- Package directories exist.
- Dependencies are declared.
- The repo layout matches the plan enough for the next phases to proceed.

Verification:

- `find src scripts notebooks -maxdepth 2 -type f | sort`
- `sed -n '1,200p' requirements.txt`

P0 completion notes:

- created `src/lab5/`, `scripts/`, and `notebooks/`
- added `src/lab5/__init__.py`
- created `data/spectral_library/`, `data/sentinel2/`, and `data/outputs/`
- expanded `requirements.txt` for the planned Lab 5 workflow
- added a `README.md` project layout section

## Phase P1. Shared ENVI Utility Layer

Goal:

- Move all reusable hyperspectral reading logic into one utility module that both
  the viewer and the notebook can call.

Files to create or edit:

- `src/lab5/envi_utils.py`
- `viewer.py`

Functions to implement:

- `find_hdr_files(directory)`
- `load_envi_image(hdr_path)`
- `parse_wavelengths(metadata)`
- `get_ignore_value(metadata)`
- `parse_map_info(metadata)`
- `pixel_to_map(row, col, map_info)`
- `find_nearest_band(wavelengths, target_nm, invalid_mask=None)`
- `build_invalid_band_mask(metadata)`
- `read_rgb(img, band_triplet, ignore_value)`
- `read_pixel_spectrum(img, row, col, ignore_value, invalid_mask=None)`
- `read_roi_cube(img, row_min, row_max, col_min, col_max)`
- `summarize_roi_spectra(roi_cube, ignore_value, invalid_mask=None)`

Important implementation details:

- Detect invalid bands from ENVI metadata if band names are marked with `*`.
- Apply `data ignore value` masking.
- Convert pixel coordinates to map coordinates using `map info`.
- Keep memory use bounded:
  - full-scene load is not allowed
  - ROI reads are allowed

Tasks:

- [x] P1.1 Extract the ENVI helper logic out of `viewer.py`.
- [x] P1.2 Add map-coordinate support.
- [x] P1.3 Add invalid-band masking support.
- [x] P1.4 Add ROI summary support returning:
  - mean spectrum
  - standard deviation spectrum
  - valid pixel count per band
  - total valid pixel count
- [x] P1.5 Update `viewer.py` imports to use the shared module.

Done when:

- `viewer.py` depends on `src/lab5/envi_utils.py` for all reusable ENVI logic.
- The utility layer can serve both single-pixel and ROI workflows.

Verification:

- `python -m compileall viewer.py src/lab5`
- manual smoke check: `python viewer.py`

P1 completion notes:

- added `src/lab5/envi_utils.py` for shared ENVI I/O and metadata handling
- moved reusable file discovery, wavelength parsing, ignore-value handling, RGB
  loading, and pixel-spectrum reading out of `viewer.py`
- added map-info parsing, pixel-to-map conversion, invalid-band masking, nearest
  wavelength lookup, ROI reading, and ROI spectrum summarization
- updated `viewer.py` to import and use the shared ENVI utility layer
- `compileall` passed
- real-data utility harness passed on
  `data/images/221000_Odra_HS_Blok_A_008_VS_join_atm.hdr`
- GUI smoke check was inconclusive in the sandbox because Tk/Matplotlib startup
  did not exit cleanly after cache initialization

## Phase P2. Spectral-Library Data Model And Rebuild Logic

Goal:

- Define the exact file format for saved samples and the logic that rebuilds
  class summaries from raw exports.

Files to create or edit:

- `src/lab5/spectral_library.py`
- `scripts/rebuild_spectral_library.py`

Data model to implement:

- `data/spectral_library/catalog.csv`
- `data/spectral_library/raw/<class_name>/<sample_id>.csv`
- `data/spectral_library/summary/<class_name>.csv`
- optional:
  - `data/spectral_library/raw_pixels/<class_name>/<sample_id>.csv`

Catalog columns:

- `sample_id`
- `class_name`
- `scene_id`
- `geometry_type`
- `row_min`
- `row_max`
- `col_min`
- `col_max`
- `row_center`
- `col_center`
- `map_x`
- `map_y`
- `pixel_count`
- `wavelength_count`
- `source_file`
- `exported_at`
- `notes`

Raw sample columns:

- `wavelength_nm`
- `mean_value`
- `std_value`
- `valid_pixel_count`

Functions to implement:

- `ensure_library_dirs(base_dir)`
- `make_sample_id(scene_id, class_name)`
- `write_roi_sample(...)`
- `append_catalog_row(...)`
- `load_catalog(...)`
- `rebuild_class_summaries(...)`

Tasks:

- [x] P2.1 Implement the library schema and directory creation.
- [x] P2.2 Implement raw ROI sample export format.
- [x] P2.3 Implement catalog append logic.
- [x] P2.4 Implement summary rebuild logic:
  - align by wavelength
  - compute mean and standard deviation by class
- [x] P2.5 Create `scripts/rebuild_spectral_library.py`.

Done when:

- A saved ROI sample can be rebuilt into class summaries without manual edits.
- Catalog and summary files are deterministic.

Verification:

- `python -m compileall scripts/rebuild_spectral_library.py src/lab5`
- create one dummy export through a small local test harness or from the viewer

P2 completion notes:

- added `src/lab5/spectral_library.py`
- added `scripts/rebuild_spectral_library.py`
- implemented catalog, raw-sample, and summary file schemas
- implemented deterministic class-summary rebuild by sorted class and sorted
  wavelength
- raw sample files now store:
  - `wavelength_nm`
  - `mean_value`
  - `std_value`
  - `valid_pixel_count`
- catalog rows now store the full P2 metadata set, including relative
  `source_file` paths
- `python3 -m compileall scripts/rebuild_spectral_library.py src/lab5` passed
- a `python3` real-data harness created temporary ROI samples from the airborne
  scene, appended catalog rows, ran the rebuild script, and produced class
  summaries for `water` and `forest`

## Phase P3. Viewer ROI Workflow

Goal:

- Upgrade the viewer from pixel-only inspection to a usable ROI-based library
  collection tool.

Files to create or edit:

- `viewer.py`
- `README.md`

Features to add:

- rectangle ROI selection on the RGB panel
- visible ROI overlay
- ROI statistics panel or status message
- export of ROI average sample with class label and notes
- keep existing pixel spectrum click workflow intact

Exact interaction design:

- left-click on image:
  - keep current pixel inspect behavior
- click-drag on image:
  - create rectangle ROI
- export actions:
  - `Export pixel spectrum to CSV...`
  - `Export ROI sample...`

Minimum ROI export metadata:

- scene id
- row and column bounds
- center pixel
- map coordinates of center pixel
- pixel count
- class label
- notes

Tasks:

- [x] P3.1 Add rectangle ROI selection state to `viewer.py`.
- [x] P3.2 Add ROI overlay drawing.
- [x] P3.3 Compute ROI summary through `src/lab5/envi_utils.py`.
- [x] P3.4 Add export flow wired to `src/lab5/spectral_library.py`.
- [x] P3.5 Rename the existing pixel export button so pixel and ROI exports are
  clearly separate.
- [x] P3.6 Update the viewer status text to show current pixel or ROI state.
- [x] P3.7 Document the ROI workflow in `README.md`.

Done when:

- The viewer supports both pixel inspection and ROI export.
- Exported ROI samples land in the spectral-library structure without manual
  moving or renaming.

Verification:

- `python viewer.py`
- manual test:
  - open default scene
  - inspect one pixel
  - draw one ROI
  - export one ROI sample
  - confirm raw sample and catalog entry were created

P3 completion notes:

- added rectangle ROI selection state and mouse press/move/release handling to
  `viewer.py`
- ROI rectangles are now drawn on the RGB panel, including drag-preview bounds
- the right-hand plot now supports ROI mean spectra with standard-deviation
  shading and ROI pixel statistics
- added a viewer-level ROI export flow that writes directly into
  `data/spectral_library/`
- renamed the pixel export button to `Export pixel spectrum to CSV...`
- added a new `Export ROI sample...` button
- updated status text for both pixel and ROI selections
- documented the ROI workflow in `README.md`
- `python3 -m compileall viewer.py src/lab5` passed
- a `python3` real-data harness used `viewer.export_roi_sample_to_library(...)`
  to create a raw ROI sample plus a catalog row in a temporary spectral-library
  directory
- a full interactive `python3 viewer.py` manual GUI test was not completed in
  the sandbox

## Phase P4. Airborne Index Pipeline

Goal:

- Implement the reusable water-quality and visualization logic for airborne
  hyperspectral scenes.

Files to create or edit:

- `src/lab5/indices.py`
- `notebooks/water_quality_analysis.ipynb`

Functions to implement in `src/lab5/indices.py`:

- `safe_ratio(numerator, denominator)`
- `band_by_wavelength(cube_or_reader, wavelengths, target_nm, invalid_mask=None)`
- `chlorophyll_red_edge_peak(r665, r705, r740)`
- `ndci(r665, r705)`
- `doc_proxy_green_red(r560, r665)`
- `ndti(r560, r665)`
- `stretch_composite(rgb)`

Notebook sections to implement in this phase:

- scene selection and metadata summary
- natural RGB
- false-color composites
- airborne index maps
- quick histogram and descriptive stats per index

Tasks:

- [x] P4.1 Implement the index functions.
- [x] P4.2 Add airborne band-selection logic by nearest wavelength.
- [x] P4.3 Add false-color composite helpers.
- [x] P4.4 Build the airborne-only notebook sections.
- [x] P4.5 Save selected airborne outputs into `data/outputs/airborne/<scene_id>/`
  if notebook code writes files.

Done when:

- The notebook can run the airborne-only workflow for one scene.
- The index formulas are reusable from code, not notebook-only.

Verification:

- `python -m compileall src/lab5`
- execute the airborne-only notebook cells manually

P4 completion notes:

- added `src/lab5/indices.py`
- implemented:
  - `safe_ratio(...)`
  - `band_by_wavelength(...)`
  - `chlorophyll_red_edge_peak(...)`
  - `ndci(...)`
  - `doc_proxy_green_red(...)`
  - `ndti(...)`
  - `composite_by_wavelengths(...)`
  - `stretch_composite(...)`
- added airborne nearest-wavelength band selection through
  `src.lab5.envi_utils.find_nearest_band(...)`
- created `notebooks/water_quality_analysis.ipynb`
- the notebook now includes airborne-only sections for:
  - scene selection and metadata summary
  - natural RGB
  - false-color composites
  - airborne index maps
  - histogram and descriptive statistics
- no notebook files are written to `data/outputs/airborne/<scene_id>/` yet, so
  P4.5 is satisfied by non-use rather than file generation
- `python3 -m compileall src/lab5` passed
- a `python3` full-scene airborne harness parsed the notebook JSON and executed
  the same composite and index workflow used by the notebook

## Phase P5. Sentinel-2 Acquisition Pipeline

Goal:

- Retrieve the closest usable Sentinel-2 scene near `2025-06-17`, cache the
  required bands locally, and expose them in a reusable loader.

Files to create or edit:

- `src/lab5/sentinel2.py`
- `notebooks/water_quality_analysis.ipynb`

Execution choice:

- Use Planetary Computer STAC because it matches the pattern already used in
  `../lab_2/aral_lab_2.ipynb`.
- If this is blocked in practice, only then fall back to another STAC source.

Functions to implement:

- `open_catalog()`
- `scene_bbox_wgs84_from_envi(scene_metadata)`
- `search_s2_items(bbox, target_date, day_window, cloud_limit)`
- `score_item(item, target_date)`
- `select_best_item(items)`
- `get_asset_href(item, band_name)`
- `reproject_to_grid(href, target_crs, target_transform, width, height)`
- `download_band(item, band_name, save_dir, target_grid)`
- `load_local_stack(band_files)`

Selection rule to implement:

1. search within `target_date +/- 5 days`
2. filter to low-cloud items
3. prefer smallest absolute day difference to `2025-06-17`
4. break ties with lower `eo:cloud_cover`
5. use one tile if possible before attempting mosaics

Required Sentinel-2 bands:

- `B03`
- `B04`
- `B05`
- `B06`
- `B08`
- `B11`

Tasks:

- [x] P5.1 Implement STAC client and search logic.
- [x] P5.2 Implement item scoring and selection.
- [x] P5.3 Implement per-band download and caching.
- [x] P5.4 Implement reproject-to-grid logic matching the airborne scene extent.
- [x] P5.5 Add the Sentinel-2 acquisition section to the notebook.

Done when:

- The code can search, pick, download, and reopen a Sentinel-2 scene for the
  airborne footprint.

Verification:

- dry run the search logic first
- then verify that expected local files appear in `data/sentinel2/`

P5 completion notes:

- added `src/lab5/sentinel2.py` with Planetary Computer STAC access, ENVI scene
  footprint-to-grid conversion, item scoring and selection, asset lookup,
  reprojection, per-band caching, and local stack loading
- extended `notebooks/water_quality_analysis.ipynb` with Sentinel-2 search,
  selection, cache, and local-load cells using the shared module
- `python3 -m compileall src/lab5` passed
- offline harness passed for grid derivation, item scoring, selection, cache
  naming, local stack loading, and notebook JSON validation
- live Planetary Computer verification passed for the airborne footprint and
  selected
  `S2C_MSIL2A_20250615T095051_R079_T33UYR_20250615T122401`
- verified cached local files under `data/sentinel2/` for `B03`, `B04`, `B05`,
  `B06`, `B08`, and `B11`, each reopened successfully on the airborne grid with
  shape `4300 x 2001`

## Phase P6. Cross-Sensor Alignment And Comparison

Goal:

- Put airborne and Sentinel-2 products onto a common basis for map comparison.

Files to create or edit:

- `src/lab5/sentinel2.py`
- `notebooks/water_quality_analysis.ipynb`

Alignment rules:

- comparison happens on a shared grid
- invalid pixels are masked out before statistics
- all comparison plots use the overlap mask only
- airborne-to-Sentinel spectral mapping uses nearest-band or explicit target
  wavelength selection, never hard-coded index positions

Tasks:

- [x] P6.1 Define the comparison grid and overlap mask.
- [x] P6.2 Resample airborne index products to the comparison grid.
- [x] P6.3 Compute Sentinel-2 index products on the same grid.
- [x] P6.4 Build comparison plots:
  - side-by-side maps
  - scatter plots
  - histograms
  - summary stats table
- [x] P6.5 Save comparison outputs under
  `data/outputs/comparison/<scene_id>/`

Done when:

- Airborne and Sentinel-2 maps can be compared over exactly the same pixels.

Verification:

- inspect overlap mask visually
- confirm comparison arrays have identical shape
- confirm statistics exclude masked pixels

P6 completion notes:

- extended `src/lab5/sentinel2.py` with local cache discovery, array-to-grid
  alignment, overlap-mask construction, pairwise comparison statistics, and
  comparison output directory helpers
- rebuilt `notebooks/water_quality_analysis.ipynb` to include Sentinel-2 cache
  fallback, shared-grid comparison setup, overlap-mask review, cross-sensor
  plots, and saved comparison outputs
- `python3 -m compileall src/lab5` passed
- notebook JSON validation passed
- end-to-end `python3` comparison harness passed against the cached Sentinel-2
  stack in `data/sentinel2/`
- verified identical comparison shapes of `4300 x 2001`, overlap mask size of
  `3174220` valid pixels, and comparison statistics columns:
  `index`, `valid_pixels`, `airborne_mean`, `sentinel_mean`,
  `bias_s2_minus_airborne`, `mae`, `rmse`, `pearson_r`
- verified saved outputs under
  `data/outputs/comparison/221000_Odra_HS_Blok_A_008_VS_join_atm/`:
  `comparison_maps.png`, `comparison_histograms.png`,
  `comparison_scatter.png`, `comparison_stats.csv`, and `overlap_mask.png`

## Phase P7. SAM And Calibration

Goal:

- Implement reusable SAM functions and a simple bandwise calibration workflow
  before applying SAM to Sentinel-2.

Files to create or edit:

- `src/lab5/sam.py`
- `notebooks/water_quality_analysis.ipynb`

Functions to implement:

- `spectral_angle(reference, spectra)`
- `spectral_angle_image(reference, image_stack)`
- `resample_airborne_spectrum_to_s2(reference_spectrum, airborne_wavelengths)`
- `fit_bandwise_linear_calibration(airborne_stack, sentinel_stack, mask)`
- `apply_bandwise_linear_calibration(sentinel_stack, calibration_params)`
- `sam_classification(angle_stack, threshold=None)`

Execution order inside this phase:

1. load class reference spectra from the spectral library summaries
2. resample airborne references to Sentinel-2-like bands
3. fit calibration over the overlap area
4. apply calibration to Sentinel-2 stack
5. run SAM for airborne
6. run SAM for calibrated Sentinel-2
7. compare angle maps and, if implemented, class maps

Tasks:

- [x] P7.1 Extract or rewrite SAM logic from `Lab_3`.
- [x] P7.2 Implement reference-spectrum resampling to Sentinel-2 bands.
- [x] P7.3 Implement bandwise gain-offset calibration.
- [x] P7.4 Implement calibrated Sentinel-2 SAM.
- [x] P7.5 Add SAM outputs to the notebook.

Done when:

- The spectral library can drive SAM on both airborne and Sentinel-2 data.
- Calibration happens before Sentinel-2 SAM.

Verification:

- `python -m compileall src/lab5`
- manual notebook run of the SAM section

P7 completion notes:

- added `src/lab5/sam.py` with spectral-library summary loading, reference
  alignment to target wavelength grids, Sentinel-2 reference resampling,
  chunked spectral-angle computation, bandwise calibration, and SAM
  classification helpers
- extended `notebooks/water_quality_analysis.ipynb` with spectral-library
  summary loading, calibration table export, resampled reference export,
  airborne SAM, calibrated Sentinel-2 SAM, per-class angle comparisons, and SAM
  class-map outputs
- `python3 -m compileall src/lab5` passed
- notebook JSON validation passed
- real-data `python3` harness passed using the cached Sentinel-2 stack and a
  temporary ROI-built spectral library from the airborne scene
- verified temporary class-summary rebuilds for `water`, `green vegetation`, and
  `forest`, calibration parameters for `B03`, `B04`, `B05`, `B06`, `B08`, and
  `B11`, and successful airborne plus Sentinel-2 SAM class assignments over a
  valid overlap subset
- verified harness output files `calibration_params.csv` and
  `sam_angle_stats.csv`
- the repo notebook SAM section is wired to `data/spectral_library/summary/`
  and will run once that library is populated; the current repo library is still
  empty

## Phase P8. Final Notebook And Documentation Polish

Goal:

- Make the project understandable and runnable end to end.

Files to create or edit:

- `notebooks/water_quality_analysis.ipynb`
- `README.md`
- `TODO.md` if useful for cross-reference only

Tasks:

- [x] P8.1 Ensure the notebook order is coherent and fully narrative.
- [x] P8.2 Add short markdown explanations before each major section.
- [x] P8.3 Document how to run the viewer and how to collect ROI samples.
- [x] P8.4 Document how to rebuild the spectral library.
- [x] P8.5 Document how Sentinel-2 scene selection works.
- [x] P8.6 Document known limitations:
  - DOC is a proxy, not a calibrated concentration
  - rectangle ROIs only in v1
  - single-scene airborne workflow in v1

Done when:

- A reviewer can understand the workflow without reading the source first.

Verification:

- open notebook and README side by side and check that all paths and commands
  are accurate

P8 completion notes:

- rewrote the notebook intro cell with a clear objective statement and numbered
  pipeline overview
- enhanced all 10 section markdown cells with explanations of why each step
  matters, what the indices measure, band-mapping tables, and algorithm
  descriptions
- added a Summary cell at the end of the notebook recapping all pipeline stages
- rewrote `README.md` with comprehensive documentation covering:
  - project layout tree
  - step-by-step viewer and ROI collection workflow (P8.3)
  - spectral library rebuild instructions (P8.4)
  - Sentinel-2 scene selection algorithm description (P8.5)
  - known limitations section (P8.6)
  - notebook execution instructions
- all P8 checkboxes marked complete in this file

## Phase P9. Algae Detection Extension

Goal:

- Add algae-specific spectral indices and update the notebook for bloom
  detection analysis. Extend the spectral library with algae and sediment
  classes.

Rationale:

- NDCI (already exists) is the primary bridge index for chlorophyll-a between
  airborne and Sentinel-2.
- NDRE adds sensitivity to dense blooms where NDCI saturates.
- FAI detects floating algae mats and surface scums via the NIR peak.
- NDWI provides a water/land mask to focus algae analysis on water pixels only.
- FLH and MCI exploit narrow-band chlorophyll fluorescence features available
  only in the airborne hyperspectral data.

Files to create or edit:

- `src/lab5/indices.py`
- `notebooks/water_quality_analysis.ipynb`
- `README.md`
- `EXECUTION_PLAN.md`
- `IMPLEMENTATION_PLAN.md`

Tasks:

- [x] P9.1 Add `ndre()`, `fai()`, `ndwi()` functions to `indices.py`.
- [x] P9.2 Add `flh()`, `mci()` (airborne-only) functions to `indices.py`.
- [x] P9.3 Update notebook imports cell to include new index functions.
- [x] P9.4 Expand airborne index computation cell:
  - extract bands at 681, 709, 753 nm
  - compute all 9 indices (4 existing + 5 new)
  - expand subplot grid from 2×2 to 3×3
- [x] P9.5 Add an algae detection analysis section to the notebook:
  - NDWI water mask
  - water-only statistics for NDCI, NDRE, FAI, FLH, MCI
  - water-masked index maps
- [x] P9.6 Expand cross-sensor comparison with NDRE, FAI, NDWI:
  - compute Sentinel-2 versions of the three new cross-sensor indices
  - add them to the comparison plots and stats table
  - FLH and MCI are excluded from comparison (airborne only)
- [ ] P9.7 Collect algae and sediment ROI samples using the viewer.
- [ ] P9.8 Rebuild spectral library with new classes.
- [x] P9.9 Update README with algae detection documentation.
- [x] P9.10 Update notebook markdown cells with algae context.

New spectral library classes to collect:

- `algae` — ROIs over visibly green or turbid water with elevated chlorophyll
  signatures
- `sediment` — ROIs over water with high suspended sediment but no algal bloom
  signatures (to help SAM distinguish turbidity from algae)

No new Sentinel-2 bands required:

- NDRE uses B05 (705 nm) + B08 (842 nm) — already downloaded
- FAI uses B04 (665 nm) + B08 (842 nm) + B11 (1610 nm) — already downloaded
- NDWI uses B03 (560 nm) + B08 (842 nm) — already downloaded

Done when:

- All 5 new index functions compile and pass basic tests.
- The notebook computes 9 airborne indices and 7 cross-sensor indices.
- The algae detection section uses NDWI to mask water and reports water-only
  statistics.
- The spectral library includes algae and sediment classes.
- SAM classification automatically picks up the new classes.

Verification:

- `python -m compileall src/lab5`
- execute the full notebook end to end
- confirm the new index maps, water mask, and comparison outputs render
  correctly

## Recommended Implementation Order By Turn

When executing with Codex, the safest turn-by-turn order is:

1. `P0`
2. `P1`
3. `P2`
4. `P3`
5. `P4`
6. `P5`
7. `P6`
8. `P7`
9. `P8`
10. `P9`

This order is intentional:

- viewer ROI export depends on the spectral-library schema
- notebook work depends on reusable index and I/O code
- Sentinel comparison depends on the acquisition pipeline
- SAM depends on both the library and the aligned cross-sensor products

## First Implementation Turn

When we start implementation, the first turn should do only this:

- complete `P0`
- start `P1`
- stop after the shared ENVI utility layer compiles and `viewer.py` still opens

That keeps the first code change bounded and prevents the viewer, notebook, and
satellite work from diverging too early.
