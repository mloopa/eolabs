# SAR Lab 7 Execution Plan

## Objective

Recreate the pre-failure deformation analysis of the June 24, 2017 Xinmo
landslide using a LiCSAR interferogram stack and MintPy SBAS processing, then
compare coherence loss, Sentinel-1 VV change, and Sentinel-2 optical changes.
The final deliverable is a PDF report containing methods, figures, results,
and discussion.

## Working Conventions

- Run terminal commands from `SAR_Lab_7/` unless a step says otherwise.
- Activate the environment first:

  ```bash
  conda activate sar-lab-7
  ```

- Keep all lab files under this directory:
  - Raw and downloaded data: `data/`
  - Processing outputs: `output/`
  - Final figures: `figures/`
  - Report files: `report/`
- Event date: `2017-06-24`
- Initial study center: approximately `103.6506 E, 32.0661 N`
- Processing AOI: `103.62, 32.04, 103.68, 32.09`

## Phase 1: Environment and Project Setup

- [x] Read `SAR_Lab_7.pdf` and identify all SAR, optical, and reporting tasks.
- [x] Create the `sar-lab-7` Conda environment with Python 3.11.
- [x] Install and verify MintPy and the required geospatial packages.
- [x] Resolve the MintPy/NumPy compatibility issue by using NumPy 1.26.4.
- [x] Create the lab directory structure:
  - `data/licsar_pre_event/`
  - `data/event_coherence/`
  - `output/mintpy/`
  - `output/earth_engine/`
  - `figures/`
  - `report/`
- [x] Install or verify notebook packages:

  ```bash
  python -c "import jupyterlab, ipykernel, ee, geemap; print('Notebook environment ready')"
  ```

- [x] Register the environment as a Jupyter kernel if it is not already listed:

  ```bash
  python -m ipykernel install --user \
    --name sar-lab-7 \
    --display-name "Python (SAR Lab 7)"
  ```

## Phase 2: LiCSAR Pre-Event Stack

- [x] Test the LiCSAR downloader.
- [x] Inspect the pre-event pair selection with a dry run.
- [x] Download unwrapped phase and coherence GeoTIFFs.
- [x] Expand the maximum temporal baseline from 72 to 108 days to connect the
  interferogram network.
- [x] Confirm that 127 interferogram directories are present.
- [x] Download LiCSAR metadata and geometry rasters.
- [x] Correct the LiCSAR `baselines` parser in `licsar_to_mintpy_h5.py`.
- [x] Preserve the unsuccessful disconnected run in `output/mintpy_failed_72d/`.

Important input locations:

```text
data/licsar_pre_event/interferograms/
data/licsar_pre_event/metadata/
data/licsar_pre_event/selected_ifg_pairs.csv
```

## Phase 3: MintPy Stack and Initial SBAS Run

- [x] Convert the LiCSAR products to:
  - `output/mintpy/inputs/ifgramStack.h5`
  - `output/mintpy/inputs/geometryGeo.h5`
- [x] Subset the data to the AOI.
- [x] Verify the interferogram network and perpendicular baselines.
- [x] Run MintPy from `modify_network` with an automatic reference point.
- [x] Produce the initial MintPy outputs:
  - `output/mintpy/timeseries.h5`
  - `output/mintpy/velocity.h5`
  - `output/mintpy/temporalCoherence.h5`
  - `output/mintpy/pic/network.pdf`
  - `output/mintpy/pic/velocity.png`

## Phase 4: MintPy Quality Review and Reference Selection

- [x] Review the network plot:

  ```bash
  cd output/mintpy
  plot_network.py inputs/ifgramStack.h5
  ```

  Confirm that the network is connected and record the acquisition period,
  number of acquisitions, number of interferograms, and baseline ranges.

  ```text
  Network status: connected
  Acquisition period: 2014-10-09 to 2017-06-07
  Number of acquisitions: 42
  Number of interferograms: 127
  Perpendicular baseline range: -178 m to +189 m
  ```

- [x] Inspect the initial velocity map:

  ```bash
  view.py velocity.h5 velocity
  ```

- [x] Open the interactive time-series viewer:

  ```bash
  tsview.py timeseries.h5
  ```

- [x] Locate pixels within the landslide and inspect their displacement
  histories.
- [x] Determine whether the pre-failure series contains acceleration, a
  seasonal signal, or both.

  ```text
  Landslide-pixel inspection: example Y/X = 29, 29 at lat/lon
  32.06133, 103.64883 shows a negative trend of about
  -1.07 +/- 0.08 cm/year in tsview.

  Interpretation: the inspected series shows a broadly negative pre-event
  trend with scatter. A clear acceleration signal is not evident from the
  current time series alone; seasonal/atmospheric scatter remains a plausible
  contributor.
  ```

- [x] Select a stable, coherent reference pixel near but outside the landslide.
- [x] Record the chosen reference pixel and its location here:

  ```text
  Reference Y/X: 28, 31
  Reference latitude/longitude: 32.06233, 103.65083
  Selection rationale: selected from the MintPy velocity/time-series review as
  a nearby stable point outside the interpreted landslide area. The pixel has
  high temporal coherence (0.904), near-zero velocity before manual
  rereferencing (0.00025 m/year, approximately 0.025 cm/year), and matches the
  supplied stable-pixel coordinates.
  ```

- [x] Update `output/mintpy/smallbaselineApp_licsar.cfg`:

  ```ini
  mintpy.reference.yx = 28,31
  ```

- [x] Rerun MintPy from the reference-point step:

  ```bash
  smallbaselineApp.py smallbaselineApp_licsar.cfg --start reference_point
  ```

  ```text
  Rerun completed normally on 2026-06-30 after applying a local MintPy 1.6.2
  compatibility patch in ifgram_inversion.py. The failure was caused by
  assigning a length-1 inversion-quality array into a scalar pixel slot during
  pixel-by-pixel inversion. The patch converts one-element inv_quality arrays
  to a scalar before returning from estimate_timeseries().

  Regenerated outputs:
  - output/mintpy/timeseries.h5
  - output/mintpy/velocity.h5
  - output/mintpy/temporalCoherence.h5
  - output/mintpy/maskTempCoh.h5

  Verified output reference: Y/X = 28,31; lat/lon =
  32.06233405524, 103.65083223745. The reference-pixel velocity is 0.0 m/year
  and temporal coherence is set to 1.0 after rereferencing.
  ```

- [x] Reinspect `velocity.h5` and `timeseries.h5`.
- [x] Save final MintPy figures in `figures/`:
  - Velocity map with landslide and reference locations
  - Landslide-pixel displacement time series
  - Reference-pixel context or coherence map
  - Interferogram network

  ```text
  Final Phase 4 figures saved:
  - figures/mintpy_velocity_manual_reference.png
  - figures/mintpy_landslide_timeseries_manual_reference.png
  - figures/mintpy_reference_context_temporal_coherence.png
  - figures/mintpy_network_overview.png
  ```

- [x] Compare the SBAS time series qualitatively with the study paper.

  ```text
  Qualitative comparison: the SBAS result supports pre-event motion at the
  landslide, but it does not resolve the late pre-failure acceleration shown in
  the published PS-based analysis. The representative landslide pixel has a
  broad negative trend of about -1.07 +/- 0.08 cm/year, while the LiCSAR stack
  ends on 2017-06-07, 17 days before failure.
  ```
- [x] Document the missing late acquisitions noted in the assignment and
  explain how they limit comparison with the published PS result.

  ```text
  Current LiCSAR/MintPy stack ends on 2017-06-07, which is before the
  2017-06-24 landslide. This prevents direct reproduction of the late
  pre-failure acceleration reported in the published PS-based analysis.
  ```

## Phase 5: Analysis Notebook

- [x] Create `SAR_Lab_7_analysis.ipynb`.
- [x] Use the `Python (SAR Lab 7)` kernel.
- [x] Add a reproducibility header containing:
  - Event name and date
  - AOI bounds
  - LiCSAR frame `062D_05831_131313`
  - Software versions
  - Input/output paths
- [x] Add notebook sections:
  1. Study area and assignment objectives
  2. LiCSAR network and MintPy processing
  3. Reference-point selection and SBAS result
  4. Event-spanning coherence
  5. Sentinel-1 VV change
  6. Sentinel-2 true color and landslide extent
  7. NDVI and BSI change
  8. Cross-method comparison
  9. Conclusions and limitations
- [x] Ensure notebook cells are executable from top to bottom.
- [x] Save reusable figures from the notebook into `figures/`.

## Phase 6: Event-Spanning LiCSAR Coherence

- [x] Identify a LiCSAR interferogram with one acquisition before and one after
  June 24, 2017.
- [x] Record the selected pair:

  ```text
  Event-spanning pair: 20170607_20170725
  Temporal baseline: 48 days
  Reason for selection: shortest available pair spanning the event, with
  acquisitions 17 days before and 31 days after June 24, 2017.
  ```

- [x] Download its `.geo.cc.tif` coherence file into
  `data/event_coherence/`.
- [x] Crop the coherence raster to the same AOI.
- [x] Plot coherence with a fixed `0-1` scale and mark the landslide location.
- [x] Compare coherence inside the mapped landslide with nearby stable terrain.
- [x] Save the final coherence figure in `figures/`.
- [x] Answer: Is coherence loss useful for mapping this landslide?
- [x] Discuss temporal decorrelation, vegetation, terrain geometry, and the
  selected acquisition interval as limitations.

Phase 6 result:

```text
Landslide-centered mean coherence: 0.046
Manual-reference mean coherence: 0.053
Landslide/reference mean ratio: 0.87
Interpretation: coherence loss is weak evidence for mapping this landslide in
the current event-spanning pair because both the landslide-centered sample and
manual-reference sample have very low coherence.
```

## Phase 7: Google Earth Engine Initialization

- [x] Configure the notebook to receive the Earth Engine project ID without
  storing or committing credentials:

  ```bash
  read -s "EARTH_ENGINE_PROJECT?Google Earth Engine project ID: "
  export EARTH_ENGINE_PROJECT
  echo
  ```

  The notebook reads `EARTH_ENGINE_PROJECT`, never prints it, and does not
  write it into the notebook or repository. `.env` files are ignored by Git.

- [x] Define the common AOI in Earth Engine.
- [x] Add a secure notebook cell that displays the AOI and landslide location
  on an interactive map after private initialization.
- [x] Verify that Earth Engine image metadata and map layers load correctly.

  Run privately from `SAR_Lab_7/`:

  ```bash
  python verify_gee_setup.py
  jupyter lab
  ```

  Then run the secure Earth Engine initialization cell. It should report a
  Sentinel-1 metadata count and display the AOI map without printing the
  project ID. After closing Jupyter, remove the variable from the shell:

  ```bash
  unset EARTH_ENGINE_PROJECT
  ```

## Phase 8: Sentinel-1 GRD VV Change

- Secure execution command:

  ```bash
  python run_sentinel1_vv_analysis.py
  ```

  Run it from the same terminal session that has `EARTH_ENGINE_PROJECT`
  exported, or use a PyCharm Run Configuration containing that environment
  variable. Alternatively, start Jupyter from that terminal and use **Run All**;
  the notebook runs the exporter automatically when Phase 8 outputs are absent.

- [x] Load `COPERNICUS/S1_GRD`.
- [x] Filter to:
  - The common AOI
  - `IW` instrument mode
  - `VV` polarization
  - Descending orbit
  - A consistent relative orbit
- [x] List all candidate acquisitions immediately before and after
  June 24, 2017.
- [x] Select comparable pre-event and post-event acquisitions or short
  composites.
- [x] Record image IDs, dates, orbit direction, and relative orbit number.
- [x] Calculate VV change in dB:

  ```text
  VV change = post-event VV - pre-event VV
  ```

- [x] Display pre-event VV, post-event VV, and VV change using consistent
  visualization ranges.
- [x] Compute summary statistics over the landslide and a stable comparison
  area.
- [x] Export the VV-change raster or final map to `output/earth_engine/`.
- [x] Save a report-ready VV-change figure in `figures/`.
- [x] Answer: Is VV change better, worse, or similar to coherence loss for
  mapping the landslide?

Phase 8 result:

```text
Selected images: Sentinel-1A descending, relative orbit 62
Pre-event date: 2017-06-19 (4 days before event)
Pre-event image: S1A_IW_GRDH_1SDV_20170619T230410_20170619T230435_017109_01C859_52F1
Post-event date: 2017-07-13 (19 days after event)
Post-event image: S1A_IW_GRDH_1SDV_20170713T230411_20170713T230436_017459_01D2E8_2197
Landslide-centered mean VV change: -0.49 dB
Stable-reference mean VV change: -0.89 dB
Local contrast (landslide - stable): +0.40 dB
Interpretation: VV change is worse than coherence loss for mapping the
landslide in this comparison because the local contrast is weak and the map is
strongly affected by speckle and terrain-related variation.
```

## Phase 9: Sentinel-2 Optical Analysis

- Secure execution command:

  ```bash
  python run_sentinel2_optical_analysis.py
  ```

  Run it from the same terminal session that has `EARTH_ENGINE_PROJECT`
  exported, or use the existing PyCharm Run Configuration containing that
  environment variable. The project ID is read only from the environment and
  is not printed or written to any lab file.

  The runner first tries `COPERNICUS/S2_SR_HARMONIZED`. If no 2017
  surface-reflectance imagery is available for the AOI, it falls back to
  `COPERNICUS/S2_HARMONIZED` Level-1C top-of-atmosphere imagery. This fallback
  is needed because early Sentinel-2 Level-2A coverage is not global. The
  runner scores candidates by clear AOI coverage, selects a pre/post pair on
  one MGRS tile, applies SCL+QA60 masking for SR or QA60 masking for TOA, and
  classifies the optical scar within 1.8 km of the landslide using a
  post-event brightness and NDVI-loss rule:

  ```text
  post brightness >= 0.115
  post NDVI <= 0.0
  NDVI change <= -0.10
  ```

  The largest connected component is retained as the affected-area estimate.
  For the TOA fallback, the script does not discard scenes by global cloud
  percentage because the scene-level value can be misleading for this small
  AOI. Pair selection uses AOI clear-fraction thresholds
  `0.85, 0.70, 0.50, 0.30, 0.20`.

- [x] Load `COPERNICUS/S2_SR_HARMONIZED`.
- [x] Inspect pre-event and post-event image availability.
- [x] Select cloud-free or minimally cloudy imagery covering the landslide.
- [x] Record image IDs, acquisition dates, and cloud conditions.
- [x] Apply a suitable cloud and cloud-shadow mask.
- [x] Create pre-event and post-event true-color images.
- [x] Delineate or classify the landslide-affected area.
- [x] Calculate the affected area in square kilometers or hectares.
- [x] Save the landslide extent as a reusable geometry or vector export.
- [x] Calculate pre-event and post-event NDVI:

  ```text
  NDVI = (B8 - B4) / (B8 + B4)
  ```

- [x] Calculate NDVI change and quantify vegetation loss.
- [x] Calculate pre-event and post-event BSI:

  ```text
  BSI = ((B11 + B4) - (B8 + B2)) /
        ((B11 + B4) + (B8 + B2))
  ```

- [x] Calculate BSI change and assess increased exposed soil or debris.
- [x] Export useful rasters or tables to `output/earth_engine/`.
- [x] Save report-ready figures in `figures/`:
  - Pre/post true color
  - Landslide extent
  - NDVI before, after, and change
  - BSI before, after, and change

Phase 9 result:

```text
Selected optical collection: COPERNICUS/S2_HARMONIZED
Processing level: Sentinel-2 Level-1C top-of-atmosphere reflectance
Reason for fallback: SR had no usable pre/post pair for the AOI.
Pre-event image: S2A 2017-02-19, tile 48SUA, AOI clear fraction 1.000
Post-event image: S2B 2017-08-13, tile 48SUA, AOI clear fraction 1.000
Cloud mask: QA60 cloud/cirrus bits only; no SCL in Level-1C TOA fallback.

Affected-area mask:
- Rule: post brightness >= 0.115, post NDVI <= 0.0, NDVI change <= -0.10
- Post brightness computed as the mean of post-event red, green, and blue TOA
  reflectance.
- Final extent is the largest connected component inside the 1.8 km
  landslide-centered search radius.
- Estimated affected area: 80.04 ha, or 0.8004 km2.

Affected-area summary statistics:
- Affected-area mean NDVI change: -0.229
- Stable-reference mean NDVI change: +0.333
- Affected-area mean BSI change: -0.108
- Stable-reference mean BSI change: -0.221

Generated outputs:
- output/earth_engine/sentinel2_candidates.csv
- output/earth_engine/sentinel2_candidates_sr.csv
- output/earth_engine/sentinel2_candidates_toa.csv
- output/earth_engine/sentinel2_selection.json
- output/earth_engine/sentinel2_pre_post_indices.tif
- output/earth_engine/sentinel2_stats.csv
- output/earth_engine/sentinel2_landslide_extent.geojson
- figures/sentinel2_true_color_extent.png
- figures/sentinel2_ndvi_change.png
- figures/sentinel2_bsi_change.png

Refinement note: the initial strict NDVI-loss plus BSI-increase rule
under-detected the visible scar. The final mask therefore uses the post-event
true-color brightness scar plus NDVI decrease, which better follows the
visible source and runout area in the Sentinel-2 panels.
```

Expected Phase 9 outputs:

```text
output/earth_engine/sentinel2_candidates.csv
output/earth_engine/sentinel2_selection.json
output/earth_engine/sentinel2_pre_post_indices.tif
output/earth_engine/sentinel2_stats.csv
output/earth_engine/sentinel2_landslide_extent.geojson
figures/sentinel2_true_color_extent.png
figures/sentinel2_ndvi_change.png
figures/sentinel2_bsi_change.png
```

Note: Sentinel-2 is multispectral, although the assignment uses the term
"hyperspectral index."

## Phase 10: Interpretation and Method Comparison

- [x] Summarize the pre-failure SBAS deformation signal.
- [x] Explain the effect of automatic versus manually chosen reference pixels.
- [x] Compare the result with the published PS-based analysis.
- [x] Compare spatial mapping performance for:
  - LiCSAR coherence loss
  - Sentinel-1 VV change
  - Sentinel-2 true color
  - NDVI change
  - BSI change
- [x] Discuss which method most clearly maps the landslide and why.
- [x] Discuss uncertainty and limitations:
  - Sparse LiCSAR acquisition coverage
  - Atmospheric and seasonal signals
  - Reference-point sensitivity
  - Low coherence and vegetation
  - Layover and shadow in steep terrain
  - Sentinel-1 speckle
  - Sentinel-2 cloud contamination
  - Different acquisition dates and spatial resolutions

Phase 10 result:

```text
Interpretation draft saved to:
- report/phase10_interpretation.md

SBAS result: the inspected landslide pixel shows a broad negative pre-failure
trend of about -1.07 +/- 0.08 cm/year. A clear late acceleration is not
resolved because the LiCSAR stack ends on 2017-06-07, before the 2017-06-24
failure.

Reference result: the manual reference point Y/X = 28,31 provides a local,
stable basis for relative displacement and reduces sensitivity to distant
automatic-reference behavior. The tradeoff is that all displacement is relative
to that selected reference pixel.

Mapping comparison: Sentinel-2 true color plus NDVI decrease provides the
clearest landslide extent. The refined optical mask estimates 80.04 ha
(0.8004 km2). Coherence loss is weak because both the landslide and manual
reference samples have very low coherence. Sentinel-1 VV change is also weak
because local contrast is small. BSI change is not reliable as a simple
positive-change threshold for this image pair.
```

## Phase 11: Final PDF Report

- [x] Draft the report in `report/`.
- [x] Include:
  1. Introduction and study objective
  2. Data and study area
  3. LiCSAR/MintPy SBAS methodology
  4. Reference-point selection
  5. Pre-failure time-series results
  6. Coherence analysis
  7. Sentinel-1 VV analysis
  8. Sentinel-2 and spectral-index analysis
  9. Comparison and discussion
  10. Conclusions
  11. References
- [x] Give every figure a number, caption, legend, units, date range, and data
  source.
- [x] Cite the supplied landslide paper, LiCSAR/COMET, MintPy, Sentinel-1,
  Sentinel-2, and Google Earth Engine.
- [x] Verify that statements about vegetation loss or affected area are
  supported by computed values.
- [x] Export the final report to:

  ```text
  report/SAR_Lab_7_report.pdf
  ```

- [x] Review the PDF for missing figures, unreadable text, incorrect units,
  and unsupported conclusions.
- [ ] Submit the final PDF.

Phase 11 result:

```text
Final report files:
- report/SAR_Lab_7_report.pdf
- report/SAR_Lab_7_report.md
- build_report_pdf.py

PDF QA:
- PDF renders to 7 pages.
- All 10 report figures are present with captions and data-source notes.
- Text extraction confirms current key values:
  - MintPy landslide trend: -1.07 +/- 0.08 cm/year
  - Event coherence means: 0.046 at landslide, 0.053 at manual reference
  - Sentinel-2 affected area: 80.04 ha / 0.8004 km2
- Rendered pages were visually inspected for clipping, unreadable text,
  incorrect units, and missing references.

Remaining manual action: upload/submit report/SAR_Lab_7_report.pdf.
```

## Immediate Next Actions

- [x] Verify notebook and Earth Engine packages.
- [x] Create `SAR_Lab_7_analysis.ipynb`.
- [x] Inspect `velocity.h5` and `timeseries.h5`.
- [x] Select and record the stable manual reference pixel.
- [x] Rerun MintPy from `reference_point`.
- [x] Begin the notebook with the MintPy network, velocity, and time-series
  figures before moving to Earth Engine.
