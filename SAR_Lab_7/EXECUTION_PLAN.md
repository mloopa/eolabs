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

- [ ] Review the network plot:

  ```bash
  cd output/mintpy
  plot_network.py inputs/ifgramStack.h5
  ```

  Confirm that the network is connected and record the acquisition period,
  number of acquisitions, number of interferograms, and baseline ranges.

- [ ] Inspect the initial velocity map:

  ```bash
  view.py velocity.h5 velocity
  ```

- [ ] Open the interactive time-series viewer:

  ```bash
  tsview.py timeseries.h5
  ```

- [ ] Locate pixels within the landslide and inspect their displacement
  histories.
- [ ] Determine whether the pre-failure series contains acceleration, a
  seasonal signal, or both.
- [ ] Select a stable, coherent reference pixel near but outside the landslide.
- [ ] Record the chosen reference pixel and its location here:

  ```text
  Reference Y/X:
  Reference latitude/longitude:
  Selection rationale:
  ```

- [ ] Update `output/mintpy/smallbaselineApp_licsar.cfg`:

  ```ini
  mintpy.reference.yx = Y, X
  ```

- [ ] Rerun MintPy from the reference-point step:

  ```bash
  smallbaselineApp.py smallbaselineApp_licsar.cfg --start reference_point
  ```

- [ ] Reinspect `velocity.h5` and `timeseries.h5`.
- [ ] Save final MintPy figures in `figures/`:
  - Velocity map with landslide and reference locations
  - Landslide-pixel displacement time series
  - Reference-pixel context or coherence map
  - Interferogram network
- [ ] Compare the SBAS time series qualitatively with the study paper.
- [ ] Document the missing late acquisitions noted in the assignment and
  explain how they limit comparison with the published PS result.

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
Stable-reference mean coherence: 0.198
Landslide/stable mean ratio: 0.23
Interpretation: coherence loss is useful supporting evidence, but not a
standalone landslide map because coherence is low across much of the AOI.
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
- [ ] Verify that Earth Engine image metadata and map layers load correctly.

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

- [ ] Load `COPERNICUS/S1_GRD`.
- [ ] Filter to:
  - The common AOI
  - `IW` instrument mode
  - `VV` polarization
  - Descending orbit
  - A consistent relative orbit
- [ ] List all candidate acquisitions immediately before and after
  June 24, 2017.
- [ ] Select comparable pre-event and post-event acquisitions or short
  composites.
- [ ] Record image IDs, dates, orbit direction, and relative orbit number.
- [ ] Calculate VV change in dB:

  ```text
  VV change = post-event VV - pre-event VV
  ```

- [ ] Display pre-event VV, post-event VV, and VV change using consistent
  visualization ranges.
- [ ] Compute summary statistics over the landslide and a stable comparison
  area.
- [ ] Export the VV-change raster or final map to `output/earth_engine/`.
- [ ] Save a report-ready VV-change figure in `figures/`.
- [ ] Answer: Is VV change better, worse, or similar to coherence loss for
  mapping the landslide?

## Phase 9: Sentinel-2 Optical Analysis

- [ ] Load `COPERNICUS/S2_SR_HARMONIZED`.
- [ ] Inspect pre-event and post-event image availability.
- [ ] Select cloud-free or minimally cloudy imagery covering the landslide.
- [ ] Record image IDs, acquisition dates, and cloud conditions.
- [ ] Apply a suitable cloud and cloud-shadow mask.
- [ ] Create pre-event and post-event true-color images.
- [ ] Delineate or classify the landslide-affected area.
- [ ] Calculate the affected area in square kilometers or hectares.
- [ ] Save the landslide extent as a reusable geometry or vector export.
- [ ] Calculate pre-event and post-event NDVI:

  ```text
  NDVI = (B8 - B4) / (B8 + B4)
  ```

- [ ] Calculate NDVI change and quantify vegetation loss.
- [ ] Calculate pre-event and post-event BSI:

  ```text
  BSI = ((B11 + B4) - (B8 + B2)) /
        ((B11 + B4) + (B8 + B2))
  ```

- [ ] Calculate BSI change and assess increased exposed soil or debris.
- [ ] Export useful rasters or tables to `output/earth_engine/`.
- [ ] Save report-ready figures in `figures/`:
  - Pre/post true color
  - Landslide extent
  - NDVI before, after, and change
  - BSI before, after, and change

Note: Sentinel-2 is multispectral, although the assignment uses the term
"hyperspectral index."

## Phase 10: Interpretation and Method Comparison

- [ ] Summarize the pre-failure SBAS deformation signal.
- [ ] Explain the effect of automatic versus manually chosen reference pixels.
- [ ] Compare the result with the published PS-based analysis.
- [ ] Compare spatial mapping performance for:
  - LiCSAR coherence loss
  - Sentinel-1 VV change
  - Sentinel-2 true color
  - NDVI change
  - BSI change
- [ ] Discuss which method most clearly maps the landslide and why.
- [ ] Discuss uncertainty and limitations:
  - Sparse LiCSAR acquisition coverage
  - Atmospheric and seasonal signals
  - Reference-point sensitivity
  - Low coherence and vegetation
  - Layover and shadow in steep terrain
  - Sentinel-1 speckle
  - Sentinel-2 cloud contamination
  - Different acquisition dates and spatial resolutions

## Phase 11: Final PDF Report

- [ ] Draft the report in `report/`.
- [ ] Include:
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
- [ ] Give every figure a number, caption, legend, units, date range, and data
  source.
- [ ] Cite the supplied landslide paper, LiCSAR/COMET, MintPy, Sentinel-1,
  Sentinel-2, and Google Earth Engine.
- [ ] Verify that statements about vegetation loss or affected area are
  supported by computed values.
- [ ] Export the final report to:

  ```text
  report/SAR_Lab_7_report.pdf
  ```

- [ ] Review the PDF for missing figures, unreadable text, incorrect units,
  and unsupported conclusions.
- [ ] Submit the final PDF.

## Immediate Next Actions

- [x] Verify notebook and Earth Engine packages.
- [x] Create `SAR_Lab_7_analysis.ipynb`.
- [x] Inspect `velocity.h5` and `timeseries.h5`.
- [ ] Select and record the stable manual reference pixel.
- [ ] Rerun MintPy from `reference_point`.
- [x] Begin the notebook with the MintPy network, velocity, and time-series
  figures before moving to Earth Engine.
