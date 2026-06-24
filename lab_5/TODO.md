# Lab 4. Hyperspectral Data and Water Quality Analysis

The aim of the project is to develop tools enabling the fusion of airborne data
(hyperspectral) and satellite data (multispectral) in order to build water
quality monitoring tools.

## Tasks To Complete

1. Use any LLM model to develop a Python tool for browsing the data cube. The
   tool should enable visualization of airborne data in RGB, allow the user to
   click a selected pixel to display its spectral signature, and export that
   signature to a CSV file.
2. Prepare spectral signatures for different types of land cover: water, green
   areas, forest, etc., and save them. These will serve as a spectral library.
3. Prepare a notebook that:
   - Displays false-color composites based on the data cube.
   - Calculates water quality indices: Chl-a, DOC, turbidity.
   - Downloads Sentinel-2 data acquired as close as possible to the acquisition
     date of the airborne data.
   - Calculates the indices from point 3b for Sentinel-2 data and compares the
     results.
4. Based on the notebook from `Lab_3`, prepare SAM. Consider how Sentinel-2
   data can be calibrated using the airborne data, and implement the solution.
