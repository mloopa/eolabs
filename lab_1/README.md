# EO Labs

Practical notebooks for exploring Earth Observation data with Python.

---

## Contents

| File | Description |
|---|---|
| `visualise_imagery.ipynb` | Quick side-by-side visualisation of VHR, Sentinel-2 and Landsat over Istanbul |
| `eo_lab_1.ipynb` | Guided lab: multi-resolution comparison, spectral indices (NDVI, NDWI), GeoTIFF export |
| `data/` | Local raster data (see structure below) |

### Data layout

```
data/
  vhr/
    istanbul_vhr.tiff          # Very high resolution RGB (~1 m), EPSG:32635
  stac_downloads/
    sentinel2/
      s2_B02.tif               # Blue  (10 m)
      s2_B03.tif               # Green (10 m)
      s2_B04.tif               # Red   (10 m)
      s2_B08.tif               # NIR   (10 m)
    landsat/
      landsat_blue.tif
      landsat_green.tif
      landsat_red.tif
      landsat_nir08.tif        # all bands 30 m
```

All rasters are in **EPSG:32635 (UTM zone 35N)**, covering Istanbul, Turkey.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mloopa/eolabs.git
cd eolabs
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `rasterio` requires GDAL. On macOS the easiest route is
> [Conda](https://conda.io) or [Miniforge](https://github.com/conda-forge/miniforge):
>
> ```bash
> conda create -n eolabs python=3.11
> conda activate eolabs
> conda install -c conda-forge rasterio numpy matplotlib jupyterlab
> ```

### 4. Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

Open `eo_lab_1.ipynb` to start the lab.

---

## Lab 1 overview

The notebook walks through:

1. Loading raster files with `rasterio` and inspecting metadata
2. Clipping all sensors to a common geographic extent
3. Visualising true colour composites at native resolution
4. Grayscale conversion and false colour (NIR/R/G)
5. Computing NDVI and comparing it across sensors
6. **Final challenge:** implement NDWI and export results to GeoTIFF for QGIS

Output files are written to `data/outputs/`.
