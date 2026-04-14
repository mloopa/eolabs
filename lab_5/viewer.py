#!/usr/bin/env python3
"""
Hyperspectral BSQ Viewer
------------------------
Browse ENVI/BSQ hyperspectral data cubes.
Click any pixel in the RGB preview to display its full spectral signature.
Export the selected spectrum to CSV.

Usage:
    python viewer.py [path/to/file.hdr]

If no path is given the tool searches data/images/ for .hdr files.
Requires Python 3.10+  (uses X | Y union type hints)
"""

import sys
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit(
        "The 'spectral' library is missing.\n"
        "Install it with:  pip install spectral"
    )

# ── Configuration ─────────────────────────────────────────────────────────────

# Default search directory (relative to this script)
DATA_DIR = Path(__file__).parent / "data" / "images"

# Fallback RGB band indices (0-based) when the header has no default_bands
FALLBACK_RGB = (30, 20, 10)

# ── ENVI header helpers ───────────────────────────────────────────────────────

def find_hdr_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.hdr"))


def parse_wavelengths(meta: dict) -> np.ndarray | None:
    """Return wavelength array from ENVI metadata, or None if absent."""
    wl = meta.get("wavelength")
    if wl:
        return np.array([float(w) for w in wl])
    return None


def get_rgb_bands(meta: dict) -> tuple[int, int, int]:
    """
    Read default_bands from the ENVI header.
    Header values are 1-based floats, so we subtract 1.
    Falls back to FALLBACK_RGB if the key is missing.
    """
    db = meta.get("default bands")
    if db and len(db) >= 3:
        return tuple(int(float(v)) - 1 for v in db[:3])
    return FALLBACK_RGB


def get_ignore_value(meta: dict) -> float | None:
    """Return the no-data / ignore value declared in the ENVI header."""
    raw = meta.get("data ignore value")
    if raw:
        try:
            return float(str(raw).strip())
        except ValueError:
            pass
    return None


# ── Image I/O ─────────────────────────────────────────────────────────────────

def load_image(hdr_path: Path):
    """
    Open an ENVI image via spectral (memory-mapped — no full load into RAM).
    The companion .bsq file is located automatically next to the .hdr.
    """
    return envi.open(str(hdr_path))


def read_rgb(img, r: int, g: int, b: int, ignore_value: float | None) -> np.ndarray:
    """
    Read three bands and return a float32 RGB array (values 0–1) for display.
    No-data pixels and negative values are masked before the stretch.
    Applies a per-channel 2–98 % percentile stretch.
    """
    # shape: (lines, samples, 3) — reads only 3 bands from disk
    rgb = img.read_bands([r, g, b]).astype(np.float32)

    if ignore_value is not None:
        rgb[rgb >= ignore_value] = np.nan
    rgb[rgb < 0] = np.nan

    for c in range(3):
        ch = rgb[:, :, c]
        p2, p98 = np.nanpercentile(ch, [2, 98])
        rgb[:, :, c] = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0, 1)

    return np.nan_to_num(rgb, nan=0.0)


def read_spectrum(img, row: int, col: int, ignore_value: float | None) -> np.ndarray:
    """
    Read the full spectrum (all bands) for a single pixel.
    BSQ layout makes this efficient: one seek per band.
    Bad / no-data values are replaced with NaN so the plot shows gaps.
    """
    spec = img.read_pixel(row, col).astype(np.float64)
    if ignore_value is not None:
        spec[spec >= ignore_value] = np.nan
    spec[spec < 0] = np.nan
    return spec


# ── Application ───────────────────────────────────────────────────────────────

class HyperspectralViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hyperspectral BSQ Viewer")
        self.root.geometry("1300x720")

        # state
        self.img = None
        self.wavelengths: np.ndarray | None = None
        self.ignore_value: float | None = None
        self.rgb_display: np.ndarray | None = None   # float32 (lines, samples, 3)
        self.spectrum: np.ndarray | None = None      # 1-D, last clicked pixel
        self.pixel_pos: tuple[int, int] | None = None

        self._build_ui()
        self._auto_load()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # top toolbar
        bar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="Open file…", command=self._open_file).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        tk.Button(bar, text="Export spectrum to CSV…", command=self._export_csv).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        self.status_var = tk.StringVar(value="No file loaded.")
        tk.Label(bar, textvariable=self.status_var, anchor=tk.W, fg="#444").pack(
            side=tk.LEFT, padx=12
        )

        # matplotlib figure embedded inside the tkinter window
        self.fig = Figure(figsize=(14, 6.5))
        self.ax_rgb = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        NavigationToolbar2Tk(self.canvas, self.root)  # adds zoom/pan/save toolbar
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # left-click to inspect a pixel
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # ── File loading ──────────────────────────────────────────────────────────

    def _auto_load(self):
        """
        Called at startup. Loads a file from DATA_DIR automatically if
        exactly one .hdr is present, otherwise shows a picker dialog.
        A path can also be passed as a command-line argument.
        """
        if len(sys.argv) > 1:
            self._load(Path(sys.argv[1]))
            return

        if not DATA_DIR.exists():
            self.status_var.set(f"Data directory not found: {DATA_DIR}")
            return

        hdrs = find_hdr_files(DATA_DIR)
        if not hdrs:
            self.status_var.set(f"No .hdr files found in {DATA_DIR}")
            return
        if len(hdrs) == 1:
            self._load(hdrs[0])
        else:
            self._pick_file(hdrs)

    def _pick_file(self, hdrs: list[Path]):
        """Small modal dialog when multiple datasets are available."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Select dataset")
        dlg.grab_set()
        tk.Label(dlg, text="Multiple datasets found — select one to open:").pack(
            padx=12, pady=8
        )
        lb = tk.Listbox(dlg, width=72, height=min(len(hdrs), 10))
        lb.pack(padx=12, pady=4)
        for h in hdrs:
            lb.insert(tk.END, h.name)
        lb.selection_set(0)

        def on_ok():
            idx = lb.curselection()
            dlg.destroy()
            self._load(hdrs[idx[0]])

        tk.Button(dlg, text="Open", command=on_ok, width=12).pack(pady=8)
        dlg.wait_window()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open ENVI header (.hdr)",
            initialdir=DATA_DIR if DATA_DIR.exists() else Path.home(),
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        if path:
            self._load(Path(path))

    def _load(self, hdr_path: Path):
        self.status_var.set(f"Loading RGB bands from  {hdr_path.name} …")
        self.root.update_idletasks()
        try:
            self.img = load_image(hdr_path)
            meta = self.img.metadata
            self.wavelengths = parse_wavelengths(meta)
            self.ignore_value = get_ignore_value(meta)
            r, g, b = get_rgb_bands(meta)
            self.rgb_display = read_rgb(self.img, r, g, b, self.ignore_value)
            self.spectrum = None
            self.pixel_pos = None
            self._refresh_plots()
            self.status_var.set(
                f"{hdr_path.name}  |  "
                f"{self.img.nrows} lines × {self.img.ncols} samples × "
                f"{self.img.nbands} bands  |  "
                "Click a pixel to inspect its spectrum."
            )
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))
            self.status_var.set("Load failed.")

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _refresh_plots(self):
        # ── left: RGB image ──
        self.ax_rgb.clear()
        if self.rgb_display is not None:
            self.ax_rgb.imshow(
                self.rgb_display, interpolation="bilinear", aspect="auto"
            )
            if self.pixel_pos:
                row, col = self.pixel_pos
                self.ax_rgb.plot(
                    col, row, "r+", markersize=14, markeredgewidth=2.5
                )
        self.ax_rgb.set_title("RGB preview — click a pixel to inspect")
        self.ax_rgb.axis("off")

        # ── right: spectral signature ──
        self.ax_spec.clear()
        if self.spectrum is not None:
            row, col = self.pixel_pos
            x = (
                self.wavelengths
                if self.wavelengths is not None
                else np.arange(len(self.spectrum))
            )
            xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
            self.ax_spec.plot(x, self.spectrum, linewidth=1.2, color="steelblue")
            self.ax_spec.set_title(f"Spectral signature — row {row},  col {col}")
            self.ax_spec.set_xlabel(xlabel)
            self.ax_spec.set_ylabel("Reflectance (× 10⁻⁴)")
            self.ax_spec.grid(True, alpha=0.3)
        else:
            self.ax_spec.set_title("Spectral signature")
            self.ax_spec.text(
                0.5, 0.5,
                "Click a pixel in the RGB image",
                ha="center", va="center",
                transform=self.ax_spec.transAxes,
                color="gray", fontsize=12,
            )

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    # ── Mouse click handler ───────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is not self.ax_rgb or self.img is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < self.img.nrows and 0 <= col < self.img.ncols):
            return

        self.pixel_pos = (row, col)
        self.spectrum = read_spectrum(self.img, row, col, self.ignore_value)
        self._refresh_plots()
        self.status_var.set(
            f"Pixel ({row}, {col})  |  "
            "Use 'Export spectrum to CSV…' to save."
        )

    # ── CSV export ────────────────────────────────────────────────────────────

    def _export_csv(self):
        if self.spectrum is None:
            messagebox.showinfo("Nothing to export", "Click on a pixel first.")
            return

        row, col = self.pixel_pos
        default_name = f"spectrum_r{row}_c{col}.csv"
        path = filedialog.asksaveasfilename(
            title="Save spectrum as CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        x = (
            self.wavelengths
            if self.wavelengths is not None
            else np.arange(len(self.spectrum))
        )
        col_header = "wavelength_nm" if self.wavelengths is not None else "band"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([col_header, "value"])
            for xi, vi in zip(x, self.spectrum):
                writer.writerow([float(xi), "" if np.isnan(vi) else float(vi)])

        self.status_var.set(f"Saved → {path}")
        messagebox.showinfo("Saved", f"Spectrum exported to:\n{path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralViewer(root)
    root.mainloop()


# PUNKT 2 - BIBLIOTEKA SPEKTRALNA


import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path("spectral_library")
OUTPUT_FILE = "spectral_library.csv"

reference_spectra = {}

for class_dir in BASE_DIR.iterdir():
    if class_dir.is_dir():
        spectra = []
        wavelengths = None

        for file in class_dir.glob("*.csv"):
            df = pd.read_csv(file)
            spectra.append(df.iloc[:, 1].values)

            if wavelengths is None:
                wavelengths = df.iloc[:, 0].values

        reference_spectra[class_dir.name] = np.array(spectra)


library_mean = {}
library_std = {}

for key in reference_spectra.keys():
    library_mean[key] = np.nanmean(reference_spectra[key], axis=0)
    library_std[key] = np.nanstd(reference_spectra[key], axis=0)

colors = {
    "water": "blue",
    "forest": "darkgreen",
    "grass": "limegreen",
    "soil": "orange",
    "urban": "red"
}

plt.figure(figsize=(10,6))

for key in library_mean:
    mean = library_mean[key]
    std = library_std[key]
    color = colors.get(key, "black")

    plt.plot(wavelengths, mean, label=key, color=color)
    plt.fill_between(wavelengths, mean-std, mean+std, alpha=0.3, color = color)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance (× 10⁻⁴)")
plt.title("Spectral Library")
plt.legend()
plt.grid(True)
plt.show()


df_out = pd.DataFrame({"wavelength": wavelengths})

for key in library_mean:
    df_out[key] = library_mean[key]

df_out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved spectral library to: {OUTPUT_FILE}")


# PUNKT 3

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import sys


def manual_pick_file():
    DATA_DIR = Path("data/images")
    hdrs = sorted(DATA_DIR.glob("*.hdr")) if DATA_DIR.exists() else []

    if not hdrs:
        root_temp = tk.Tk()
        root_temp.withdraw()
        path = filedialog.askopenfilename(
            title="Open ENVI header (.hdr)",
            initialdir=DATA_DIR if DATA_DIR.exists() else Path.home(),
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        root_temp.destroy()
        return path

    if len(hdrs) == 1:
        return str(hdrs[0])

    selected_container = []
    dlg = tk.Tk()
    dlg.title("Select dataset")

    tk.Label(dlg, text="Multiple datasets found — select one to open:").pack(padx=12, pady=8)
    lb = tk.Listbox(dlg, width=72, height=min(len(hdrs), 10))
    lb.pack(padx=12, pady=4)

    for h in hdrs:
        lb.insert(tk.END, h.name)
    lb.selection_set(0)

    def on_ok():
        idx = lb.curselection()
        if idx:
            selected_container.append(str(hdrs[idx[0]]))
        dlg.destroy()

    tk.Button(dlg, text="Open", command=on_ok, width=12).pack(pady=8)
    dlg.mainloop()

    return selected_container[0] if selected_container else None


HDR_PATH = manual_pick_file()

img = envi.open(HDR_PATH)

meta = img.metadata

wavelengths = np.array([float(w) for w in meta["wavelength"]])

def plot_index(data, title, cmap="viridis"):
    plt.figure(figsize=(10, 8))
    vmin, vmax = np.nanpercentile(data, [2, 98])
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Index value")
    plt.title(title)
    plt.axis("off")
    plt.show()

def get_band_index(target_wavelength):
    return np.argmin(np.abs(wavelengths - target_wavelength))

nir = get_band_index(800)
red = get_band_index(650)
red1 = get_band_index(670)
green = get_band_index(550)

# Maska danych

sample_band = img.read_band(get_band_index(650)).astype(float)
data_mask = sample_band > 0

# Maska wody

water_mask = img.read_bands([nir, red1]).astype(float)
ndvi = (water_mask[:,:,0] - water_mask[:,:,1]) / (water_mask[:,:,0] + water_mask[:,:,1] + 1e-6)
mask = (ndvi < 0.1) & data_mask

#FALSE COLOR

false_color = img.read_bands([nir, red, green]).astype(float)
false_color = (false_color - np.min(false_color)) / (np.max(false_color) - np.min(false_color))
false_color[~data_mask] = 1.0

# CHL-A

chl_a1 = get_band_index(706)
chl_a2 = get_band_index(750)

chl_a = img.read_bands([chl_a1, chl_a2]).astype(float)
chl_a = (chl_a[:, :, 0] - chl_a[:, :, 1]) / (chl_a[:, :, 0] + chl_a[:, :, 1] + 1e-6)
chl_a = (chl_a - np.min(chl_a)) / (np.max(chl_a) - np.min(chl_a))
chl_a[~mask] = np.nan

# DOC

doc1 = get_band_index(480)
doc2 = get_band_index(660)

doc = img.read_bands([doc1, doc2]).astype(float)
doc = doc[:,:,0] / (doc[:,:,1] + 1e-6)
doc[~mask] = np.nan

# Turbidity

tur1 = get_band_index(710)
tur = img.read_bands([tur1]).astype(float)[:,:,0]
tur[~mask] = np.nan

fig, axes = plt.subplots(1, 4, figsize=(16, 12), facecolor='white')
fig.suptitle("False color and water quality indices", fontsize=20)

plots = [
    (false_color, "False Color", None),
    (chl_a, "Chlorophyll-a", "turbo"),
    (doc, "DOC", "gist_earth"),
    (tur, "Turbidity", "magma")
]

for i, (data, title, cmap) in enumerate(plots):
    ax = axes[i]
    ax.set_facecolor("white")

    if cmap is None:
        ax.imshow(data)
    else:
        vmin, vmax = np.nanpercentile(data, [5, 95])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=14)
    ax.axis("off")

plt.tight_layout()
plt.show()


# SENTINEL 2

# LOAD2 - funkcja pomocnicza do zdjęć sentinel

from osgeo import gdal
import os
import numpy as np
from glob import glob


def save_tif(tif, data, trans, proj, nodata=0):

    NP2GDAL_CONVERSION = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11,
    }

    [rows, cols] = data.shape
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(tif, cols, rows, 1, NP2GDAL_CONVERSION[str(data.dtype)])
    raster.SetGeoTransform(trans)
    raster.SetProjection(proj)
    band = raster.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(nodata)
    raster.FlushCache()


def save_tif_all(tif, data_list, trans, proj, nodata=0):

    NP2GDAL_CONVERSION = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11,
    }

    [rows, cols] = data_list[0].shape
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(tif, cols, rows, len(data_list), NP2GDAL_CONVERSION[str(data_list[0].dtype)])
    raster.SetGeoTransform(trans)
    raster.SetProjection(proj)

    for i, image in enumerate(data_list):
        band = raster.GetRasterBand(i + 1)
        band.WriteArray(image)
        band.SetNoDataValue(nodata)

    raster.FlushCache()


def load_tif(tif, data_only=0, channel=1):
    raster = gdal.Open(tif, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(channel)
    data = np.array(band.ReadAsArray())
    nodata = band.GetNoDataValue()
    trans = raster.GetGeoTransform()
    proj = raster.GetProjection()
    if data_only:
        return np.array(data)
    else:
        return np.array(data), nodata, trans, proj


def load_tif_all(tif, data_only=0, channels=None):
    raster = gdal.Open(tif, gdal.GA_ReadOnly)
    raster_count = raster.RasterCount
    output = []
    nodata_output = []
    for i in range(raster_count):
        if channels:
            if not (i+1) in channels:
                continue
        band = raster.GetRasterBand(i + 1)
        data = np.array(band.ReadAsArray())
        output.append(data)
        nodata = band.GetNoDataValue()
        nodata_output.append(nodata)
    trans = raster.GetGeoTransform()
    proj = raster.GetProjection()
    if data_only:
        return output
    else:
        return output, nodata_output, trans, proj

# LOAD2_V3 - funkcja pomocnicza do zdjęć sentinel

from osgeo import gdal
from skimage.transform import resize
import numpy as np
import sys
import math
import zipfile
from glob import glob
import os
from datetime import datetime


def load_s2(s2_zip, bands=None, resolution=10, interpolation=0):
    ds = gdal.Open(s2_zip)
    result_list = {}
    for sub_ds_info in ds.GetSubDatasets():
        if ':TCI:' in sub_ds_info[0]:  # skip true color images
            continue
        sub_ds = gdal.Open(sub_ds_info[0])
        for i in range(1, sub_ds.RasterCount + 1):
            raster = sub_ds.GetRasterBand(i)
            res = sub_ds.GetGeoTransform()[1]
            band_name = raster.GetMetadata()['BANDNAME']
            band_name_zerofilled = 'B' + band_name[1:].zfill(2)
            print(band_name_zerofilled)
            if band_name == 'B8A':  # skip B8A
                continue
            if isinstance(bands, list) and (band_name not in bands):
                continue
            if resolution == res:
                if band_name_zerofilled in result_list:
                    continue
                trans = sub_ds.GetGeoTransform()
                proj = sub_ds.GetProjection()
                result_list[band_name_zerofilled] = raster.ReadAsArray()
            else:
                if band_name_zerofilled in result_list:
                    continue
                new_res = res / resolution
                print('loading with resizing, band:', band_name, res)
                result_list[band_name_zerofilled] = resize(raster.ReadAsArray(),
                                                (math.ceil(raster.XSize * new_res), math.ceil(raster.YSize * new_res)),
                                                order=0)
    return trans, proj, result_list


def load_s2_cloud_mask(s2_zip, resolution=10, interpolation=0):
    zz = zipfile.ZipFile(s2_zip)
    scls = [f.filename
            for f in zz.filelist
            if f.filename.find("SCL_20m") >= 0]
    ds = gdal.Open(f"/vsizip/{s2_zip}/{scls[0]}")
    raster = ds.GetRasterBand(1)
    res = ds.GetGeoTransform()[1]
    new_res = res / resolution
    return resize(raster.ReadAsArray(),
                  (math.ceil(raster.XSize * new_res), math.ceil(raster.YSize * new_res)),
                  order=0)

import load2
import os
os.environ["GDAL_DRIVER_PATH"] = r"D:\APD\miniconda\envs\apdenv2\Library\lib\gdalplugins"
from osgeo import gdal
import loads2_v3 as ls2
gdal.DontUseExceptions()

trans, proj, real_s2 = ls2.load_s2(r"D:\agh\remote\eolabs\lab_1\lab_5\data\images\S2B_MSIL2A_20221017T095919_N0510_R122_T33UYR_20240723T160612.SAFE.zip", bands=['B2', 'B3',  'B4', 'B5', 'B8', 'B11', 'B12'])

b02 = real_s2['B02'].astype(float)
b04 = real_s2['B04'].astype(float)
b05 = real_s2['B05'].astype(float)
b08 = real_s2['B08'].astype(float)

# CROP IMAGINE

from pyproj import Transformer
from affine import Affine

affine_trans = Affine.from_gdal(*trans)

transformer = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
utm_min_x, utm_min_y = transformer.transform(18.127, 50.315)
utm_max_x, utm_max_y = transformer.transform(18.309, 50.378)

inv_trans = ~affine_trans
col_min, row_max = [int(round(x)) for x in inv_trans * (utm_min_x, utm_min_y)]
col_max, row_min = [int(round(x)) for x in inv_trans * (utm_max_x, utm_max_y)]

r1, r2 = sorted([row_min, row_max])
c1, c2 = sorted([col_min, col_max])

cropped_s2 = {}


for band_name, band_data in real_s2.items():
    final_r1 = max(0, r1)
    final_r2 = min(band_data.shape[0], r2)
    final_c1 = max(0, c1)
    final_c2 = min(band_data.shape[1], c2)

    cropped_s2[band_name] = band_data[final_r1:final_r2, final_c1:final_c2]


def get_b(data_dict, band_num):
    candidates = [f'B{band_num}', f'B0{band_num}', str(band_num)]
    for c in candidates:
        if c in data_dict:
            return data_dict[c].astype(float)
    raise KeyError(f"ERROR")

# INDICES FOR SENTINEL

b2 = get_b(cropped_s2, '2')
b3 = get_b(cropped_s2, '3')
b4 = get_b(cropped_s2, '4')
b5 = get_b(cropped_s2, '5')
b8 = get_b(cropped_s2, '8')


ndvi_s2 = (b8 - b4) / (b8 + b4 + 1e-6)
water_mask_s2 = ndvi_s2 < 0.1


chl_a_s2 = (b5 - b4) / (b5 + b4 + 1e-6)
chl_a_s2[~water_mask_s2] = np.nan

doc_s2 = b2 / (b4 + 1e-6)
doc_s2[~water_mask_s2] = np.nan

tur_s2 = b5
tur_s2[~water_mask_s2] = np.nan


def plot_water_indices(ch, d, t):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='white')

    indices = [
        (ch, 'Chlorophyll-a', 'turbo'),
        (d, 'DOC', 'gist_earth'),
        (t, 'Turbidity', 'magma')
    ]

    for i, (data, title, cmap) in enumerate(indices):
        ax = axes[i]
        vmin, vmax = np.nanpercentile(data, [2, 98])

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

plot_water_indices(chl_a_s2, doc_s2, tur_s2)


# PUNKT 4


from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    s2_b2  = get_b(cropped_s2, '2')
    s2_b3  = get_b(cropped_s2, '3')
    s2_b4  = get_b(cropped_s2, '4')
    s2_b5  = get_b(cropped_s2, '5')
    s2_b8  = get_b(cropped_s2, '8')
    s2_b11 = get_b(cropped_s2, '11')
except KeyError:
    s2_b2, s2_b3, s2_b4, s2_b5, s2_b8, s2_b11 = [cropped_s2[k].astype(float) for k in ['B02','B03','B04','B05','B08','B11']]

rows, cols = s2_b2.shape

s2_spectra = np.stack([
    s2_b2.flatten(),
    s2_b3.flatten(),
    s2_b4.flatten(),
    s2_b5.flatten(),
    s2_b8.flatten(),
    s2_b11.flatten()
], axis=0).T

mask_flat = water_mask_s2.flatten()
ref_water = np.nanmedian(s2_spectra[mask_flat], axis=0)

dot_product = np.dot(s2_spectra, ref_water)
norm_product = np.linalg.norm(s2_spectra, axis=1) * np.linalg.norm(ref_water)

sam = np.arccos(np.clip(dot_product / (norm_product + 1e-6), -1, 1))
sam_map = sam.reshape((rows, cols))

sam_map[~water_mask_s2] = np.nan

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111)
ax1.set_title('Sentinel-2: Spectral Angle Mapper')
vmax = np.nanpercentile(sam_map, 95)
im = ax1.imshow(sam_map, cmap='jet_r', vmin=0, vmax=vmax)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2%", pad=0.05)
fig.colorbar(im, cax=cax, label='Angle (radians)')
plt.show()

# SENTINEL2 CALIBRATION USING HYPERSPECTRAL DATA
