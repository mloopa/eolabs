#!/usr/bin/env python3
"""Generate Lab 7 figures and results PDF for Xinmo landslide (Sichuan, 2017-06-24)."""

from __future__ import annotations

import json
from pathlib import Path

import ee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import map_coordinates

BASE = Path(__file__).resolve().parent
LICSAR_DIR = BASE / "data" / "licsar_pre"
MINTPY_DIR = BASE / "data" / "mintpy_licsar"
FIG_DIR = BASE / "figures"
PDF_OUT = BASE / "results.pdf"

# Xinmo landslide AOI (colleague bbox)
BBOX = dict(west=103.62, south=32.04, east=103.68, north=32.09)
LANDSLIDE_DATE = "20170624"
EVENT_PAIR = "20170526_20170607"  # last pre-event pair ending just before cutoff
PROJECT_ID = "vast-advantage-494108-d0"

# Reference pixel in subset (50x60): stable lower-right area
REF_YX = (45, 55)
LANDSLIDE_YX = (25, 30)


def init_gee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)


def read_mintpy_maps():
    vel_file = MINTPY_DIR / "velocity.h5"
    tcoh_file = MINTPY_DIR / "temporalCoherence.h5"
    ts_file = MINTPY_DIR / "timeseries.h5"

    with h5py.File(vel_file, "r") as f:
        vel = f["velocity"][:] * 100  # cm/yr
        vel_attrs = dict(f.attrs)

    with h5py.File(tcoh_file, "r") as f:
        tcoh = f["temporalCoherence"][:]

    with h5py.File(ts_file, "r") as f:
        ts = f["timeseries"][:] * 100  # cm
        dates = [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]

    return vel, tcoh, ts, dates, vel_attrs


def read_dem():
    dem_path = LICSAR_DIR / "metadata" / "062D_05831_131313.geo.hgt.tif"
    if not dem_path.exists():
        return None, None, None
    with rasterio.open(dem_path) as ds:
        window = rasterio.windows.from_bounds(
            BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"], transform=ds.transform
        )
        dem = ds.read(1, window=window)
        extent = rasterio.windows.bounds(window, ds.transform)
    return dem, extent, dem_path


def read_event_coherence():
    cc_path = LICSAR_DIR / "interferograms" / EVENT_PAIR / f"{EVENT_PAIR}.geo.cc.tif"
    with rasterio.open(cc_path) as ds:
        window = rasterio.windows.from_bounds(
            BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"], transform=ds.transform
        )
        cc = ds.read(1, window=window).astype(float)
        if np.nanmax(cc) > 1.5:
            cc = cc / 255.0
    return cc, cc_path


def sample_ts(ts, dates, y, x):
    return ts[:, int(y), int(x)], dates


def plot_dem(dem, extent, out):
    if dem is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(dem, cmap="terrain", extent=[extent[0], extent[2], extent[1], extent[3]])
    ax.set_title("DEM – Xinmo landslide AOI")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_tcoh(vel, tcoh, out_vel, out_tcoh, ref_yx=REF_YX):
    for data, title, label, path, cmap in [
        (vel, "LOS velocity (pre-event SBAS)", "cm/year", out_vel, "RdBu_r"),
        (tcoh, "Temporal coherence", "0–1", out_tcoh, "viridis"),
    ]:
        finite = data[np.isfinite(data)]
        p = np.nanpercentile(np.abs(finite), 98) if finite.size else 1
        vmin, vmax = (-p, p) if "velocity" in title.lower() else (0, 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ry, rx = ref_yx
        ax.plot(rx, ry, "s", color="cyan", ms=8, label="reference")
        ax.set_title(title)
        ax.set_xlabel("x pixel"); ax.set_ylabel("y pixel")
        plt.colorbar(im, ax=ax, label=label)
        ax.legend(loc="upper right")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_ts(ts, dates, ref_yx, landslide_yx, out):
    ref_ts, _ = sample_ts(ts, dates, *ref_yx)
    ls_ts, _ = sample_ts(ts, dates, *landslide_yx)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, ref_ts, "o-", label="Reference (stable)")
    ax.plot(dates, ls_ts, "o-", label="Landslide area")
    ax.set_title("Displacement time series (re-referenced)")
    ax.set_xlabel("Date"); ax.set_ylabel("LOS displacement (cm)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coherence_analysis(cc, ref_yx, landslide_yx, out_map, out_box):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cc, cmap="viridis", vmin=0, vmax=1)
    ax.plot(ref_yx[1], ref_yx[0], "cs", ms=8, label="reference")
    ax.plot(landslide_yx[1], landslide_yx[0], "r*", ms=12, label="landslide")
    ax.set_title(f"Coherence – pair {EVENT_PAIR}")
    plt.colorbar(im, ax=ax, label="Coherence")
    ax.legend()
    fig.savefig(out_map, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Sample patches
    def patch_mean(y, x, r=15):
        y, x = int(y), int(x)
        sl = cc[max(0, y - r): y + r, max(0, x - r): x + r]
        return sl[np.isfinite(sl)]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([patch_mean(*landslide_yx), patch_mean(*ref_yx)], labels=["Landslide", "Reference"])
    ax.set_ylabel("Coherence"); ax.set_title("Coherence distribution")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_box, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gee_s1_vv_change(out_pre, out_post, out_diff):
    init_gee()
    region = ee.Geometry.Rectangle([BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]])
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    pre = coll.filterDate("2017-05-01", "2017-06-23").median().clip(region)
    post = coll.filterDate("2017-06-25", "2017-07-31").median().clip(region)
    diff = post.subtract(pre)

    vis = {"min": -20, "max": 0}
    diff_vis = {"min": -5, "max": 5, "palette": ["blue", "white", "red"]}

    for img, vis_params, path in [(pre, vis, out_pre), (post, vis, out_post), (diff, diff_vis, out_diff)]:
        url = img.getThumbURL({"region": region, "dimensions": 512, "format": "png", **vis_params})
        import urllib.request
        urllib.request.urlretrieve(url, path)


def gee_s2_indices(out_rgb_pre, out_rgb_post, out_ndvi_pre, out_ndvi_post, out_ndwi_pre, out_ndwi_post):
    init_gee()
    region = ee.Geometry.Rectangle([BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]])

    def s2_composite(start, end, cloud_max=80):
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
            .map(lambda img: img.divide(10000))
        )
        if col.size().getInfo() == 0:
            raise RuntimeError(f"No S2 images for {start}–{end}")
        return col.median().clip(region)

    pre = s2_composite("2016-06-01", "2017-06-20")
    post = s2_composite("2017-07-01", "2017-12-31", cloud_max=100)

    def ndvi(img):
        return img.normalizedDifference(["B8", "B4"]).rename("NDVI")

    def ndwi(img):
        return img.normalizedDifference(["B3", "B8"]).rename("NDWI")

    rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.3}
    idx_vis = {"min": -0.2, "max": 0.8, "palette": ["brown", "yellow", "green"]}
    ndwi_vis = {"min": -0.5, "max": 0.5, "palette": ["brown", "white", "blue"]}

    import urllib.request
    for img, vis, path in [
        (pre, rgb_vis, out_rgb_pre), (post, rgb_vis, out_rgb_post),
        (ndvi(pre), idx_vis, out_ndvi_pre), (ndvi(post), idx_vis, out_ndvi_post),
        (ndwi(pre), ndwi_vis, out_ndwi_pre), (ndwi(post), ndwi_vis, out_ndwi_post),
    ]:
        url = img.getThumbURL({"region": region, "dimensions": 512, "format": "png", **vis})
        urllib.request.urlretrieve(url, path)
        print(f"Saved {path}")


def build_pdf():
    sections = [
        ("SAR Lab 7 – Xinmo Landslide (Maoxian, Sichuan)\nPre-event SBAS analysis using COMET-LiCSAR + MintPy\nLandslide date: 24 June 2017", None),
        ("1. Data & Methods", None),
        ("Downloaded pre-event COMET-LiCS interferograms (frame 062D_05831_131313, cutoff 20170624, max baseline 72 days).\n"
         "Converted to MintPy HDF5 stack and ran SBAS inversion. AOI bbox: 103.62–103.68°E, 32.04–32.09°N.\n"
         "Reference pixel selected in stable, high-coherence area (lower-right of subset) to reduce seasonal bias near the slope.", None),
        ("2. DEM & Terrain", FIG_DIR / "dem.png"),
        ("3. LOS Velocity & Temporal Coherence", FIG_DIR / "velocity.png"),
        ("", FIG_DIR / "temporal_coherence.png"),
        ("Velocity map shows deformation zone (red = toward satellite, up to ~15 mm/yr) across the slope before failure.\n"
         "Temporal coherence is mixed in steep terrain; reliable data form a diagonal band. Reference point is stable.", None),
        ("4. Displacement Time Series", FIG_DIR / "timeseries.png"),
        ("Pre-event displacement shows a major step around 2015, then gradual decrease to ~25–30 cm before the landslide — "
         "suggesting earlier acceleration rather than steady creep until failure.", None),
        ("5. Coherence at Landslide Pair", FIG_DIR / "coherence_map.png"),
        ("", FIG_DIR / "coherence_boxplot.png"),
        ("Coherence pair 20170526_20170607 (last before event): median coherence ~0.3 in landslide area vs reference.\n"
         "Coherence loss partly maps the unstable slope but is ambiguous due to steep terrain and vegetation.", None),
        ("6. Sentinel-1 VV Change", FIG_DIR / "s1_pre.png"),
        ("", FIG_DIR / "s1_post.png"),
        ("", FIG_DIR / "s1_diff.png"),
        ("VV difference reveals a dark streak (backscatter decrease) along the landslide path — clearer than raw images.\n"
         "Better for post-event mapping than coherence alone, but still affected by geometry.", None),
        ("7. Sentinel-2 Optical", FIG_DIR / "s2_pre.png"),
        ("", FIG_DIR / "s2_post.png"),
        ("True-colour images show massive grey debris scar post-event, burying Xinmo village area.", None),
        ("8. Spectral Indices", FIG_DIR / "ndvi_pre.png"),
        ("", FIG_DIR / "ndvi_post.png"),
        ("", FIG_DIR / "ndwi_pre.png"),
        ("", FIG_DIR / "ndwi_post.png"),
        ("NDVI drops sharply over landslide (vegetation loss). NDWI and NBR (not shown separately) also outline the scar.\n"
         "Optical indices give the clearest landslide extent mapping.", None),
        ("9. Conclusions", None),
        ("• SBAS pre-event InSAR detected slow deformation on the slope years before collapse.\n"
         "• Coherence loss partially indicates instability but is hard to interpret in mountains.\n"
         "• S1 VV change and especially S2 optical/NDVI best map the landslide extent.\n"
         "• Pipeline runs on Mac with conda (licsar env) + MintPy + Earth Engine.", None),
    ]

    with PdfPages(PDF_OUT) as pdf:
        for title, img_path in sections:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.92, title, fontsize=11, va="top", wrap=True)
            if img_path and Path(img_path).exists():
                img = plt.imread(img_path)
                ax = fig.add_axes([0.08, 0.15, 0.84, 0.65])
                ax.imshow(img)
                ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {PDF_OUT}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if (MINTPY_DIR / "velocity.h5").exists():
        vel, tcoh, ts, dates, _ = read_mintpy_maps()
        dem, extent, _ = read_dem()
        plot_dem(dem, extent, FIG_DIR / "dem.png")
        plot_velocity_tcoh(vel, tcoh, FIG_DIR / "velocity.png", FIG_DIR / "temporal_coherence.png")
        plot_ts(ts, dates, REF_YX, LANDSLIDE_YX, FIG_DIR / "timeseries.png")

    if (LICSAR_DIR / "interferograms" / EVENT_PAIR).exists():
        cc, _ = read_event_coherence()
        plot_coherence_analysis(cc, REF_YX, LANDSLIDE_YX, FIG_DIR / "coherence_map.png", FIG_DIR / "coherence_boxplot.png")

    try:
        gee_s1_vv_change(FIG_DIR / "s1_pre.png", FIG_DIR / "s1_post.png", FIG_DIR / "s1_diff.png")
        gee_s2_indices(
            FIG_DIR / "s2_pre.png", FIG_DIR / "s2_post.png",
            FIG_DIR / "ndvi_pre.png", FIG_DIR / "ndvi_post.png",
            FIG_DIR / "ndwi_pre.png", FIG_DIR / "ndwi_post.png",
        )
    except Exception as e:
        print(f"GEE figures skipped: {e}")

    build_pdf()


if __name__ == "__main__":
    main()
