#!/usr/bin/env python3
"""Run the Phase 8 Sentinel-1 VV analysis without storing the GEE project ID."""

from __future__ import annotations

import json
import os
from pathlib import Path

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


LAB_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = LAB_DIR / "output" / "earth_engine"
FIGURES_DIR = LAB_DIR / "figures"

AOI = [103.62, 32.04, 103.68, 32.09]
EVENT_DATE = pd.Timestamp("2017-06-24", tz="UTC")
LANDSLIDE_LON = 103.6506
LANDSLIDE_LAT = 32.0661
REFERENCE_LON = 103.6458322414
REFERENCE_LAT = 32.07133404813
SAMPLE_RADIUS_M = 250

CANDIDATES_FILE = OUTPUT_DIR / "sentinel1_vv_candidates.csv"
SELECTION_FILE = OUTPUT_DIR / "sentinel1_vv_selection.json"
RASTER_FILE = OUTPUT_DIR / "sentinel1_vv_pre_post_change.tif"
STATS_FILE = OUTPUT_DIR / "sentinel1_vv_stats.csv"
FIGURE_FILE = FIGURES_DIR / "sentinel1_vv_change.png"


def collection_to_dataframe(collection: ee.ImageCollection) -> pd.DataFrame:
    properties = [
        "system:index",
        "system:time_start",
        "relativeOrbitNumber_start",
        "orbitProperties_pass",
        "platform_number",
        "instrumentMode",
        "transmitterReceiverPolarisation",
        "resolution_meters",
    ]
    records = collection.select(["VV"]).toList(collection.size()).map(
        lambda image: ee.Image(image).toDictionary(properties)
    ).getInfo()
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["system:time_start"], unit="ms", utc=True)
    frame["relativeOrbitNumber_start"] = frame[
        "relativeOrbitNumber_start"
    ].astype(int)
    return frame.sort_values("date").reset_index(drop=True)


def select_pair(candidates: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    options: list[tuple[pd.Timedelta, pd.Timedelta, int, pd.Series, pd.Series]] = []
    for orbit, group in candidates.groupby("relativeOrbitNumber_start"):
        pre = group[group["date"] < EVENT_DATE]
        post = group[group["date"] > EVENT_DATE]
        if pre.empty or post.empty:
            continue
        pre_row = pre.iloc[-1]
        post_row = post.iloc[0]
        options.append(
            (
                EVENT_DATE - pre_row["date"],
                post_row["date"] - EVENT_DATE,
                orbit,
                pre_row,
                post_row,
            )
        )

    if not options:
        raise RuntimeError(
            "No descending Sentinel-1 VV pre/post pair was found on a common relative orbit."
        )

    _, _, _, pre_row, post_row = min(
        options,
        key=lambda item: (
            item[0] + item[1],
            max(item[0], item[1]),
        ),
    )
    return pre_row, post_row


def image_from_index(collection: ee.ImageCollection, index: str) -> ee.Image:
    return ee.Image(collection.filter(ee.Filter.eq("system:index", index)).first())


def sample_stats(
    image: ee.Image,
    sample_name: str,
    lon: float,
    lat: float,
) -> dict[str, float | str]:
    region = ee.Geometry.Point([lon, lat]).buffer(SAMPLE_RADIUS_M)
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.percentile([25, 75]), sharedInputs=True)
    )
    values = image.reduceRegion(
        reducer=reducer,
        geometry=region,
        scale=10,
        bestEffort=True,
        maxPixels=1_000_000,
    ).getInfo()
    return {
        "sample": sample_name,
        "center_lon": lon,
        "center_lat": lat,
        "radius_m": SAMPLE_RADIUS_M,
        **values,
    }


def plot_outputs(
    raster_file: Path,
    stats: pd.DataFrame,
    pre_date: str,
    post_date: str,
) -> None:
    with rasterio.open(raster_file) as source:
        # Earth Engine exports this AOI in its native UTM projection. Read it
        # through a WGS84 virtual raster so imagery, markers, and axis labels
        # use the same longitude/latitude coordinate system.
        with WarpedVRT(
            source,
            crs="EPSG:4326",
            resampling=Resampling.bilinear,
            nodata=np.nan,
        ) as dataset:
            data = dataset.read(masked=True).filled(np.nan).astype(float)
            pre, post, change = data
            bounds = dataset.bounds

    vv_values = np.concatenate([pre[np.isfinite(pre)], post[np.isfinite(post)]])
    vv_min, vv_max = np.percentile(vv_values, [2, 98])
    change_limit = max(1.0, float(np.percentile(np.abs(change[np.isfinite(change)]), 98)))
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    panels = [
        (pre, f"Pre-event VV\n{pre_date}", "gray", vv_min, vv_max, "VV backscatter (dB)"),
        (post, f"Post-event VV\n{post_date}", "gray", vv_min, vv_max, "VV backscatter (dB)"),
        (
            change,
            "VV change (post - pre)",
            "RdBu_r",
            -change_limit,
            change_limit,
            "VV change (dB)",
        ),
    ]

    for axis, (data, title, cmap, vmin, vmax, colorbar_label) in zip(axes, panels):
        image = axis.imshow(
            data,
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.scatter(
            LANDSLIDE_LON,
            LANDSLIDE_LAT,
            marker="*",
            s=130,
            facecolor="yellow",
            edgecolor="black",
            linewidth=0.8,
            label="Landslide",
        )
        axis.scatter(
            REFERENCE_LON,
            REFERENCE_LAT,
            marker="s",
            s=45,
            facecolor="none",
            edgecolor="cyan",
            linewidth=1.2,
            label="Stable reference",
        )
        radius_deg = SAMPLE_RADIUS_M / 111_320
        axis.add_patch(
            Circle(
                (LANDSLIDE_LON, LANDSLIDE_LAT),
                radius_deg,
                fill=False,
                color="yellow",
                linestyle="--",
                linewidth=1,
            )
        )
        axis.add_patch(
            Circle(
                (REFERENCE_LON, REFERENCE_LAT),
                radius_deg,
                fill=False,
                color="cyan",
                linestyle="--",
                linewidth=1,
            )
        )
        axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(6))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        fig.colorbar(
            image,
            ax=axis,
            label=colorbar_label,
            shrink=0.86,
        )

    axes[0].legend(loc="lower left", fontsize=8)
    landslide_change = stats.loc[
        stats["sample"] == "Landslide-centered sample", "vv_change_db_mean"
    ].iloc[0]
    stable_change = stats.loc[
        stats["sample"] == "Stable reference sample", "vv_change_db_mean"
    ].iloc[0]
    fig.suptitle(
        "Sentinel-1 Descending VV Change\n"
        f"Landslide mean: {landslide_change:.2f} dB; "
        f"stable reference mean: {stable_change:.2f} dB",
        fontsize=14,
    )
    fig.savefig(FIGURE_FILE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_id = os.environ.get("EARTH_ENGINE_PROJECT", "").strip()
    if not project_id:
        raise SystemExit(
            "EARTH_ENGINE_PROJECT is not set. Configure it privately in this run configuration."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        ee.Initialize(project=project_id)
    except Exception as exc:
        raise SystemExit(
            f"Earth Engine initialization failed with {type(exc).__name__}."
        ) from None

    aoi = ee.Geometry.Rectangle(AOI)
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate("2017-04-01", "2017-09-01")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.eq("resolution_meters", 10))
    )

    candidates = collection_to_dataframe(collection)
    if candidates.empty:
        raise SystemExit("No Sentinel-1 descending IW VV candidates were found.")
    candidates.to_csv(CANDIDATES_FILE, index=False)

    pre_row, post_row = select_pair(candidates)
    pre_image = image_from_index(collection, pre_row["system:index"]).select("VV")
    post_image = image_from_index(collection, post_row["system:index"]).select("VV")
    change_image = post_image.subtract(pre_image).rename("vv_change_db")
    export_image = (
        pre_image.rename("pre_vv_db")
        .addBands(post_image.rename("post_vv_db"))
        .addBands(change_image)
        .clip(aoi)
        .toFloat()
    )

    selection = {
        "collection": "COPERNICUS/S1_GRD",
        "orbit_pass": "DESCENDING",
        "relative_orbit": int(pre_row["relativeOrbitNumber_start"]),
        "pre_image_id": pre_row["system:index"],
        "pre_date": pre_row["date"].date().isoformat(),
        "post_image_id": post_row["system:index"],
        "post_date": post_row["date"].date().isoformat(),
        "days_before_event": int((EVENT_DATE - pre_row["date"]).days),
        "days_after_event": int((post_row["date"] - EVENT_DATE).days),
        "formula": "post-event VV - pre-event VV (dB)",
        "aoi": AOI,
    }
    SELECTION_FILE.write_text(json.dumps(selection, indent=2) + "\n")

    stats_image = export_image
    stats = pd.DataFrame(
        [
            sample_stats(
                stats_image,
                "Landslide-centered sample",
                LANDSLIDE_LON,
                LANDSLIDE_LAT,
            ),
            sample_stats(
                stats_image,
                "Stable reference sample",
                REFERENCE_LON,
                REFERENCE_LAT,
            ),
        ]
    )
    stats.to_csv(STATS_FILE, index=False)

    geemap.ee_export_image(
        export_image,
        filename=str(RASTER_FILE),
        scale=10,
        region=aoi,
        file_per_band=False,
    )
    if not RASTER_FILE.exists():
        raise SystemExit("Earth Engine export did not create the expected GeoTIFF.")

    plot_outputs(
        RASTER_FILE,
        stats,
        selection["pre_date"],
        selection["post_date"],
    )

    print("Sentinel-1 VV analysis completed.")
    print(f"Candidate count: {len(candidates)}")
    print(f"Relative orbit: {selection['relative_orbit']}")
    print(
        "Selected dates:",
        selection["pre_date"],
        "and",
        selection["post_date"],
    )
    print(f"Saved raster: {RASTER_FILE.relative_to(LAB_DIR)}")
    print(f"Saved statistics: {STATS_FILE.relative_to(LAB_DIR)}")
    print(f"Saved figure: {FIGURE_FILE.relative_to(LAB_DIR)}")
    print("Project ID was loaded from the environment and was not displayed.")


if __name__ == "__main__":
    main()
