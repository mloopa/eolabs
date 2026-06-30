#!/usr/bin/env python3
"""Run the Phase 9 Sentinel-2 analysis without storing the GEE project ID."""

from __future__ import annotations

import json
import os
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(LAB_DIR / "output" / ".matplotlib"),
)

import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


OUTPUT_DIR = LAB_DIR / "output" / "earth_engine"
FIGURES_DIR = LAB_DIR / "figures"

COLLECTION_SR = "COPERNICUS/S2_SR_HARMONIZED"
COLLECTION_TOA = "COPERNICUS/S2_HARMONIZED"
START_DATE = "2017-01-01"
END_DATE = "2017-10-01"
MAX_CLOUDY_PIXEL_PERCENTAGE = 100
PAIR_CLEAR_FRACTION_LEVELS = (0.85, 0.70, 0.50, 0.30, 0.20)
AOI = [103.62, 32.04, 103.68, 32.09]
EVENT_DATE = pd.Timestamp("2017-06-24", tz="UTC")
LANDSLIDE_LON = 103.6506
LANDSLIDE_LAT = 32.0661
REFERENCE_LON = 103.6458322414
REFERENCE_LAT = 32.07133404813
SEARCH_RADIUS_M = 1_800
REFERENCE_RADIUS_M = 250
OPTICAL_EXTENT_RULES = (
    {
        "post_brightness_gte": 0.115,
        "post_ndvi_lte": 0.0,
        "ndvi_change_lte": -0.10,
    },
    {
        "post_brightness_gte": 0.110,
        "post_ndvi_lte": 0.10,
        "ndvi_change_lte": -0.20,
    },
    {
        "post_brightness_gte": 0.125,
        "post_ndvi_lte": 0.05,
        "ndvi_change_lte": -0.10,
    },
)
MIN_COMPONENT_PIXELS = 4

CANDIDATES_FILE = OUTPUT_DIR / "sentinel2_candidates.csv"
CANDIDATES_SR_FILE = OUTPUT_DIR / "sentinel2_candidates_sr.csv"
CANDIDATES_TOA_FILE = OUTPUT_DIR / "sentinel2_candidates_toa.csv"
SELECTION_FILE = OUTPUT_DIR / "sentinel2_selection.json"
RASTER_FILE = OUTPUT_DIR / "sentinel2_pre_post_indices.tif"
STATS_FILE = OUTPUT_DIR / "sentinel2_stats.csv"
EXTENT_FILE = OUTPUT_DIR / "sentinel2_landslide_extent.geojson"
TRUE_COLOR_FIGURE = FIGURES_DIR / "sentinel2_true_color_extent.png"
NDVI_FIGURE = FIGURES_DIR / "sentinel2_ndvi_change.png"
BSI_FIGURE = FIGURES_DIR / "sentinel2_bsi_change.png"

EXPORT_BANDS = [
    "pre_red",
    "pre_green",
    "pre_blue",
    "post_red",
    "post_green",
    "post_blue",
    "pre_ndvi",
    "post_ndvi",
    "ndvi_change",
    "pre_bsi",
    "post_bsi",
    "bsi_change",
    "affected_area",
]


def clear_mask(image: ee.Image, use_scl: bool) -> ee.Image:
    """Return a clear-pixel mask for Sentinel-2 imagery.

    Level-2A SR has the SCL scene-classification band. Level-1C TOA does not,
    so the fallback uses QA60 only.
    """
    qa60 = image.select("QA60")
    qa_clear = (
        qa60.bitwiseAnd(1 << 10)
        .eq(0)
        .And(qa60.bitwiseAnd(1 << 11).eq(0))
    )
    if not use_scl:
        return qa_clear.rename("clear")

    scl = image.select("SCL")
    scl_clear = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return scl_clear.And(qa_clear).rename("clear")


def add_aoi_clear_fraction(
    image: ee.Image,
    aoi: ee.Geometry,
    use_scl: bool,
) -> ee.Image:
    clear_fraction = clear_mask(image, use_scl).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=20,
        bestEffort=True,
        maxPixels=1_000_000,
    ).get("clear")
    return image.set("aoi_clear_fraction", clear_fraction)


def collection_to_dataframe(collection: ee.ImageCollection) -> pd.DataFrame:
    properties = [
        "system:index",
        "system:time_start",
        "PRODUCT_ID",
        "MGRS_TILE",
        "CLOUDY_PIXEL_PERCENTAGE",
        "SENSING_ORBIT_NUMBER",
        "aoi_clear_fraction",
    ]
    size = int(collection.size().getInfo())
    if size == 0:
        return pd.DataFrame(columns=properties + ["date"])

    records = (
        collection.toList(size)
        .map(lambda image: ee.Image(image).toDictionary(properties))
        .getInfo()
    )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["system:time_start"], unit="ms", utc=True)
    frame["CLOUDY_PIXEL_PERCENTAGE"] = pd.to_numeric(
        frame["CLOUDY_PIXEL_PERCENTAGE"], errors="coerce"
    )
    frame["aoi_clear_fraction"] = pd.to_numeric(
        frame["aoi_clear_fraction"], errors="coerce"
    )
    frame["SENSING_ORBIT_NUMBER"] = pd.to_numeric(
        frame["SENSING_ORBIT_NUMBER"], errors="coerce"
    ).astype("Int64")
    return frame.sort_values(["date", "MGRS_TILE"]).reset_index(drop=True)


def select_pair(candidates: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """Select the closest clear pre/post pair from one MGRS tile."""
    for minimum_clear_fraction in PAIR_CLEAR_FRACTION_LEVELS:
        usable = candidates[
            candidates["aoi_clear_fraction"].ge(minimum_clear_fraction)
        ].copy()
        options: list[
            tuple[float, pd.Timedelta, pd.Timedelta, pd.Series, pd.Series]
        ] = []
        for _, tile_group in usable.groupby("MGRS_TILE"):
            pre = tile_group[tile_group["date"] < EVENT_DATE]
            post = tile_group[tile_group["date"] > EVENT_DATE]
            if pre.empty or post.empty:
                continue
            for _, pre_row in pre.iterrows():
                for _, post_row in post.iterrows():
                    days_before = EVENT_DATE - pre_row["date"]
                    days_after = post_row["date"] - EVENT_DATE
                    min_clear = min(
                        float(pre_row["aoi_clear_fraction"]),
                        float(post_row["aoi_clear_fraction"]),
                    )
                    mean_clear = (
                        float(pre_row["aoi_clear_fraction"])
                        + float(post_row["aoi_clear_fraction"])
                    ) / 2
                    cloud_sum = (
                        float(pre_row["CLOUDY_PIXEL_PERCENTAGE"])
                        + float(post_row["CLOUDY_PIXEL_PERCENTAGE"])
                    )
                    total_days = days_before.days + days_after.days
                    max_days = max(days_before.days, days_after.days)
                    score = (
                        2.0 * min_clear
                        + mean_clear
                        - 0.003 * total_days
                        - 0.001 * max_days
                        - 0.001 * cloud_sum
                    )
                    options.append(
                        (score, days_before, days_after, pre_row, post_row)
                    )
        if options:
            _, _, _, pre_row, post_row = max(
                options,
                key=lambda item: (
                    item[0],
                    -((item[1] + item[2]).days),
                ),
            )
            return pre_row, post_row, minimum_clear_fraction

    summary = candidate_availability_summary(candidates)
    raise RuntimeError(
        "No Sentinel-2 pre/post pair was found on a common MGRS tile using "
        f"clear-fraction thresholds {PAIR_CLEAR_FRACTION_LEVELS}.\n{summary}"
    )


def candidate_availability_summary(candidates: pd.DataFrame) -> str:
    if candidates.empty:
        return "Candidate table is empty."

    rows = ["Candidate availability by tile and event side:"]
    frame = candidates.copy()
    frame["event_side"] = np.where(frame["date"] < EVENT_DATE, "pre", "post")
    for (tile, side), group in frame.groupby(["MGRS_TILE", "event_side"]):
        best = group.sort_values(
            ["aoi_clear_fraction", "date"],
            ascending=[False, True],
        ).iloc[0]
        rows.append(
            f"- {tile} {side}: count={len(group)}, "
            f"best_clear={best['aoi_clear_fraction']:.3f}, "
            f"best_date={best['date'].date().isoformat()}, "
            f"best_global_cloud={best['CLOUDY_PIXEL_PERCENTAGE']:.1f}%"
        )
    rows.append(f"Candidate table written to: {CANDIDATES_FILE}")
    return "\n".join(rows)


def image_from_index(collection: ee.ImageCollection, index: str) -> ee.Image:
    return ee.Image(collection.filter(ee.Filter.eq("system:index", index)).first())


def masked_reflectance(image: ee.Image, use_scl: bool) -> ee.Image:
    return (
        image.select(["B2", "B3", "B4", "B8", "B11"])
        .multiply(0.0001)
        .updateMask(clear_mask(image, use_scl))
        .toFloat()
    )


def normalized_indices(image: ee.Image) -> tuple[ee.Image, ee.Image]:
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
    bsi = image.expression(
        "((swir + red) - (nir + blue)) / "
        "((swir + red) + (nir + blue))",
        {
            "swir": image.select("B11"),
            "red": image.select("B4"),
            "nir": image.select("B8"),
            "blue": image.select("B2"),
        },
    ).rename("bsi")
    return ndvi, bsi


def build_collection(
    collection_id: str,
    aoi: ee.Geometry,
    use_scl: bool,
) -> tuple[ee.ImageCollection, pd.DataFrame]:
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUDY_PIXEL_PERCENTAGE))
        .map(lambda image: add_aoi_clear_fraction(image, aoi, use_scl))
        .filter(ee.Filter.notNull(["aoi_clear_fraction"]))
    )
    candidates = collection_to_dataframe(collection)
    if not candidates.empty:
        candidates.insert(0, "collection", collection_id)
    return collection, candidates


def choose_collection_and_pair(
    aoi: ee.Geometry,
) -> tuple[ee.ImageCollection, pd.DataFrame, pd.Series, pd.Series, float, str, bool]:
    attempts: list[str] = []

    for collection_id, use_scl, candidate_file in (
        (COLLECTION_SR, True, CANDIDATES_SR_FILE),
        (COLLECTION_TOA, False, CANDIDATES_TOA_FILE),
    ):
        collection, candidates = build_collection(collection_id, aoi, use_scl)
        if not candidates.empty:
            candidates.to_csv(candidate_file, index=False)
            candidates.to_csv(CANDIDATES_FILE, index=False)

        if candidates.empty:
            attempts.append(f"{collection_id}: no candidates")
            if collection_id == COLLECTION_SR:
                print(
                    f"No candidates found in {COLLECTION_SR}; falling back to "
                    f"{COLLECTION_TOA}."
                )
            continue

        try:
            pre_row, post_row, minimum_clear_fraction = select_pair(candidates)
            if collection_id == COLLECTION_TOA:
                print(
                    f"Using {COLLECTION_TOA}. The fallback uses QA60 cloud/cirrus "
                    "masking because Level-1C TOA imagery has no SCL band."
                )
            return (
                collection,
                candidates,
                pre_row,
                post_row,
                minimum_clear_fraction,
                collection_id,
                use_scl,
            )
        except RuntimeError as exc:
            attempts.append(f"{collection_id}: {exc}")
            if collection_id == COLLECTION_SR:
                print(
                    f"No usable pre/post pair found in {COLLECTION_SR}; "
                    f"falling back to {COLLECTION_TOA}."
                )

    raise SystemExit(
        "No usable Sentinel-2 pre/post pair was found.\n\n"
        + "\n\n".join(attempts)
    )


def masked_mean(
    image: ee.Image,
    mask: ee.Image,
    geometry: ee.Geometry,
) -> dict[str, float]:
    return (
        image.updateMask(mask)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=20,
            bestEffort=True,
            maxPixels=2_000_000,
        )
        .getInfo()
    )


def classify_affected_area(
    post: ee.Image,
    post_ndvi: ee.Image,
    ndvi_change: ee.Image,
    common_valid: ee.Image,
    search_region: ee.Geometry,
) -> tuple[ee.Image, float, dict[str, float]]:
    post_brightness = (
        post.select(["B4", "B3", "B2"])
        .reduce(ee.Reducer.mean())
        .rename("post_brightness")
    )
    for rule in OPTICAL_EXTENT_RULES:
        raw_affected = (
            post_brightness.gte(rule["post_brightness_gte"])
            .And(post_ndvi.lte(rule["post_ndvi_lte"]))
            .And(ndvi_change.lte(rule["ndvi_change_lte"]))
            .And(common_valid)
            .clip(search_region)
            .rename("affected_area")
        )
        connected_pixels = raw_affected.selfMask().connectedPixelCount(
            100, eightConnected=True
        )
        affected = (
            raw_affected.updateMask(connected_pixels.gte(MIN_COMPONENT_PIXELS))
            .selfMask()
            .rename("affected_area")
        )
        area_m2 = ee.Number(
            ee.Image.pixelArea()
            .updateMask(affected)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=search_region,
                scale=20,
                bestEffort=True,
                maxPixels=2_000_000,
            )
            .get("area")
        ).getInfo()
        if area_m2 and area_m2 > 0:
            if rule != OPTICAL_EXTENT_RULES[0]:
                print(
                    "Initial optical classification found no affected pixels; "
                    f"using relaxed rule {rule}."
                )
            return affected, float(area_m2), rule

    raise SystemExit(
        "The optical extent rules classified no affected pixels. Inspect candidate "
        "imagery manually before proceeding."
    )


def read_exported_raster(raster_file: Path) -> tuple[np.ndarray, list[float]]:
    with rasterio.open(raster_file) as source:
        with WarpedVRT(
            source,
            crs="EPSG:4326",
            resampling=Resampling.bilinear,
            nodata=np.nan,
        ) as dataset:
            data = dataset.read(masked=True).filled(np.nan).astype(float)
            bounds = dataset.bounds
    return data, [bounds.left, bounds.right, bounds.bottom, bounds.top]


def stretch_rgb(rgb: np.ndarray, low: float, high: float) -> np.ndarray:
    stretched = np.moveaxis(rgb, 0, -1)
    stretched = np.clip((stretched - low) / (high - low), 0, 1)
    return np.power(stretched, 0.9)


def format_map_axis(axis: plt.Axes, title: str) -> None:
    axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(6))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def plot_true_color(
    data: np.ndarray,
    extent: list[float],
    pre_date: str,
    post_date: str,
    affected_area_ha: float,
) -> None:
    pre_rgb = data[0:3]
    post_rgb = data[3:6]
    affected = data[12]
    reflectance = np.concatenate(
        [
            pre_rgb[np.isfinite(pre_rgb)],
            post_rgb[np.isfinite(post_rgb)],
        ]
    )
    low, high = np.percentile(reflectance, [1, 99])
    pre_display = stretch_rgb(pre_rgb, float(low), float(high))
    post_display = stretch_rgb(post_rgb, float(low), float(high))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for axis, rgb, title in (
        (axes[0], pre_display, f"Pre-event true color\n{pre_date}"),
        (axes[1], post_display, f"Post-event true color\n{post_date}"),
        (axes[2], post_display, "Post-event with classified extent"),
    ):
        axis.imshow(rgb, extent=extent, origin="upper")
        axis.scatter(
            LANDSLIDE_LON,
            LANDSLIDE_LAT,
            marker="*",
            s=130,
            facecolor="yellow",
            edgecolor="black",
            linewidth=0.8,
            label="Approximate landslide location",
        )
        format_map_axis(axis, title)

    axes[2].contour(
        affected,
        levels=[0.5],
        colors=["magenta"],
        linewidths=1.3,
        extent=extent,
        origin="upper",
    )
    axes[2].plot([], [], color="magenta", label="Classified affected extent")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[2].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Sentinel-2 True Color and Xinmo Landslide Classification\n"
        f"Classified affected area: {affected_area_ha:.1f} ha",
        fontsize=14,
    )
    fig.savefig(TRUE_COLOR_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_index(
    data: np.ndarray,
    extent: list[float],
    band_indices: tuple[int, int, int],
    index_name: str,
    pre_date: str,
    post_date: str,
    output_file: Path,
) -> None:
    pre, post, change = data[list(band_indices)]
    change_values = change[np.isfinite(change)]
    change_limit = max(
        0.1,
        float(np.percentile(np.abs(change_values), 98)),
    )
    affected = data[12]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    panels = [
        (pre, f"Pre-event {index_name}\n{pre_date}", "RdYlGn", -1, 1),
        (post, f"Post-event {index_name}\n{post_date}", "RdYlGn", -1, 1),
        (
            change,
            f"{index_name} change (post - pre)",
            "RdBu",
            -change_limit,
            change_limit,
        ),
    ]
    for axis, (values, title, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(
            values,
            extent=extent,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.contour(
            affected,
            levels=[0.5],
            colors=["magenta"],
            linewidths=1.0,
            extent=extent,
            origin="upper",
        )
        axis.scatter(
            LANDSLIDE_LON,
            LANDSLIDE_LAT,
            marker="*",
            s=100,
            facecolor="yellow",
            edgecolor="black",
            linewidth=0.7,
        )
        format_map_axis(axis, title)
        fig.colorbar(image, ax=axis, label=index_name, shrink=0.86)

    fig.suptitle(
        f"Sentinel-2 {index_name} Before and After the Xinmo Landslide",
        fontsize=14,
    )
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_id = os.environ.get("EARTH_ENGINE_PROJECT", "").strip()
    if not project_id:
        raise SystemExit(
            "EARTH_ENGINE_PROJECT is not set. Configure it privately in this "
            "run configuration."
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
    (
        collection,
        candidates,
        pre_row,
        post_row,
        minimum_clear_fraction,
        collection_id,
        use_scl,
    ) = choose_collection_and_pair(aoi)

    pre_source = image_from_index(collection, str(pre_row["system:index"]))
    post_source = image_from_index(collection, str(post_row["system:index"]))
    pre = masked_reflectance(pre_source, use_scl)
    post = masked_reflectance(post_source, use_scl)
    pre_ndvi, pre_bsi = normalized_indices(pre)
    post_ndvi, post_bsi = normalized_indices(post)
    ndvi_change = post_ndvi.subtract(pre_ndvi).rename("ndvi_change")
    bsi_change = post_bsi.subtract(pre_bsi).rename("bsi_change")

    search_region = ee.Geometry.Point(
        [LANDSLIDE_LON, LANDSLIDE_LAT]
    ).buffer(SEARCH_RADIUS_M)
    common_valid = pre_ndvi.mask().And(post_ndvi.mask())
    (
        affected,
        area_m2,
        extent_rule_used,
    ) = classify_affected_area(
        post,
        post_ndvi,
        ndvi_change,
        common_valid,
        search_region,
    )

    extent_vectors = affected.toByte().reduceToVectors(
        geometry=search_region,
        scale=20,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="affected",
        reducer=ee.Reducer.countEvery(),
        maxPixels=2_000_000,
    )
    EXTENT_FILE.write_text(json.dumps(extent_vectors.getInfo(), indent=2) + "\n")

    export_image = (
        pre.select(["B4", "B3", "B2"], ["pre_red", "pre_green", "pre_blue"])
        .addBands(
            post.select(
                ["B4", "B3", "B2"],
                ["post_red", "post_green", "post_blue"],
            )
        )
        .addBands(pre_ndvi.rename("pre_ndvi"))
        .addBands(post_ndvi.rename("post_ndvi"))
        .addBands(ndvi_change)
        .addBands(pre_bsi.rename("pre_bsi"))
        .addBands(post_bsi.rename("post_bsi"))
        .addBands(bsi_change)
        .addBands(affected.unmask(0).toFloat())
        .clip(aoi)
        .toFloat()
    )

    affected_means = masked_mean(
        export_image.select(
            ["pre_ndvi", "post_ndvi", "ndvi_change", "pre_bsi", "post_bsi", "bsi_change"]
        ),
        affected,
        search_region,
    )
    reference_region = ee.Geometry.Point(
        [REFERENCE_LON, REFERENCE_LAT]
    ).buffer(REFERENCE_RADIUS_M)
    reference_means = (
        export_image.select(
            ["pre_ndvi", "post_ndvi", "ndvi_change", "pre_bsi", "post_bsi", "bsi_change"]
        )
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=reference_region,
            scale=20,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    stats = pd.DataFrame(
        [
            {
                "sample": "Classified affected area",
                "area_m2": float(area_m2),
                "area_ha": float(area_m2) / 10_000,
                "area_km2": float(area_m2) / 1_000_000,
                **affected_means,
            },
            {
                "sample": "Stable reference sample",
                "area_m2": np.nan,
                "area_ha": np.nan,
                "area_km2": np.nan,
                **reference_means,
            },
        ]
    )
    stats.to_csv(STATS_FILE, index=False)

    selection = {
        "collection": collection_id,
        "processing_level": "Level-2A surface reflectance" if use_scl else "Level-1C top-of-atmosphere reflectance",
        "pre_image_id": str(pre_row["system:index"]),
        "pre_product_id": str(pre_row["PRODUCT_ID"]),
        "pre_date": pre_row["date"].date().isoformat(),
        "pre_cloudy_pixel_percentage": float(
            pre_row["CLOUDY_PIXEL_PERCENTAGE"]
        ),
        "pre_aoi_clear_fraction": float(pre_row["aoi_clear_fraction"]),
        "post_image_id": str(post_row["system:index"]),
        "post_product_id": str(post_row["PRODUCT_ID"]),
        "post_date": post_row["date"].date().isoformat(),
        "post_cloudy_pixel_percentage": float(
            post_row["CLOUDY_PIXEL_PERCENTAGE"]
        ),
        "post_aoi_clear_fraction": float(post_row["aoi_clear_fraction"]),
        "mgrs_tile": str(pre_row["MGRS_TILE"]),
        "sensing_orbit": int(pre_row["SENSING_ORBIT_NUMBER"]),
        "minimum_pair_clear_fraction": minimum_clear_fraction,
        "cloud_mask": (
            "SCL classes 0, 1, 3, 8, 9, 10, and 11 excluded; "
            "QA60 cloud and cirrus bits excluded"
            if use_scl
            else "QA60 cloud and cirrus bits excluded; no SCL band available in Level-1C TOA fallback"
        ),
        "ndvi_formula": "(B8 - B4) / (B8 + B4)",
        "bsi_formula": (
            "((B11 + B4) - (B8 + B2)) / "
            "((B11 + B4) + (B8 + B2))"
        ),
        "affected_rule": (
            "post brightness >= "
            f"{extent_rule_used['post_brightness_gte']}, post NDVI <= "
            f"{extent_rule_used['post_ndvi_lte']}, and NDVI change <= "
            f"{extent_rule_used['ndvi_change_lte']}"
        ),
        "affected_rule_type": "post-event brightness plus NDVI-loss optical scar mask",
        "initial_affected_rule": (
            "post brightness >= "
            f"{OPTICAL_EXTENT_RULES[0]['post_brightness_gte']}, post NDVI <= "
            f"{OPTICAL_EXTENT_RULES[0]['post_ndvi_lte']}, and NDVI change <= "
            f"{OPTICAL_EXTENT_RULES[0]['ndvi_change_lte']}"
        ),
        "classification_rules_tried": list(OPTICAL_EXTENT_RULES),
        "search_radius_m": SEARCH_RADIUS_M,
        "minimum_component_pixels": MIN_COMPONENT_PIXELS,
        "affected_area_m2": float(area_m2),
        "affected_area_ha": float(area_m2) / 10_000,
        "affected_area_km2": float(area_m2) / 1_000_000,
        "aoi": AOI,
        "export_bands": EXPORT_BANDS,
    }
    SELECTION_FILE.write_text(json.dumps(selection, indent=2) + "\n")

    geemap.ee_export_image(
        export_image,
        filename=str(RASTER_FILE),
        scale=20,
        region=aoi,
        file_per_band=False,
    )
    if not RASTER_FILE.exists():
        raise SystemExit("Earth Engine export did not create the expected GeoTIFF.")

    data, plot_extent = read_exported_raster(RASTER_FILE)
    if data.shape[0] != len(EXPORT_BANDS):
        raise SystemExit(
            f"Expected {len(EXPORT_BANDS)} exported bands, found {data.shape[0]}."
        )
    plot_true_color(
        data,
        plot_extent,
        selection["pre_date"],
        selection["post_date"],
        selection["affected_area_ha"],
    )
    plot_index(
        data,
        plot_extent,
        (6, 7, 8),
        "NDVI",
        selection["pre_date"],
        selection["post_date"],
        NDVI_FIGURE,
    )
    plot_index(
        data,
        plot_extent,
        (9, 10, 11),
        "BSI",
        selection["pre_date"],
        selection["post_date"],
        BSI_FIGURE,
    )

    print("Sentinel-2 optical analysis completed.")
    print(f"Candidate count: {len(candidates)}")
    print(
        "Selected dates:",
        selection["pre_date"],
        "and",
        selection["post_date"],
    )
    print(f"Classified affected area: {selection['affected_area_ha']:.1f} ha")
    print(f"Saved raster: {RASTER_FILE.relative_to(LAB_DIR)}")
    print(f"Saved vector: {EXTENT_FILE.relative_to(LAB_DIR)}")
    print(f"Saved statistics: {STATS_FILE.relative_to(LAB_DIR)}")
    print("Saved figures:")
    for figure_file in (TRUE_COLOR_FIGURE, NDVI_FIGURE, BSI_FIGURE):
        print(f"  {figure_file.relative_to(LAB_DIR)}")
    print("Project ID was loaded from the environment and was not displayed.")


if __name__ == "__main__":
    main()
