#!/usr/bin/env python3
"""Refine the Phase 9 Sentinel-2 landslide extent from the exported raster."""

from __future__ import annotations

import json
import os
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(LAB_DIR / "output" / ".matplotlib"))

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.warp import transform, transform_geom
from scipy import ndimage

from run_sentinel2_optical_analysis import (
    BSI_FIGURE,
    EXTENT_FILE,
    FIGURES_DIR,
    LANDSLIDE_LAT,
    LANDSLIDE_LON,
    NDVI_FIGURE,
    OPTICAL_EXTENT_RULES,
    OUTPUT_DIR,
    RASTER_FILE,
    REFERENCE_LAT,
    REFERENCE_LON,
    REFERENCE_RADIUS_M,
    SEARCH_RADIUS_M,
    SELECTION_FILE,
    STATS_FILE,
    TRUE_COLOR_FIGURE,
    plot_index,
    plot_true_color,
    read_exported_raster,
)

RULE = OPTICAL_EXTENT_RULES[0]
BAND_NAMES = [
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


def coordinate_grid(dataset: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((dataset.height, dataset.width))
    transform_ = dataset.transform
    xs = transform_.c + (cols + 0.5) * transform_.a + (rows + 0.5) * transform_.b
    ys = transform_.f + (cols + 0.5) * transform_.d + (rows + 0.5) * transform_.e
    return xs, ys


def circular_mask(
    dataset: rasterio.io.DatasetReader,
    lon: float,
    lat: float,
    radius_m: float,
) -> np.ndarray:
    x, y = transform("EPSG:4326", dataset.crs, [lon], [lat])
    xs, ys = coordinate_grid(dataset)
    return np.sqrt((xs - x[0]) ** 2 + (ys - y[0]) ** 2) <= radius_m


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        raise SystemExit("Refined Sentinel-2 mask contains no connected components.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def stats_for_mask(data: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    names = ["pre_ndvi", "post_ndvi", "ndvi_change", "pre_bsi", "post_bsi", "bsi_change"]
    band_indices = [6, 7, 8, 9, 10, 11]
    return {
        name: float(np.nanmean(data[index][mask]))
        for name, index in zip(names, band_indices)
    }


def write_geojson(mask: np.ndarray, dataset: rasterio.io.DatasetReader, area_m2: float) -> None:
    geometries = []
    for geom, value in features.shapes(
        mask.astype("uint8"),
        mask=mask,
        transform=dataset.transform,
    ):
        if value != 1:
            continue
        geometries.append(transform_geom(dataset.crs, "EPSG:4326", geom))

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "affected": 1,
                    "area_m2": area_m2,
                    "area_ha": area_m2 / 10_000,
                    "area_km2": area_m2 / 1_000_000,
                    "rule": (
                        f"post_brightness >= {RULE['post_brightness_gte']}; "
                        f"post_ndvi <= {RULE['post_ndvi_lte']}; "
                        f"ndvi_change <= {RULE['ndvi_change_lte']}; "
                        "largest connected component"
                    ),
                },
                "geometry": geom,
            }
            for geom in geometries
        ],
    }
    EXTENT_FILE.write_text(json.dumps(feature_collection, indent=2) + "\n")


def main() -> None:
    if not RASTER_FILE.exists():
        raise SystemExit(f"Missing Sentinel-2 raster: {RASTER_FILE}")
    if not SELECTION_FILE.exists():
        raise SystemExit(f"Missing Sentinel-2 selection metadata: {SELECTION_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(RASTER_FILE, "r+") as dataset:
        data = dataset.read(masked=True).filled(np.nan).astype("float32")
        search = circular_mask(dataset, LANDSLIDE_LON, LANDSLIDE_LAT, SEARCH_RADIUS_M)
        post_brightness = np.nanmean(data[3:6], axis=0)
        post_ndvi = data[7]
        ndvi_change = data[8]
        valid = (
            search
            & np.isfinite(post_brightness)
            & np.isfinite(post_ndvi)
            & np.isfinite(ndvi_change)
        )
        raw_mask = (
            valid
            & (post_brightness >= RULE["post_brightness_gte"])
            & (post_ndvi <= RULE["post_ndvi_lte"])
            & (ndvi_change <= RULE["ndvi_change_lte"])
        )
        refined_mask = largest_component(raw_mask)
        data[12] = refined_mask.astype("float32")
        dataset.write(data[12], 13)

        pixel_area_m2 = abs(dataset.transform.a * dataset.transform.e)
        area_m2 = float(refined_mask.sum() * pixel_area_m2)
        reference_mask = circular_mask(
            dataset,
            REFERENCE_LON,
            REFERENCE_LAT,
            REFERENCE_RADIUS_M,
        )
        affected_stats = stats_for_mask(data, refined_mask)
        reference_stats = stats_for_mask(data, reference_mask)
        write_geojson(refined_mask, dataset, area_m2)

    stats = pd.DataFrame(
        [
            {
                "sample": "Classified affected area",
                "area_m2": area_m2,
                "area_ha": area_m2 / 10_000,
                "area_km2": area_m2 / 1_000_000,
                **affected_stats,
            },
            {
                "sample": "Stable reference sample",
                "area_m2": np.nan,
                "area_ha": np.nan,
                "area_km2": np.nan,
                **reference_stats,
            },
        ]
    )
    stats.to_csv(STATS_FILE, index=False)

    selection = json.loads(SELECTION_FILE.read_text())
    selection.update(
        {
            "affected_rule_type": "post-event brightness plus NDVI-loss optical scar mask",
            "affected_rule": (
                f"post brightness >= {RULE['post_brightness_gte']}, "
                f"post NDVI <= {RULE['post_ndvi_lte']}, "
                f"NDVI change <= {RULE['ndvi_change_lte']}, "
                "largest connected component"
            ),
            "affected_rule_parameters": RULE,
            "initial_affected_rule": (
                f"post brightness >= {OPTICAL_EXTENT_RULES[0]['post_brightness_gte']}, "
                f"post NDVI <= {OPTICAL_EXTENT_RULES[0]['post_ndvi_lte']}, "
                f"NDVI change <= {OPTICAL_EXTENT_RULES[0]['ndvi_change_lte']}"
            ),
            "classification_rules_tried": list(OPTICAL_EXTENT_RULES),
            "affected_area_m2": area_m2,
            "affected_area_ha": area_m2 / 10_000,
            "affected_area_km2": area_m2 / 1_000_000,
            "export_bands": BAND_NAMES,
        }
    )
    SELECTION_FILE.write_text(json.dumps(selection, indent=2) + "\n")

    plot_data, plot_extent = read_exported_raster(RASTER_FILE)
    plot_true_color(
        plot_data,
        plot_extent,
        selection["pre_date"],
        selection["post_date"],
        selection["affected_area_ha"],
    )
    plot_index(
        plot_data,
        plot_extent,
        (6, 7, 8),
        "NDVI",
        selection["pre_date"],
        selection["post_date"],
        NDVI_FIGURE,
    )
    plot_index(
        plot_data,
        plot_extent,
        (9, 10, 11),
        "BSI",
        selection["pre_date"],
        selection["post_date"],
        BSI_FIGURE,
    )

    print("Refined Sentinel-2 extent completed.")
    print(f"Area: {area_m2 / 10_000:.1f} ha ({area_m2 / 1_000_000:.3f} km2)")
    print(f"Updated raster mask band: {RASTER_FILE.relative_to(LAB_DIR)}")
    print(f"Updated vector: {EXTENT_FILE.relative_to(LAB_DIR)}")
    print(f"Updated stats: {STATS_FILE.relative_to(LAB_DIR)}")
    print(f"Updated figures: {TRUE_COLOR_FIGURE.name}, {NDVI_FIGURE.name}, {BSI_FIGURE.name}")


if __name__ == "__main__":
    main()
