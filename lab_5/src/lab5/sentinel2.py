from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.lab5.envi_utils import parse_map_info


PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL2_COLLECTION = "sentinel-2-l2a"
SENTINEL2_REQUIRED_BANDS = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
)

SENTINEL2_BAND_ALIASES = {
    "B02": ("B02", "blue"),
    "B03": ("B03", "green"),
    "B04": ("B04", "red"),
    "B05": ("B05",),
    "B06": ("B06",),
    "B07": ("B07",),
    "B08": ("B08", "nir", "nir08"),
    "B8A": ("B8A", "nir09"),
    "B11": ("B11", "swir16"),
    "B12": ("B12", "swir22"),
}


@dataclass(frozen=True)
class TargetGrid:
    crs: Any
    transform: Any
    width: int
    height: int
    bounds: tuple[float, float, float, float]


def _import_catalog_clients():
    import planetary_computer
    import pystac_client

    return planetary_computer, pystac_client


def _import_rasterio():
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds

    return rasterio, CRS, from_origin, Resampling, reproject, transform_bounds


def _coerce_target_date(target_date: str | date | datetime) -> date:
    if isinstance(target_date, datetime):
        return target_date.date()
    if isinstance(target_date, date):
        return target_date
    return date.fromisoformat(str(target_date))


def _scene_crs_from_envi(scene_metadata: Mapping[str, Any]):
    _, CRS, _, _, _, _ = _import_rasterio()
    coordinate_system = scene_metadata.get("coordinate system string")
    if not coordinate_system:
        raise ValueError("ENVI metadata is missing 'coordinate system string'.")
    if isinstance(coordinate_system, (list, tuple, np.ndarray)):
        wkt = ",".join(str(part) for part in coordinate_system).strip()
    else:
        wkt = str(coordinate_system).strip()
    if wkt.startswith("{") and wkt.endswith("}"):
        wkt = wkt[1:-1]
    try:
        return CRS.from_wkt(wkt)
    except Exception:
        import pyproj

        normalized_wkt = pyproj.CRS.from_wkt(wkt).to_wkt()
        return CRS.from_wkt(normalized_wkt)


def scene_grid_from_envi(scene_metadata: Mapping[str, Any]) -> TargetGrid:
    _, _, from_origin, _, _, _ = _import_rasterio()

    map_info = parse_map_info(dict(scene_metadata))
    if map_info is None:
        raise ValueError("ENVI metadata is missing usable 'map info'.")

    width = int(str(scene_metadata["samples"]).strip())
    height = int(str(scene_metadata["lines"]).strip())
    pixel_width = abs(map_info.pixel_width)
    pixel_height = abs(map_info.pixel_height)

    left = map_info.origin_x + (map_info.x_start - map_info.ref_pixel_x - 0.5) * pixel_width
    top = map_info.origin_y - (map_info.y_start - map_info.ref_pixel_y - 0.5) * pixel_height
    transform = from_origin(left, top, pixel_width, pixel_height)
    bounds = (
        left,
        top - height * pixel_height,
        left + width * pixel_width,
        top,
    )

    return TargetGrid(
        crs=_scene_crs_from_envi(scene_metadata),
        transform=transform,
        width=width,
        height=height,
        bounds=bounds,
    )


def scene_bbox_wgs84_from_envi(scene_metadata: Mapping[str, Any]) -> tuple[float, float, float, float]:
    _, CRS, _, _, _, transform_bounds = _import_rasterio()

    grid = scene_grid_from_envi(scene_metadata)
    return tuple(
        transform_bounds(
            grid.crs,
            CRS.from_epsg(4326),
            *grid.bounds,
            densify_pts=21,
        )
    )


def open_catalog(url: str = PLANETARY_COMPUTER_STAC_URL):
    planetary_computer, pystac_client = _import_catalog_clients()
    return pystac_client.Client.open(
        url,
        modifier=planetary_computer.sign_inplace,
    )


def _item_datetime(item) -> datetime:
    dt = getattr(item, "datetime", None)
    if isinstance(dt, datetime):
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

    raw = item.properties.get("datetime") or item.properties.get("start_datetime")
    if raw is None:
        raise ValueError(f"STAC item '{item.id}' is missing a datetime property.")

    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def search_s2_items(
    bbox: tuple[float, float, float, float],
    target_date: str | date | datetime,
    day_window: int = 5,
    cloud_limit: float = 20.0,
    catalog=None,
) -> list[Any]:
    target = _coerce_target_date(target_date)
    start = target - timedelta(days=int(day_window))
    end = target + timedelta(days=int(day_window))

    if catalog is None:
        catalog = open_catalog()

    search = catalog.search(
        collections=[SENTINEL2_COLLECTION],
        bbox=bbox,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lt": float(cloud_limit)}},
    )
    items = list(search.items())
    return sorted(items, key=lambda item: score_item(item, target))


def score_item(item, target_date: str | date | datetime) -> tuple[int, float, str, str]:
    target = _coerce_target_date(target_date)
    item_dt = _item_datetime(item)
    cloud_cover = float(item.properties.get("eo:cloud_cover", 100.0))
    days_from_target = abs((item_dt.date() - target).days)
    return (
        days_from_target,
        cloud_cover,
        item_dt.isoformat(),
        str(item.id),
    )


def select_best_item(items: list[Any], target_date: str | date | datetime):
    if not items:
        raise ValueError("No Sentinel-2 items available for selection.")
    return min(items, key=lambda item: score_item(item, target_date))


def describe_item(item, target_date: str | date | datetime) -> dict[str, Any]:
    item_dt = _item_datetime(item)
    target = _coerce_target_date(target_date)
    return {
        "id": item.id,
        "datetime": item_dt.isoformat(),
        "days_from_target": abs((item_dt.date() - target).days),
        "eo_cloud_cover": float(item.properties.get("eo:cloud_cover", np.nan)),
        "platform": item.properties.get("platform", ""),
        "mgrs_tile": item.properties.get("s2:mgrs_tile", ""),
    }


def _asset_for_band(item, band_name: str):
    aliases = SENTINEL2_BAND_ALIASES.get(band_name, (band_name,))

    for alias in aliases:
        asset = item.assets.get(alias)
        if asset is not None:
            return asset

    for asset in item.assets.values():
        if asset.extra_fields.get("title") in aliases:
            return asset

        eo_bands = asset.extra_fields.get("eo:bands", [])
        raster_bands = asset.extra_fields.get("raster:bands", [])
        for band in eo_bands + raster_bands:
            common_name = band.get("common_name")
            name = band.get("name")
            if common_name in aliases or name in aliases:
                return asset

    raise KeyError(f"Band '{band_name}' not found. Available assets: {list(item.assets.keys())}")


def get_asset_href(item, band_name: str) -> str:
    return str(_asset_for_band(item, band_name).href)


def _asset_scale_offset(asset, array: np.ndarray) -> tuple[float, float]:
    for extra_key in ("raster:bands", "eo:bands"):
        bands = asset.extra_fields.get(extra_key, [])
        if bands:
            scale = bands[0].get("scale")
            offset = bands[0].get("offset")
            if scale is not None or offset is not None:
                return float(scale if scale is not None else 1.0), float(offset if offset is not None else 0.0)

    finite_values = array[np.isfinite(array)]
    if finite_values.size and float(np.nanmax(finite_values)) > 2.0:
        return 0.0001, 0.0
    return 1.0, 0.0


def reproject_to_grid(
    href: str,
    target_crs,
    target_transform,
    width: int,
    height: int,
    resampling=None,
):
    rasterio, _, _, Resampling, reproject, _ = _import_rasterio()

    if resampling is None:
        resampling = Resampling.bilinear

    out = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=0.0,
        )
    return out


def download_band(
    item,
    band_name: str,
    save_dir: Path | str,
    target_grid: TargetGrid,
) -> Path:
    rasterio, _, _, _, _, _ = _import_rasterio()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    local_path = save_path / f"{item.id}__{band_name}.tif"
    if local_path.exists():
        return local_path

    asset = _asset_for_band(item, band_name)
    arr = reproject_to_grid(
        str(asset.href),
        target_grid.crs,
        target_grid.transform,
        target_grid.width,
        target_grid.height,
    )
    scale, offset = _asset_scale_offset(asset, arr)
    valid_mask = np.isfinite(arr) & (arr != 0.0)
    arr = arr.astype(np.float32, copy=False)
    arr[valid_mask] = np.clip(arr[valid_mask] * scale + offset, 0.0, 1.0)

    with rasterio.open(
        local_path,
        "w",
        driver="GTiff",
        height=target_grid.height,
        width=target_grid.width,
        count=1,
        dtype="float32",
        crs=target_grid.crs,
        transform=target_grid.transform,
        nodata=0.0,
    ) as dst:
        dst.write(arr, 1)

    return local_path


def find_cached_band_files(
    save_dir: Path | str,
    item_id: str | None = None,
    required_bands: Sequence[str] = SENTINEL2_REQUIRED_BANDS,
) -> dict[str, Path]:
    save_path = Path(save_dir)
    if not save_path.exists():
        return {}

    if item_id is not None:
        candidate_prefixes = [str(item_id)]
    else:
        candidate_prefixes = sorted(
            {
                path.name.rsplit("__", 1)[0]
                for path in save_path.glob("*__*.tif")
            },
            key=lambda prefix: max(
                (candidate.stat().st_mtime for candidate in save_path.glob(f"{prefix}__*.tif")),
                default=0.0,
            ),
            reverse=True,
        )

    required = tuple(str(band_name) for band_name in required_bands)
    for prefix in candidate_prefixes:
        band_files = {
            band_name: save_path / f"{prefix}__{band_name}.tif"
            for band_name in required
        }
        if all(path.exists() for path in band_files.values()):
            return band_files

    return {}


def load_local_stack(band_files: Mapping[str, Path | str]) -> dict[str, np.ndarray]:
    rasterio, _, _, _, _, _ = _import_rasterio()

    stack: dict[str, np.ndarray] = {}
    for band_name, path in band_files.items():
        with rasterio.open(path) as src:
            stack[band_name] = src.read(1).astype(np.float32)
    return stack


def align_array_to_grid(
    array: np.ndarray,
    source_grid: TargetGrid,
    target_grid: TargetGrid,
    resampling=None,
    nodata: float = np.nan,
) -> np.ndarray:
    _, _, _, Resampling, reproject, _ = _import_rasterio()

    src_array = np.asarray(array, dtype=np.float32)
    expected_shape = (source_grid.height, source_grid.width)
    if src_array.shape != expected_shape:
        raise ValueError(f"Array shape {src_array.shape} does not match source grid {expected_shape}.")

    if (
        source_grid.crs == target_grid.crs
        and source_grid.transform == target_grid.transform
        and source_grid.width == target_grid.width
        and source_grid.height == target_grid.height
    ):
        return src_array.copy()

    if resampling is None:
        resampling = Resampling.bilinear

    fill_value = -9999.0 if np.isnan(nodata) else float(nodata)
    src_prepared = np.where(np.isfinite(src_array), src_array, fill_value).astype(np.float32, copy=False)
    out = np.full((target_grid.height, target_grid.width), fill_value, dtype=np.float32)

    reproject(
        source=src_prepared,
        destination=out,
        src_transform=source_grid.transform,
        src_crs=source_grid.crs,
        dst_transform=target_grid.transform,
        dst_crs=target_grid.crs,
        resampling=resampling,
        src_nodata=fill_value,
        dst_nodata=fill_value,
    )

    if np.isnan(nodata):
        out = out.astype(np.float32, copy=False)
        out[out == fill_value] = np.nan

    return out


def build_overlap_mask(*product_groups: Mapping[str, np.ndarray]) -> np.ndarray:
    overlap_mask: np.ndarray | None = None
    expected_shape: tuple[int, int] | None = None

    for product_group in product_groups:
        for name, array in product_group.items():
            arr = np.asarray(array)
            if arr.ndim != 2:
                raise ValueError(f"Product '{name}' must be 2-D, got shape {arr.shape}.")
            if expected_shape is None:
                expected_shape = arr.shape
            elif arr.shape != expected_shape:
                raise ValueError(f"Product '{name}' has shape {arr.shape}, expected {expected_shape}.")

            finite_mask = np.isfinite(arr)
            overlap_mask = finite_mask if overlap_mask is None else overlap_mask & finite_mask

    if overlap_mask is None:
        raise ValueError("At least one product group is required to build an overlap mask.")

    return overlap_mask


def summarize_pair(
    airborne: np.ndarray,
    sentinel: np.ndarray,
    overlap_mask: np.ndarray,
) -> dict[str, float]:
    airborne_arr = np.asarray(airborne, dtype=np.float64)
    sentinel_arr = np.asarray(sentinel, dtype=np.float64)
    mask = np.asarray(overlap_mask, dtype=bool)

    if airborne_arr.shape != sentinel_arr.shape:
        raise ValueError(f"Shape mismatch: airborne {airborne_arr.shape}, sentinel {sentinel_arr.shape}.")
    if airborne_arr.shape != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match product shape {airborne_arr.shape}.")

    airborne_values = airborne_arr[mask]
    sentinel_values = sentinel_arr[mask]
    delta = sentinel_values - airborne_values

    if airborne_values.size == 0:
        return {
            "valid_pixels": 0,
            "airborne_mean": np.nan,
            "sentinel_mean": np.nan,
            "bias_s2_minus_airborne": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "pearson_r": np.nan,
        }

    if airborne_values.size < 2 or np.std(airborne_values) == 0.0 or np.std(sentinel_values) == 0.0:
        pearson_r = np.nan
    else:
        pearson_r = float(np.corrcoef(airborne_values, sentinel_values)[0, 1])

    return {
        "valid_pixels": int(airborne_values.size),
        "airborne_mean": float(np.mean(airborne_values)),
        "sentinel_mean": float(np.mean(sentinel_values)),
        "bias_s2_minus_airborne": float(np.mean(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "pearson_r": pearson_r,
    }


def comparison_output_dir(root_dir: Path | str, scene_id: str) -> Path:
    output_dir = Path(root_dir) / "comparison" / str(scene_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


__all__ = [
    "PLANETARY_COMPUTER_STAC_URL",
    "SENTINEL2_BAND_ALIASES",
    "SENTINEL2_COLLECTION",
    "SENTINEL2_REQUIRED_BANDS",
    "TargetGrid",
    "align_array_to_grid",
    "build_overlap_mask",
    "comparison_output_dir",
    "describe_item",
    "download_band",
    "find_cached_band_files",
    "get_asset_href",
    "load_local_stack",
    "open_catalog",
    "reproject_to_grid",
    "scene_bbox_wgs84_from_envi",
    "scene_grid_from_envi",
    "score_item",
    "search_s2_items",
    "select_best_item",
    "summarize_pair",
]
