from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import spectral.io.envi as envi
except ImportError as exc:  # pragma: no cover - import guard for runtime
    raise ImportError(
        "The 'spectral' library is missing.\n"
        "Install it with: pip install spectral"
    ) from exc


FALLBACK_RGB = (30, 20, 10)


@dataclass(frozen=True)
class MapInfo:
    projection_name: str
    ref_pixel_x: float
    ref_pixel_y: float
    origin_x: float
    origin_y: float
    pixel_width: float
    pixel_height: float
    x_start: float = 1.0
    y_start: float = 1.0


def find_hdr_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.hdr"))


def _metadata_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return [str(item).strip() for item in raw.tolist()]
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw]
    text = str(raw).strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    if not text:
        return []
    return [item.strip() for item in text.split(",")]


def parse_wavelengths(metadata: dict[str, Any]) -> np.ndarray | None:
    raw_values = _metadata_list(metadata.get("wavelength"))
    if not raw_values:
        return None
    return np.array([float(value) for value in raw_values], dtype=np.float64)


def get_rgb_bands(
    metadata: dict[str, Any],
    fallback: tuple[int, int, int] = FALLBACK_RGB,
) -> tuple[int, int, int]:
    raw_values = _metadata_list(metadata.get("default bands"))
    if len(raw_values) >= 3:
        return tuple(int(float(value)) - 1 for value in raw_values[:3])
    return fallback


def get_ignore_value(metadata: dict[str, Any]) -> float | None:
    raw = metadata.get("data ignore value")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def parse_map_info(metadata: dict[str, Any]) -> MapInfo | None:
    values = _metadata_list(metadata.get("map info"))
    if len(values) < 6:
        return None
    try:
        return MapInfo(
            projection_name=values[0],
            ref_pixel_x=float(values[1]),
            ref_pixel_y=float(values[2]),
            origin_x=float(values[3]),
            origin_y=float(values[4]),
            pixel_width=float(values[5]),
            pixel_height=float(values[6]) if len(values) >= 7 else float(values[5]),
            x_start=float(str(metadata.get("x start", 1)).strip()),
            y_start=float(str(metadata.get("y start", 1)).strip()),
        )
    except ValueError:
        return None


def pixel_to_map(row: int, col: int, map_info: MapInfo | None) -> tuple[float, float] | None:
    if map_info is None:
        return None
    image_x = col + map_info.x_start
    image_y = row + map_info.y_start
    delta_x = image_x - map_info.ref_pixel_x
    delta_y = image_y - map_info.ref_pixel_y
    map_x = map_info.origin_x + delta_x * map_info.pixel_width
    map_y = map_info.origin_y - delta_y * abs(map_info.pixel_height)
    return map_x, map_y


def build_invalid_band_mask(metadata: dict[str, Any]) -> np.ndarray | None:
    band_count_raw = metadata.get("bands")
    try:
        band_count = int(str(band_count_raw).strip()) if band_count_raw is not None else None
    except ValueError:
        band_count = None

    bbl_values = _metadata_list(metadata.get("bbl"))
    if bbl_values:
        mask = np.array(
            [float(value) <= 0 for value in bbl_values],
            dtype=bool,
        )
        if band_count is not None and len(mask) == band_count:
            return mask

    band_names = _metadata_list(metadata.get("band names"))
    if band_names:
        mask = np.array(
            [name.lower().lstrip().startswith("band*") for name in band_names],
            dtype=bool,
        )
        if band_count is not None and len(mask) == band_count:
            return mask
        if band_count is None:
            return mask

    if band_count is None:
        return None
    return np.zeros(band_count, dtype=bool)


def find_nearest_band(
    wavelengths: np.ndarray | None,
    target_nm: float,
    invalid_mask: np.ndarray | None = None,
) -> int:
    if wavelengths is None or len(wavelengths) == 0:
        raise ValueError("No wavelength metadata available.")

    candidate_idx = np.arange(len(wavelengths))
    if invalid_mask is not None:
        candidate_idx = candidate_idx[~invalid_mask]
    if len(candidate_idx) == 0:
        raise ValueError("No valid bands available for wavelength lookup.")

    best = candidate_idx[np.argmin(np.abs(wavelengths[candidate_idx] - target_nm))]
    return int(best)


def load_envi_image(hdr_path: Path):
    return envi.open(str(hdr_path))


def _mask_spectra(
    values: np.ndarray,
    ignore_value: float | None,
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    masked = values.astype(np.float64, copy=True)
    if ignore_value is not None:
        masked[masked >= ignore_value] = np.nan
    masked[masked < 0] = np.nan
    if invalid_mask is not None:
        masked[..., invalid_mask] = np.nan
    return masked


def read_rgb(
    img,
    band_triplet: tuple[int, int, int],
    ignore_value: float | None,
) -> np.ndarray:
    rgb = img.read_bands(list(band_triplet)).astype(np.float32)
    if ignore_value is not None:
        rgb[rgb >= ignore_value] = np.nan
    rgb[rgb < 0] = np.nan

    for channel_idx in range(3):
        channel = rgb[:, :, channel_idx]
        if np.all(np.isnan(channel)):
            rgb[:, :, channel_idx] = 0.0
            continue
        p2, p98 = np.nanpercentile(channel, [2, 98])
        rgb[:, :, channel_idx] = np.clip(
            (channel - p2) / max(p98 - p2, 1e-6),
            0,
            1,
        )

    return np.nan_to_num(rgb, nan=0.0)


def read_pixel_spectrum(
    img,
    row: int,
    col: int,
    ignore_value: float | None,
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    spectrum = img.read_pixel(row, col)
    return _mask_spectra(spectrum, ignore_value, invalid_mask)


def read_roi_cube(
    img,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> np.ndarray:
    row_start, row_stop = sorted((int(row_min), int(row_max)))
    col_start, col_stop = sorted((int(col_min), int(col_max)))
    if row_start < 0 or col_start < 0:
        raise ValueError("ROI bounds must be non-negative.")
    return img.read_subregion(
        (row_start, row_stop + 1),
        (col_start, col_stop + 1),
    )


def summarize_roi_spectra(
    roi_cube: np.ndarray,
    ignore_value: float | None,
    invalid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | int]:
    if roi_cube.ndim != 3:
        raise ValueError("ROI cube must have shape (rows, cols, bands).")

    spectra = roi_cube.reshape(-1, roi_cube.shape[-1])
    masked = _mask_spectra(spectra, ignore_value, invalid_mask)
    valid_mask = ~np.isnan(masked)
    valid_pixel_count_by_band = valid_mask.sum(axis=0).astype(np.int32)

    mean_spectrum = np.full(masked.shape[1], np.nan, dtype=np.float64)
    sum_by_band = np.nansum(masked, axis=0)
    np.divide(
        sum_by_band,
        valid_pixel_count_by_band,
        out=mean_spectrum,
        where=valid_pixel_count_by_band > 0,
    )

    variance = np.full(masked.shape[1], np.nan, dtype=np.float64)
    centered = np.where(valid_mask, masked - mean_spectrum, np.nan)
    squared_diff_sum = np.nansum(centered ** 2, axis=0)
    np.divide(
        squared_diff_sum,
        valid_pixel_count_by_band,
        out=variance,
        where=valid_pixel_count_by_band > 0,
    )
    std_spectrum = np.sqrt(variance)

    return {
        "mean_spectrum": mean_spectrum,
        "std_spectrum": std_spectrum,
        "valid_pixel_count_by_band": valid_pixel_count_by_band,
        "total_valid_pixel_count": int(np.any(valid_mask, axis=1).sum()),
        "pixel_count": int(masked.shape[0]),
    }


__all__ = [
    "FALLBACK_RGB",
    "MapInfo",
    "build_invalid_band_mask",
    "find_hdr_files",
    "find_nearest_band",
    "get_ignore_value",
    "get_rgb_bands",
    "load_envi_image",
    "parse_map_info",
    "parse_wavelengths",
    "pixel_to_map",
    "read_pixel_spectrum",
    "read_rgb",
    "read_roi_cube",
    "summarize_roi_spectra",
]
