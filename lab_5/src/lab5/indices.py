from __future__ import annotations

from typing import Sequence

import numpy as np

from src.lab5.envi_utils import find_nearest_band


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _mask_band(band, ignore_value: float | None = None) -> np.ndarray:
    masked = _as_float_array(band).copy()
    if ignore_value is not None:
        masked[masked >= ignore_value] = np.nan
    masked[masked < 0] = np.nan
    return masked


def _read_band(cube_or_reader, band_index: int) -> np.ndarray:
    if hasattr(cube_or_reader, "read_band"):
        return np.asarray(cube_or_reader.read_band(int(band_index)), dtype=np.float64)

    cube = np.asarray(cube_or_reader)
    if cube.ndim != 3:
        raise ValueError("Expected a 3-D cube with shape (rows, cols, bands).")
    return np.asarray(cube[:, :, int(band_index)], dtype=np.float64)


def safe_ratio(numerator, denominator):
    numerator_arr, denominator_arr = np.broadcast_arrays(
        _as_float_array(numerator),
        _as_float_array(denominator),
    )
    result = np.full(numerator_arr.shape, np.nan, dtype=np.float64)
    valid_mask = (
        np.isfinite(numerator_arr)
        & np.isfinite(denominator_arr)
        & (np.abs(denominator_arr) > 1e-12)
    )
    np.divide(
        numerator_arr,
        denominator_arr,
        out=result,
        where=valid_mask,
    )
    return result.item() if result.shape == () else result


def band_by_wavelength(
    cube_or_reader,
    wavelengths: np.ndarray,
    target_nm: float,
    invalid_mask: np.ndarray | None = None,
    ignore_value: float | None = None,
) -> np.ndarray:
    band_index = find_nearest_band(wavelengths, target_nm, invalid_mask)
    return _mask_band(_read_band(cube_or_reader, band_index), ignore_value)


def chlorophyll_red_edge_peak(r665, r705, r740):
    r665_arr = _as_float_array(r665)
    r705_arr = _as_float_array(r705)
    r740_arr = _as_float_array(r740)
    baseline = r665_arr + ((705.0 - 665.0) / (740.0 - 665.0)) * (r740_arr - r665_arr)
    result = r705_arr - baseline
    return result.item() if result.shape == () else result


def ndci(r665, r705):
    return safe_ratio(_as_float_array(r705) - _as_float_array(r665), _as_float_array(r705) + _as_float_array(r665))


def doc_proxy_green_red(r560, r665):
    return safe_ratio(r560, r665)


def ndti(r560, r665):
    return safe_ratio(_as_float_array(r665) - _as_float_array(r560), _as_float_array(r665) + _as_float_array(r560))


def ndre(r705, r842):
    """Normalized Difference Red Edge: (R842 - R705) / (R842 + R705).

    Sensitive to chlorophyll in dense blooms. S2 bands: B05, B08.
    """
    return safe_ratio(
        _as_float_array(r842) - _as_float_array(r705),
        _as_float_array(r842) + _as_float_array(r705),
    )


def fai(r665, r842, r1610):
    """Floating Algae Index: NIR peak above RED-SWIR baseline.

    Detects surface scums and floating algal mats. S2 bands: B04, B08, B11.
    """
    r665_arr = _as_float_array(r665)
    r842_arr = _as_float_array(r842)
    r1610_arr = _as_float_array(r1610)
    slope = (842.0 - 665.0) / (1610.0 - 665.0)
    baseline = r665_arr + slope * (r1610_arr - r665_arr)
    result = r842_arr - baseline
    return result.item() if result.shape == () else result


def ndwi(r560, r842):
    """Normalized Difference Water Index: (R560 - R842) / (R560 + R842).

    Water mask: positive → water, negative → land. S2 bands: B03, B08.
    """
    return safe_ratio(
        _as_float_array(r560) - _as_float_array(r842),
        _as_float_array(r560) + _as_float_array(r842),
    )


def flh(r665, r681, r709):
    """Fluorescence Line Height at 681 nm above the 665-709 nm baseline.

    Airborne only — Sentinel-2 lacks a 681 nm band.
    """
    r665_arr = _as_float_array(r665)
    r681_arr = _as_float_array(r681)
    r709_arr = _as_float_array(r709)
    slope = (681.0 - 665.0) / (709.0 - 665.0)
    baseline = r665_arr + slope * (r709_arr - r665_arr)
    result = r681_arr - baseline
    return result.item() if result.shape == () else result


def mci(r681, r709, r753):
    """Maximum Chlorophyll Index: 709 nm peak above the 681-753 nm baseline.

    Airborne only — Sentinel-2 lacks a 681 nm band.
    """
    r681_arr = _as_float_array(r681)
    r709_arr = _as_float_array(r709)
    r753_arr = _as_float_array(r753)
    slope = (709.0 - 681.0) / (753.0 - 681.0)
    baseline = r681_arr + slope * (r753_arr - r681_arr)
    result = r709_arr - baseline
    return result.item() if result.shape == () else result


def composite_by_wavelengths(
    cube_or_reader,
    wavelengths: np.ndarray,
    targets_nm: Sequence[float],
    invalid_mask: np.ndarray | None = None,
    ignore_value: float | None = None,
) -> np.ndarray:
    if len(targets_nm) != 3:
        raise ValueError("False-color composites require exactly three target wavelengths.")

    channels = [
        band_by_wavelength(
            cube_or_reader,
            wavelengths,
            target_nm,
            invalid_mask=invalid_mask,
            ignore_value=ignore_value,
        )
        for target_nm in targets_nm
    ]
    return np.stack(channels, axis=-1)


def stretch_composite(rgb: np.ndarray) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=np.float64)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
        raise ValueError("RGB composite must have shape (rows, cols, 3).")

    stretched = rgb_arr.copy()
    for channel_idx in range(3):
        channel = stretched[:, :, channel_idx]
        valid_values = channel[np.isfinite(channel)]
        if valid_values.size == 0:
            stretched[:, :, channel_idx] = 0.0
            continue
        lo, hi = np.percentile(valid_values, [2, 98])
        stretched[:, :, channel_idx] = np.clip(
            (channel - lo) / max(hi - lo, 1e-6),
            0.0,
            1.0,
        )

    return np.nan_to_num(stretched, nan=0.0).astype(np.float32)


__all__ = [
    "band_by_wavelength",
    "chlorophyll_red_edge_peak",
    "composite_by_wavelengths",
    "doc_proxy_green_red",
    "fai",
    "flh",
    "mci",
    "ndci",
    "ndre",
    "ndti",
    "ndwi",
    "safe_ratio",
    "stretch_composite",
]
