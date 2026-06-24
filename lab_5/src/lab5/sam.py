from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.lab5.spectral_library import CATALOG_FILENAME, SUMMARY_DIRNAME


SENTINEL2_SAM_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
SENTINEL2_TARGET_WAVELENGTHS = np.array(
    (490.0, 560.0, 665.0, 705.0, 740.0, 783.0, 842.0, 865.0, 1610.0, 2190.0),
    dtype=np.float64,
)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _summary_dir(base_dir: Path | str) -> Path:
    root = Path(base_dir)
    if root.name == SUMMARY_DIRNAME:
        return root
    return root / SUMMARY_DIRNAME


def _catalog_label_map(base_dir: Path | str) -> dict[str, str]:
    root = Path(base_dir)
    if root.name == SUMMARY_DIRNAME:
        root = root.parent

    catalog_path = root / CATALOG_FILENAME
    if not catalog_path.exists():
        return {}

    label_map: dict[str, str] = {}
    with catalog_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_name = str(row.get("class_name", "")).strip()
            if class_name:
                label_map.setdefault(_slugify(class_name), class_name)
    return label_map


def load_library_summaries(base_dir: Path | str) -> dict[str, dict[str, Any]]:
    summary_dir = _summary_dir(base_dir)
    if not summary_dir.exists():
        return {}

    label_map = _catalog_label_map(base_dir)
    references: dict[str, dict[str, Any]] = {}

    for path in sorted(summary_dir.glob("*.csv")):
        wavelengths: list[float] = []
        mean_values: list[float] = []
        std_values: list[float] = []
        sample_counts: list[int] = []

        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                wavelengths.append(float(str(row.get("wavelength_nm", "")).strip()))
                mean_values.append(float(str(row.get("mean_value", "")).strip()))
                std_text = str(row.get("std_value", "")).strip()
                count_text = str(row.get("sample_count", "")).strip()
                std_values.append(float(std_text) if std_text else np.nan)
                sample_counts.append(int(count_text) if count_text else 0)

        if not wavelengths:
            continue

        class_slug = path.stem
        class_name = label_map.get(class_slug, class_slug.replace("-", " "))
        references[class_name] = {
            "class_name": class_name,
            "class_slug": class_slug,
            "summary_path": path,
            "wavelengths": np.asarray(wavelengths, dtype=np.float64),
            "mean_spectrum": np.asarray(mean_values, dtype=np.float64),
            "std_spectrum": np.asarray(std_values, dtype=np.float64),
            "sample_count": int(max(sample_counts)) if sample_counts else 0,
        }

    return references


def stack_named_bands(band_stack: Mapping[str, np.ndarray], band_order: Sequence[str]) -> np.ndarray:
    arrays = [np.asarray(band_stack[band_name], dtype=np.float32) for band_name in band_order]
    if not arrays:
        raise ValueError("band_order must contain at least one band.")

    first_shape = arrays[0].shape
    for band_name, array in zip(band_order, arrays, strict=True):
        if array.ndim != 2:
            raise ValueError(f"Band '{band_name}' must be 2-D, got shape {array.shape}.")
        if array.shape != first_shape:
            raise ValueError(f"Band '{band_name}' has shape {array.shape}, expected {first_shape}.")

    return np.stack(arrays, axis=-1)


def align_reference_to_wavelengths(
    reference_spectrum: Sequence[float],
    reference_wavelengths: Sequence[float],
    target_wavelengths: Sequence[float],
    max_delta_nm: float = 2.0,
) -> np.ndarray:
    spectrum = np.asarray(reference_spectrum, dtype=np.float64).reshape(-1)
    wavelengths = np.asarray(reference_wavelengths, dtype=np.float64).reshape(-1)
    targets = np.asarray(target_wavelengths, dtype=np.float64).reshape(-1)

    if spectrum.size != wavelengths.size:
        raise ValueError("reference_spectrum and reference_wavelengths must have the same length.")

    aligned = np.full(targets.size, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(spectrum) & np.isfinite(wavelengths)
    if not np.any(valid_mask):
        return aligned

    for wavelength, value in zip(wavelengths[valid_mask], spectrum[valid_mask], strict=True):
        best_idx = int(np.argmin(np.abs(targets - wavelength)))
        if abs(float(targets[best_idx] - wavelength)) <= float(max_delta_nm):
            aligned[best_idx] = float(value)

    return aligned


def resample_airborne_spectrum_to_s2(
    reference_spectrum: Sequence[float],
    airborne_wavelengths: Sequence[float],
) -> np.ndarray:
    spectrum = np.asarray(reference_spectrum, dtype=np.float64).reshape(-1)
    wavelengths = np.asarray(airborne_wavelengths, dtype=np.float64).reshape(-1)
    if spectrum.size != wavelengths.size:
        raise ValueError("reference_spectrum and airborne_wavelengths must have the same length.")

    valid_mask = np.isfinite(spectrum) & np.isfinite(wavelengths)
    if valid_mask.sum() < 2:
        raise ValueError("At least two finite wavelength/value pairs are required for resampling.")

    wavelengths_valid = wavelengths[valid_mask]
    spectrum_valid = spectrum[valid_mask]
    order = np.argsort(wavelengths_valid)
    wavelengths_valid = wavelengths_valid[order]
    spectrum_valid = spectrum_valid[order]

    resampled = np.interp(
        SENTINEL2_TARGET_WAVELENGTHS,
        wavelengths_valid,
        spectrum_valid,
    )
    within_bounds = (
        (SENTINEL2_TARGET_WAVELENGTHS >= wavelengths_valid.min())
        & (SENTINEL2_TARGET_WAVELENGTHS <= wavelengths_valid.max())
    )
    resampled = resampled.astype(np.float64, copy=False)
    resampled[~within_bounds] = np.nan
    return resampled


def spectral_angle(reference, spectra):
    reference_arr = np.asarray(reference, dtype=np.float64).reshape(-1)
    spectra_arr = np.asarray(spectra, dtype=np.float64)

    squeeze = False
    if spectra_arr.ndim == 1:
        spectra_arr = spectra_arr.reshape(1, -1)
        squeeze = True
    elif spectra_arr.ndim != 2:
        raise ValueError("spectra must be a 1-D or 2-D array.")

    if spectra_arr.shape[1] != reference_arr.size:
        raise ValueError(
            f"Spectra have {spectra_arr.shape[1]} bands, but reference has {reference_arr.size}."
        )

    reference_valid = np.isfinite(reference_arr)
    spectra_valid = np.isfinite(spectra_arr)
    valid_mask = spectra_valid & reference_valid[np.newaxis, :]

    masked_reference = np.where(valid_mask, reference_arr[np.newaxis, :], 0.0)
    masked_spectra = np.where(valid_mask, spectra_arr, 0.0)

    dot_product = np.sum(masked_spectra * masked_reference, axis=1)
    reference_norm = np.linalg.norm(masked_reference, axis=1)
    spectra_norm = np.linalg.norm(masked_spectra, axis=1)
    denominator = reference_norm * spectra_norm

    angles = np.full(spectra_arr.shape[0], np.nan, dtype=np.float64)
    valid_rows = (valid_mask.sum(axis=1) > 0) & (denominator > 0.0)
    if np.any(valid_rows):
        cosines = np.clip(dot_product[valid_rows] / denominator[valid_rows], -1.0, 1.0)
        angles[valid_rows] = np.arccos(cosines)

    return angles[0] if squeeze else angles


def spectral_angle_image(
    reference,
    image_stack,
    ignore_value: float | None = None,
    invalid_mask: np.ndarray | None = None,
    pixel_mask: np.ndarray | None = None,
    chunk_rows: int = 128,
) -> np.ndarray:
    reference_arr = np.asarray(reference, dtype=np.float64).reshape(-1)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")

    stack_shape = getattr(image_stack, "shape", None)
    if stack_shape is None or len(stack_shape) != 3:
        raise ValueError("image_stack must have shape (rows, cols, bands).")

    rows, cols, band_count = stack_shape
    if band_count != reference_arr.size:
        raise ValueError(f"image_stack has {band_count} bands, but reference has {reference_arr.size}.")

    band_selector = np.isfinite(reference_arr)
    if invalid_mask is not None:
        invalid_arr = np.asarray(invalid_mask, dtype=bool).reshape(-1)
        if invalid_arr.size != band_count:
            raise ValueError("invalid_mask length must match the band dimension.")
        band_selector &= ~invalid_arr

    if not np.any(band_selector):
        raise ValueError("No valid reference bands remain for SAM.")

    selected_reference = reference_arr[band_selector]
    output = np.full((rows, cols), np.nan, dtype=np.float32)
    pixel_mask_arr = None if pixel_mask is None else np.asarray(pixel_mask, dtype=bool)

    if pixel_mask_arr is not None and pixel_mask_arr.shape != (rows, cols):
        raise ValueError(f"pixel_mask has shape {pixel_mask_arr.shape}, expected {(rows, cols)}.")

    for row_start in range(0, rows, chunk_rows):
        row_stop = min(row_start + chunk_rows, rows)
        chunk_mask = None if pixel_mask_arr is None else pixel_mask_arr[row_start:row_stop]
        if chunk_mask is not None and not np.any(chunk_mask):
            continue

        chunk = np.asarray(image_stack[row_start:row_stop, :, :][:, :, band_selector], dtype=np.float64)
        if ignore_value is not None:
            chunk[chunk >= ignore_value] = np.nan
        chunk[chunk < 0.0] = np.nan

        flat_chunk = chunk.reshape(-1, selected_reference.size)
        if chunk_mask is None:
            angles = spectral_angle(selected_reference, flat_chunk)
            output[row_start:row_stop] = np.asarray(angles, dtype=np.float32).reshape(row_stop - row_start, cols)
        else:
            flat_mask = chunk_mask.reshape(-1)
            row_output = np.full(flat_mask.shape, np.nan, dtype=np.float32)
            if np.any(flat_mask):
                row_output[flat_mask] = np.asarray(
                    spectral_angle(selected_reference, flat_chunk[flat_mask]),
                    dtype=np.float32,
                )
            output[row_start:row_stop] = row_output.reshape(row_stop - row_start, cols)

    return output


def _trim_fit_samples(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size < 8:
        return x, y

    x_lo, x_hi = np.percentile(x, [1, 99])
    y_lo, y_hi = np.percentile(y, [1, 99])
    keep = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
    if int(keep.sum()) < 2:
        return x, y
    return x[keep], y[keep]


def _gain_only_fit(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        return 1.0
    return float(np.dot(x, y) / denominator)


def fit_bandwise_linear_calibration(
    airborne_stack,
    sentinel_stack,
    mask,
) -> dict[str, dict[str, float | int | str]]:
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 2:
        raise ValueError("mask must be 2-D.")

    if isinstance(airborne_stack, Mapping) and isinstance(sentinel_stack, Mapping):
        band_names = list(airborne_stack.keys())
        if set(band_names) != set(sentinel_stack.keys()):
            raise ValueError("airborne_stack and sentinel_stack must contain the same band names.")
        airborne_arr = stack_named_bands(airborne_stack, band_names).astype(np.float64)
        sentinel_arr = stack_named_bands(sentinel_stack, band_names).astype(np.float64)
    else:
        airborne_arr = np.asarray(airborne_stack, dtype=np.float64)
        sentinel_arr = np.asarray(sentinel_stack, dtype=np.float64)
        if airborne_arr.ndim != 3 or sentinel_arr.ndim != 3:
            raise ValueError("airborne_stack and sentinel_stack must both be 3-D arrays when not using mappings.")
        if airborne_arr.shape != sentinel_arr.shape:
            raise ValueError(
                f"Stack shape mismatch: airborne {airborne_arr.shape}, sentinel {sentinel_arr.shape}."
            )
        band_names = [f"band_{idx + 1}" for idx in range(airborne_arr.shape[-1])]

    if airborne_arr.shape[:2] != mask_arr.shape:
        raise ValueError(
            f"mask shape {mask_arr.shape} does not match stack shape {airborne_arr.shape[:2]}."
        )

    calibration: dict[str, dict[str, float | int | str]] = {}

    for band_idx, band_name in enumerate(band_names):
        airborne_band = airborne_arr[:, :, band_idx]
        sentinel_band = sentinel_arr[:, :, band_idx]
        valid_mask = mask_arr & np.isfinite(airborne_band) & np.isfinite(sentinel_band)

        x = sentinel_band[valid_mask]
        y = airborne_band[valid_mask]
        valid_pixels = int(x.size)

        if valid_pixels == 0:
            calibration[band_name] = {
                "gain": 1.0,
                "offset": 0.0,
                "valid_pixels": 0,
                "fit_pixels": 0,
                "method": "identity",
                "rmse_before": np.nan,
                "rmse_after": np.nan,
            }
            continue

        fit_x, fit_y = _trim_fit_samples(x, y)
        method = "gain_offset"

        if fit_x.size < 2 or np.std(fit_x) <= 1e-12:
            gain = _gain_only_fit(x, y)
            offset = 0.0
            method = "gain_only"
        else:
            design = np.column_stack([fit_x, np.ones(fit_x.size, dtype=np.float64)])
            gain, offset = np.linalg.lstsq(design, fit_y, rcond=None)[0]
            gain = float(gain)
            offset = float(offset)

            if (not np.isfinite(gain)) or (not np.isfinite(offset)) or gain <= 0.0 or abs(offset) > 0.25:
                gain = _gain_only_fit(fit_x, fit_y)
                offset = 0.0
                method = "gain_only"

        calibrated = x * gain + offset
        calibration[band_name] = {
            "gain": float(gain),
            "offset": float(offset),
            "valid_pixels": valid_pixels,
            "fit_pixels": int(fit_x.size),
            "method": method,
            "rmse_before": float(np.sqrt(np.mean((x - y) ** 2))),
            "rmse_after": float(np.sqrt(np.mean((calibrated - y) ** 2))),
        }

    return calibration


def apply_bandwise_linear_calibration(sentinel_stack, calibration_params):
    if isinstance(sentinel_stack, Mapping):
        calibrated: dict[str, np.ndarray] = {}
        for band_name, array in sentinel_stack.items():
            params = calibration_params.get(band_name)
            if params is None:
                raise KeyError(f"Missing calibration parameters for band '{band_name}'.")
            gain = float(params["gain"])
            offset = float(params["offset"])
            output = np.asarray(array, dtype=np.float32).copy()
            finite_mask = np.isfinite(output)
            output[finite_mask] = output[finite_mask] * gain + offset
            calibrated[band_name] = output
        return calibrated

    stack_arr = np.asarray(sentinel_stack, dtype=np.float32)
    if stack_arr.ndim != 3:
        raise ValueError("sentinel_stack must be a mapping or a 3-D array.")

    output = stack_arr.copy()
    band_names = [f"band_{idx + 1}" for idx in range(output.shape[-1])]
    for band_idx, band_name in enumerate(band_names):
        params = calibration_params.get(band_name)
        if params is None:
            raise KeyError(f"Missing calibration parameters for band '{band_name}'.")
        gain = float(params["gain"])
        offset = float(params["offset"])
        finite_mask = np.isfinite(output[:, :, band_idx])
        output[:, :, band_idx][finite_mask] = output[:, :, band_idx][finite_mask] * gain + offset

    return output


def sam_classification(angle_stack, threshold: float | None = None) -> dict[str, Any]:
    if isinstance(angle_stack, Mapping):
        class_names = list(angle_stack.keys())
        angle_cube = stack_named_bands(angle_stack, class_names).astype(np.float64)
    else:
        angle_cube = np.asarray(angle_stack, dtype=np.float64)
        if angle_cube.ndim != 3:
            raise ValueError("angle_stack must be a mapping or a 3-D array.")
        class_names = [f"class_{idx + 1}" for idx in range(angle_cube.shape[-1])]

    finite_any = np.any(np.isfinite(angle_cube), axis=-1)
    working_cube = np.where(np.isfinite(angle_cube), angle_cube, np.inf)
    best_idx0 = np.argmin(working_cube, axis=-1)
    best_angle = np.take_along_axis(working_cube, best_idx0[..., np.newaxis], axis=-1)[..., 0]

    class_index = np.where(finite_any, best_idx0 + 1, 0).astype(np.int32)
    if threshold is not None:
        threshold_value = float(threshold)
        keep = finite_any & (best_angle <= threshold_value)
        class_index = np.where(keep, class_index, 0).astype(np.int32)
        best_angle = np.where(keep, best_angle, np.nan)
    else:
        best_angle = np.where(finite_any, best_angle, np.nan)

    class_labels = np.empty(class_index.shape, dtype=object)
    class_labels[:] = ""
    for idx, class_name in enumerate(class_names, start=1):
        class_labels[class_index == idx] = class_name

    return {
        "class_names": class_names,
        "class_index": class_index,
        "class_labels": class_labels,
        "best_angle": best_angle.astype(np.float32),
    }


__all__ = [
    "SENTINEL2_SAM_BANDS",
    "SENTINEL2_TARGET_WAVELENGTHS",
    "align_reference_to_wavelengths",
    "apply_bandwise_linear_calibration",
    "fit_bandwise_linear_calibration",
    "load_library_summaries",
    "resample_airborne_spectrum_to_s2",
    "sam_classification",
    "spectral_angle",
    "spectral_angle_image",
    "stack_named_bands",
]
