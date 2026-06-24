from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CATALOG_FILENAME = "catalog.csv"
RAW_DIRNAME = "raw"
SUMMARY_DIRNAME = "summary"
RAW_PIXELS_DIRNAME = "raw_pixels"

CATALOG_COLUMNS = [
    "sample_id",
    "class_name",
    "scene_id",
    "geometry_type",
    "row_min",
    "row_max",
    "col_min",
    "col_max",
    "row_center",
    "col_center",
    "map_x",
    "map_y",
    "pixel_count",
    "wavelength_count",
    "source_file",
    "exported_at",
    "notes",
]

RAW_SAMPLE_COLUMNS = [
    "wavelength_nm",
    "mean_value",
    "std_value",
    "valid_pixel_count",
]

SUMMARY_COLUMNS = [
    "wavelength_nm",
    "mean_value",
    "std_value",
    "sample_count",
]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "sample"


def _to_path(base_dir: Path | str) -> Path:
    return Path(base_dir)


def _catalog_path(base_dir: Path | str) -> Path:
    return _to_path(base_dir) / CATALOG_FILENAME


def _raw_sample_path(base_dir: Path | str, class_name: str, sample_id: str) -> Path:
    class_slug = _slugify(class_name)
    return _to_path(base_dir) / RAW_DIRNAME / class_slug / f"{sample_id}.csv"


def _summary_path(base_dir: Path | str, class_name: str) -> Path:
    class_slug = _slugify(class_name)
    return _to_path(base_dir) / SUMMARY_DIRNAME / f"{class_slug}.csv"


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        return f"{float(value):.10g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _read_float(raw: str) -> float:
    text = raw.strip()
    return float(text) if text else np.nan


def _read_int(raw: str) -> int | None:
    text = raw.strip()
    return int(text) if text else None


def ensure_library_dirs(base_dir: Path | str) -> dict[str, Path]:
    root = _to_path(base_dir)
    raw_dir = root / RAW_DIRNAME
    summary_dir = root / SUMMARY_DIRNAME
    raw_pixels_dir = root / RAW_PIXELS_DIRNAME

    for path in (root, raw_dir, summary_dir, raw_pixels_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": root,
        "raw_dir": raw_dir,
        "summary_dir": summary_dir,
        "raw_pixels_dir": raw_pixels_dir,
        "catalog_path": _catalog_path(root),
    }


def make_sample_id(scene_id: str, class_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{_slugify(scene_id)}__{_slugify(class_name)}__{timestamp}"


def write_roi_sample(
    base_dir: Path | str,
    class_name: str,
    sample_id: str,
    wavelengths: Sequence[float],
    mean_spectrum: Sequence[float],
    std_spectrum: Sequence[float],
    valid_pixel_count: Sequence[int],
) -> Path:
    ensure_library_dirs(base_dir)

    wavelengths_arr = np.asarray(wavelengths, dtype=np.float64)
    mean_arr = np.asarray(mean_spectrum, dtype=np.float64)
    std_arr = np.asarray(std_spectrum, dtype=np.float64)
    valid_arr = np.asarray(valid_pixel_count, dtype=np.int32)

    n_values = len(wavelengths_arr)
    if not (len(mean_arr) == len(std_arr) == len(valid_arr) == n_values):
        raise ValueError("Sample arrays must all have the same length.")

    path = _raw_sample_path(base_dir, class_name, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_SAMPLE_COLUMNS)
        writer.writeheader()
        for wavelength, mean_value, std_value, valid_count in zip(
            wavelengths_arr,
            mean_arr,
            std_arr,
            valid_arr,
            strict=True,
        ):
            writer.writerow(
                {
                    "wavelength_nm": _format_scalar(wavelength),
                    "mean_value": _format_scalar(mean_value),
                    "std_value": _format_scalar(std_value),
                    "valid_pixel_count": _format_scalar(valid_count),
                }
            )

    return path


def append_catalog_row(base_dir: Path | str, row: dict[str, Any]) -> Path:
    paths = ensure_library_dirs(base_dir)
    catalog_path = paths["catalog_path"]

    normalized_row = {column: _format_scalar(row.get(column)) for column in CATALOG_COLUMNS}

    source_file = normalized_row.get("source_file", "")
    if not source_file:
        raise ValueError("Catalog row requires 'source_file'.")

    if not normalized_row.get("exported_at"):
        normalized_row["exported_at"] = datetime.now(timezone.utc).isoformat()

    write_header = not catalog_path.exists()
    with catalog_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)

    return catalog_path


def load_catalog(base_dir: Path | str) -> list[dict[str, str]]:
    catalog_path = _catalog_path(base_dir)
    if not catalog_path.exists():
        return []

    with catalog_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_raw_sample(sample_path: Path) -> list[dict[str, float | int | None]]:
    with sample_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "wavelength_nm": _read_float(row["wavelength_nm"]),
                    "mean_value": _read_float(row["mean_value"]),
                    "std_value": _read_float(row["std_value"]),
                    "valid_pixel_count": _read_int(row["valid_pixel_count"]),
                }
            )
    return rows


def _write_summary_file(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_scalar(row.get(column)) for column in SUMMARY_COLUMNS})


def rebuild_class_summaries(base_dir: Path | str) -> dict[str, Path]:
    ensure_library_dirs(base_dir)
    catalog_rows = load_catalog(base_dir)
    grouped_rows: dict[str, list[dict[str, str]]] = {}

    for row in sorted(catalog_rows, key=lambda item: (item.get("class_name", ""), item.get("sample_id", ""))):
        class_name = row.get("class_name", "").strip()
        if not class_name:
            continue
        grouped_rows.setdefault(class_name, []).append(row)

    written_paths: dict[str, Path] = {}

    for class_name, rows in grouped_rows.items():
        sample_by_wavelength: dict[float, list[float]] = {}

        for row in rows:
            source_file = row.get("source_file", "").strip()
            if not source_file:
                raise ValueError(f"Catalog row for class '{class_name}' is missing source_file.")

            sample_path = _to_path(base_dir) / source_file
            if not sample_path.exists():
                raise FileNotFoundError(f"Raw sample file not found: {sample_path}")

            for sample_row in _load_raw_sample(sample_path):
                wavelength = float(sample_row["wavelength_nm"])
                mean_value = float(sample_row["mean_value"])
                if np.isnan(mean_value):
                    continue
                sample_by_wavelength.setdefault(wavelength, []).append(mean_value)

        summary_rows = []
        for wavelength in sorted(sample_by_wavelength):
            values = np.asarray(sample_by_wavelength[wavelength], dtype=np.float64)
            summary_rows.append(
                {
                    "wavelength_nm": wavelength,
                    "mean_value": np.mean(values),
                    "std_value": np.std(values),
                    "sample_count": len(values),
                }
            )

        summary_path = _summary_path(base_dir, class_name)
        _write_summary_file(summary_path, summary_rows)
        written_paths[class_name] = summary_path

    return written_paths


__all__ = [
    "CATALOG_COLUMNS",
    "CATALOG_FILENAME",
    "RAW_DIRNAME",
    "RAW_PIXELS_DIRNAME",
    "RAW_SAMPLE_COLUMNS",
    "SUMMARY_COLUMNS",
    "SUMMARY_DIRNAME",
    "append_catalog_row",
    "ensure_library_dirs",
    "load_catalog",
    "make_sample_id",
    "rebuild_class_summaries",
    "write_roi_sample",
]
