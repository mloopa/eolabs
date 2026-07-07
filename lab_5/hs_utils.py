"""Helpers for Lab 5 — Odra hyperspectral cube + Sentinel-2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import spectral.io.envi as envi
except ImportError as exc:
    raise ImportError("pip install spectral") from exc

HDR_NAME = "221000_Odra_HS_Blok_A_008_VS_join_atm.hdr"
S2_BANDS_NM = {"B2": 490, "B3": 560, "B4": 665, "B5": 705, "B8": 842}
# Airborne Odra campaign flown in 2022 (file dated Nov 2022). Exact flight day is
# not in the metadata, so we search the 2022 season and pick the nearest cloud-free S2.
ACQ_DATE = "2022-09-15"
EPSG_HS = "EPSG:2177"    # ETRS89 / Poland CS2000 zone 6 (Central Meridian 18E)


def open_odra(data_dir: Path):
    hdr = data_dir / HDR_NAME
    if not hdr.exists():
        raise FileNotFoundError(f"Place {HDR_NAME} in {data_dir}")
    img = envi.open(str(hdr))
    wl = np.array([float(x) for x in img.metadata["wavelength"]])
    ignore = float(img.metadata["data ignore value"])
    return img, wl, ignore


def mask_invalid(arr: np.ndarray, ignore: float) -> np.ndarray:
    out = arr.astype(np.float64)
    out[out >= ignore] = np.nan
    out[out < 0] = np.nan
    return out


def band_index(wl: np.ndarray, nm: float) -> int:
    return int(np.argmin(np.abs(wl - nm)))


def read_bands(img, wl: np.ndarray, targets_nm: list[float], ignore: float) -> np.ndarray:
    idx = [band_index(wl, nm) for nm in targets_nm]
    data = img.read_bands(idx).astype(np.float32)
    return mask_invalid(data, ignore)


def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    for c in range(3):
        ch = out[:, :, c]
        p2, p98 = np.nanpercentile(ch, [2, 98])
        out[:, :, c] = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0, 1)
    return np.nan_to_num(out)


def chl_a(cube: np.ndarray, wl: np.ndarray) -> np.ndarray:
    b665 = cube[:, :, band_index(wl, 665)]
    b705 = cube[:, :, band_index(wl, 705)]
    return (b705 - b665) / np.maximum(b705 + b665, 1e-6)


def doc(cube: np.ndarray, wl: np.ndarray) -> np.ndarray:
    b440 = cube[:, :, band_index(wl, 440)]
    b560 = cube[:, :, band_index(wl, 560)]
    return b440 / np.maximum(b560, 1e-6)


def turbidity(cube: np.ndarray, wl: np.ndarray) -> np.ndarray:
    b670 = cube[:, :, band_index(wl, 670)]
    b560 = cube[:, :, band_index(wl, 560)]
    return b670 / np.maximum(b560, 1e-6)


def sample_patch(img, row: int, col: int, size: int = 7) -> np.ndarray:
    r0, r1 = max(0, row - size // 2), min(img.nrows, row + size // 2 + 1)
    c0, c1 = max(0, col - size // 2), min(img.ncols, col + size // 2 + 1)
    return np.array([img.read_pixel(r, c) for r in range(r0, r1) for c in range(c0, c1)])


def sam_map(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    rows, cols, bands = cube.shape
    ref = np.nan_to_num(ref.astype(np.float64))
    specs = cube.reshape(-1, bands)
    dots = specs @ ref
    norms = np.linalg.norm(specs, axis=1) * (np.linalg.norm(ref) + 1e-12)
    return np.arccos(np.clip(dots / norms, -1, 1)).reshape(rows, cols)


def sam_classify(cube: np.ndarray, library: dict[str, np.ndarray]) -> np.ndarray:
    maps = {k: sam_map(cube, v) for k, v in library.items()}
    stack = np.stack(list(maps.values()))
    best = np.argmin(stack, axis=0)
    names = np.array(list(maps.keys()), dtype=object)
    return names[best]


def s2_indices_np(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Chl-a (NDCI), DOC (B2/B3), turbidity (B4/B3) from S2 reflectance arrays."""
    b3, b4, b5, b2 = bands["B3"], bands["B4"], bands["B5"], bands["B2"]
    return {
        "Chl-a": (b5 - b4) / np.maximum(b5 + b4, 1e-6),
        "DOC": b2 / np.maximum(b3, 1e-6),
        "Turbidity": b4 / np.maximum(b3, 1e-6),
    }


def hs_indices_at_s2(cube: np.ndarray, wl: np.ndarray) -> dict[str, np.ndarray]:
    """Same index formulas as S2 but from HS bands resampled to S2 centres."""
    b2 = cube[:, :, band_index(wl, S2_BANDS_NM["B2"])]
    b3 = cube[:, :, band_index(wl, S2_BANDS_NM["B3"])]
    b4 = cube[:, :, band_index(wl, S2_BANDS_NM["B4"])]
    b5 = cube[:, :, band_index(wl, S2_BANDS_NM["B5"])]
    return {
        "Chl-a": (b5 - b4) / np.maximum(b5 + b4, 1e-6),
        "DOC": b2 / np.maximum(b3, 1e-6),
        "Turbidity": b4 / np.maximum(b3, 1e-6),
    }


def fetch_s2_arrays(image, region) -> dict[str, np.ndarray]:
    """Download S2 reflectance bands over AOI as numpy arrays on a common 10 m grid."""
    import ee

    bands = list(S2_BANDS_NM.keys())
    grid = image.select("B2").projection()  # 10 m reference grid
    resampled = image.select(bands).resample("bilinear").reproject(grid)
    sample = resampled.sampleRectangle(region=region, defaultValue=0)
    info = sample.getInfo()["properties"]
    arrs = {b: np.array(info[b], dtype=np.float64) for b in bands}
    # Guard: crop all to the smallest common shape (sub-pixel rounding at edges)
    h = min(a.shape[0] for a in arrs.values())
    w = min(a.shape[1] for a in arrs.values())
    return {b: a[:h, :w] for b, a in arrs.items()}


def aoi_wgs84(img) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    mi = img.metadata["map info"]
    parts = [str(x).strip() for x in mi] if isinstance(mi, list) else [p.strip() for p in str(mi).replace("{", "").replace("}", "").split(",")]
    e0, n0, px, py = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
    e1 = e0 + img.ncols * px
    n1 = n0 - img.nrows * py
    tr = Transformer.from_crs(EPSG_HS, "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for e, n in [(e0, n0), (e1, n0), (e1, n1), (e0, n1)]:
        lon, lat = tr.transform(e, n)
        lons.append(lon)
        lats.append(lat)
    return min(lons), min(lats), max(lons), max(lats)
