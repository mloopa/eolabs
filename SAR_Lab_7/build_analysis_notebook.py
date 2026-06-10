from pathlib import Path
from textwrap import dedent

import nbformat as nbf


LAB_DIR = Path(__file__).resolve().parent
OUT_FILE = LAB_DIR / "SAR_Lab_7_analysis.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python (SAR Lab 7)",
        "language": "python",
        "name": "sar-lab-7",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

nb.cells = [
    markdown(
        """
        # SAR Lab 7: Xinmo Landslide Analysis

        This notebook documents the June 24, 2017 Xinmo landslide laboratory
        workflow. It combines a pre-event LiCSAR/MintPy SBAS time series with
        event-spanning coherence, Sentinel-1 GRD VV change, and Sentinel-2
        optical change analysis.

        **Current status:** the LiCSAR stack and initial MintPy processing are
        complete. Later sections are prepared for the event and Earth Engine
        analyses and remain safely executable until their inputs are supplied.
        """
    ),
    markdown(
        """
        ## Reproducibility Header

        - **Event:** Xinmo landslide, Sichuan Province, China
        - **Event date:** 2017-06-24
        - **Approximate landslide location:** 103.6506 E, 32.0661 N
        - **Processing AOI:** west 103.62, south 32.04, east 103.68, north 32.09
        - **LiCSAR frame:** `062D_05831_131313`
        - **MintPy inputs:** `output/mintpy/inputs/`
        - **MintPy results:** `output/mintpy/`
        - **Reusable figures:** `figures/`

        The code below discovers the lab directory automatically, whether the
        notebook is launched from `SAR_Lab_7/` or its parent repository.
        """
    ),
    code(
        """
        import os
        import platform
        import sys
        from pathlib import Path

        os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "output" / ".matplotlib"))

        import h5py
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import rasterio
        import scipy
        from matplotlib.patches import Circle
        from matplotlib.dates import DateFormatter, YearLocator
        from matplotlib.ticker import FormatStrFormatter
        from rasterio.windows import from_bounds

        EVENT_DATE = pd.Timestamp("2017-06-24")
        LANDSLIDE_LON = 103.6506
        LANDSLIDE_LAT = 32.0661
        AOI = (103.62, 32.04, 103.68, 32.09)
        LICSAR_FRAME = "062D_05831_131313"

        cwd = Path.cwd().resolve()
        if (cwd / "SAR_Lab_7.pdf").exists():
            LAB_DIR = cwd
        elif (cwd / "SAR_Lab_7" / "SAR_Lab_7.pdf").exists():
            LAB_DIR = cwd / "SAR_Lab_7"
        else:
            raise FileNotFoundError("Run this notebook from SAR_Lab_7/ or its parent repository.")

        MINTPY_DIR = LAB_DIR / "output" / "mintpy"
        INPUTS_DIR = MINTPY_DIR / "inputs"
        FIGURES_DIR = LAB_DIR / "figures"
        EE_OUTPUT_DIR = LAB_DIR / "output" / "earth_engine"
        EVENT_COH_DIR = LAB_DIR / "data" / "event_coherence"
        EVENT_COH_OUTPUT_DIR = LAB_DIR / "output" / "event_coherence"

        for directory in (
            FIGURES_DIR,
            EE_OUTPUT_DIR,
            EVENT_COH_DIR,
            EVENT_COH_OUTPUT_DIR,
            Path(os.environ["MPLCONFIGDIR"]),
        ):
            directory.mkdir(parents=True, exist_ok=True)

        required_files = [
            INPUTS_DIR / "ifgramStack.h5",
            INPUTS_DIR / "geometryGeo.h5",
            MINTPY_DIR / "timeseries.h5",
            MINTPY_DIR / "velocity.h5",
            MINTPY_DIR / "temporalCoherence.h5",
        ]
        missing = [str(path) for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required MintPy files:\\n" + "\\n".join(missing))

        versions = pd.Series(
            {
                "Python": sys.version.split()[0],
                "Platform": platform.platform(),
                "NumPy": np.__version__,
                "Pandas": pd.__version__,
                "SciPy": scipy.__version__,
                "Matplotlib": matplotlib.__version__,
                "h5py": h5py.__version__,
                "Rasterio": rasterio.__version__,
            },
            name="Version",
        )
        versions.to_frame()
        """
    ),
    markdown(
        """
        ## 1. Study Area and Assignment Objectives

        The assignment has three linked objectives:

        1. Reconstruct pre-failure deformation with an SBAS time series.
        2. Compare event mapping using LiCSAR coherence loss and Sentinel-1 VV
           backscatter change.
        3. Map the affected area with Sentinel-2 and quantify land-surface
           change using indices such as NDVI and BSI.

        The plots below verify the MintPy subset and place the approximate
        landslide location inside the processed terrain.
        """
    ),
    code(
        """
        def read_attrs(h5_file):
            return {key: value.decode() if isinstance(value, bytes) else value
                    for key, value in h5_file.attrs.items()}


        def pixel_centers(attrs, shape):
            rows, cols = shape
            x_first = float(attrs["X_FIRST"])
            y_first = float(attrs["Y_FIRST"])
            x_step = float(attrs["X_STEP"])
            y_step = float(attrs["Y_STEP"])
            lon = x_first + x_step * (np.arange(cols) + 0.5)
            lat = y_first + y_step * (np.arange(rows) + 0.5)
            return lon, lat


        def nearest_pixel(lon, lat, target_lon, target_lat):
            x = int(np.argmin(np.abs(lon - target_lon)))
            y = int(np.argmin(np.abs(lat - target_lat)))
            return y, x


        with h5py.File(INPUTS_DIR / "geometryGeo.h5", "r") as h5:
            geometry_attrs = read_attrs(h5)
            height = h5["height"][:]
            longitude = h5["longitude"][:]
            latitude = h5["latitude"][:]

        lon_axis = longitude[0, :]
        lat_axis = latitude[:, 0]
        landslide_yx = nearest_pixel(
            lon_axis, lat_axis, LANDSLIDE_LON, LANDSLIDE_LAT
        )

        dem_data = np.ma.masked_where(height == 0, height)
        dem_cmap = plt.get_cmap("terrain").copy()
        dem_cmap.set_bad("#d9d9d9")

        fig, ax = plt.subplots(figsize=(8, 6))
        dem = ax.imshow(
            dem_data,
            extent=[lon_axis.min(), lon_axis.max(), lat_axis.min(), lat_axis.max()],
            origin="upper",
            cmap=dem_cmap,
        )
        ax.scatter(
            LANDSLIDE_LON,
            LANDSLIDE_LAT,
            marker="*",
            s=180,
            color="red",
            edgecolor="white",
            linewidth=0.8,
            label="Approximate landslide location",
        )
        ax.set(
            title="Xinmo Landslide Study Area and MintPy DEM",
            xlabel="Longitude",
            ylabel="Latitude",
        )
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.legend(loc="lower left")
        fig.colorbar(dem, ax=ax, label="Elevation (m)")
        fig.tight_layout()
        study_area_figure = FIGURES_DIR / "study_area_dem.png"
        fig.savefig(study_area_figure, dpi=220, bbox_inches="tight")
        plt.show()

        print(f"Nearest MintPy pixel to the landslide: Y/X = {landslide_yx}")
        print(f"Saved: {study_area_figure.relative_to(LAB_DIR)}")
        """
    ),
    markdown(
        """
        ## 2. LiCSAR Network and MintPy Processing

        LiCSAR frame `062D_05831_131313` was filtered to interferograms ending
        before the event. A 108-day maximum temporal baseline was used because
        the original 72-day selection produced two disconnected network
        components. The expanded selection contains a bridging interferogram
        and can be inverted as one connected network.
        """
    ),
    code(
        """
        with h5py.File(INPUTS_DIR / "ifgramStack.h5", "r") as h5:
            stack_attrs = read_attrs(h5)
            date_pairs = h5["date"][:].astype(str)
            pair_bperp = h5["bperp"][:].astype(float)
            drop_ifgram = h5["dropIfgram"][:].astype(bool)

        pair_table = pd.DataFrame(date_pairs, columns=["reference_date", "secondary_date"])
        pair_table["reference_date"] = pd.to_datetime(pair_table["reference_date"])
        pair_table["secondary_date"] = pd.to_datetime(pair_table["secondary_date"])
        pair_table["temporal_baseline_days"] = (
            pair_table["secondary_date"] - pair_table["reference_date"]
        ).dt.days
        pair_table["perpendicular_baseline_m"] = pair_bperp
        pair_table["kept"] = drop_ifgram

        acquisitions = pd.DatetimeIndex(
            sorted(set(pair_table["reference_date"]) | set(pair_table["secondary_date"]))
        )
        baseline_by_date = {}
        for row in pair_table.itertuples():
            baseline_by_date.setdefault(row.reference_date, 0.0)
            baseline_by_date.setdefault(
                row.secondary_date,
                baseline_by_date[row.reference_date] + row.perpendicular_baseline_m,
            )

        network_summary = pd.Series(
            {
                "Acquisition start": acquisitions.min().date(),
                "Acquisition end": acquisitions.max().date(),
                "Number of acquisitions": len(acquisitions),
                "Number of interferograms": len(pair_table),
                "Maximum temporal baseline (days)": int(
                    pair_table["temporal_baseline_days"].max()
                ),
                "Perpendicular baseline min (m)": float(pair_bperp.min()),
                "Perpendicular baseline max (m)": float(pair_bperp.max()),
            },
            name="Value",
        )
        display(network_summary.to_frame())

        fig, ax = plt.subplots(figsize=(11, 5.5))
        for row in pair_table.itertuples():
            color = "#3973ac" if row.kept else "#b0b0b0"
            ax.plot(
                [row.reference_date, row.secondary_date],
                [
                    baseline_by_date.get(row.reference_date, 0.0),
                    baseline_by_date.get(row.secondary_date, row.perpendicular_baseline_m),
                ],
                color=color,
                linewidth=0.7,
                alpha=0.6,
            )
        ax.scatter(
            acquisitions,
            [baseline_by_date.get(date, 0.0) for date in acquisitions],
            color="#111111",
            s=18,
            zorder=3,
        )
        ax.axvline(EVENT_DATE, color="red", linestyle="--", linewidth=1.2, label="Event date")
        ax.set(
            title="LiCSAR Interferogram Network Used for MintPy SBAS",
            xlabel="Acquisition date",
            ylabel="Relative perpendicular baseline (m)",
        )
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        network_figure = FIGURES_DIR / "mintpy_network_overview.png"
        fig.savefig(network_figure, dpi=220, bbox_inches="tight")
        plt.show()

        print(f"Saved: {network_figure.relative_to(LAB_DIR)}")
        """
    ),
    markdown(
        """
        ## 3. Reference-Point Selection and SBAS Result

        MintPy was first run with an automatic reference point. A nearby stable
        pixel outside the landslide was then selected manually and the workflow
        rerun from `reference_point`. The current HDF5 metadata records the
        final reference at Y/X `19/26`.

        Velocity is shown in millimeters per year for readability. The
        time-series plot compares the approximate landslide pixel with the
        final reference pixel.
        """
    ),
    code(
        """
        with h5py.File(MINTPY_DIR / "velocity.h5", "r") as h5:
            velocity_attrs = read_attrs(h5)
            velocity = h5["velocity"][:]
            velocity_std = h5["velocityStd"][:]

        with h5py.File(MINTPY_DIR / "temporalCoherence.h5", "r") as h5:
            temporal_coherence = h5["temporalCoherence"][:]

        ref_y = int(velocity_attrs["REF_Y"])
        ref_x = int(velocity_attrs["REF_X"])
        ref_lon = float(velocity_attrs["REF_LON"])
        ref_lat = float(velocity_attrs["REF_LAT"])

        velocity_mm = velocity * 1000.0
        valid_velocity = velocity_mm[np.isfinite(velocity_mm)]
        vmax = max(5.0, float(np.nanpercentile(np.abs(valid_velocity), 98)))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
        extent = [lon_axis.min(), lon_axis.max(), lat_axis.min(), lat_axis.max()]

        im0 = axes[0].imshow(
            velocity_mm,
            extent=extent,
            origin="upper",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[0].scatter(
            LANDSLIDE_LON, LANDSLIDE_LAT, marker="*", s=150,
            color="yellow", edgecolor="black", linewidth=0.8, label="Landslide"
        )
        axes[0].scatter(
            ref_lon, ref_lat, marker="s", s=55,
            facecolor="none", edgecolor="black", linewidth=1.4, label="Reference"
        )
        axes[0].set(title="MintPy LOS Velocity", xlabel="Longitude", ylabel="Latitude")
        axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axes[0].legend(loc="lower left")
        fig.colorbar(im0, ax=axes[0], label="LOS velocity (mm/year)")

        im1 = axes[1].imshow(
            temporal_coherence,
            extent=extent,
            origin="upper",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[1].scatter(
            LANDSLIDE_LON, LANDSLIDE_LAT, marker="*", s=150,
            color="red", edgecolor="white", linewidth=0.8
        )
        axes[1].scatter(
            ref_lon, ref_lat, marker="s", s=55,
            facecolor="none", edgecolor="white", linewidth=1.4
        )
        axes[1].set(
            title="MintPy Temporal Coherence",
            xlabel="Longitude",
            ylabel="Latitude",
        )
        axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        fig.colorbar(im1, ax=axes[1], label="Temporal coherence")

        result_map_figure = FIGURES_DIR / "mintpy_velocity_temporal_coherence.png"
        fig.savefig(result_map_figure, dpi=220, bbox_inches="tight")
        plt.show()

        reference_summary = pd.Series(
            {
                "Reference Y": ref_y,
                "Reference X": ref_x,
                "Reference latitude": ref_lat,
                "Reference longitude": ref_lon,
                "Landslide Y": landslide_yx[0],
                "Landslide X": landslide_yx[1],
                "Landslide velocity (mm/year)": velocity_mm[landslide_yx],
                "Landslide temporal coherence": temporal_coherence[landslide_yx],
            },
            name="Value",
        )
        display(reference_summary.to_frame())
        print(f"Saved: {result_map_figure.relative_to(LAB_DIR)}")
        """
    ),
    code(
        """
        with h5py.File(MINTPY_DIR / "timeseries.h5", "r") as h5:
            timeseries_attrs = read_attrs(h5)
            dates = pd.to_datetime(h5["date"][:].astype(str))
            displacement = h5["timeseries"][:]

        landslide_series_mm = displacement[:, landslide_yx[0], landslide_yx[1]] * 1000.0
        reference_series_mm = displacement[:, ref_y, ref_x] * 1000.0

        time_series_table = pd.DataFrame(
            {
                "date": dates,
                "landslide_displacement_mm": landslide_series_mm,
                "reference_displacement_mm": reference_series_mm,
            }
        ).set_index("date")

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(
            time_series_table.index,
            time_series_table["landslide_displacement_mm"],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=f"Landslide pixel Y/X {landslide_yx[0]}/{landslide_yx[1]}",
        )
        ax.plot(
            time_series_table.index,
            time_series_table["reference_displacement_mm"],
            color="black",
            linewidth=1.0,
            alpha=0.7,
            label=f"Reference pixel Y/X {ref_y}/{ref_x}",
        )
        ax.axvline(EVENT_DATE, color="red", linestyle="--", linewidth=1.2, label="Event date")
        ax.set(
            title="Pre-Event MintPy LOS Displacement Time Series",
            xlabel="Date",
            ylabel="LOS displacement (mm)",
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        time_series_figure = FIGURES_DIR / "mintpy_landslide_timeseries.png"
        fig.savefig(time_series_figure, dpi=220, bbox_inches="tight")
        plt.show()

        display(time_series_table.tail(10).round(2))
        print(f"Saved: {time_series_figure.relative_to(LAB_DIR)}")
        print(
            "The final LiCSAR acquisition is",
            dates.max().date(),
            "which precedes the June 24 event and omits the final observations shown in the paper.",
        )
        """
    ),
    markdown(
        """
        ### Initial SBAS Interpretation

        Complete this narrative after reviewing the velocity map and several
        nearby pixels in `tsview.py`:

        - Describe the sign and magnitude of deformation at the landslide.
        - State whether the series appears linear, seasonal, accelerating, or
          noisy.
        - Explain why the manual reference pixel improves interpretation.
        - Compare the timing qualitatively with the PS-based result in the
          supplied paper.
        - Note that the LiCSAR stack ends on June 7, 2017, so it cannot reproduce
          every late pre-failure point shown in the paper.
        """
    ),
    markdown(
        """
        ## 4. Event-Spanning Coherence

        The selected LiCSAR pair is `20170607_20170725`, with acquisitions 17
        days before and 31 days after the event. Its 48-day temporal baseline is
        the shortest available pair spanning June 24, 2017.

        LiCSAR stores this coherence raster as unsigned bytes. Valid values are
        normalized by 255 to the conventional `0-1` coherence range. The
        comparison below uses equal-radius samples centered on:

        - the approximate landslide location; and
        - the manually selected stable MintPy reference point.

        These samples are transparent local proxies, not a final mapped
        landslide polygon.
        """
    ),
    code(
        """
        EVENT_PAIR = "20170607_20170725"
        EVENT_COH_FILE = EVENT_COH_DIR / f"{EVENT_PAIR}.geo.cc.tif"
        EVENT_COH_CROP = EVENT_COH_OUTPUT_DIR / f"{EVENT_PAIR}_aoi.geo.cc.tif"
        EVENT_COH_STATS = EVENT_COH_OUTPUT_DIR / f"{EVENT_PAIR}_coherence_stats.csv"
        SAMPLE_RADIUS_DEG = 0.0025

        if not EVENT_COH_FILE.exists():
            raise FileNotFoundError(
                f"Missing {EVENT_COH_FILE.relative_to(LAB_DIR)}. "
                "Download the selected event-spanning coherence raster first."
            )

        with rasterio.open(EVENT_COH_FILE) as src:
            crop_window = from_bounds(*AOI, transform=src.transform)
            crop_window = crop_window.round_offsets().round_lengths()
            coherence_raw = src.read(1, window=crop_window).astype(np.float32)
            coherence_transform = src.window_transform(crop_window)
            coherence_profile = src.profile.copy()
            coherence_profile.update(
                height=coherence_raw.shape[0],
                width=coherence_raw.shape[1],
                transform=coherence_transform,
                dtype="float32",
                nodata=np.nan,
                compress="deflate",
            )

        coherence = coherence_raw / 255.0
        coherence[coherence_raw == 0] = np.nan

        with rasterio.open(EVENT_COH_CROP, "w", **coherence_profile) as dst:
            dst.write(coherence.astype(np.float32), 1)

        coh_rows, coh_cols = np.indices(coherence.shape)
        coh_lon = coherence_transform.c + coherence_transform.a * (coh_cols + 0.5)
        coh_lat = coherence_transform.f + coherence_transform.e * (coh_rows + 0.5)

        sample_centers = {
            "Landslide-centered sample": (LANDSLIDE_LON, LANDSLIDE_LAT),
            "Stable reference sample": (ref_lon, ref_lat),
        }
        sample_values = {}
        stats_rows = []

        for sample_name, (sample_lon, sample_lat) in sample_centers.items():
            sample_mask = (
                (coh_lon - sample_lon) ** 2 + (coh_lat - sample_lat) ** 2
                <= SAMPLE_RADIUS_DEG**2
            )
            values = coherence[sample_mask & np.isfinite(coherence)]
            sample_values[sample_name] = values
            stats_rows.append(
                {
                    "sample": sample_name,
                    "center_lon": sample_lon,
                    "center_lat": sample_lat,
                    "radius_degrees": SAMPLE_RADIUS_DEG,
                    "valid_pixels": values.size,
                    "mean_coherence": values.mean(),
                    "median_coherence": np.median(values),
                    "std_coherence": values.std(),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75),
                    "fraction_below_0_15": np.mean(values < 0.15),
                }
            )

        coherence_stats = pd.DataFrame(stats_rows).set_index("sample")
        coherence_stats.to_csv(EVENT_COH_STATS)

        landslide_mean = coherence_stats.loc[
            "Landslide-centered sample", "mean_coherence"
        ]
        stable_mean = coherence_stats.loc[
            "Stable reference sample", "mean_coherence"
        ]
        mean_ratio = landslide_mean / stable_mean

        display(coherence_stats.round(3))
        print(f"Landslide/stable mean coherence ratio: {mean_ratio:.2f}")
        print(f"Saved crop: {EVENT_COH_CROP.relative_to(LAB_DIR)}")
        print(f"Saved statistics: {EVENT_COH_STATS.relative_to(LAB_DIR)}")
        """
    ),
    code(
        """
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13, 5.5),
            gridspec_kw={"width_ratios": [1.35, 0.75]},
            constrained_layout=True,
        )

        coherence_map = axes[0].imshow(
            coherence,
            extent=[AOI[0], AOI[2], AOI[1], AOI[3]],
            origin="upper",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        colors = {
            "Landslide-centered sample": "#ff3b30",
            "Stable reference sample": "#ffffff",
        }
        for sample_name, (sample_lon, sample_lat) in sample_centers.items():
            axes[0].scatter(
                sample_lon,
                sample_lat,
                marker="*" if "Landslide" in sample_name else "s",
                s=140 if "Landslide" in sample_name else 55,
                facecolor=colors[sample_name] if "Landslide" in sample_name else "none",
                edgecolor="white",
                linewidth=1.2,
                label=sample_name,
                zorder=4,
            )
            axes[0].add_patch(
                Circle(
                    (sample_lon, sample_lat),
                    SAMPLE_RADIUS_DEG,
                    fill=False,
                    edgecolor=colors[sample_name],
                    linewidth=1.2,
                    linestyle="--",
                )
            )

        axes[0].set(
            title=f"Event-Spanning LiCSAR Coherence\\n{EVENT_PAIR}",
            xlabel="Longitude",
            ylabel="Latitude",
        )
        axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axes[0].legend(loc="lower left", fontsize=8)
        fig.colorbar(coherence_map, ax=axes[0], label="Coherence", shrink=0.92)

        box_data = [
            sample_values["Landslide-centered sample"],
            sample_values["Stable reference sample"],
        ]
        box = axes[1].boxplot(
            box_data,
            tick_labels=["Landslide", "Stable reference"],
            patch_artist=True,
            widths=0.6,
        )
        box["boxes"][0].set(facecolor="#ffb3ad")
        box["boxes"][1].set(facecolor="#c7dcef")
        axes[1].scatter(
            np.repeat(1, box_data[0].size),
            box_data[0],
            color="#a71910",
            s=16,
            alpha=0.65,
        )
        axes[1].scatter(
            np.repeat(2, box_data[1].size),
            box_data[1],
            color="#245b8a",
            s=16,
            alpha=0.65,
        )
        axes[1].set(
            title="Local Coherence Comparison",
            ylabel="Coherence",
            ylim=(0, 1),
        )
        axes[1].grid(axis="y", alpha=0.25)

        coherence_figure = FIGURES_DIR / "event_spanning_coherence.png"
        fig.savefig(coherence_figure, dpi=220, bbox_inches="tight")
        plt.show()

        print(f"Saved: {coherence_figure.relative_to(LAB_DIR)}")
        """
    ),
    markdown(
        """
        ### Coherence Interpretation

        The event-spanning pair shows substantially lower coherence in the
        landslide-centered sample than around the stable reference pixel. For
        the equal-radius samples used here, mean coherence is approximately
        `0.046` at the landslide and `0.198` at the stable reference, so the
        landslide sample retains only about 23% of the stable sample's mean
        coherence.

        This indicates that coherence loss is useful as supporting evidence for
        the disturbed area in this case. It is not sufficient as a standalone
        landslide map because coherence is low across much of the mountainous
        AOI. The 48-day interval also includes temporal decorrelation unrelated
        to the failure, while vegetation, moisture change, steep-terrain
        layover/shadow, and geometric differences can all reduce coherence.
        The point-centered sample is also only a proxy until the optical
        analysis provides a mapped landslide polygon.
        """
    ),
    markdown(
        """
        ### Secure Google Earth Engine Initialization

        The Google Earth Engine project ID is read only from the process
        environment and is never printed or written into this notebook:

        ```bash
        export EARTH_ENGINE_PROJECT="your-private-project-id"
        jupyter lab
        ```

        Run both commands in the same terminal. Do not paste the project ID into
        a notebook cell, configuration file, Git commit, or report.
        """
    ),
    code(
        """
        EARTH_ENGINE_PROJECT = os.environ.get("EARTH_ENGINE_PROJECT", "").strip()
        EARTH_ENGINE_READY = False
        ee_aoi = None
        ee_landslide = None
        gee_map = None

        try:
            import ee
            import geemap
        except ImportError:
            print("Install earthengine-api and geemap in the sar-lab-7 environment.")
        else:
            if not EARTH_ENGINE_PROJECT:
                print(
                    "Earth Engine is not initialized in this run. "
                    "Set EARTH_ENGINE_PROJECT privately before starting Jupyter."
                )
            else:
                try:
                    ee.Initialize(project=EARTH_ENGINE_PROJECT)
                    ee_aoi = ee.Geometry.Rectangle(list(AOI))
                    ee_landslide = ee.Geometry.Point(
                        [LANDSLIDE_LON, LANDSLIDE_LAT]
                    )

                    metadata_probe = (
                        ee.ImageCollection("COPERNICUS/S1_GRD")
                        .filterBounds(ee_aoi)
                        .filterDate("2017-06-01", "2017-08-01")
                    )
                    metadata_count = metadata_probe.size().getInfo()

                    gee_map = geemap.Map()
                    gee_map.centerObject(ee_aoi, 12)
                    gee_map.addLayer(
                        ee_aoi,
                        {"color": "yellow"},
                        "Common analysis AOI",
                    )
                    gee_map.addLayer(
                        ee_landslide,
                        {"color": "red"},
                        "Approximate landslide location",
                    )
                    EARTH_ENGINE_READY = True
                    print(
                        "Earth Engine initialized successfully using the private "
                        "environment variable. Project ID is intentionally hidden."
                    )
                    print(
                        "Sentinel-1 metadata probe count for June-July 2017:",
                        metadata_count,
                    )
                    display(gee_map)
                except Exception as exc:
                    print(
                        "Earth Engine initialization or metadata validation failed "
                        f"with {type(exc).__name__}. Check authentication, project "
                        "registration, and permissions. Details are hidden to avoid "
                        "leaking identifiers."
                    )
        """
    ),
    markdown(
        """
        ## 5. Sentinel-1 VV Change

        This section will compare consistent descending-orbit Sentinel-1 GRD
        acquisitions before and after the event. The Earth Engine project ID is
        intentionally not stored in the repository.
        """
    ),
    markdown(
        """
        When Phase 8 is executed, record the selected image IDs, dates, orbit
        direction, and relative orbit. Calculate:

        `VV change (dB) = post-event VV - pre-event VV`

        Use identical visualization ranges for the before/after images and
        compare statistics inside the landslide with a stable control area.
        """
    ),
    markdown(
        """
        ## 6. Sentinel-2 True Color and Landslide Extent

        Phase 9 will use `COPERNICUS/S2_SR_HARMONIZED`. Select cloud-free or
        minimally cloudy pre-event and post-event imagery, apply a cloud and
        shadow mask, display true-color composites, and delineate the affected
        area. Record acquisition IDs, dates, cloud conditions, and calculated
        area.
        """
    ),
    code(
        """
        sentinel2_inputs = {
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "event_date": EVENT_DATE.date().isoformat(),
            "aoi": AOI,
            "true_color_bands": ["B4", "B3", "B2"],
        }
        pd.Series(sentinel2_inputs, name="Value").to_frame()
        """
    ),
    markdown(
        """
        ## 7. NDVI and BSI Change

        The optical analysis will quantify vegetation loss and increased bare
        soil/debris using:

        `NDVI = (B8 - B4) / (B8 + B4)`

        `BSI = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))`

        Compute pre-event, post-event, and difference layers using the same
        cloud-masked images selected in the previous section.
        """
    ),
    code(
        """
        index_definitions = pd.DataFrame(
            {
                "Index": ["NDVI", "BSI"],
                "Formula": [
                    "(B8 - B4) / (B8 + B4)",
                    "((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))",
                ],
                "Expected landslide response": [
                    "Decrease where vegetation was removed",
                    "Increase where soil and debris became exposed",
                ],
            }
        )
        index_definitions
        """
    ),
    markdown(
        """
        ## 8. Cross-Method Comparison

        Complete this table after Phases 6-9. Use both visual evidence and
        computed statistics rather than ranking methods by appearance alone.
        """
    ),
    code(
        """
        comparison = pd.DataFrame(
            [
                ["MintPy SBAS", "Pre-failure motion", "Pending interpretation", "Sparse dates; reference sensitivity"],
                [
                    "LiCSAR coherence",
                    "Event disturbance",
                    "Strong local loss: mean 0.046 at landslide vs 0.198 at stable reference",
                    "AOI is broadly low coherence; 48-day temporal decorrelation; terrain",
                ],
                ["Sentinel-1 VV", "Backscatter change", "Pending Phase 8", "Speckle; geometry; moisture"],
                ["Sentinel-2 true color", "Visible extent", "Pending Phase 9", "Clouds; illumination"],
                ["NDVI change", "Vegetation loss", "Pending Phase 9", "Seasonality; cloud mask"],
                ["BSI change", "Exposed soil/debris", "Pending Phase 9", "Mixed pixels; spectral ambiguity"],
            ],
            columns=["Method", "Primary signal", "Current assessment", "Key limitation"],
        )
        comparison
        """
    ),
    markdown(
        """
        ## 9. Conclusions and Limitations

        The final conclusions should answer:

        - Does the SBAS analysis show meaningful pre-failure deformation?
        - How did reference-point selection affect the interpreted time series?
        - Does coherence loss map the failure effectively?
        - Is Sentinel-1 VV change better, worse, or similar to coherence?
        - Which Sentinel-2 product best maps the affected area?
        - What do NDVI and BSI changes imply about vegetation and exposed soil?

        Limitations to discuss include sparse LiCSAR acquisitions, the June 7
        end date, atmospheric and seasonal signals, low coherence, steep-terrain
        layover/shadow, Sentinel-1 speckle, optical cloud contamination, and
        differing acquisition dates and spatial resolutions.
        """
    ),
    code(
        """
        generated_figures = sorted(FIGURES_DIR.glob("*.png"))
        figure_inventory = pd.DataFrame(
            {
                "figure": [path.name for path in generated_figures],
                "relative_path": [str(path.relative_to(LAB_DIR)) for path in generated_figures],
                "size_kb": [round(path.stat().st_size / 1024, 1) for path in generated_figures],
            }
        )
        display(figure_inventory)
        print("Notebook setup and MintPy analysis completed successfully.")
        """
    ),
]

nbf.write(nb, OUT_FILE)
print(f"Wrote {OUT_FILE}")
