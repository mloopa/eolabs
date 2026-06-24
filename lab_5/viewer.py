#!/usr/bin/env python3
"""
Hyperspectral BSQ Viewer
------------------------
Browse ENVI/BSQ hyperspectral data cubes.
Click any pixel in the RGB preview to display its full spectral signature.
Export the selected spectrum to CSV.

Usage:
    python viewer.py [path/to/file.hdr]

If no path is given the tool searches data/images/ for .hdr files.
Requires Python 3.10+  (uses X | Y union type hints)
"""

import sys
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle

from src.lab5.envi_utils import (
    find_hdr_files,
    get_ignore_value,
    get_rgb_bands,
    load_envi_image,
    parse_map_info,
    parse_wavelengths,
    pixel_to_map,
    read_pixel_spectrum,
    read_rgb,
    read_roi_cube,
    summarize_roi_spectra,
)
from src.lab5.spectral_library import (
    append_catalog_row,
    make_sample_id,
    write_roi_sample,
)

# ── Configuration ─────────────────────────────────────────────────────────────

# Default search directory (relative to this script)
DATA_DIR = Path(__file__).parent / "data" / "images"
SPECTRAL_LIBRARY_DIR = Path(__file__).parent / "data" / "spectral_library"


def export_roi_sample_to_library(
    *,
    base_dir: Path,
    hdr_path: Path,
    class_name: str,
    notes: str,
    wavelengths: np.ndarray | None,
    map_info,
    roi_bounds: tuple[int, int, int, int],
    roi_summary: dict[str, np.ndarray | int],
) -> tuple[Path, Path]:
    row_min, row_max, col_min, col_max = roi_bounds
    row_center = (row_min + row_max) // 2
    col_center = (col_min + col_max) // 2
    map_coords = pixel_to_map(row_center, col_center, map_info)

    mean_spectrum = np.asarray(roi_summary["mean_spectrum"], dtype=np.float64)
    std_spectrum = np.asarray(roi_summary["std_spectrum"], dtype=np.float64)
    valid_pixel_count = np.asarray(
        roi_summary["valid_pixel_count_by_band"],
        dtype=np.int32,
    )
    x = wavelengths if wavelengths is not None else np.arange(len(mean_spectrum))

    sample_id = make_sample_id(hdr_path.stem, class_name)
    raw_path = write_roi_sample(
        base_dir,
        class_name,
        sample_id,
        x,
        mean_spectrum,
        std_spectrum,
        valid_pixel_count,
    )
    catalog_path = append_catalog_row(
        base_dir,
        {
            "sample_id": sample_id,
            "class_name": class_name,
            "scene_id": hdr_path.stem,
            "geometry_type": "rectangle",
            "row_min": row_min,
            "row_max": row_max,
            "col_min": col_min,
            "col_max": col_max,
            "row_center": row_center,
            "col_center": col_center,
            "map_x": None if map_coords is None else map_coords[0],
            "map_y": None if map_coords is None else map_coords[1],
            "pixel_count": roi_summary["pixel_count"],
            "wavelength_count": len(x),
            "source_file": raw_path.relative_to(base_dir).as_posix(),
            "notes": notes,
        },
    )
    return raw_path, catalog_path

# ── Application ───────────────────────────────────────────────────────────────

class HyperspectralViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hyperspectral BSQ Viewer")
        self.root.geometry("1300x720")

        # state
        self.img = None
        self.hdr_path: Path | None = None
        self.wavelengths: np.ndarray | None = None
        self.ignore_value: float | None = None
        self.map_info = None
        self.rgb_display: np.ndarray | None = None   # float32 (lines, samples, 3)
        self.spectrum: np.ndarray | None = None      # 1-D, last clicked pixel
        self.pixel_pos: tuple[int, int] | None = None
        self.roi_bounds: tuple[int, int, int, int] | None = None
        self.roi_summary: dict[str, np.ndarray | int] | None = None
        self.active_selection: str | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None

        self._build_ui()
        self._auto_load()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # top toolbar
        bar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="Open file…", command=self._open_file).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        tk.Button(
            bar,
            text="Export pixel spectrum to CSV…",
            command=self._export_csv,
        ).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        tk.Button(bar, text="Export ROI sample…", command=self._export_roi).pack(
            side=tk.LEFT, padx=4, pady=3
        )
        self.status_var = tk.StringVar(value="No file loaded.")
        tk.Label(bar, textvariable=self.status_var, anchor=tk.W, fg="#444").pack(
            side=tk.LEFT, padx=12
        )

        # matplotlib figure embedded inside the tkinter window
        self.fig = Figure(figsize=(14, 6.5))
        self.ax_rgb = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        NavigationToolbar2Tk(self.canvas, self.root)  # adds zoom/pan/save toolbar
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # click to inspect a pixel, drag to define an ROI
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)

    # ── File loading ──────────────────────────────────────────────────────────

    def _auto_load(self):
        """
        Called at startup. Loads a file from DATA_DIR automatically if
        exactly one .hdr is present, otherwise shows a picker dialog.
        A path can also be passed as a command-line argument.
        """
        if len(sys.argv) > 1:
            self._load(Path(sys.argv[1]))
            return

        if not DATA_DIR.exists():
            self.status_var.set(f"Data directory not found: {DATA_DIR}")
            return

        hdrs = find_hdr_files(DATA_DIR)
        if not hdrs:
            self.status_var.set(f"No .hdr files found in {DATA_DIR}")
            return
        if len(hdrs) == 1:
            self._load(hdrs[0])
        else:
            self._pick_file(hdrs)

    def _pick_file(self, hdrs: list[Path]):
        """Small modal dialog when multiple datasets are available."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Select dataset")
        dlg.grab_set()
        tk.Label(dlg, text="Multiple datasets found — select one to open:").pack(
            padx=12, pady=8
        )
        lb = tk.Listbox(dlg, width=72, height=min(len(hdrs), 10))
        lb.pack(padx=12, pady=4)
        for h in hdrs:
            lb.insert(tk.END, h.name)
        lb.selection_set(0)

        def on_ok():
            idx = lb.curselection()
            dlg.destroy()
            self._load(hdrs[idx[0]])

        tk.Button(dlg, text="Open", command=on_ok, width=12).pack(pady=8)
        dlg.wait_window()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open ENVI header (.hdr)",
            initialdir=DATA_DIR if DATA_DIR.exists() else Path.home(),
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        if path:
            self._load(Path(path))

    def _load(self, hdr_path: Path):
        self.status_var.set(f"Loading RGB bands from  {hdr_path.name} …")
        self.root.update_idletasks()
        try:
            self.hdr_path = hdr_path
            self.img = load_envi_image(hdr_path)
            meta = self.img.metadata
            self.wavelengths = parse_wavelengths(meta)
            self.ignore_value = get_ignore_value(meta)
            self.map_info = parse_map_info(meta)
            r, g, b = get_rgb_bands(meta)
            self.rgb_display = read_rgb(self.img, (r, g, b), self.ignore_value)
            self.spectrum = None
            self.pixel_pos = None
            self.roi_bounds = None
            self.roi_summary = None
            self.active_selection = None
            self.drag_start = None
            self.drag_current = None
            self._refresh_plots()
            self.status_var.set(
                f"{hdr_path.name}  |  "
                f"{self.img.nrows} lines × {self.img.ncols} samples × "
                f"{self.img.nbands} bands  |  "
                "Click for a pixel or drag a rectangle ROI."
            )
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))
            self.status_var.set("Load failed.")

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _refresh_plots(self):
        # ── left: RGB image ──
        self.ax_rgb.clear()
        if self.rgb_display is not None:
            self.ax_rgb.imshow(
                self.rgb_display, interpolation="bilinear", aspect="auto"
            )
            preview_bounds = self._current_preview_bounds()
            if preview_bounds is not None:
                row_min, row_max, col_min, col_max = preview_bounds
                self.ax_rgb.add_patch(
                    Rectangle(
                        (col_min - 0.5, row_min - 0.5),
                        col_max - col_min + 1,
                        row_max - row_min + 1,
                        linewidth=1.6,
                        edgecolor="gold",
                        facecolor="none",
                        linestyle="--" if self.drag_start is not None else "-",
                    )
                )
            if self.pixel_pos is not None:
                row, col = self.pixel_pos
                self.ax_rgb.plot(
                    col, row, "r+", markersize=14, markeredgewidth=2.5
                )
        self.ax_rgb.set_title("RGB preview — click for pixel, drag for ROI")
        self.ax_rgb.axis("off")

        # ── right: spectral signature ──
        self.ax_spec.clear()
        if self.active_selection == "roi" and self.roi_summary is not None:
            self._draw_roi_spectrum()
        elif self.spectrum is not None and self.pixel_pos is not None:
            row, col = self.pixel_pos
            x = (
                self.wavelengths
                if self.wavelengths is not None
                else np.arange(len(self.spectrum))
            )
            xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
            self.ax_spec.plot(x, self.spectrum, linewidth=1.2, color="steelblue")
            self._set_robust_ylim(self.ax_spec, self.spectrum)
            self.ax_spec.set_title(f"Spectral signature — row {row},  col {col}")
            self.ax_spec.set_xlabel(xlabel)
            self.ax_spec.set_ylabel("Reflectance (× 10⁻⁴)")
            self.ax_spec.grid(True, alpha=0.3)
        else:
            self.ax_spec.set_title("Spectral signature")
            self.ax_spec.text(
                0.5, 0.5,
                "Click a pixel in the RGB image",
                ha="center", va="center",
                transform=self.ax_spec.transAxes,
                color="gray", fontsize=12,
            )

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    # ── Spectrum plot helpers ─────────────────────────────────────────────────

    def _set_robust_ylim(self, ax, spectrum: np.ndarray):
        """Set y-axis limits to the 2nd–98th percentile of valid values."""
        valid = spectrum[np.isfinite(spectrum)]
        if valid.size == 0:
            return
        lo, hi = np.percentile(valid, [2, 98])
        margin = max((hi - lo) * 0.08, 1e-6)
        ax.set_ylim(lo - margin, hi + margin)

    # ── Mouse click handler ───────────────────────────────────────────────────

    def _draw_roi_spectrum(self):
        assert self.roi_bounds is not None
        assert self.roi_summary is not None

        row_min, row_max, col_min, col_max = self.roi_bounds
        x = (
            self.wavelengths
            if self.wavelengths is not None
            else np.arange(len(self.roi_summary["mean_spectrum"]))
        )
        mean_spectrum = np.asarray(self.roi_summary["mean_spectrum"], dtype=np.float64)
        std_spectrum = np.asarray(self.roi_summary["std_spectrum"], dtype=np.float64)
        valid_mask = ~np.isnan(mean_spectrum)

        self.ax_spec.plot(x, mean_spectrum, linewidth=1.4, color="darkorange")
        self.ax_spec.fill_between(
            x,
            mean_spectrum - std_spectrum,
            mean_spectrum + std_spectrum,
            where=valid_mask,
            color="orange",
            alpha=0.2,
        )
        self._set_robust_ylim(self.ax_spec, mean_spectrum)
        self.ax_spec.set_title(
            "ROI spectrum"
            f" — rows {row_min}:{row_max}, cols {col_min}:{col_max}"
        )
        self.ax_spec.set_xlabel(
            "Wavelength (nm)" if self.wavelengths is not None else "Band index"
        )
        self.ax_spec.set_ylabel("Reflectance (× 10⁻⁴)")
        self.ax_spec.grid(True, alpha=0.3)
        self.ax_spec.text(
            0.02,
            0.98,
            (
                f"Pixels: {self.roi_summary['pixel_count']}\n"
                f"Valid pixels: {self.roi_summary['total_valid_pixel_count']}"
            ),
            transform=self.ax_spec.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#444",
        )

    def _event_to_pixel(self, event) -> tuple[int, int] | None:
        if (
            self.img is None
            or event.inaxes is not self.ax_rgb
            or event.xdata is None
            or event.ydata is None
        ):
            return None
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < self.img.nrows and 0 <= col < self.img.ncols):
            return None
        return row, col

    def _normalize_bounds(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        row_min, row_max = sorted((start[0], end[0]))
        col_min, col_max = sorted((start[1], end[1]))
        return row_min, row_max, col_min, col_max

    def _current_preview_bounds(self) -> tuple[int, int, int, int] | None:
        if self.drag_start is not None and self.drag_current is not None:
            return self._normalize_bounds(self.drag_start, self.drag_current)
        return self.roi_bounds

    def _select_pixel(self, row: int, col: int):
        self.pixel_pos = (row, col)
        # Do not pass invalid_band_mask — the "bad" bands in the O2-A
        # absorption region (689-737 nm) contain usable reflectance data.
        self.spectrum = read_pixel_spectrum(
            self.img,
            row,
            col,
            self.ignore_value,
        )
        self.active_selection = "pixel"
        self._refresh_plots()
        self.status_var.set(
            f"Pixel ({row}, {col})  |  "
            "Use 'Export pixel spectrum to CSV…' to save."
        )

    def _select_roi(self, bounds: tuple[int, int, int, int]):
        row_min, row_max, col_min, col_max = bounds
        roi_cube = read_roi_cube(self.img, row_min, row_max, col_min, col_max)
        self.roi_summary = summarize_roi_spectra(
            roi_cube,
            self.ignore_value,
        )
        self.roi_bounds = bounds
        self.active_selection = "roi"
        self._refresh_plots()
        self.status_var.set(
            f"ROI rows {row_min}:{row_max}, cols {col_min}:{col_max}  |  "
            f"Pixels: {self.roi_summary['pixel_count']}  |  "
            f"Valid: {self.roi_summary['total_valid_pixel_count']}  |  "
            "Use 'Export ROI sample…' to save."
        )

    def _on_mouse_press(self, event):
        if event.button != 1:
            return
        pixel = self._event_to_pixel(event)
        if pixel is None:
            return
        self.drag_start = pixel
        self.drag_current = pixel

    def _on_mouse_move(self, event):
        if self.drag_start is None:
            return
        pixel = self._event_to_pixel(event)
        if pixel is None:
            return
        self.drag_current = pixel
        self._refresh_plots()

    def _on_mouse_release(self, event):
        if event.button != 1 or self.drag_start is None:
            return

        end_pixel = self._event_to_pixel(event) or self.drag_current or self.drag_start
        start_pixel = self.drag_start
        self.drag_start = None
        self.drag_current = None

        if end_pixel is None:
            self._refresh_plots()
            return

        row_delta = abs(end_pixel[0] - start_pixel[0])
        col_delta = abs(end_pixel[1] - start_pixel[1])
        if max(row_delta, col_delta) <= 1:
            self._select_pixel(*start_pixel)
            return

        self._select_roi(self._normalize_bounds(start_pixel, end_pixel))

    # ── CSV export ────────────────────────────────────────────────────────────

    def _export_csv(self):
        if self.spectrum is None:
            messagebox.showinfo("Nothing to export", "Click on a pixel first.")
            return

        row, col = self.pixel_pos
        default_name = f"spectrum_r{row}_c{col}.csv"
        path = filedialog.asksaveasfilename(
            title="Save spectrum as CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        x = (
            self.wavelengths
            if self.wavelengths is not None
            else np.arange(len(self.spectrum))
        )
        col_header = "wavelength_nm" if self.wavelengths is not None else "band"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([col_header, "value"])
            for xi, vi in zip(x, self.spectrum):
                writer.writerow([float(xi), "" if np.isnan(vi) else float(vi)])

        self.status_var.set(f"Saved → {path}")
        messagebox.showinfo("Saved", f"Spectrum exported to:\n{path}")

    def _export_roi(self):
        if self.roi_bounds is None or self.roi_summary is None or self.hdr_path is None:
            messagebox.showinfo("Nothing to export", "Draw an ROI first.")
            return

        class_name = simpledialog.askstring(
            "ROI class label",
            "Class label for this ROI sample:",
            parent=self.root,
        )
        if class_name is None:
            return
        class_name = class_name.strip()
        if not class_name:
            messagebox.showinfo("Missing class label", "A class label is required.")
            return

        notes = simpledialog.askstring(
            "ROI notes",
            "Optional notes for this ROI sample:",
            parent=self.root,
        )
        if notes is None:
            notes = ""

        try:
            raw_path, _ = export_roi_sample_to_library(
                base_dir=SPECTRAL_LIBRARY_DIR,
                hdr_path=self.hdr_path,
                class_name=class_name,
                notes=notes.strip(),
                wavelengths=self.wavelengths,
                map_info=self.map_info,
                roi_bounds=self.roi_bounds,
                roi_summary=self.roi_summary,
            )
        except Exception as exc:
            messagebox.showerror("ROI export failed", str(exc))
            return

        self.status_var.set(f"Saved ROI sample → {raw_path}")
        messagebox.showinfo(
            "Saved",
            "ROI sample exported to:\n"
            f"{raw_path}\n\n"
            f"Class: {class_name}",
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralViewer(root)
    root.mainloop()
