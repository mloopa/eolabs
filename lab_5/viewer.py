#!/usr/bin/env python3
"""
Hyperspectral BSQ Viewer + Spectral Library Builder
----------------------------------------------------
• Click pixels on the RGB image — each click drops a yellow marker.
• Select a terrain type, then click "Add to Library" to assign the whole
  pending batch to that class (markers turn class-coloured).
• "Save Library CSV" writes data/spectral_library.csv.

Usage:
    python viewer.py [path/to/file.hdr]

Requires Python 3.10+
"""

import sys
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit("The 'spectral' library is missing.\nInstall: pip install spectral")

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent / "data" / "Obrazy lotnicze"
LIBRARY_PATH = Path(__file__).parent / "data" / "spectral_library.csv"
FALLBACK_RGB = (30, 20, 10)

TERRAIN_TYPES = ["Water", "Vegetation", "Forest", "Urban", "Soil"]

# Marker colours per class (also used for map dots)
CLASS_COLORS: dict[str, str] = {
    "Water":      "#1f77b4",
    "Vegetation": "#2ca02c",
    "Forest":     "#17becf",
    "Urban":      "#d62728",
    "Soil":       "#8c564b",
}
PENDING_COLOR = "#ffdd00"   # yellow: clicked but not yet assigned

# Pre-collected pixels loaded automatically when a matching file is opened.
PRESET_PIXELS: dict[str, dict[str, list[tuple[int, int]]]] = {
    "221000_Odra_HS_Blok_A_008_VS_join_atm.hdr": {
        "Water": [
            (1183, 691), (1712, 829), (1967, 787), (2298, 818), (2527, 1020),
            (2807, 1098), (3078, 1170), (3316, 1364), (3519, 1532), (3629, 1689),
            (3689, 1791), (364,  221),  (508,  575),  (508,  703),  (525,  791),
            (2077, 721),  (1891, 818),  (2069, 780),
        ],
    },
}

# ── ENVI helpers ──────────────────────────────────────────────────────────────

def find_hdr_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.hdr"))


def parse_wavelengths(meta: dict) -> np.ndarray | None:
    wl = meta.get("wavelength")
    return np.array([float(w) for w in wl]) if wl else None


def get_rgb_bands(meta: dict) -> tuple[int, int, int]:
    db = meta.get("default bands")
    if db and len(db) >= 3:
        return tuple(int(float(v)) - 1 for v in db[:3])
    return FALLBACK_RGB


def get_ignore_value(meta: dict) -> float | None:
    raw = meta.get("data ignore value")
    if raw:
        try:
            return float(str(raw).strip())
        except ValueError:
            pass
    return None


def load_image(hdr_path: Path):
    return envi.open(str(hdr_path))


def read_rgb(img, r: int, g: int, b: int, ignore_value: float | None) -> np.ndarray:
    rgb = img.read_bands([r, g, b]).astype(np.float32)
    if ignore_value is not None:
        rgb[rgb >= ignore_value] = np.nan
    rgb[rgb < 0] = np.nan
    for c in range(3):
        ch = rgb[:, :, c]
        p2, p98 = np.nanpercentile(ch, [2, 98])
        rgb[:, :, c] = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0, 1)
    return np.nan_to_num(rgb, nan=0.0)


def read_spectrum(img, row: int, col: int, ignore_value: float | None) -> np.ndarray:
    spec = img.read_pixel(row, col).astype(np.float64)
    if ignore_value is not None:
        spec[spec >= ignore_value] = np.nan
    spec[spec < 0] = np.nan
    return spec


# ── Application ───────────────────────────────────────────────────────────────

class HyperspectralViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hyperspectral BSQ Viewer")
        self.root.geometry("1300x780")

        # image state
        self.img          = None
        self.wavelengths: np.ndarray | None = None
        self.ignore_value: float | None     = None
        self.rgb_display:  np.ndarray | None = None

        # last clicked pixel — drives the spectrum chart
        self.pixel_pos: tuple[int, int] | None = None
        self.spectrum:  np.ndarray | None       = None

        # pending batch: pixels clicked but not yet assigned to a class
        self.pending_pixels: list[tuple[int, int]] = []

        # library: list of dicts {class, row, col, wl_XXX, ...}
        self.library_records: list[dict] = []

        self._build_ui()
        self._auto_load()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── toolbar row 1: file controls ──
        bar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="Open file…", command=self._open_file).pack(
            side=tk.LEFT, padx=4, pady=3)
        tk.Button(bar, text="Export spectrum to CSV…", command=self._export_csv).pack(
            side=tk.LEFT, padx=4, pady=3)
        self.status_var = tk.StringVar(value="No file loaded.")
        tk.Label(bar, textvariable=self.status_var, anchor=tk.W, fg="#444").pack(
            side=tk.LEFT, padx=12)

        # ── toolbar row 2: library builder ──
        lib_bar = tk.Frame(self.root, bd=1, relief=tk.GROOVE, bg="#f0f4f0")
        lib_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(lib_bar, text="Terrain type:", bg="#f0f4f0",
                 font=("", 9, "bold")).pack(side=tk.LEFT, padx=(8, 2), pady=4)

        self.terrain_var = tk.StringVar(value=TERRAIN_TYPES[0])
        ttk.Combobox(
            lib_bar, textvariable=self.terrain_var,
            values=TERRAIN_TYPES, state="readonly", width=12,
        ).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Button(
            lib_bar, text="Add to Library",
            command=self._add_batch_to_library,
            bg="#d4edda", activebackground="#a8d5b5",
        ).pack(side=tk.LEFT, padx=6, pady=4)

        tk.Button(
            lib_bar, text="Clear pending",
            command=self._clear_pending,
            bg="#fff3cd", activebackground="#ffe69c",
        ).pack(side=tk.LEFT, padx=2, pady=4)

        tk.Button(
            lib_bar, text="Save Library CSV",
            command=self._save_library,
            bg="#cce5ff", activebackground="#99caff",
        ).pack(side=tk.LEFT, padx=6, pady=4)

        tk.Button(
            lib_bar, text="Clear Library",
            command=self._clear_library,
            bg="#f8d7da", activebackground="#f1aeb5",
        ).pack(side=tk.LEFT, padx=6, pady=4)

        self.lib_counts_var = tk.StringVar(value=self._counts_text())
        tk.Label(lib_bar, textvariable=self.lib_counts_var,
                 bg="#f0f4f0", fg="#555", font=("", 9)).pack(side=tk.LEFT, padx=16)

        # ── matplotlib figure ──
        self.fig    = Figure(figsize=(14, 6.5))
        self.ax_rgb  = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        NavigationToolbar2Tk(self.canvas, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # ── File loading ──────────────────────────────────────────────────────────

    def _auto_load(self):
        if len(sys.argv) > 1:
            self._load(Path(sys.argv[1]))
            return
        if not DATA_DIR.exists():
            self.status_var.set(f"Data directory not found: {DATA_DIR}")
            return
        hdrs = find_hdr_files(DATA_DIR)
        if not hdrs:
            self.status_var.set(f"No .hdr files found in {DATA_DIR}")
        elif len(hdrs) == 1:
            self._load(hdrs[0])
        else:
            self._pick_file(hdrs)

    def _pick_file(self, hdrs: list[Path]):
        dlg = tk.Toplevel(self.root)
        dlg.title("Select dataset")
        dlg.grab_set()
        tk.Label(dlg, text="Multiple datasets found — select one:").pack(padx=12, pady=8)
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
        self.status_var.set(f"Loading {hdr_path.name} …")
        self.root.update_idletasks()
        try:
            self.img          = load_image(hdr_path)
            meta              = self.img.metadata
            self.wavelengths  = parse_wavelengths(meta)
            self.ignore_value = get_ignore_value(meta)
            r, g, b           = get_rgb_bands(meta)
            self.rgb_display  = read_rgb(self.img, r, g, b, self.ignore_value)
            self.spectrum     = None
            self.pixel_pos    = None
            self.pending_pixels.clear()
            self._load_preset_pixels(hdr_path.name)
            self._refresh_plots()
            self.status_var.set(
                f"{hdr_path.name}  |  "
                f"{self.img.nrows} × {self.img.ncols} × {self.img.nbands} bands  |  "
                "Click pixels to mark them, then 'Add to Library'."
            )
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))
            self.status_var.set("Load failed.")

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _refresh_plots(self):
        # ── left: RGB + markers ──
        self.ax_rgb.clear()
        if self.rgb_display is not None:
            self.ax_rgb.imshow(self.rgb_display, interpolation="bilinear", aspect="auto")

            # library pixels — coloured by class
            for rec in self.library_records:
                color = CLASS_COLORS.get(rec["class"], "#888888")
                self.ax_rgb.plot(
                    rec["col"], rec["row"], "o",
                    color=color, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.4,
                )

            # pending pixels — yellow
            for row, col in self.pending_pixels:
                self.ax_rgb.plot(
                    col, row, "o",
                    color=PENDING_COLOR, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5,
                )

        self.ax_rgb.set_title("RGB preview — click pixels to mark  |  "
                              "● library  ● pending")
        self.ax_rgb.axis("off")

        # ── right: spectrum of last clicked pixel ──
        self.ax_spec.clear()
        if self.spectrum is not None:
            row, col = self.pixel_pos
            x      = self.wavelengths if self.wavelengths is not None \
                     else np.arange(len(self.spectrum))
            xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
            self.ax_spec.plot(x, self.spectrum, linewidth=1.2, color="steelblue")
            self.ax_spec.set_title(f"Spectral signature — row {row}, col {col}")
            self.ax_spec.set_xlabel(xlabel)
            self.ax_spec.set_ylabel("Reflectance (× 10⁻⁴)")
            self.ax_spec.grid(True, alpha=0.3)
        else:
            self.ax_spec.set_title("Spectral signature")
            self.ax_spec.text(0.5, 0.5, "Click a pixel in the RGB image",
                              ha="center", va="center",
                              transform=self.ax_spec.transAxes,
                              color="gray", fontsize=12)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    # ── Mouse click ───────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is not self.ax_rgb or self.img is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < self.img.nrows and 0 <= col < self.img.ncols):
            return

        self.pending_pixels.append((row, col))
        self.pixel_pos = (row, col)
        self.spectrum  = read_spectrum(self.img, row, col, self.ignore_value)
        print(f"Clicked pixel — row: {row}, col: {col}  "
              f"(pending: {len(self.pending_pixels)})")
        self._refresh_plots()
        self.status_var.set(
            f"Pixel ({row}, {col}) added to pending batch  |  "
            f"{len(self.pending_pixels)} pending  |  "
            "Select terrain type and click 'Add to Library'."
        )

    # ── Library builder ───────────────────────────────────────────────────────

    def _make_record(self, terrain: str, row: int, col: int,
                     spectrum: np.ndarray) -> dict:
        record: dict = {"class": terrain, "row": row, "col": col}
        if self.wavelengths is not None:
            for wl, val in zip(self.wavelengths, spectrum):
                record[f"wl_{wl:.2f}"] = "" if np.isnan(val) else float(val)
        else:
            for i, val in enumerate(spectrum):
                record[f"band_{i}"] = "" if np.isnan(val) else float(val)
        return record

    def _add_batch_to_library(self):
        """Assign all pending pixels to the selected terrain type, then clear pending."""
        if self.img is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        if not self.pending_pixels:
            messagebox.showinfo("No pending pixels",
                                "Click some pixels on the image first.")
            return

        terrain = self.terrain_var.get()
        for row, col in self.pending_pixels:
            spec   = read_spectrum(self.img, row, col, self.ignore_value)
            record = self._make_record(terrain, row, col, spec)
            self.library_records.append(record)

        n = len(self.pending_pixels)
        print(f"Added {n} × {terrain} to library  (total: {len(self.library_records)})")
        self.pending_pixels.clear()
        self._update_lib_counts()
        self._refresh_plots()
        self.status_var.set(
            f"Added {n} {terrain} pixel(s) to library  |  {self._counts_text()}"
        )

    def _clear_pending(self):
        if self.pending_pixels:
            self.pending_pixels.clear()
            self._refresh_plots()
            self.status_var.set("Pending batch cleared.")

    def _load_preset_pixels(self, hdr_name: str):
        """Add pre-collected pixels for this file to the library immediately."""
        presets = PRESET_PIXELS.get(hdr_name)
        if not presets:
            return
        total = sum(len(v) for v in presets.values())
        print(f"Loading {total} preset pixels for {hdr_name}…")
        for terrain, pixels in presets.items():
            for row, col in pixels:
                if not (0 <= row < self.img.nrows and 0 <= col < self.img.ncols):
                    print(f"  ⚠ skipping out-of-bounds preset ({row}, {col})")
                    continue
                spec   = read_spectrum(self.img, row, col, self.ignore_value)
                record = self._make_record(terrain, row, col, spec)
                self.library_records.append(record)
        self._update_lib_counts()
        print(f"Presets loaded.  {self._counts_text()}")

    def _counts_text(self) -> str:
        counts: dict[str, int] = {}
        for r in self.library_records:
            counts[r["class"]] = counts.get(r["class"], 0) + 1
        if not counts:
            return "Library: empty"
        return "Library — " + "  ".join(f"{c}: {n}" for c, n in counts.items())

    def _update_lib_counts(self):
        self.lib_counts_var.set(self._counts_text())

    def _save_library(self):
        if not self.library_records:
            messagebox.showinfo("Empty library",
                                "No pixels in the library yet.\n"
                                "Click pixels → select type → Add to Library.")
            return

        LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Collect all column names preserving order
        all_keys: list[str] = []
        seen: set[str] = set()
        for rec in self.library_records:
            for k in rec:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(LIBRARY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for rec in self.library_records:
                writer.writerow(rec)

        msg = (f"Saved {len(self.library_records)} spectra\n"
               f"{self._counts_text()}\n\n→ {LIBRARY_PATH}")
        print(f"Library saved → {LIBRARY_PATH}  ({self._counts_text()})")
        self.status_var.set(f"Library saved → {LIBRARY_PATH}")
        messagebox.showinfo("Library saved", msg)

    def _clear_library(self):
        if not self.library_records:
            return
        if messagebox.askyesno("Clear library",
                               f"Remove all {len(self.library_records)} records?\n"
                               "(The CSV file on disk is not deleted.)"):
            self.library_records.clear()
            self._update_lib_counts()
            self._refresh_plots()
            self.status_var.set("Library cleared.")

    # ── Single-spectrum CSV export ────────────────────────────────────────────

    def _export_csv(self):
        if self.spectrum is None:
            messagebox.showinfo("Nothing to export", "Click on a pixel first.")
            return

        row, col = self.pixel_pos
        path = filedialog.asksaveasfilename(
            title="Save spectrum as CSV",
            initialfile=f"spectrum_r{row}_c{col}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        x          = self.wavelengths if self.wavelengths is not None \
                     else np.arange(len(self.spectrum))
        col_header = "wavelength_nm" if self.wavelengths is not None else "band"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([col_header, "value"])
            for xi, vi in zip(x, self.spectrum):
                writer.writerow([float(xi), "" if np.isnan(vi) else float(vi)])

        self.status_var.set(f"Saved → {path}")
        messagebox.showinfo("Saved", f"Spectrum exported to:\n{path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralViewer(root)
    root.mainloop()
