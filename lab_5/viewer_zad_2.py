#!/usr/bin/env python3
"""
Hyperspectral BSQ Viewer + Spectral Library
--------------------------------------------

Browse ENVI/BSQ hyperspectral data cubes, collect labelled spectral
signatures and export them as a shared spectral library CSV.

Controls:
  Left-click  — select a pixel and display its spectral signature
  Add to lib  — save the current spectrum to the in-memory library
  Export lib  — write the whole library to one CSV (class + wavelengths)
  Clear lib   — remove all collected spectra

Usage:
    python viewer.py [path/to/file.hdr]

If no path is given the tool searches data/images/ for .hdr files.
Requires Python 3.10+
"""

import sys
import csv
import dataclasses
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
from itertools import cycle

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit(
        "The 'spectral' library is missing.\n"
        "Install it with:  pip install spectral"
    )

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).parent / "data" / "images"
FALLBACK_RGB = (30, 20, 10)

# Predefined land-cover classes shown in the dropdown
LAND_COVER_CLASSES = [
    "water",
    "vegetation",
    "forest",
    "bare soil",
    "urban",
    "shadow",
    "other",
]

# Colours used to draw library spectra on the plot (cycles if > 8 entries)
LIBRARY_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]

# ── Data container ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class LibraryEntry:
    label:   str            # land-cover class
    row:     int
    col:     int
    spectrum: np.ndarray   # 1-D float64, NaN for bad values

# ── ENVI header helpers ────────────────────────────────────────────────────────

def find_hdr_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.hdr"))


def parse_wavelengths(meta: dict) -> np.ndarray | None:
    wl = meta.get("wavelength")
    if wl:
        return np.array([float(w) for w in wl])
    return None


def get_rgb_bands(meta: dict) -> tuple[int, int, int]:
    db = meta.get("default_bands") or meta.get("default bands")
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

# ── Image I/O ──────────────────────────────────────────────────────────────────

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

# ── Application ────────────────────────────────────────────────────────────────

class HyperspectralViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hyperspectral BSQ Viewer — Spectral Library")
        self.root.geometry("1500x780")

        # image state
        self.img          = None
        self.wavelengths: np.ndarray | None = None
        self.ignore_value: float | None     = None
        self.rgb_display:  np.ndarray | None = None
        self.spectrum:     np.ndarray | None = None
        self.pixel_pos:    tuple[int, int] | None = None

        # spectral library
        self.library: list[LibraryEntry] = []

        self._build_ui()
        self._auto_load()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top toolbar ──
        bar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="Open file…",        command=self._open_file).pack(side=tk.LEFT, padx=4, pady=3)
        tk.Button(bar, text="Export spectrum…",  command=self._export_single_csv).pack(side=tk.LEFT, padx=4, pady=3)

        tk.Frame(bar, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=3)

        # class selector
        tk.Label(bar, text="Class:").pack(side=tk.LEFT, padx=(4, 2))
        self.class_var = tk.StringVar(value=LAND_COVER_CLASSES[0])
        class_menu = tk.OptionMenu(bar, self.class_var, *LAND_COVER_CLASSES)
        class_menu.config(width=10)
        class_menu.pack(side=tk.LEFT)

        tk.Button(bar, text="Add to library ＋", bg="#2ecc71", fg="white",
                  command=self._add_to_library).pack(side=tk.LEFT, padx=6, pady=3)
        tk.Button(bar, text="Export library…",   bg="#3498db", fg="white",
                  command=self._export_library_csv).pack(side=tk.LEFT, padx=4, pady=3)
        tk.Button(bar, text="Clear library",     bg="#e74c3c", fg="white",
                  command=self._clear_library).pack(side=tk.LEFT, padx=4, pady=3)

        self.status_var = tk.StringVar(value="No file loaded.")
        tk.Label(bar, textvariable=self.status_var, anchor=tk.W, fg="#444").pack(
            side=tk.LEFT, padx=12)

        # ── main area: plots (left) + library panel (right) ──
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # matplotlib canvas
        plot_frame = tk.Frame(main)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(12, 6.5))
        self.ax_rgb  = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        NavigationToolbar2Tk(self.canvas, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # library side panel
        lib_frame = tk.Frame(main, width=260, bd=1, relief=tk.SUNKEN)
        lib_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)
        lib_frame.pack_propagate(False)

        tk.Label(lib_frame, text="Spectral Library", font=("", 11, "bold")).pack(pady=(8, 2))
        self.lib_count_var = tk.StringVar(value="0 entries")
        tk.Label(lib_frame, textvariable=self.lib_count_var, fg="#666").pack()

        # scrollable listbox
        scroll = tk.Scrollbar(lib_frame, orient=tk.VERTICAL)
        self.lib_listbox = tk.Listbox(lib_frame, yscrollcommand=scroll.set,
                                       width=32, font=("Courier", 9))
        scroll.config(command=self.lib_listbox.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.lib_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # remove selected button
        tk.Button(lib_frame, text="Remove selected",
                  command=self._remove_selected).pack(pady=(0, 6))

    # ── File loading ───────────────────────────────────────────────────────────

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
            return
        if len(hdrs) == 1:
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
            if not idx:
                return
            dlg.destroy()
            self._load(hdrs[idx[0]])

        tk.Button(dlg, text="Open", command=on_ok, width=12).pack(pady=8)
        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
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
        self.status_var.set(f"Loading  {hdr_path.name} …")
        self.root.update_idletasks()
        try:
            self.img          = load_image(hdr_path)
            meta              = self.img.metadata
            self.wavelengths  = parse_wavelengths(meta)
            self.ignore_value = get_ignore_value(meta)
            r, g, b           = get_rgb_bands(meta)

            # validate band indices
            nb = self.img.nbands
            if not all(0 <= idx < nb for idx in (r, g, b)):
                r, g, b = min(FALLBACK_RGB[0], nb-1), min(FALLBACK_RGB[1], nb-1), min(FALLBACK_RGB[2], nb-1)

            self.rgb_display = read_rgb(self.img, r, g, b, self.ignore_value)
            self.spectrum    = None
            self.pixel_pos   = None
            self._refresh_plots()
            self.status_var.set(
                f"{hdr_path.name}  |  "
                f"{self.img.nrows} × {self.img.ncols} px  ×  {nb} bands  |  "
                "Click a pixel to inspect."
            )
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))
            self.status_var.set("Load failed.")

    # ── Plotting ───────────────────────────────────────────────────────────────

    def _refresh_plots(self):
        colour_cycle = cycle(LIBRARY_COLOURS)

        # ── left: RGB image with all library markers ──
        self.ax_rgb.clear()
        if self.rgb_display is not None:
            self.ax_rgb.imshow(self.rgb_display, interpolation="bilinear", aspect="auto")

            # draw saved library points
            for entry in self.library:
                colour = next(colour_cycle)
                self.ax_rgb.plot(entry.col, entry.row, "o",
                                  markersize=7, color=colour,
                                  markeredgecolor="white", markeredgewidth=0.8)

            # draw current (unsaved) pixel
            if self.pixel_pos:
                row, col = self.pixel_pos
                self.ax_rgb.plot(col, row, "r+", markersize=14, markeredgewidth=2.5)

        self.ax_rgb.set_title("RGB preview — click a pixel")
        self.ax_rgb.axis("off")

        # ── right: spectral signatures ──
        self.ax_spec.clear()
        colour_cycle = cycle(LIBRARY_COLOURS)

        x       = (self.wavelengths if self.wavelengths is not None
                   else (np.arange(len(self.library[0].spectrum))
                         if self.library else np.array([])))
        xlabel  = "Wavelength (nm)" if self.wavelengths is not None else "Band index"

        # draw saved library spectra (thin, coloured)
        for entry in self.library:
            colour = next(colour_cycle)
            xi = (self.wavelengths if self.wavelengths is not None
                  else np.arange(len(entry.spectrum)))
            self.ax_spec.plot(xi, entry.spectrum,
                               linewidth=1.0, alpha=0.75, color=colour,
                               label=f"{entry.label} ({entry.row},{entry.col})")

        # draw current spectrum (thicker, blue)
        if self.spectrum is not None:
            row, col = self.pixel_pos
            xi = (self.wavelengths if self.wavelengths is not None
                  else np.arange(len(self.spectrum)))
            self.ax_spec.plot(xi, self.spectrum,
                               linewidth=2.0, color="steelblue",
                               label=f"current ({row},{col})", zorder=5)

        if self.library or self.spectrum is not None:
            self.ax_spec.set_xlabel(xlabel)
            self.ax_spec.set_ylabel("Reflectance (× 10⁻⁴)")
            self.ax_spec.grid(True, alpha=0.3)
            if len(self.library) <= 8:
                self.ax_spec.legend(fontsize=7, loc="upper right")
            self.ax_spec.set_title(
                f"Spectral signatures  ({len(self.library)} saved)"
            )
        else:
            self.ax_spec.set_title("Spectral signatures")
            self.ax_spec.text(0.5, 0.5, "Click a pixel in the RGB image",
                               ha="center", va="center",
                               transform=self.ax_spec.transAxes,
                               color="gray", fontsize=12)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    # ── Mouse click ────────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is not self.ax_rgb or self.img is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < self.img.nrows and 0 <= col < self.img.ncols):
            return
        self.pixel_pos = (row, col)
        self.spectrum  = read_spectrum(self.img, row, col, self.ignore_value)
        self._refresh_plots()
        self.status_var.set(
            f"Pixel ({row}, {col})  |  "
            f"Class: {self.class_var.get()}  |  "
            "Press 'Add to library ＋' to save."
        )

    # ── Spectral library ───────────────────────────────────────────────────────

    def _add_to_library(self):
        if self.spectrum is None:
            messagebox.showinfo("No spectrum", "Click a pixel first.")
            return
        row, col = self.pixel_pos
        label    = self.class_var.get().strip()

        # allow free-text label for "other"
        if label == "other":
            custom = simpledialog.askstring(
                "Custom class", "Enter class name:", parent=self.root)
            if custom:
                label = custom.strip() or "other"

        entry = LibraryEntry(label=label, row=row, col=col,
                              spectrum=self.spectrum.copy())
        self.library.append(entry)
        self._refresh_library_panel()
        self._refresh_plots()
        self.status_var.set(
            f"Added  [{label}]  pixel ({row}, {col})  —  "
            f"{len(self.library)} spectra in library."
        )

    def _refresh_library_panel(self):
        self.lib_listbox.delete(0, tk.END)
        for i, e in enumerate(self.library):
            valid = int(np.sum(~np.isnan(e.spectrum)))
            self.lib_listbox.insert(
                tk.END,
                f"{i+1:>2}. {e.label:<12} ({e.row:4},{e.col:4})  {valid}b"
            )
        self.lib_count_var.set(f"{len(self.library)} entries")

    def _remove_selected(self):
        sel = self.lib_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        removed = self.library.pop(idx)
        self._refresh_library_panel()
        self._refresh_plots()
        self.status_var.set(
            f"Removed [{removed.label}] ({removed.row},{removed.col})  —  "
            f"{len(self.library)} spectra remaining."
        )

    def _clear_library(self):
        if not self.library:
            return
        if messagebox.askyesno("Clear library",
                                f"Remove all {len(self.library)} spectra?"):
            self.library.clear()
            self._refresh_library_panel()
            self._refresh_plots()
            self.status_var.set("Library cleared.")

    # ── CSV export ─────────────────────────────────────────────────────────────

    def _x_axis(self, n_bands: int) -> tuple[np.ndarray, str]:
        if self.wavelengths is not None:
            return self.wavelengths, "wavelength_nm"
        return np.arange(n_bands), "band"

    def _export_single_csv(self):
        """Export only the currently selected spectrum (original behaviour)."""
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
        x, col_header = self._x_axis(len(self.spectrum))
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([col_header, "value"])
            for xi, vi in zip(x, self.spectrum):
                writer.writerow([float(xi), "" if np.isnan(vi) else float(vi)])
        self.status_var.set(f"Saved → {path}")
        messagebox.showinfo("Saved", f"Spectrum exported to:\n{path}")

    def _export_library_csv(self):
        """
        Export the full spectral library as a wide-format CSV:
            class, row, col, wl_400.0, wl_401.5, …
        Each row = one spectrum (one pixel sample).
        """
        if not self.library:
            messagebox.showinfo("Library empty",
                                 "Add at least one spectrum to the library first.")
            return

        n_bands = len(self.library[0].spectrum)
        x, wl_label = self._x_axis(n_bands)

        path = filedialog.asksaveasfilename(
            title="Save spectral library as CSV",
            initialfile="spectral_library.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        # header: class, row, col, wl_XXX.X, wl_XXX.X, …
        band_headers = [f"{wl_label}_{xi:.1f}" for xi in x]
        header = ["class", "row", "col"] + band_headers

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for entry in self.library:
                values = [
                    "" if np.isnan(v) else float(v)
                    for v in entry.spectrum
                ]
                writer.writerow([entry.label, entry.row, entry.col] + values)

        self.status_var.set(f"Library ({len(self.library)} spectra) saved → {path}")
        messagebox.showinfo(
            "Library exported",
            f"{len(self.library)} spectra saved to:\n{path}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralViewer(root)
    root.mainloop()