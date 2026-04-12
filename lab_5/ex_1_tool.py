#!/usr/bin/env python3
"""
SpectralCube Pro | Airborne Hyperspectral Data Browser
------------------------------------------------------
A high-performance, polished GUI for exploring ENVI hyperspectral cubes.
Optimized for efficiency using memory-mapping and spatial sampling.

Features:
- Instant loading of large cubes via memory mapping.
- Interactive RGB preview with percentile stretching.
- Real-time spectral signature on hover.
- Spectrum pinning for multi-pixel comparison.
- Automatic wavelength-to-band matching.
- CSV export of spectral signatures.
"""

from __future__ import annotations

import argparse
import csv
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import spectral.io.envi as envi
except ImportError as err:
    raise SystemExit("The 'spectral' library is required. Install via: pip install spectral") from err

# --- Configuration & Defaults ---
DEFAULT_DATA_DIR = Path("lab_5/data/you-shall-not-pass/Obrazy lotnicze")
SEARCH_PATHS = [DEFAULT_DATA_DIR, Path("data"), Path(".")]
RGB_PRESETS = {
    "Natural Color": (670.0, 560.0, 490.0),
    "Color Infrared": (840.0, 670.0, 560.0),
    "False Color (Veg)": (1600.0, 840.0, 670.0),
}

class HyperspectralData:
    """Efficiently manages ENVI data access via memory mapping."""
    def __init__(self, hdr_path: Path):
        self.path = hdr_path
        self.img = envi.open(str(hdr_path))
        self.mm = self.img.open_memmap(writable=False)
        self.meta = self.img.metadata
        
        self.nrows, self.ncols, self.nbands = self.img.nrows, self.img.ncols, self.img.nbands
        
        # Extract Wavelengths
        self.wavelengths = None
        if "wavelength" in self.meta:
            try:
                self.wavelengths = np.array([float(x) for x in self.meta["wavelength"]])
            except (ValueError, TypeError): pass
        
        # Extract Ignore Value
        self.ignore_value = None
        for key in ["data ignore value", "data_ignore_value"]:
            if key in self.meta:
                try:
                    self.ignore_value = float(self.meta[key])
                    break
                except (ValueError, TypeError): pass

    def get_band_idx(self, target_nm: float) -> int:
        """Finds closest band index for a target wavelength."""
        if self.wavelengths is None: return 0
        return int(np.argmin(np.abs(self.wavelengths - target_nm)))

    def get_rgb(self, bands: tuple[int, int, int], stretch: tuple[float, float] = (2, 98)) -> np.ndarray:
        """Generates a normalized RGB array using memory mapping and sampling."""
        # Read only required bands from disk
        data = self.mm[:, :, bands].astype(np.float32)
        
        if self.ignore_value is not None:
            data[data == self.ignore_value] = np.nan
        
        rgb = np.zeros_like(data)
        sampling = max(1, min(self.nrows, self.ncols) // 200)
        
        for i in range(3):
            band = data[:, :, i]
            # Use sampling for fast percentile calculation on large cubes
            sample = band[::sampling, ::sampling]
            vmin, vmax = np.nanpercentile(sample, stretch)
            
            if vmax > vmin:
                rgb[:, :, i] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
        
        return np.nan_to_num(rgb)

    def get_spectrum(self, row: int, col: int) -> np.ndarray:
        """Retrieves full spectral vector for a pixel."""
        spec = self.mm[row, col, :].astype(np.float64)
        if self.ignore_value is not None:
            spec[spec == self.ignore_value] = np.nan
        return spec

class HyperspectralBrowser:
    """Main Application GUI."""
    def __init__(self, root: tk.Tk, start_path: Path | None = None):
        self.root = root
        self.root.title("SpectralCube Pro | Hyperspectral Data Browser")
        self.root.geometry("1400x850")
        
        self.data: HyperspectralData | None = None
        self.rgb_img: np.ndarray | None = None
        self.selected_pixel: tuple[int, int] | None = None
        self.pinned_spectra: list[dict] = []
        self.live_mode = tk.BooleanVar(value=True)

        self._setup_ui()
        
        if start_path:
            self.load_cube(start_path)
        else:
            self.root.after(200, self.auto_discover)

    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Toolbar
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Group: File
        file_frame = ttk.LabelFrame(toolbar, text=" File ", padding=5)
        file_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Open Cube", command=self.on_open).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="Metadata", command=self.show_meta).pack(side=tk.LEFT, padx=2)

        # Group: View
        view_frame = ttk.LabelFrame(toolbar, text=" Visualization ", padding=5)
        view_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(view_frame, text="Bands:").pack(side=tk.LEFT)
        self.band_ent = ttk.Entry(view_frame, width=15)
        self.band_ent.pack(side=tk.LEFT, padx=2)
        self.band_ent.bind("<Return>", lambda e: self.apply_visuals())
        
        self.preset_var = tk.StringVar()
        cb = ttk.Combobox(view_frame, textvariable=self.preset_var, values=list(RGB_PRESETS.keys()), state="readonly", width=18)
        cb.pack(side=tk.LEFT, padx=2)
        cb.bind("<<ComboboxSelected>>", self.on_preset)

        # Group: Analysis
        an_frame = ttk.LabelFrame(toolbar, text=" Analysis ", padding=5)
        an_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(an_frame, text="Pin Spectrum", command=self.pin_spec).pack(side=tk.LEFT, padx=2)
        ttk.Button(an_frame, text="Clear Pins", command=self.clear_pins).pack(side=tk.LEFT, padx=2)
        ttk.Button(an_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(an_frame, text="Live Hover", variable=self.live_mode).pack(side=tk.LEFT, padx=5)

        # Main Area
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Image Plot
        img_container = ttk.Frame(self.paned)
        self.paned.add(img_container, weight=3)
        self.fig_img = Figure(figsize=(6, 6), facecolor="#f0f0f0")
        self.ax_img = self.fig_img.add_subplot(111)
        self.ax_img.axis("off")
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=img_container)
        self.canvas_img.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_img.mpl_connect("button_press_event", self.on_click)
        self.canvas_img.mpl_connect("motion_notify_event", self.on_hover)
        NavigationToolbar2Tk(self.canvas_img, img_container)

        # Spectrum Plot
        spec_container = ttk.Frame(self.paned)
        self.paned.add(spec_container, weight=2)
        self.fig_spec = Figure(figsize=(5, 5), facecolor="#f0f0f0")
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=spec_container)
        self.canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status
        self.stat_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.stat_var, relief=tk.SUNKEN, anchor=tk.W, padding=3).pack(side=tk.BOTTOM, fill=tk.X)

    def auto_discover(self):
        found = []
        for p in SEARCH_PATHS:
            if p.exists(): found.extend(list(p.glob("**/*.hdr")))
        if found: self.load_cube(found[0])

    def on_open(self):
        path = filedialog.askopenfilename(filetypes=[("ENVI Header", "*.hdr"), ("All", "*.*")])
        if path: self.load_cube(Path(path))

    def load_cube(self, path: Path):
        try:
            self.root.config(cursor="watch")
            self.root.update()
            self.data = HyperspectralData(path)
            self.pinned_spectra = []
            
            # Default visualization
            if self.data.wavelengths is not None:
                self.preset_var.set("Natural Color")
                self.on_preset()
            else:
                self.band_ent.delete(0, tk.END)
                self.band_ent.insert(0, "30, 20, 10")
                self.apply_visuals()
            
            self.selected_pixel = (self.data.nrows // 2, self.data.ncols // 2)
            self.update_plots()
            self.stat_var.set(f"Loaded: {path.name} | {self.data.nrows}x{self.data.ncols}x{self.data.nbands}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load cube: {e}")
        finally:
            self.root.config(cursor="")

    def on_preset(self, event=None):
        if not self.data: return
        p = RGB_PRESETS.get(self.preset_var.get())
        if p:
            indices = [self.data.get_band_idx(w) for p_w in p for w in [p_w]] # flattening not needed but clear
            # Wait, easier:
            r, g, b = [self.data.get_band_idx(w) for w in p]
            self.band_ent.delete(0, tk.END)
            self.band_ent.insert(0, f"{r}, {g}, {b}")
            self.apply_visuals()

    def apply_visuals(self):
        if not self.data: return
        try:
            b_str = self.band_ent.get().split(",")
            bands = tuple(int(x.strip()) for x in b_str)
            self.rgb_img = self.data.get_rgb(bands)
            self.update_plots()
        except Exception:
            messagebox.showwarning("Input", "Enter 3 band indices separated by commas.")

    def update_plots(self, hover_pixel=None):
        if not self.data: return
        
        # Update Image
        self.ax_img.clear()
        if self.rgb_img is not None:
            self.ax_img.imshow(self.rgb_img)
            if self.selected_pixel:
                self.ax_img.plot(self.selected_pixel[1], self.selected_pixel[0], 'r+', ms=12, mew=2)
        self.ax_img.axis("off")
        self.canvas_img.draw_idle()

        # Update Spectrum
        self.ax_spec.clear()
        target = hover_pixel or self.selected_pixel
        if target:
            x = self.data.wavelengths if self.data.wavelengths is not None else np.arange(self.data.nbands)
            
            # Pinned
            for p in self.pinned_spectra:
                self.ax_spec.plot(x, p["data"], alpha=0.3, label=f"Pin {p['pos']}")
            
            # Active
            spec = self.data.get_spectrum(*target)
            color = "blue" if hover_pixel else "red"
            self.ax_spec.plot(x, spec, color=color, lw=2, label=f"Active {target}")
            
            self.ax_spec.set_title(f"Spectral Signature at {target}")
            self.ax_spec.set_xlabel("Wavelength (nm)" if self.data.wavelengths is not None else "Band Index")
            self.ax_spec.set_ylabel("Reflectance / Value")
            self.ax_spec.grid(True, alpha=0.3)
            self.ax_spec.legend(fontsize=8)
        self.canvas_spec.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax_img or not self.data: return
        self.selected_pixel = (int(event.ydata), int(event.xdata))
        self.update_plots()
        self.stat_var.set(f"Selected: Row {self.selected_pixel[0]}, Col {self.selected_pixel[1]}")

    def on_hover(self, event):
        if not self.live_mode.get() or event.inaxes != self.ax_img or not self.data: return
        self.update_plots(hover_pixel=(int(event.ydata), int(event.xdata)))

    def pin_spec(self):
        if not self.selected_pixel: return
        d = self.data.get_spectrum(*self.selected_pixel)
        self.pinned_spectra.append({"pos": self.selected_pixel, "data": d})
        self.update_plots()

    def clear_pins(self):
        self.pinned_spectra = []
        self.update_plots()

    def export_csv(self):
        if not self.selected_pixel: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        spec = self.data.get_spectrum(*self.selected_pixel)
        x = self.data.wavelengths if self.data.wavelengths is not None else np.arange(self.data.nbands)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Band/Wavelength", "Value"])
            for xi, vi in zip(x, spec): w.writerow([xi, vi])
        self.stat_var.set(f"Exported to {os.path.basename(path)}")

    def show_meta(self):
        if not self.data: return
        win = tk.Toplevel(self.root)
        win.title("Header Metadata")
        txt = tk.Text(win, font=("Consolas", 10))
        txt.pack(fill=tk.BOTH, expand=True)
        import json
        txt.insert(tk.END, json.dumps(self.data.meta, indent=2))
        txt.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralBrowser(root)
    root.mainloop()
