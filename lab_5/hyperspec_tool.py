#!/usr/bin/env python3
"""
Airborne Hyperspectral Data Cube Browser
---------------------------------------

This tool is a polished, user-friendly alternative to the example provided.
It supports:
- automatic discovery of ENVI .hdr/.bsq cubes under data/you-shall-not-pass/Obrazy lotniczne
- optional command-line path override
- interactive RGB preview with per-channel percentile stretch
- click on a pixel to plot the full spectral signature (wavelengths if present)
- exports selected spectrum to CSV
- presets for common RGB bands and fast manual band entry
- arrow-key navigation between neighbouring pixels
- built-in status messages and robust error handling

Usage:
    python hyperspec_tool.py
    python hyperspec_tool.py --file /path/to/cube.hdr
    python hyperspec_tool.py --list

Requirements:
    pip install spectral numpy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import spectral.io.envi as envi
except ImportError as err:
    raise SystemExit("The 'spectral' library is required. Install via: pip install spectral") from err

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "you-shall-not-pass" / "Obrazy lotniczne"
ADDITIONAL_DATA_DIRS = [
    Path(__file__).resolve().parent / "data" / "mock",
    Path(__file__).resolve().parent / "data" / "images",
    Path("data/images"),
]
FALLBACK_RGB = (30, 20, 10)
RGB_PRESETS = {
    "natural": (30, 20, 10),
    "vegetation": (80, 40, 20),
    "ncc": (115, 45, 20),
    "custom": None,
}


def find_hdr_files(directory: Path, recursive: bool = True) -> list[Path]:
    """Find all .hdr files in the given directory."""
    if not directory.exists() or not directory.is_dir():
        return []
    globs = ["**/*.hdr", "**/*.bsq"] if recursive else ["*.hdr", "*.bsq"]
    paths = []
    for pattern in globs:
        paths.extend(directory.glob(pattern))
    return sorted({p for p in paths})


def parse_wavelengths(meta: dict) -> np.ndarray | None:
    """Parse wavelength information from ENVI metadata."""
    wl = meta.get("wavelength")
    if not wl:
        return None
    try:
        return np.array([float(x) for x in wl])
    except Exception:
        return None


def get_rgb_bands(meta: dict) -> tuple[int, int, int]:
    """Retrieve default RGB bands from metadata or fallback to defaults."""
    db = meta.get("default bands") or meta.get("default_bands")
    if db and len(db) >= 3:
        try:
            return tuple(int(float(x)) - 1 for x in db[:3])
        except Exception:
            pass
    return FALLBACK_RGB


def parse_ignore_value(meta: dict) -> float | None:
    """Parse the data ignore value from ENVI metadata."""
    raw = meta.get("data ignore value") or meta.get("data_ignore_value")
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def load_image(path: Path) -> envi.SpectralImage:
    """Load an ENVI image from the specified path."""
    if not path.exists():
        raise FileNotFoundError(path)
    return envi.open(str(path))


def clamp_band_idx(idx: int, nbands: int) -> int:
    """Clamp the band index within the valid range [0, nbands-1]."""
    if idx < 0:
        return 0
    if idx >= nbands:
        return nbands - 1
    return idx


def to_rgb_display(img: envi.SpectralImage, bands: tuple[int, int, int],
                   ignore_value: float | None, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Process three bands into an RGB display array with percentile stretching."""
    r, g, b = bands
    arr = img.read_bands([r, g, b]).astype(np.float32)

    if ignore_value is not None:
        arr[arr >= ignore_value] = np.nan
    arr[arr < 0] = np.nan

    out = np.zeros_like(arr)
    for ch in range(3):
        channel = arr[:, :, ch]
        p_min, p_max = np.nanpercentile(channel, [p_low, p_high])
        out[:, :, ch] = np.clip((channel - p_min) / max(p_max - p_min, 1e-9), 0.0, 1.0)

    return np.nan_to_num(out, nan=0.0)


def read_spectrum(img: envi.SpectralImage, row: int, col: int, ignore_value: float | None) -> np.ndarray:
    """Read the full spectral signature for a specific pixel."""
    sp = np.array(img.read_pixel(row, col), dtype=np.float64)
    if ignore_value is not None:
        sp[sp >= ignore_value] = np.nan
    sp[sp < 0] = np.nan
    return sp


class HyperspectralBrowser:
    """A GUI application for browsing and analyzing airborne hyperspectral data cubes."""
    def __init__(self, root: tk.Tk, default_file: Path | None):
        self.root = root
        self.root.title("Airborne Hyperspectral Browser ")
        self.root.geometry("1400x760")

        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass

        self.img = None
        self.meta = {}
        self.wavelengths = None
        self.ignore_value = None
        self.rgb_bands = FALLBACK_RGB
        self.rgb_image = None
        self.spectrum = None
        self.selected_pixel = None
        self.pinned_spectra = []  # List of (row, col, spectrum)

        self.default_file = default_file
        self.data_dir = DEFAULT_DATA_DIR

        self._build_ui()
        self._startup_load()

    def _build_ui(self) -> None:
        """Construct the main user interface components."""
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open file...", command=self._on_open).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Reload", command=self._reload).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Export spectrum", command=self._export_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Metadata", command=self._show_metadata).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Save RGB", command=self._save_rgb).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Pin", command=self._pin_spectrum).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Clear Pins", command=self._clear_pins).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="RGB bands:").pack(side=tk.LEFT, padx=(12, 3))
        self.rgb_entry = ttk.Entry(top, width=20)
        self.rgb_entry.pack(side=tk.LEFT, padx=2)
        self.rgb_entry.bind('<Return>', lambda e: self._apply_rgb_text())
        ttk.Button(top, text="Apply", command=self._apply_rgb_text).pack(side=tk.LEFT, padx=2)

        ttk.Label(top, text="Preset:").pack(side=tk.LEFT, padx=(12, 3))
        self.preset_var = tk.StringVar(value='natural')
        self.preset_menu = ttk.OptionMenu(top, self.preset_var, 'natural', *RGB_PRESETS.keys(), command=self._on_preset)
        self.preset_menu.pack(side=tk.LEFT, padx=2)

        ttk.Label(top, text="Jump to row:").pack(side=tk.LEFT, padx=(12, 2))
        self.row_spin = ttk.Spinbox(top, from_=0, to=1000, width=6, command=lambda: self._jump_pixel())
        self.row_spin.pack(side=tk.LEFT, padx=2)
        ttk.Label(top, text="col:").pack(side=tk.LEFT, padx=(4, 2))
        self.col_spin = ttk.Spinbox(top, from_=0, to=1000, width=6, command=lambda: self._jump_pixel())
        self.col_spin.pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text='Go', command=self._jump_pixel).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Stretch %:").pack(side=tk.LEFT, padx=(12, 2))
        self.stretch_low = ttk.Spinbox(top, from_=0, to=49, width=4)
        self.stretch_low.set(2)
        self.stretch_low.pack(side=tk.LEFT, padx=1)
        ttk.Label(top, text="-").pack(side=tk.LEFT)
        self.stretch_high = ttk.Spinbox(top, from_=51, to=100, width=4)
        self.stretch_high.set(98)
        self.stretch_high.pack(side=tk.LEFT, padx=1)

        self.interpolate_var = tk.BooleanVar(value=True)
        self.interp_cb = ttk.Checkbutton(top, text="Interpolate", variable=self.interpolate_var, command=self._refresh_plots)
        self.interp_cb.pack(side=tk.LEFT, padx=(12, 2))

        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(top, textvariable=self.status_var, anchor='w', foreground='#1a1a1a')
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # Main resizable area
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left pane: RGB preview
        rgb_frame = ttk.Frame(self.paned)
        self.paned.add(rgb_frame, weight=1)
        self.fig_rgb = Figure(figsize=(6, 6), tight_layout=True)
        self.ax_rgb = self.fig_rgb.add_subplot(1, 1, 1)
        self.canvas_rgb = FigureCanvasTkAgg(self.fig_rgb, master=rgb_frame)
        self.canvas_rgb.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_rgb = NavigationToolbar2Tk(self.canvas_rgb, rgb_frame)
        self.toolbar_rgb.update()
        self.canvas_rgb.mpl_connect('button_press_event', self._handle_click)

        # Right pane: Spectral plot
        spec_frame = ttk.Frame(self.paned)
        self.paned.add(spec_frame, weight=1)
        self.fig_spec = Figure(figsize=(6, 6), tight_layout=True)
        self.ax_spectrum = self.fig_spec.add_subplot(1, 1, 1)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=spec_frame)
        self.canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar_spec = NavigationToolbar2Tk(self.canvas_spec, spec_frame)
        self.toolbar_spec.update()

        self.root.bind('<Left>', lambda e: self._move_pixel(-1, 0))
        self.root.bind('<Right>', lambda e: self._move_pixel(1, 0))
        self.root.bind('<Up>', lambda e: self._move_pixel(0, -1))
        self.root.bind('<Down>', lambda e: self._move_pixel(0, 1))

    def _startup_load(self) -> None:
        """Handle initial data loading on application startup."""
        if self.default_file:
            self._load_file(self.default_file)
            return

        search_dirs = [self.data_dir] + ADDITIONAL_DATA_DIRS
        candidates = []
        for d in search_dirs:
            candidates.extend(find_hdr_files(d, recursive=True))

        # Remove duplicates while preserving order
        seen = set()
        candidates = [x for x in candidates if not (x in seen or seen.add(x))]

        if not candidates:
            self.status_var.set("No .hdr found in default or local data directories")
            return

        if len(candidates) == 1:
            self._load_file(candidates[0])
            return

        self._pick_file_dialog(candidates)

    def _pick_file_dialog(self, options: list[Path]):
        dlg = tk.Toplevel(self.root)
        dlg.title("Select a dataset")
        dlg.geometry('650x320')
        dlg.grab_set()

        tk.Label(dlg, text='Multiple data cubes found. Select one:').pack(anchor='w', pady=8, padx=8)
        lb = tk.Listbox(dlg, width=90, height=10)
        lb.pack(padx=8, pady=4, fill=tk.BOTH, expand=True)

        for p in options:
            lb.insert(tk.END, str(p))
        lb.select_set(0)

        def choose():
            sel = lb.curselection()
            if not sel:
                return
            path = options[sel[0]]
            dlg.destroy()
            self._load_file(path)

        btn = tk.Button(dlg, text="Open", command=choose, width=12)
        btn.pack(pady=6)
        dlg.wait_window()

    def _on_open(self):
        initial_dir = self.data_dir
        if not initial_dir.exists():
            for d in ADDITIONAL_DATA_DIRS:
                if d.exists():
                    initial_dir = d
                    break
        if not initial_dir.exists():
            initial_dir = Path.home()

        path = filedialog.askopenfilename(
            title='Open ENVI header file',
            initialdir=str(initial_dir),
            filetypes=[('ENVI header', '*.hdr'), ('All files', '*.*')]
        )
        if path:
            self._load_file(Path(path))

    def _reload(self):
        img_path = getattr(self, 'img_path', None)
        if img_path is not None:
            self._load_file(img_path)

    def _on_preset(self, preset_name):
        p = RGB_PRESETS.get(preset_name)
        if p is None:
            self.status_var.set('Custom preset selected, please enter values and press Apply')
            return
        self.rgb_bands = p
        self.rgb_entry.delete(0, tk.END)
        self.rgb_entry.insert(0, ', '.join(str(x) for x in p))
        self._update_rgb_display(p)

    def _apply_rgb_text(self):
        raw = self.rgb_entry.get().strip()
        if not raw:
            messagebox.showinfo('RGB bands', 'Enter three band indices separated by commas.')
            return
        parts = [x.strip() for x in raw.replace(';', ',').split(',') if x.strip()]
        if len(parts) != 3:
            messagebox.showerror('Error', 'Specify exactly 3 integer band indices: R, G, B')
            return
        try:
            rgb = tuple(int(p) for p in parts)
        except ValueError:
            messagebox.showerror('Error', 'RGB band values must be integers')
            return
        self.rgb_bands = rgb
        self._update_rgb_display(rgb)

    def _jump_pixel(self):
        if self.img is None:
            return
        try:
            r = int(self.row_spin.get())
            c = int(self.col_spin.get())
        except ValueError:
            return
        r = int(np.clip(r, 0, self.img.nrows - 1))
        c = int(np.clip(c, 0, self.img.ncols - 1))
        self.selected_pixel = (r, c)
        self.spectrum = read_spectrum(self.img, r, c, self.ignore_value)
        self.status_var.set(f'Pixel selected: row {r}, col {c}')
        self._refresh_plots()

    def _update_rgb_display(self, rgb_bands):
        if self.img is None:
            return
        try:
            pl = float(self.stretch_low.get())
            ph = float(self.stretch_high.get())
        except ValueError:
            pl, ph = 2, 98

        rgb_key = (rgb_bands, self.ignore_value, pl, ph)
        if getattr(self, '_cache_rgb_key', None) != rgb_key:
            self.rgb_image = to_rgb_display(self.img, rgb_bands, self.ignore_value, p_low=pl, p_high=ph)
            self._cache_rgb_key = rgb_key
        self.selected_pixel = self.selected_pixel or (self.img.nrows // 2, self.img.ncols // 2)
        self.spectrum = read_spectrum(self.img, *self.selected_pixel, self.ignore_value)
        self._refresh_plots()

    def _load_file(self, path: Path):
        try:
            self.img_path = path
            self.status_var.set(f'Loading {path}...')
            self.root.update_idletasks()
            self.img = load_image(path)
            self.meta = self.img.metadata or {}
            self.wavelengths = parse_wavelengths(self.meta)
            self.ignore_value = parse_ignore_value(self.meta)

            r, g, b = get_rgb_bands(self.meta)
            self.rgb_bands = (clamp_band_idx(r, self.img.nbands), clamp_band_idx(g, self.img.nbands), clamp_band_idx(b, self.img.nbands))
            self.rgb_entry.delete(0, tk.END)
            self.rgb_entry.insert(0, ', '.join(str(x) for x in self.rgb_bands))

            self.selected_pixel = (self.img.nrows // 2, self.img.ncols // 2)
            self.spectrum = read_spectrum(self.img, *self.selected_pixel, self.ignore_value)
            self._update_rgb_display(self.rgb_bands)
            self.status_var.set(f'Loaded: {path.name} ({self.img.nrows}x{self.img.ncols}x{self.img.nbands})')

        except Exception as err:
            messagebox.showerror('Load failed', str(err))
            self.status_var.set('Error loading file')

    def _refresh_plots(self) -> None:
        """Update the RGB preview and spectral plot based on current selection and settings."""
        self.ax_rgb.clear()
        if self.rgb_image is not None:
            interp = "bilinear" if self.interpolate_var.get() else "none"
            self.ax_rgb.imshow(self.rgb_image, origin="upper", interpolation=interp)
            if self.selected_pixel is not None:
                r, c = self.selected_pixel
                self.ax_rgb.plot(c, r, 'r+', markersize=12, markeredgewidth=2)
        self.ax_rgb.set_title('RGB preview (click pixel)')
        self.ax_rgb.axis('off')

        self.ax_spectrum.clear()
        if self.spectrum is not None:
            r, c = self.selected_pixel
            x = self.wavelengths if self.wavelengths is not None else np.arange(len(self.spectrum))
            label = 'Wavelength (nm)' if self.wavelengths is not None else 'Band'
            # Plot pinned spectra
            for i, (pr, pc, psp) in enumerate(self.pinned_spectra):
                self.ax_spectrum.plot(x, psp, alpha=0.5, linewidth=1.0, label=f'Pinned ({pr}, {pc})')

            # Plot current spectrum
            self.ax_spectrum.plot(x, self.spectrum, color='darkblue', linewidth=1.5, label=f'Current ({r}, {c})')

            minv = float(np.nanmin(self.spectrum))
            maxv = float(np.nanmax(self.spectrum))
            self.ax_spectrum.set_title(f'Spectrum | Current: (row={r}, col={c})')
            self.ax_spectrum.set_xlabel(label)
            self.ax_spectrum.set_ylabel('Reflectance')
            self.ax_spectrum.grid(True, alpha=0.35)
            self.ax_spectrum.legend(fontsize='small', loc='upper right')
        else:
            self.ax_spectrum.set_title('Spectral signature (click pixel in RGB view)')
            self.ax_spectrum.text(0.5, 0.5, 'Select pixel first', ha='center', va='center', transform=self.ax_spectrum.transAxes, color='gray', fontsize=12)
            self.ax_spectrum.axis('off')

        self.fig_rgb.tight_layout(pad=1.5)
        self.fig_spec.tight_layout(pad=1.5)
        self.canvas_rgb.draw_idle()
        self.canvas_spec.draw_idle()

    def _handle_click(self, event):
        if event.inaxes is not self.ax_rgb or self.img is None:
            return
        row = int(np.clip(round(event.ydata), 0, self.img.nrows - 1))
        col = int(np.clip(round(event.xdata), 0, self.img.ncols - 1))
        self.selected_pixel = (row, col)
        self.spectrum = read_spectrum(self.img, row, col, self.ignore_value)
        self.status_var.set(f'Pixel selected: row {row}, col {col}, click Export to save')
        self._refresh_plots()

    def _move_pixel(self, dx: int, dy: int):
        if self.img is None or self.selected_pixel is None:
            return
        r, c = self.selected_pixel
        r = int(np.clip(r + dy, 0, self.img.nrows - 1))
        c = int(np.clip(c + dx, 0, self.img.ncols - 1))
        self.selected_pixel = (r, c)
        self.spectrum = read_spectrum(self.img, r, c, self.ignore_value)
        self.status_var.set(f'Pixel moved: row {r}, col {c}')
        self._refresh_plots()

    def _show_metadata(self):
        if not self.meta:
            messagebox.showinfo('Metadata', 'No metadata available. Load a file first.')
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("ENVI Metadata")
        dlg.geometry('600x500')

        txt = tk.Text(dlg, wrap=tk.NONE, padx=10, pady=10)
        vsb = ttk.Scrollbar(dlg, orient="vertical", command=txt.yview)
        hsb = ttk.Scrollbar(dlg, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        import json
        try:
            # Try to format nicely if possible, but some ENVI meta might not be JSON serializable
            content = json.dumps(self.meta, indent=2, default=str)
        except Exception:
            content = str(self.meta)

        txt.insert(tk.END, content)
        txt.config(state=tk.DISABLED)

    def _save_rgb(self):
        if self.rgb_image is None:
            messagebox.showwarning('No image', 'No RGB image to save.')
            return

        path = filedialog.asksaveasfilename(
            title='Save RGB Preview',
            defaultextension='.png',
            filetypes=[('PNG file', '*.png'), ('JPEG file', '*.jpg'), ('All files', '*.*')]
        )
        if path:
            try:
                self.fig_rgb.savefig(path)
                messagebox.showinfo('Saved', f'RGB image saved to {path}')
            except Exception as e:
                messagebox.showerror('Save failed', str(e))

    def _pin_spectrum(self):
        if self.spectrum is None or self.selected_pixel is None:
            return
        r, c = self.selected_pixel
        self.pinned_spectra.append((r, c, self.spectrum.copy()))
        self.status_var.set(f'Pinned spectrum at ({r}, {c})')
        self._refresh_plots()

    def _clear_pins(self):
        self.pinned_spectra = []
        self.status_var.set('Cleared all pinned spectra')
        self._refresh_plots()

    def _export_csv(self):
        if self.spectrum is None or self.selected_pixel is None:
            tk.messagebox.showwarning('No spectrum', 'Select a pixel first by clicking on RGB preview.')
            return

        row, col = self.selected_pixel
        default_name = f'spectrum_r{row}_c{col}.csv'
        path = filedialog.asksaveasfilename(
            title='Save spectrum',
            initialfile=default_name,
            defaultextension='.csv',
            filetypes=[('CSV file', '*.csv'), ('All files', '*.*')]
        )
        if not path:
            return

        x = self.wavelengths if self.wavelengths is not None else np.arange(len(self.spectrum))
        col_name = 'wavelength_nm' if self.wavelengths is not None else 'band'
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([col_name, 'value'])
            for xi, yi in zip(x, self.spectrum):
                w.writerow([float(xi), '' if np.isnan(yi) else float(yi)])

        tk.messagebox.showinfo('Saved', f'Spectrum exported to {path}')
        self.status_var.set(f'Saved spectrum to {path}')


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description='Airborne Hyperspectral Browser ')
    parser.add_argument('--file', '-f', type=Path, help='Path to ENVI header (.hdr) file')
    parser.add_argument('--list', action='store_true', help='List discovered .hdr files and exit')
    args = parser.parse_args(argv)

    if args.list:
        search_dirs = [DEFAULT_DATA_DIR] + ADDITIONAL_DATA_DIRS
        candidates = []
        for d in search_dirs:
            candidates.extend(find_hdr_files(d, recursive=True))
        seen = set()
        candidates = [x for x in candidates if not (x in seen or seen.add(x))]

        if not candidates:
            print('No .hdr found in default or local data directories')
            return 0
        for x in candidates:
            print(x)
        return 0

    root = tk.Tk()
    HyperspectralBrowser(root, default_file=args.file)
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
