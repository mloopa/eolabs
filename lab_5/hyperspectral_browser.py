#!/usr/bin/env python3
"""
Hyperspectral Data Cube Browser
================================
Browse ENVI BSQ hyperspectral images, visualize RGB previews,
inspect spectral signatures by clicking pixels, and export to CSV.
"""

import os
import sys
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from pathlib import Path

# ── optional fast image display ───────────────────────────────────────────────
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── optional matplotlib for spectrum plot ─────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─────────────────────────────────────────────────────────────────────────────
# ENVI header reader
# ─────────────────────────────────────────────────────────────────────────────

DTYPE_MAP = {
    1:  np.uint8,
    2:  np.int16,
    3:  np.int32,
    4:  np.float32,
    5:  np.float64,
    6:  np.complex64,
    9:  np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def parse_envi_header(hdr_path: str) -> dict:
    """Parse an ENVI .hdr file and return a dict of key -> value."""
    meta = {}
    with open(hdr_path, "r", errors="replace") as f:
        text = f.read()

    # Collapse multi-line brace blocks into single lines
    text = re.sub(
        r"\{([^}]*)\}",
        lambda m: "{" + m.group(1).replace("\n", " ") + "}",
        text,
    )

    for line in text.splitlines():
        line = line.strip()
        if "=" not in line or line.startswith(";"):
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower()
        val = val.strip()

        if val.startswith("{") and val.endswith("}"):
            inner = val[1:-1].strip()
            items = [v.strip() for v in inner.split(",") if v.strip()]
            try:
                val = [float(v) for v in items]
            except ValueError:
                val = items
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        meta[key] = val
    return meta


def find_data_file(hdr_path: str) -> str:
    """Find the binary data file paired with an ENVI header."""
    base = os.path.splitext(hdr_path)[0]
    for ext in ("", ".bsq", ".img", ".raw", ".dat", ".bil", ".bip"):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No data file found for header: {hdr_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cube loader
# ─────────────────────────────────────────────────────────────────────────────

class HyperspectralCube:
    def __init__(self, hdr_path: str):
        self.hdr_path = hdr_path
        self.meta = parse_envi_header(hdr_path)
        self.data_path = find_data_file(hdr_path)
        self._wavelengths_are_indices = False
        self._validate()
        self._open_memmap()

    def _validate(self):
        m = self.meta
        for key in ("samples", "lines", "bands"):
            if key not in m:
                raise ValueError(f"Header missing required key: '{key}'")
        interleave = str(m.get("interleave", "bsq")).lower()
        if interleave != "bsq":
            raise ValueError(
                f"Only BSQ interleave is supported; got '{interleave}'")

    def _open_memmap(self):
        m = self.meta
        self.samples = int(m["samples"])
        self.lines   = int(m["lines"])
        self.bands   = int(m["bands"])

        dtype_code = int(m.get("data type", 4))
        dtype      = DTYPE_MAP.get(dtype_code, np.float32)
        byte_order = int(m.get("byte order", 0))
        order_char = ">" if byte_order == 1 else "<"
        # Build dtype with explicit endianness (e.g. "<f4", ">i2")
        native_char = np.dtype(dtype).str[1:]   # strip existing endian prefix
        dt = np.dtype(f"{order_char}{native_char}")

        self.data = np.memmap(
            self.data_path,
            dtype=dt,
            mode="r",
            shape=(self.bands, self.lines, self.samples),
        )

        # Wavelengths ──────────────────────────────────────────────────────────
        wl = m.get("wavelength")
        if isinstance(wl, list) and len(wl) == self.bands:
            self.wavelengths = np.array(wl, dtype=np.float64)
        else:
            self.wavelengths = np.arange(1, self.bands + 1, dtype=np.float64)
            self._wavelengths_are_indices = True

        # No-data sentinel ─────────────────────────────────────────────────────
        ignore = m.get("data ignore value")
        self.ignore_value = float(ignore) if ignore is not None else None

        # Default RGB bands (header is 1-based → convert to 0-based) ──────────
        db = m.get("default bands")
        if isinstance(db, list) and len(db) >= 3:
            self.default_rgb = [
                int(db[0]) - 1,
                int(db[1]) - 1,
                int(db[2]) - 1,
            ]
        else:
            step = max(1, self.bands // 4)
            self.default_rgb = [
                min(step,         self.bands - 1),
                min(2 * step,     self.bands - 1),
                min(3 * step,     self.bands - 1),
            ]

        self.scale_factor = m.get("reflectance scale factor", 1.0)

    # ── no-data mask ──────────────────────────────────────────────────────────

    def _nodata_mask(self, arr: np.ndarray) -> np.ndarray:
        """
        Return boolean mask (True = no-data) using exact equality so that
        valid near-zero pixels are never accidentally masked.
        A tiny absolute tolerance (0.5 DN) handles floating-point storage.
        """
        if self.ignore_value is None:
            return np.zeros(arr.shape, dtype=bool)
        return np.isclose(arr, self.ignore_value, rtol=0.0, atol=0.5)

    # ── public API ────────────────────────────────────────────────────────────

    def read_rgb(self, r_idx=None, g_idx=None, b_idx=None) -> np.ndarray:
        """Return (lines, samples, 3) uint8 array ready for display."""
        if r_idx is None:
            r_idx, g_idx, b_idx = self.default_rgb

        def band_u8(idx: int) -> np.ndarray:
            # Read as float32; force a real copy out of the memmap
            raw  = np.array(self.data[idx], dtype=np.float32)
            mask = self._nodata_mask(raw)

            # Replace no-data with NaN for percentile calculation
            band = raw.copy()
            band[mask] = np.nan

            valid_px = band[~mask]
            if valid_px.size == 0:
                return np.zeros((self.lines, self.samples), dtype=np.uint8)

            lo, hi = float(np.percentile(valid_px, 2)), \
                     float(np.percentile(valid_px, 98))
            if hi == lo:
                hi = lo + 1.0

            out = (band - lo) / (hi - lo)
            out = np.clip(out, 0.0, 1.0)
            out[mask] = 0.0          # no-data pixels → black
            return (out * 255).astype(np.uint8)

        r = band_u8(r_idx)
        g = band_u8(g_idx)
        b = band_u8(b_idx)
        return np.stack([r, g, b], axis=2)

    def spectrum_at(self, col: int, row: int) -> np.ndarray:
        """Return (bands,) float64 array for pixel at (col, row)."""
        # np.array() forces a copy out of the memmap
        return np.array(self.data[:, row, col], dtype=np.float64)

    @property
    def name(self) -> str:
        return Path(self.hdr_path).stem


# ─────────────────────────────────────────────────────────────────────────────
# Picker dialog
# ─────────────────────────────────────────────────────────────────────────────

class FilePicker(tk.Toplevel):
    def __init__(self, parent, hdr_files: list):
        super().__init__(parent)
        self.title("Select Hyperspectral Image")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self.result = None
        self._files = hdr_files

        tk.Label(
            self,
            text="Multiple .hdr files found.\nSelect one to open:",
            bg="#1a1a2e", fg="#e0e0ff",
            font=("Courier", 11), pady=10,
        ).pack(padx=20)

        self.listbox = tk.Listbox(
            self,
            bg="#0f0f23", fg="#a0d8ef",
            selectbackground="#4040aa",
            font=("Courier", 10),
            height=min(len(hdr_files), 10),
            width=60,
            relief="flat",
        )
        for f in hdr_files:
            self.listbox.insert(tk.END, os.path.basename(f))
        self.listbox.selection_set(0)
        self.listbox.pack(padx=20, pady=5)

        tk.Button(
            self, text="Open →",
            command=self._confirm,
            bg="#3a3aaa", fg="white",
            font=("Courier", 11, "bold"),
            relief="flat", cursor="hand2",
            pady=6, padx=20,
        ).pack(pady=(5, 15))

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Return>", lambda _: self._confirm())
        self.bind("<Escape>", lambda _: self._cancel())
        self.wait_window()

    def _confirm(self):
        idx = self.listbox.curselection()
        if idx:
            self.result = self._files[idx[0]]
        self.destroy()

    def _cancel(self):
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    CANVAS_MAX = 640     # max display side length (px)

    # Colour palette
    DARK   = "#0d0d1a"
    PANEL  = "#13132b"
    ACCENT = "#4f8ef7"
    TEXT   = "#d0d8ff"
    MUTED  = "#7080aa"
    BORDER = "#2a2a4a"

    def __init__(self, hdr_path: str):
        super().__init__()
        self.title("Hyperspectral Cube Browser")
        self.configure(bg=self.DARK)
        self.resizable(True, True)

        self.cube: HyperspectralCube = None
        self._photo            = None   # keep ImageTk.PhotoImage alive
        self._rgb_pil          = None   # full-res PIL Image (source for zoom)
        self._canvas_w         = self.CANVAS_MAX
        self._canvas_h         = self.CANVAS_MAX
        self._base_scale       = 1.0    # fit-to-canvas scale (set in _render_rgb)
        self._zoom             = 1.0    # user zoom multiplier on top of _base_scale
        self._pan_x            = 0.0   # canvas-px offset of image origin
        self._pan_y            = 0.0
        self._drag_start       = None
        self._drag_pan_start   = None
        self._current_spectrum = None
        self._current_xy       = None

        self._build_ui()
        self.update_idletasks()
        self._load_cube(hdr_path)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar ──────────────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg=self.PANEL, pady=6)
        toolbar.pack(fill="x")

        tk.Label(
            toolbar, text="⬡  CUBE BROWSER",
            bg=self.PANEL, fg=self.ACCENT,
            font=("Courier", 13, "bold"),
        ).pack(side="left", padx=16)

        self.lbl_file = tk.Label(
            toolbar, text="—",
            bg=self.PANEL, fg=self.MUTED,
            font=("Courier", 10),
        )
        self.lbl_file.pack(side="left", padx=8)

        self.btn_export = tk.Button(
            toolbar, text="Export spectrum to CSV…",
            command=self._export_csv,
            bg="#2a4a8a", fg=self.TEXT,
            font=("Courier", 10),
            relief="flat", cursor="hand2",
            padx=12, pady=4,
            state="disabled",
        )
        self.btn_export.pack(side="right", padx=16)

        btn_reset = tk.Button(
            toolbar, text="⌂ Reset zoom",
            command=self._reset_zoom,
            bg="#1e1e3a", fg=self.MUTED,
            font=("Courier", 10),
            relief="flat", cursor="hand2",
            padx=10, pady=4,
        )
        btn_reset.pack(side="right", padx=(0, 4))

        self.lbl_zoom = tk.Label(
            toolbar, text="100%",
            bg=self.PANEL, fg=self.MUTED,
            font=("Courier", 10), width=6,
        )
        self.lbl_zoom.pack(side="right", padx=(0, 2))

        tk.Frame(self, bg=self.BORDER, height=1).pack(fill="x")

        # Main area (image | spectrum) ─────────────────────────────────────────
        main = tk.Frame(self, bg=self.DARK)
        main.pack(fill="both", expand=True)

        # Left: image canvas
        left_frame = tk.Frame(main, bg=self.DARK)
        left_frame.pack(side="left", fill="both", expand=False,
                        padx=(12, 6), pady=12)

        self.canvas = tk.Canvas(
            left_frame,
            bg="#07071a", cursor="crosshair",
            highlightthickness=1,
            highlightbackground=self.BORDER,
            width=self.CANVAS_MAX, height=self.CANVAS_MAX,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<Motion>",          self._on_motion)
        self.canvas.bind("<Double-Button-1>", lambda e: self._reset_zoom())
        # zoom: mouse wheel (Windows/macOS) and Button-4/5 (Linux X11)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)
        self.canvas.bind("<Button-4>",        self._on_scroll)
        self.canvas.bind("<Button-5>",        self._on_scroll)
        # pan: middle-button drag
        self.canvas.bind("<ButtonPress-2>",   self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonPress-3>",   self._on_pan_start)
        self.canvas.bind("<B3-Motion>",       self._on_pan_drag)

        self.lbl_coords = tk.Label(
            left_frame,
            text="Scroll=zoom  ·  Middle/Right-drag=pan  ·  Double-click=reset  ·  Click=spectrum",
            bg=self.DARK, fg=self.MUTED,
            font=("Courier", 8),
        )
        self.lbl_coords.pack(pady=(4, 0), anchor="w")

        # Right: spectrum panel
        right_frame = tk.Frame(
            main, bg=self.PANEL,
            highlightthickness=1,
            highlightbackground=self.BORDER,
        )
        right_frame.pack(side="left", fill="both", expand=True,
                         padx=(0, 12), pady=12)

        tk.Label(
            right_frame, text="SPECTRAL SIGNATURE",
            bg=self.PANEL, fg=self.ACCENT,
            font=("Courier", 11, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 2))

        self.lbl_pixel = tk.Label(
            right_frame, text="",
            bg=self.PANEL, fg=self.TEXT,
            font=("Courier", 9),
        )
        self.lbl_pixel.pack(anchor="w", padx=14)

        if HAS_MPL:
            self._fig = Figure(figsize=(5, 3.8), dpi=96,
                               facecolor=self.PANEL)
            self.ax = self._fig.add_subplot(111)
            self._style_axes()
            self.mpl_canvas = FigureCanvasTkAgg(self._fig, master=right_frame)
            self.mpl_canvas.get_tk_widget().pack(
                fill="both", expand=True, padx=8, pady=8)
        else:
            self.spec_canvas = tk.Canvas(
                right_frame, bg="#07071a", height=260,
                highlightthickness=0,
            )
            self.spec_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # Status bar ───────────────────────────────────────────────────────────
        status_bar = tk.Frame(self, bg="#080814", pady=3)
        status_bar.pack(fill="x", side="bottom")

        self.lbl_status = tk.Label(
            status_bar, text="Loading…",
            bg="#080814", fg=self.MUTED,
            font=("Courier", 9), anchor="w",
        )
        self.lbl_status.pack(side="left", padx=10)

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor("#07071a")
        ax.tick_params(colors=self.MUTED, labelsize=8)
        ax.spines["bottom"].set_color(self.BORDER)
        ax.spines["left"].set_color(self.BORDER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── cube loading ──────────────────────────────────────────────────────────

    def _load_cube(self, hdr_path: str):
        try:
            self.lbl_status.config(
                text=f"Opening {os.path.basename(hdr_path)}…")
            self.update()
            self.cube = HyperspectralCube(hdr_path)
            self.lbl_file.config(text=os.path.basename(hdr_path))
            self.title(f"Cube Browser — {self.cube.name}")
            self._render_rgb()
            m    = self.cube.meta
            info = (f"{self.cube.samples}×{self.cube.lines} px, "
                    f"{self.cube.bands} bands")
            if "reflectance scale factor" in m:
                info += f"  |  scale = {m['reflectance scale factor']}"
            self.lbl_status.config(text=info)
        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror("Error loading file", str(exc))
            self.lbl_status.config(text="Failed to load file.")

    def _render_rgb(self):
        """Build the source PIL image once, then show the initial (fit) view."""
        if self.cube is None:
            return
        self.lbl_status.config(text="Rendering RGB preview…")
        self.update()

        rgb = self.cube.read_rgb()          # (lines, samples, 3) uint8
        lines, samples = rgb.shape[:2]

        # Fixed canvas size
        cw = ch = self.CANVAS_MAX
        self._canvas_w = cw
        self._canvas_h = ch
        self.canvas.config(width=cw, height=ch)

        if HAS_PIL:
            self._rgb_pil = Image.fromarray(rgb, "RGB")
        else:
            self._rgb_pil = None

        # Base scale: how many canvas pixels correspond to one image pixel
        # when the whole image just fits inside the canvas (zoom = 1)
        self._base_scale = min(cw / samples, ch / lines)

        # Centre the image within the canvas at zoom=1
        fit_w = samples * self._base_scale
        fit_h = lines   * self._base_scale
        self._pan_x = (cw - fit_w) / 2.0
        self._pan_y = (ch - fit_h) / 2.0
        self._zoom  = 1.0

        self._refresh_view()
        self._update_zoom_label()

    def _refresh_view(self):
        """Redraw the canvas from the cached PIL image at current zoom/pan."""
        cw, ch = self._canvas_w, self._canvas_h
        self.canvas.delete("img")

        if not HAS_PIL or self._rgb_pil is None:
            # No-PIL fallback: static text placeholder
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, cw, ch,
                                          fill="#1a2a3a", outline="", tags="img")
            if self.cube:
                self.canvas.create_text(
                    cw // 2, ch // 2,
                    text=(f"{self.cube.samples}×{self.cube.lines} px\n"
                          f"{self.cube.bands} bands\n\n"
                          "Install Pillow for image preview:\n"
                          "pip install Pillow"),
                    fill="#5060aa", font=("Courier", 10),
                    justify="center", tags="img",
                )
            return

        img_w, img_h = self._rgb_pil.size
        eff = self._base_scale * self._zoom      # canvas px per image px

        # Viewport in image coordinates
        x0 = -self._pan_x / eff
        y0 = -self._pan_y / eff
        x1 = x0 + cw / eff
        y1 = y0 + ch / eff

        # Clamp crop to image bounds
        cx0 = max(0.0, x0);  cy0 = max(0.0, y0)
        cx1 = min(float(img_w), x1);  cy1 = min(float(img_h), y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return

        cropped = self._rgb_pil.crop(
            (int(cx0), int(cy0), int(cx1), int(cy1)))

        # Destination size on canvas
        dest_w = max(1, int((cx1 - cx0) * eff))
        dest_h = max(1, int((cy1 - cy0) * eff))

        resample = Image.NEAREST if eff >= 3 else (
            Image.BILINEAR if eff >= 1 else Image.LANCZOS)
        resized = cropped.resize((dest_w, dest_h), resample)

        # Paste onto a black canvas-sized image
        canvas_img = Image.new("RGB", (cw, ch), (7, 7, 26))
        dest_x = max(0, int(cx0 * eff + self._pan_x))
        dest_y = max(0, int(cy0 * eff + self._pan_y))
        canvas_img.paste(resized, (dest_x, dest_y))

        self._photo = ImageTk.PhotoImage(canvas_img)
        self.canvas.create_image(0, 0, anchor="nw",
                                  image=self._photo, tags="img")
        # Keep crosshair on top
        self._redraw_crosshair()

    # ── coordinate mapping ────────────────────────────────────────────────────

    def _canvas_to_data(self, cx: int, cy: int):
        """Convert a canvas pixel position to data (col, row)."""
        eff = self._base_scale * self._zoom
        col = int((cx - self._pan_x) / eff)
        row = int((cy - self._pan_y) / eff)
        col = max(0, min(col, self.cube.samples - 1))
        row = max(0, min(row, self.cube.lines   - 1))
        return col, row

    def _data_to_canvas(self, col: int, row: int):
        """Convert data (col, row) to canvas pixel position."""
        eff = self._base_scale * self._zoom
        cx = col * eff + self._pan_x
        cy = row * eff + self._pan_y
        return cx, cy

    # ── zoom / pan ────────────────────────────────────────────────────────────

    ZOOM_MIN = 0.5
    ZOOM_MAX = 64.0
    ZOOM_STEP = 1.3

    def _on_scroll(self, event):
        if self.cube is None:
            return
        # Determine direction (Windows/macOS vs Linux X11)
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            factor = self.ZOOM_STEP
        else:
            factor = 1.0 / self.ZOOM_STEP

        new_zoom = max(self.ZOOM_MIN, min(self._zoom * factor, self.ZOOM_MAX))
        if new_zoom == self._zoom:
            return

        # Zoom centered on cursor: the image pixel under the cursor stays fixed
        old_eff = self._base_scale * self._zoom
        new_eff = self._base_scale * new_zoom
        self._pan_x = event.x - (event.x - self._pan_x) * (new_eff / old_eff)
        self._pan_y = event.y - (event.y - self._pan_y) * (new_eff / old_eff)
        self._zoom  = new_zoom

        self._refresh_view()
        self._update_zoom_label()

    def _on_pan_start(self, event):
        self._drag_start     = (event.x, event.y)
        self._drag_pan_start = (self._pan_x, self._pan_y)

    def _on_pan_drag(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._pan_x = self._drag_pan_start[0] + dx
        self._pan_y = self._drag_pan_start[1] + dy
        self._refresh_view()

    def _reset_zoom(self):
        if self.cube is None:
            return
        s = self.cube.samples
        l = self.cube.lines
        cw, ch = self._canvas_w, self._canvas_h
        self._base_scale = min(cw / s, ch / l)
        fit_w = s * self._base_scale
        fit_h = l * self._base_scale
        self._pan_x = (cw - fit_w) / 2.0
        self._pan_y = (ch - fit_h) / 2.0
        self._zoom  = 1.0
        self._refresh_view()
        self._update_zoom_label()

    def _update_zoom_label(self):
        pct = int(round(self._zoom * 100))
        self.lbl_zoom.config(text=f"{pct}%")

    # ── mouse click / motion ──────────────────────────────────────────────────

    def _on_motion(self, event):
        if self.cube is None:
            return
        col, row = self._canvas_to_data(event.x, event.y)
        self.lbl_coords.config(
            text=f"col={col}  row={row}"
                 f"  ·  zoom {int(round(self._zoom*100))}%"
                 f"  ·  Scroll=zoom  Middle/Right-drag=pan  Dbl-click=reset")

    def _on_click(self, event):
        if self.cube is None:
            return
        col, row = self._canvas_to_data(event.x, event.y)
        spectrum = self.cube.spectrum_at(col, row)
        self._current_spectrum = spectrum
        self._current_xy       = (col, row)
        self._refresh_view()       # redraws image + crosshair
        self._plot_spectrum(spectrum, col, row)
        self.btn_export.config(state="normal")

    def _draw_crosshair(self, cx: float, cy: float):
        """Draw crosshair at canvas position (cx, cy)."""
        self.canvas.delete("crosshair")
        # Don't draw if outside canvas bounds
        if not (0 <= cx <= self._canvas_w and 0 <= cy <= self._canvas_h):
            return
        colour = "#ff4444"
        r = 8
        self.canvas.create_line(cx - r, cy, cx + r, cy,
                                 fill=colour, width=1, tags="crosshair")
        self.canvas.create_line(cx, cy - r, cx, cy + r,
                                 fill=colour, width=1, tags="crosshair")
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 outline=colour, width=1, tags="crosshair")

    def _redraw_crosshair(self):
        """Reposition crosshair after zoom/pan using stored data coords."""
        if self._current_xy is None:
            return
        col, row = self._current_xy
        cx, cy = self._data_to_canvas(col, row)
        self._draw_crosshair(cx, cy)

    # ── spectrum ──────────────────────────────────────────────────────────────

    def _plot_spectrum(self, spectrum: np.ndarray, col: int, row: int):
        wl     = self.cube.wavelengths
        values = spectrum.copy()

        # Mask no-data using the same tolerance as read_rgb
        if self.cube.ignore_value is not None:
            mask   = np.isclose(values, self.cube.ignore_value,
                                rtol=0.0, atol=0.5)
            values = np.where(mask, np.nan, values)

        self.lbl_pixel.config(
            text=f"Pixel  col={col}  row={row}  —  {self.cube.bands} bands")

        if HAS_MPL:
            self._plot_mpl(wl, values, col, row)
        else:
            self._plot_canvas(wl, values)

    def _plot_mpl(self, wl, values, col, row):
        self.ax.cla()
        self._style_axes()

        valid = ~np.isnan(values)
        if valid.any():
            self.ax.plot(wl[valid], values[valid],
                         color=self.ACCENT, linewidth=1.2, zorder=3)
            self.ax.fill_between(wl[valid], values[valid],
                                 alpha=0.15, color=self.ACCENT, zorder=2)
        else:
            self.ax.text(
                0.5, 0.5, "All no-data",
                transform=self.ax.transAxes,
                ha="center", va="center",
                color=self.MUTED, fontsize=10,
            )

        xlabel = ("Wavelength (nm)"
                  if not self.cube._wavelengths_are_indices
                  else "Band index")
        self.ax.set_xlabel(xlabel, color=self.MUTED, fontsize=9)
        self.ax.set_ylabel("DN", color=self.MUTED, fontsize=9)
        self.ax.set_title(f"col={col}  row={row}",
                          color="#a0b0dd", fontsize=9)
        self.ax.tick_params(colors=self.MUTED, labelsize=8)
        self._fig.tight_layout(pad=1.2)
        self.mpl_canvas.draw()

    def _plot_canvas(self, wl, values):
        """Fallback when matplotlib is unavailable."""
        c = self.spec_canvas
        c.delete("all")
        c.update_idletasks()
        w   = max(c.winfo_width(),  200)
        h   = max(c.winfo_height(), 150)
        pad = 35

        valid = ~np.isnan(values)
        if not valid.any():
            c.create_text(w // 2, h // 2, text="No valid data",
                          fill=self.MUTED, font=("Courier", 10))
            return

        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        wmin, wmax = float(wl[0]), float(wl[-1])
        if vmax == vmin: vmax = vmin + 1
        if wmax == wmin: wmax = wmin + 1

        def tx(x): return pad + (x - wmin) / (wmax - wmin) * (w - 2 * pad)
        def ty(y): return (h - pad) - (y - vmin) / (vmax - vmin) * (h - 2 * pad)

        c.create_line(pad, pad, pad, h - pad, fill=self.BORDER)
        c.create_line(pad, h - pad, w - pad, h - pad, fill=self.BORDER)
        c.create_text(w // 2, h - 8,
                      text="Wavelength / Band index",
                      fill=self.MUTED, font=("Courier", 8))

        pts = []
        for xi, yi in zip(wl, values):
            if np.isnan(yi):
                if len(pts) >= 4:
                    c.create_line(*pts, fill=self.ACCENT,
                                  width=1, smooth=True)
                pts = []
            else:
                pts += [tx(xi), ty(yi)]
        if len(pts) >= 4:
            c.create_line(*pts, fill=self.ACCENT, width=1, smooth=True)

    # ── CSV export ────────────────────────────────────────────────────────────

    def _export_csv(self):
        if self._current_spectrum is None:
            return
        col, row = self._current_xy
        default_name = (f"spectrum_{self.cube.name}"
                        f"_col{col}_row{row}.csv")
        path = filedialog.asksaveasfilename(
            title="Save Spectrum CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if not path:
            return

        wl     = self.cube.wavelengths
        values = self._current_spectrum
        ignore = self.cube.ignore_value
        is_idx = self.cube._wavelengths_are_indices
        wl_col = "band_index" if is_idx else "wavelength_nm"

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["# Hyperspectral spectrum export"])
                writer.writerow([f"# Source: {self.cube.hdr_path}"])
                writer.writerow([f"# Pixel: col={col}  row={row}"])
                if ignore is not None:
                    writer.writerow([f"# data ignore value: {ignore}"])
                sf = self.cube.meta.get("reflectance scale factor")
                if sf is not None:
                    writer.writerow([f"# reflectance scale factor: {sf}"])
                writer.writerow([wl_col, "dn_value", "valid"])
                for w_val, v in zip(wl, values):
                    is_valid = 1
                    if ignore is not None and np.isclose(
                            v, ignore, rtol=0.0, atol=0.5):
                        is_valid = 0
                    writer.writerow([f"{w_val:.6g}", f"{v:.6g}", is_valid])

            self.lbl_status.config(
                text=f"Exported → {os.path.basename(path)}")
            messagebox.showinfo("Export complete",
                                f"Spectrum saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def scan_for_headers(directory: str = "data/images") -> list:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("*.hdr"))


def main():
    headers = scan_for_headers("data/images")

    if not headers:
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askopenfilename(
            title="No .hdr files found in data/images/ — select one manually",
            filetypes=[("ENVI headers", "*.hdr"), ("All files", "*.*")],
        )
        root.destroy()
        if not chosen:
            print("No file selected. Exiting.")
            sys.exit(0)
        hdr_path = chosen

    elif len(headers) == 1:
        hdr_path = headers[0]

    else:
        root = tk.Tk()
        root.withdraw()
        picker = FilePicker(root, headers)
        root.destroy()
        hdr_path = picker.result
        if not hdr_path:
            print("No file selected. Exiting.")
            sys.exit(0)

    app = App(hdr_path)
    app.mainloop()


if __name__ == "__main__":
    main()
