"""
Hyperspectral Data Cube Viewer
──────────────────────────────
  Two-finger scroll / scroll wheel → zoom in/out (centred on cursor)
  Left-click drag                  → pan
  Left-click (no drag)             → read spectral signature
  Reset Zoom button                → back to full view
  Export to CSV                    → save current pixel spectrum
  Open File                        → load another cube

Scroll is bound directly to the Tk canvas widget so it works on
Linux (Button-4/5), macOS (MouseWheel delta), and Windows (MouseWheel).

Memory: header-only open; 3 bands for RGB; 1 pixel vector per click.
"""

import sys, os, csv, platform
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from tkinter import Tk, filedialog, messagebox
import spectral.io.envi as envi

# ── globals ───────────────────────────────────────────
spy_img     = None
n_rows = n_cols = n_bands = 0
wavelengths = None
last_pixel  = None


# ── file loading ──────────────────────────────────────
def load_hdr_file(path):
    global spy_img, n_rows, n_cols, n_bands, wavelengths
    spy_img = envi.open(path)
    n_rows, n_cols, n_bands = spy_img.nrows, spy_img.ncols, spy_img.nbands
    meta = spy_img.metadata
    wavelengths = (
        np.array([float(w) for w in meta["wavelength"]])
        if "wavelength" in meta
        else np.arange(n_bands, dtype=np.float32)
    )


# ── RGB band selection ────────────────────────────────
def band_for_nm(target_nm):
    wl = wavelengths.copy()
    if wl.max() < 10:
        wl *= 1000.0
    if wl.max() < 200 or wl.min() > 3000:
        return int(n_bands * {660: .75, 550: .50, 460: .25}.get(int(target_nm), .5))
    return int(np.argmin(np.abs(wl - target_nm)))


def make_rgb():
    r, g, b = band_for_nm(660), band_for_nm(550), band_for_nm(460)
    if len({r, g, b}) < 3:
        r, g, b = int(n_bands*.75), int(n_bands*.50), int(n_bands*.25)
    print(f"  RGB bands: R={r} G={g} B={b}  "
          f"λ≈{wavelengths[r]:.1f}/{wavelengths[g]:.1f}/{wavelengths[b]:.1f}")
    chs = []
    for idx in (r, g, b):
        band = spy_img.read_band(idx).astype(np.float32)
        lo, hi = np.percentile(band, (2, 98))
        chs.append(np.clip((band - lo) / (hi - lo + 1e-9), 0, 1))
    return np.stack(chs, axis=-1)


def read_pixel(row, col):
    return spy_img.read_pixel(row, col).astype(np.float64)


# ── export ────────────────────────────────────────────
def export_to_csv():
    if last_pixel is None:
        messagebox.showwarning("No pixel selected", "Click a pixel first.")
        return
    row, col = last_pixel
    spectrum = read_pixel(row, col)
    root = Tk(); root.withdraw()
    path = filedialog.asksaveasfilename(
        title="Save spectrum", defaultextension=".csv",
        filetypes=[("CSV", "*.csv")],
        initialfile=f"spectrum_r{row}_c{col}.csv")
    root.destroy()
    if not path:
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Wavelength", "Reflectance"])
        for wl, v in zip(wavelengths, spectrum):
            w.writerow([wl, float(v)])
    messagebox.showinfo("Exported", f"Saved to:\n{path}")


# ── zoom helper ───────────────────────────────────────
ZOOM = 1.20   # factor per scroll step

def _zoom_ax(ax, cx, cy, factor):
    """Zoom ax by factor centred on data-coords (cx, cy)."""
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(cx + (xl[0]-cx)*factor, cx + (xl[1]-cx)*factor)
    ax.set_ylim(cy + (yl[0]-cy)*factor, cy + (yl[1]-cy)*factor)


def _tk_to_data(ax, fig, tk_x, tk_y):
    """Convert Tk pixel coords to matplotlib data coords for ax."""
    # Tk gives coords relative to the widget; matplotlib uses figure pixels
    # fig.canvas.get_tk_widget() is the Tk widget
    w = fig.canvas.get_tk_widget()
    # widget position inside the Tk window
    wx = w.winfo_rootx()
    wy = w.winfo_rooty()
    fig_x = tk_x - wx
    fig_y = tk_y - wy
    # convert figure pixels → display coords → data coords
    display_pt = (fig_x, fig.bbox.height - fig_y)   # flip y
    inv = ax.transData.inverted()
    try:
        return inv.transform(ax.transData.transform(
            ax.transAxes.inverted().transform(
                ax.transAxes.transform([0, 0])   # dummy
            )
        ))
    except Exception:
        return None, None


def _fig_px_to_data(ax, fig, fig_x, fig_y):
    """figure-pixel coords → ax data coords (or None if outside ax)."""
    display = (fig_x, fig.bbox.height - fig_y)
    try:
        data = ax.transData.inverted().transform(display)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        # check inside axes bounding box in display space
        axbb = ax.get_window_extent()
        if axbb.contains(*display):
            return data[0], data[1]
    except Exception:
        pass
    return None, None


# ── main viewer ───────────────────────────────────────
def open_viewer():
    global last_pixel

    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Open hyperspectral HDR file",
        filetypes=[("ENVI Header", "*.hdr"), ("All files", "*.*")])
    root.destroy()
    if not path:
        print("No file selected."); sys.exit(0)

    print(f"Opening: {path}")
    load_hdr_file(path)
    print(f"  {n_rows}×{n_cols} px | {n_bands} bands "
          f"| λ {wavelengths[0]:.1f}–{wavelengths[-1]:.1f}")
    print("Building RGB …")
    rgb = make_rgb()
    print("  Done.")

    # ── layout ───────────────────────────────────────
    fig = plt.figure(figsize=(14, 6.5), facecolor="#1e1e2e")
    fig.canvas.manager.set_window_title("Hyperspectral Viewer")
    # hide the default toolbar (it has its own conflicting zoom)
    try:
        fig.canvas.manager.toolbar.pack_forget()
    except Exception:
        pass

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.05, right=0.97,
                           top=0.92,  bottom=0.13,
                           wspace=0.35, hspace=0.45)
    ax_img  = fig.add_subplot(gs[:, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])

    for ax in (ax_img, ax_spec, ax_info):
        ax.set_facecolor("#13131f")

    ax_img.imshow(rgb, interpolation="nearest", aspect="auto")
    ax_img.set_title("scroll=zoom  |  drag=pan  |  click=spectrum",
                     color="white", fontsize=9, pad=6)
    ax_img.tick_params(colors="gray", labelsize=7)
    for sp in ax_img.spines.values():
        sp.set_edgecolor("#444")

    marker_pt, = ax_img.plot([], [], "+", color="#ff4f4f",
                             markersize=14, markeredgewidth=1.8, zorder=5)
    full_xlim = ax_img.get_xlim()
    full_ylim = ax_img.get_ylim()

    ax_spec.set_title("Spectral Signature", color="white", fontsize=10, pad=6)
    ax_spec.set_xlabel("Wavelength", color="#aaa", fontsize=8)
    ax_spec.set_ylabel("Reflectance", color="#aaa", fontsize=8)
    ax_spec.tick_params(colors="gray", labelsize=7)
    for sp in ax_spec.spines.values():
        sp.set_edgecolor("#444")
    ph_txt = ax_spec.text(0.5, 0.5, "Click a pixel →",
                          transform=ax_spec.transAxes,
                          ha="center", va="center", color="#555", fontsize=11)
    spec_line, = ax_spec.plot([], [], color="#7ec8e3", linewidth=1.2)

    ax_info.axis("off")
    ax_info.text(0.05, 0.85,
                 f"File:  {os.path.basename(path)}\n"
                 f"Shape: {n_rows}×{n_cols} px  |  {n_bands} bands\n"
                 f"λ: {wavelengths[0]:.1f}–{wavelengths[-1]:.1f}",
                 transform=ax_info.transAxes,
                 color="#aaa", fontsize=8, va="top", fontfamily="monospace")
    pix_lbl  = ax_info.text(0.05, 0.42, "Pixel: —",
                            transform=ax_info.transAxes,
                            color="#ccc", fontsize=9, va="top")
    zoom_lbl = ax_info.text(0.05, 0.12, "Zoom: full view",
                            transform=ax_info.transAxes,
                            color="#888", fontsize=8, va="top",
                            fontfamily="monospace")

    def make_btn(rect, label):
        axb = fig.add_axes(rect)
        b = Button(axb, label, color="#2a2a3e", hovercolor="#3d3d5c")
        b.label.set_color("white"); b.label.set_fontsize(9)
        return b

    btn_reset   = make_btn([0.06, 0.03, 0.10, 0.055], "Reset")
    btn_zoomin  = make_btn([0.17, 0.03, 0.07, 0.055], "+")
    btn_zoomout = make_btn([0.25, 0.03, 0.07, 0.055], "-")
    btn_open    = make_btn([0.52, 0.03, 0.18, 0.055], "Open File...")
    btn_export  = make_btn([0.72, 0.03, 0.18, 0.055], "Export to CSV")

    btn_export.on_clicked(lambda _: export_to_csv())
    btn_open.on_clicked(lambda _: [plt.close(fig), open_viewer()])

    def reset_zoom(_):
        ax_img.set_xlim(full_xlim)
        ax_img.set_ylim(full_ylim)
        zoom_lbl.set_text("Zoom: full view")
        fig.canvas.draw_idle()
    btn_reset.on_clicked(reset_zoom)

    def btn_zoom(factor, _):
        xl, yl = ax_img.get_xlim(), ax_img.get_ylim()
        cx = (xl[0] + xl[1]) / 2
        cy = (yl[0] + yl[1]) / 2
        _zoom_ax(ax_img, cx, cy, factor)
        update_zoom_lbl()
        fig.canvas.draw_idle()

    btn_zoomin.on_clicked(lambda e: btn_zoom(1.0/ZOOM, e))
    btn_zoomout.on_clicked(lambda e: btn_zoom(ZOOM, e))

    def update_zoom_lbl():
        xl, yl = ax_img.get_xlim(), ax_img.get_ylim()
        zoom_lbl.set_text(
            f"col {int(xl[0])}–{int(xl[1])}, row {int(yl[1])}–{int(yl[0])}")

    # ── pan state ────────────────────────────────────
    pan = {"active": False, "x0": 0, "y0": 0,
           "xl0": None, "yl0": None, "moved": False}

    # ── matplotlib mouse handlers (pan + click) ──────
    def on_mpl_press(event):
        if event.inaxes != ax_img or event.button != 1:
            return
        pan["active"] = True
        pan["x0"]   = event.xdata
        pan["y0"]   = event.ydata
        pan["xl0"]  = ax_img.get_xlim()
        pan["yl0"]  = ax_img.get_ylim()
        pan["moved"] = False

    def on_mpl_motion(event):
        if not pan["active"] or event.inaxes != ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - pan["x0"]
        dy = event.ydata - pan["y0"]
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            pan["moved"] = True
        ax_img.set_xlim(pan["xl0"][0]-dx, pan["xl0"][1]-dx)
        ax_img.set_ylim(pan["yl0"][0]-dy, pan["yl0"][1]-dy)
        fig.canvas.draw_idle()

    def on_mpl_release(event):
        if event.button != 1:
            return
        was_pan = pan["moved"]
        pan["active"] = False
        pan["moved"]  = False

        if was_pan:
            update_zoom_lbl()
            fig.canvas.draw_idle()
            return

        # ── it was a click: read spectrum ─────────────
        if event.inaxes != ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = max(0, min(int(round(event.xdata)), n_cols-1))
        row = max(0, min(int(round(event.ydata)), n_rows-1))

        global last_pixel
        last_pixel = (row, col)
        spectrum   = read_pixel(row, col)

        marker_pt.set_data([col], [row])
        ph_txt.set_visible(False)
        spec_line.set_data(wavelengths, spectrum)
        ax_spec.relim(); ax_spec.autoscale_view()
        for c in ax_spec.collections:
            c.remove()
        ax_spec.fill_between(wavelengths, spectrum, alpha=0.18, color="#7ec8e3")
        pix_lbl.set_text(
            f"Pixel:  row={row}, col={col}\n"
            f"Min={float(spectrum.min()):.4f}  "
            f"Max={float(spectrum.max()):.4f}  "
            f"Mean={float(spectrum.mean()):.4f}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event",   on_mpl_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_mpl_motion)
    fig.canvas.mpl_connect("button_release_event", on_mpl_release)

    # ── Tk-native scroll bindings ─────────────────────
    # Must be done after the figure is drawn so get_tk_widget() exists.
    # We zoom in data-space using the event's x/y in display pixels.

    def _zoom(direction, event_x, event_y):
        """direction: +1 = zoom in, -1 = zoom out."""
        factor = (1.0/ZOOM) if direction > 0 else ZOOM
        # convert Tk widget pixels → matplotlib display pixels
        tk_w = fig.canvas.get_tk_widget()
        # event x/y are relative to the widget in Tk coords
        fig_x = event_x
        fig_y = tk_w.winfo_height() - event_y   # flip y for mpl
        # display coords → data coords via ax transform
        display = np.array([[fig_x, fig_y]])
        try:
            data = ax_img.transData.inverted().transform(display)[0]
        except Exception:
            return
        # check cursor is inside the image axes
        axbb = ax_img.get_window_extent()
        if not axbb.contains(fig_x, fig_y):
            return
        cx, cy = data[0], data[1]
        _zoom_ax(ax_img, cx, cy, factor)
        update_zoom_lbl()
        fig.canvas.draw_idle()

    OS = platform.system()

    def _bind_scroll():
        tk_w = fig.canvas.get_tk_widget()
        if OS == "Linux":
            # X11: Button-4 = scroll up, Button-5 = scroll down
            tk_w.bind("<Button-4>",
                      lambda e: _zoom(+1, e.x, e.y), add="+")
            tk_w.bind("<Button-5>",
                      lambda e: _zoom(-1, e.x, e.y), add="+")
        elif OS == "Darwin":
            # macOS: delta is in pixels of trackpad movement (small floats).
            # Scale factor smoothly so fast swipes zoom faster.
            def _mw_mac(e):
                if e.delta == 0:
                    return
                # clamp to avoid huge jumps on fast swipes
                d = max(-8, min(8, e.delta))
                factor = (1.0 / ZOOM) ** (d / 3.0)
                axbb = ax_img.get_window_extent()
                tk_w2 = fig.canvas.get_tk_widget()
                fig_x = e.x
                fig_y = tk_w2.winfo_height() - e.y
                if not axbb.contains(fig_x, fig_y):
                    return
                try:
                    data = ax_img.transData.inverted().transform(
                        [[fig_x, fig_y]])[0]
                except Exception:
                    return
                _zoom_ax(ax_img, data[0], data[1], factor)
                update_zoom_lbl()
                fig.canvas.draw_idle()
            tk_w.bind("<MouseWheel>", _mw_mac, add="+")
        else:
            # Windows: delta is ±120 per notch
            def _mw_win(e):
                direction = +1 if e.delta > 0 else -1
                _zoom(direction, e.x, e.y)
            tk_w.bind("<MouseWheel>", _mw_win, add="+")

    # Bind after the window is mapped so winfo_height() is valid
    fig.canvas.get_tk_widget().after(100, _bind_scroll)

    plt.suptitle("Hyperspectral Viewer", color="white",
                 fontsize=12, fontweight="bold", y=0.97)
    plt.show()


if __name__ == "__main__":
    open_viewer()