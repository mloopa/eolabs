import sys
import csv
import numpy as np
from pathlib import Path

# PyQt6 - Nowoczesny interfejs użytkownika
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt

# Matplotlib - Integracja z PyQt6
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

import spectral.io.envi as envi
import mplcursors

# --- Konfiguracja Ścieżek ---
DEFAULT_PATH = Path(r"E:\Remote_sensing\eolabs-main\lab_5\Obrazy lotnicze")
FALLBACK_RGB = (30, 20, 10)

class SpectralApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Airborne Data Browser - Professional PyQt6 Edition")
        self.resize(1400, 850)

        # Stan aplikacji
        self.img = None
        self.current_spectrum = None
        self.pixel_coords = None
        self.rgb_image = None
        self.wavelengths = None
        self.cursor = None

        self._setup_ui()
        
        # Sprawdzenie domyślnej lokalizacji
        if DEFAULT_PATH.exists():
            self.status_label.setText(f"Folder: {DEFAULT_PATH.name}")

    def _setup_ui(self):
        """Tworzy układ: Lewy sidebar i prawy obszar z wykresami."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- PANEL LEWY (Sidebar) ---
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("background-color: #2c3e50; color: white;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        side_layout.setContentsMargins(15, 20, 15, 20)

        title = QLabel("NAWIGACJA")
        title.setStyleSheet("font-weight: bold; font-size: 18px; color: #ecf0f1; margin-bottom: 20px;")
        side_layout.addWidget(title)

        # Stylizacja przycisków CSS
        btn_css = """
            QPushButton { 
                background-color: #34495e; color: white; padding: 12px; 
                border: none; border-radius: 6px; font-size: 13px; text-align: left;
            }
            QPushButton:hover { background-color: #1abc9c; }
            QPushButton:pressed { background-color: #16a085; }
        """

        self.btn_open = QPushButton("📁  Otwórz Plik HDR")
        self.btn_csv = QPushButton("💾  Eksportuj do CSV")
        self.btn_reset = QPushButton("🔄  Reset Widoku Mapy")

        for btn in [self.btn_open, self.btn_csv, self.btn_reset]:
            btn.setStyleSheet(btn_css)
            side_layout.addWidget(btn)
            side_layout.addSpacing(10)

        self.btn_open.clicked.connect(self._open_file)
        self.btn_csv.clicked.connect(self._export_csv)
        self.btn_reset.clicked.connect(self._reset_view)

        side_layout.addStretch() # Popycha status na dół

        self.status_label = QLabel("Oczekiwanie na dane...")
        self.status_label.setStyleSheet("color: #bdc3c7; font-style: italic; font-size: 12px;")
        self.status_label.setWordWrap(True)
        side_layout.addWidget(self.status_label)

        main_layout.addWidget(sidebar)

        # --- PANEL PRAWY (Wykresy i Toolbar) ---
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        
        # Tworzenie wykresów Matplotlib
        self.fig = Figure(figsize=(12, 7), facecolor='white')
        self.canvas = FigureCanvas(self.fig)
        self.ax_img = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        
        # Pasek narzędzi Matplotlib (Zoom, Pan, Save)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: white; border-bottom: 1px solid #ddd;")
        
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        main_layout.addWidget(viz_container)

        # Podpięcie zdarzeń myszy
        self.canvas.mpl_connect("button_press_event", self._handle_click)
        self.canvas.mpl_connect("scroll_event", self._handle_scroll)

    # --- Logika Aplikacji ---

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik HDR", str(DEFAULT_PATH), "ENVI Header (*.hdr)"
        )
        if path:
            self._load_data(Path(path))

    def _load_data(self, path: Path):
        try:
            self.status_label.setText(f"Wczytywanie: {path.name}...")
            QApplication.processEvents() # Odśwież UI podczas ładowania

            self.img = envi.open(str(path))
            meta = self.img.metadata
            
            # Pobieranie kanałów RGB (default bands lub fallback)
            db = meta.get("default bands")
            bands = [int(float(v)) - 1 for v in db[:3]] if (db and len(db) >= 3) else list(FALLBACK_RGB)
            
            # Odczyt danych i stretch kontrastu
            rgb = self.img.read_bands(bands).astype(np.float32)
            for i in range(3):
                p2, p98 = np.nanpercentile(rgb[:,:,i], [2, 98])
                rgb[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2 + 1e-6), 0, 1)
            
            self.rgb_image = np.nan_to_num(rgb)
            wl = meta.get("wavelength")
            self.wavelengths = np.array([float(w) for w in wl]) if wl else np.arange(self.img.nbands)
            
            self.current_spectrum = None
            self.pixel_coords = None
            self._update_plots(full_refresh=True)
            self._reset_view()
            self.status_label.setText(f"Załadowano: {path.name}\nPasma: {self.img.nbands}\nRozmiar: {self.img.nrows}x{self.img.ncols}")
            
        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Błąd krytyczny:\n{str(e)}")

    def _update_plots(self, full_refresh=False):
        """Aktualizacja wykresów z usuwaniem starego kursora."""
        if self.cursor:
            self.cursor.remove()
            self.cursor = None

        if full_refresh:
            self.ax_img.clear()
            self.ax_img.imshow(self.rgb_image, aspect='equal', interpolation='nearest')
            self.ax_img.set_title("Podgląd Mapy Lotniczej", pad=10)
            self.ax_img.axis("off")

        # Rysowanie krzyżyka na obrazie
        for line in self.ax_img.lines: line.remove()
        if self.pixel_coords:
            self.ax_img.plot(self.pixel_coords[1], self.pixel_coords[0], 'r+', ms=15, mew=2)

        # Wykres widma
        self.ax_spec.clear()
        if self.current_spectrum is not None:
            line_plot, = self.ax_spec.plot(self.wavelengths, self.current_spectrum, color='#3498db', lw=2)
            self.ax_spec.set_title(f"Sygnatura Spektralna [x:{self.pixel_coords[1]}, y:{self.pixel_coords[0]}]")
            self.ax_spec.grid(True, alpha=0.3, linestyle='--')
            
            # Konfiguracja interaktywnego kursora (Transient - znika po zabraniu myszy)
            self.cursor = mplcursors.cursor(line_plot, hover=mplcursors.HoverMode.Transient)
            @self.cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(f"λ: {sel.target[0]:.2f}\nVal: {sel.target[1]:.4f}")
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, edgecolor='#3498db', boxstyle="round")
        else:
            self.ax_spec.text(0.5, 0.5, "Kliknij w mapę lotniczą", ha='center', color='gray')

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw_idle()

    def _reset_view(self):
        if self.rgb_image is not None:
            ny, nx, _ = self.rgb_image.shape
            self.ax_img.set_xlim(0, nx)
            self.ax_img.set_ylim(ny, 0)
            self.canvas.draw_idle()

    def _handle_click(self, event):
        if event.inaxes != self.ax_img or self.img is None: return
        # Blokada kliknięcia jeśli aktywne narzędzia Toolbaru
        if self.toolbar.mode != "": return

        col, row = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= row < self.img.nrows and 0 <= col < self.img.ncols:
            self.pixel_coords = (row, col)
            self.current_spectrum = self.img.read_pixel(row, col)
            self._update_plots()

    def _handle_scroll(self, event):
        """Poprawiony zoom kółkiem myszy - bez obracania zdjęcia."""
        if event.inaxes != self.ax_img or self.rgb_image is None:
            return

        # Pobierz aktualne limity
        cur_xlim = self.ax_img.get_xlim()
        cur_ylim = self.ax_img.get_ylim() # W obrazach zazwyczaj (max, 0)
        
        # Współczynnik przybliżenia
        scale_factor = 1/1.3 if event.button == 'up' else 1.3

        # Oblicz nowe szerokości/wysokości
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor # Tu różnica będzie ujemna

        # Wyznacz relatywną pozycję myszy wewnątrz aktualnego widoku (0-1)
        rel_x = (event.xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (event.ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        # Ustaw nowe limity zachowując punkt pod myszką w tym samym miejscu
        self.ax_img.set_xlim([event.xdata - new_width * rel_x, 
                             event.xdata + new_width * (1 - rel_x)])
        
        # Kluczowe: set_ylim musi zachować oryginalną "orientację" (np. od 500 do 0)
        self.ax_img.set_ylim([event.ydata - new_height * rel_y, 
                             event.ydata + new_height * (1 - rel_y)])
        
        self.canvas.draw_idle()
    def _export_csv(self):
        if self.current_spectrum is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz Widmo jako CSV", "", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Wavelength", "Value"])
                for wl, val in zip(self.wavelengths, self.current_spectrum):
                    writer.writerow([wl, val])
            QMessageBox.information(self, "Sukces", "Plik CSV został pomyślnie zapisany!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Stylizacja aplikacji (ciemny sidebar, jasna reszta)
    app.setStyle("Fusion")
    window = SpectralApp()
    window.show()
    sys.exit(app.exec())