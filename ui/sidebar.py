"""
ui/sidebar.py
-------------
VS Code-inspired Activity Bar and Side Panel for CIRCA.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import (
    Qt,
    pyqtSignal,
    pyqtSlot,
    QPropertyAnimation,
    QEasingCurve,
    QRectF
)
from PyQt6.QtGui import QFont, QPainter
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QTabWidget
)

from core.models import InferenceParams, PreprocessParams
from ui.theme import (
    COLOR_ACCENT_CYAN,
    SIDE_PANEL_WIDTH_EXPANDED,
    FONT_MONO,
    FONT_SIZE_LABEL,
    FONT_SIZE_MONO_LIVE,
    FONT_UI,
    SPACING_LG,
    SPACING_MD,
    SPACING_SM,
    SPACING_XS,
    ThemeMode,
    AnimationMode,
    ThemeManager,
    ICONS_SVG
)

logger = logging.getLogger(__name__)

class IconWidget(QWidget):
    """Simple widget to render a Lucide SVG icon."""
    def __init__(self, icon_name: str, size: int = 16, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._icon_name = icon_name
        self._renderer = QSvgRenderer()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt C++ API override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = ThemeManager().get_palette()
        color = palette["TEXT_SECONDARY"]
        svg_xml = ICONS_SVG.get(self._icon_name, "").format(color=color)
        if svg_xml:
            self._renderer.load(svg_xml.encode('utf-8'))
            self._renderer.render(painter, QRectF(self.rect()))

class SidebarButton(QPushButton):
    """Activity Bar button with SVG icon resized for VS Code proportions."""
    def __init__(self, icon_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ActivityButton")
        self._icon_name = icon_name
        self.setFixedSize(48, 48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._renderer = QSvgRenderer()
        self._active = False

    def set_active(self, active: bool):
        self._active = active
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt C++ API override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        palette = ThemeManager().get_palette()
        color = COLOR_ACCENT_CYAN if self._active else palette["TEXT_SECONDARY"]

        svg_xml = ICONS_SVG.get(self._icon_name, "").format(color=color)
        if svg_xml:
            self._renderer.load(svg_xml.encode('utf-8'))
            margin = 15
            self._renderer.render(painter, QRectF(self.rect().adjusted(margin, margin, -margin, -margin)))

class PreprocessingSlider(QWidget):
    value_changed = pyqtSignal(float)
    def __init__(self, icon_name: str, name: str, min_val: float, max_val: float, default_val: float, precision: int = 1, suffix: str = "", parent=None):
        super().__init__(parent)
        self._precision, self._scale, self._suffix = precision, 10**precision, suffix
        self._build_ui(icon_name, name, min_val, max_val, default_val)
        self._slider.valueChanged.connect(self._on_slider_changed)

    def _build_ui(self, icon_name, name, min_val, max_val, default_val):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, SPACING_SM)
        root.setSpacing(SPACING_XS)

        header = QHBoxLayout()
        self._icon = IconWidget(icon_name, 14)
        header.addWidget(self._icon)

        self._name_label = QLabel(name)
        self._name_label.setFont(QFont(FONT_UI, FONT_SIZE_LABEL, QFont.Weight.Medium))
        header.addWidget(self._name_label, stretch=1)

        self._value_label = QLabel()
        self._value_label.setFont(QFont(FONT_MONO, FONT_SIZE_MONO_LIVE))
        self._value_label.setStyleSheet(f"color: {COLOR_ACCENT_CYAN};")
        header.addWidget(self._value_label)
        root.addLayout(header)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(min_val * self._scale))
        self._slider.setMaximum(int(max_val * self._scale))
        self._slider.setValue(int(default_val * self._scale))
        root.addWidget(self._slider)
        self._refresh_label()

    def value(self):
        return self._slider.value() / self._scale

    def _on_slider_changed(self, v):
        self._refresh_label()
        self.value_changed.emit(self.value())

    def _refresh_label(self):
        self._value_label.setText(f"{self.value():.{self._precision}f}{self._suffix}")

class ActivityBar(QWidget):
    tab_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ActivityBar")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACING_MD, 0, SPACING_MD)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.optimization_btn = SidebarButton("activity")
        self.optimization_btn.setToolTip("Optimisation")
        self.optimization_btn.clicked.connect(lambda: self._on_btn_clicked("optimization"))
        layout.addWidget(self.optimization_btn)

        layout.addStretch(1)

        self.settings_btn = SidebarButton("settings")
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.clicked.connect(lambda: self._on_btn_clicked("prefs"))
        layout.addWidget(self.settings_btn)

        self._buttons = {"optimization": self.optimization_btn, "prefs": self.settings_btn}
        self.set_active("optimization")

    def _on_btn_clicked(self, key: str):
        self.set_active(key)
        self.tab_selected.emit(key)

    def set_active(self, key: str):
        for k, btn in self._buttons.items():
            btn.set_active(k == key)

    @pyqtSlot(bool, str)
    def handle_panel_toggle(self, expanded: bool, key: str):
        """Update button highlights based on panel expansion state."""
        if not expanded:
            for btn in self._buttons.values():
                btn.set_active(False)
        else:
            self.set_active(key)

class SidePanel(QWidget):
    preprocessing_params_changed = pyqtSignal(object)
    inference_params_changed = pyqtSignal(object)
    camera_selected = pyqtSignal(int)
    theme_changed = pyqtSignal(object)
    panel_toggled = pyqtSignal(bool, str)  # expanded, active_key

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SidePanel")
        self._expanded = False
        self.setMinimumWidth(0)
        self.setMaximumWidth(0)

        self._build_ui()
        self._setup_animations()
        self._wire_signals()

    def _build_ui(self):
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(SPACING_MD, SPACING_LG, SPACING_MD, SPACING_LG)
        self.root_layout.setSpacing(SPACING_LG)

        # Header
        self.title_label = QLabel("OPTIMISATION")
        self.title_label.setFont(QFont(FONT_UI, 10, QFont.Weight.Bold))
        self.title_label.setStyleSheet("letter-spacing: 0.5px; color: #888;")
        self.root_layout.addWidget(self.title_label)

        self.stack = QTabWidget()
        self.stack.setDocumentMode(True)
        self.stack.tabBar().hide()

        # Optimisation Page
        opt_page = QWidget()
        ol = QVBoxLayout(opt_page)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(SPACING_MD)
        ol.addWidget(self._section_header("VISION SOURCE"))
        self.camera_combo = QComboBox()
        ol.addWidget(self.camera_combo)
        ol.addWidget(self._section_header("IMAGE OPTIMISATION"))
        self.clahe_slider = PreprocessingSlider("contrast", "Contrast", 1.0, 8.0, 2.0)
        self.gamma_slider = PreprocessingSlider("sun", "Brightness", 0.5, 2.5, 1.0)
        self.sharpness_slider = PreprocessingSlider("focus", "Sharpness Gate", 20.0, 300.0, 100.0, 0)
        ol.addWidget(self.clahe_slider)
        ol.addWidget(self.gamma_slider)
        ol.addWidget(self.sharpness_slider)
        ol.addWidget(self._section_header("AI INTELLIGENCE"))
        self.confidence_slider = PreprocessingSlider("brain", "Sensitivity", 10.0, 95.0, 50.0, 0, "%")
        ol.addWidget(self.confidence_slider)
        ol.addStretch(1)

        # Preferences Page
        prefs_page = QWidget()
        pl = QVBoxLayout(prefs_page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(SPACING_MD)
        pl.addWidget(self._section_header("THEME"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Mode", "Light Mode", "System Sync"])
        pl.addWidget(self.theme_combo)
        pl.addWidget(self._section_header("UI DYNAMICS"))
        self.anim_combo = QComboBox()
        self.anim_combo.addItems(["Fluid Physics", "Snappy"])
        pl.addWidget(self.anim_combo)
        pl.addStretch(1)

        self.stack.addTab(opt_page, "optimization")
        self.stack.addTab(prefs_page, "prefs")
        self.root_layout.addWidget(self.stack)

    def _setup_animations(self):
        self._anim = QPropertyAnimation(self, b"maximumWidth")
        self._anim.setEasingCurve(QEasingCurve.Type.OutExpo)

    def _wire_signals(self):
        self.clahe_slider.value_changed.connect(self._emit_preprocessing_params)
        self.gamma_slider.value_changed.connect(self._emit_preprocessing_params)
        self.sharpness_slider.value_changed.connect(self._emit_preprocessing_params)
        self.confidence_slider.value_changed.connect(self._emit_inference_params)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.anim_combo.currentIndexChanged.connect(self._on_anim_changed)

    def set_tab(self, key: str):
        idx = 0 if key == "optimization" else 1
        if self._expanded and self.stack.currentIndex() == idx:
            self.toggle()
        else:
            self.stack.setCurrentIndex(idx)
            self.title_label.setText(key.upper() if key != "prefs" else "SETTINGS")
            if not self._expanded:
                self.toggle()
        self.panel_toggled.emit(self._expanded, key)

    def toggle(self):
        config = ThemeManager().get_anim_config()
        self._anim.setDuration(config["duration"])
        if self._expanded:
            self._anim.setStartValue(SIDE_PANEL_WIDTH_EXPANDED)
            self._anim.setEndValue(0)
        else:
            self._anim.setStartValue(0)
            self._anim.setEndValue(SIDE_PANEL_WIDTH_EXPANDED)
        self._expanded = not self._expanded
        self._anim.start()

    def _on_theme_changed(self, index):
        mode = [ThemeMode.DARK, ThemeMode.LIGHT, ThemeMode.SYSTEM][index]
        ThemeManager().mode = mode
        self.theme_changed.emit(mode)

    def _on_anim_changed(self, index):
        ThemeManager().anim_mode = AnimationMode.FLUID if index == 0 else AnimationMode.SNAPPY

    def populate_cameras(self, cameras):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for idx, name in cameras:
            self.camera_combo.addItem(name, idx)
        self.camera_combo.blockSignals(False)

    def _on_camera_changed(self, idx):
        if idx >= 0:
            self.camera_selected.emit(self.camera_combo.itemData(idx))

    def _emit_preprocessing_params(self, _=0.0):
        self.preprocessing_params_changed.emit(PreprocessParams(
            clahe_clip_limit=self.clahe_slider.value(),
            gamma=self.gamma_slider.value(),
            blur_threshold=self.sharpness_slider.value(),
        ))

    def _emit_inference_params(self, _=0.0):
        self.inference_params_changed.emit(InferenceParams(
            confidence_threshold=self.confidence_slider.value() / 100.0
        ))

    @staticmethod
    def _section_header(text):
        label = QLabel(text)
        label.setFont(QFont(FONT_UI, 9, QFont.Weight.Bold))
        label.setProperty("secondary", "true")
        return label