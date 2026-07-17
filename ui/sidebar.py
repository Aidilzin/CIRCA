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
    QTabWidget,
    QCheckBox,
    QFrame,
    QScrollArea
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
        color = palette["TEXT_DISABLED"] if not self.isEnabled() else palette["TEXT_SECONDARY"]
        svg_xml = ICONS_SVG.get(self._icon_name, "").format(color=color)
        if svg_xml:
            self._renderer.load(svg_xml.encode('utf-8'))
            self._renderer.render(painter, QRectF(self.rect()))



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
        self._value_label.setObjectName("PreprocessingValueLabel")
        self._value_label.setFont(QFont(FONT_MONO, FONT_SIZE_MONO_LIVE))
        header.addWidget(self._value_label)
        root.addLayout(header)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(min_val * self._scale))
        self._slider.setMaximum(int(max_val * self._scale))
        self._slider.setValue(int(default_val * self._scale))
        root.addWidget(self._slider)
        self._refresh_label()

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self._slider.setEnabled(enabled)
        self._value_label.setEnabled(enabled)
        self._name_label.setEnabled(enabled)
        self._icon.setEnabled(enabled)
        self.update()

    def value(self):
        return self._slider.value() / self._scale

    def set_value(self, val: float):
        self._slider.setValue(int(val * self._scale))
        self._refresh_label()

    def blockSignals(self, b: bool):
        self._slider.blockSignals(b)

    def _on_slider_changed(self, v):
        self._refresh_label()
        self.value_changed.emit(self.value())

    def _refresh_label(self):
        self._value_label.setText(f"{self.value():.{self._precision}f}{self._suffix}")



class SegmentedControl(QWidget):
    index_changed = pyqtSignal(int)

    def __init__(self, items: list[str], parent=None):
        super().__init__(parent)
        self.setObjectName("SegmentedControl")
        self._items = items
        self._buttons: list[QPushButton] = []
        self._active_index = 0

        # Background indicator widget
        self.indicator = QWidget(self)
        self.indicator.setObjectName("SegmentedIndicator")

        # Layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(0)

        for i, text in enumerate(items):
            btn = QPushButton(text, self)
            btn.setProperty("class", "SegmentedOption")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(self._make_callback(i))
            self.layout.addWidget(btn)
            self._buttons.append(btn)

        self.indicator.lower()  # Force background indicator behind buttons
        self._update_button_states()

        # Animation
        self._anim = QPropertyAnimation(self.indicator, b"geometry")
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _make_callback(self, index):
        return lambda: self.set_index(index)

    def set_index(self, index: int, animate: bool = True):
        if index == self._active_index:
            return
        self._active_index = index
        self._update_button_states()
        self.index_changed.emit(index)
        self._move_indicator(animate)

    def _update_button_states(self):
        for i, btn in enumerate(self._buttons):
            btn.setProperty("active", "true" if i == self._active_index else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._move_indicator(animate=False)

    def _move_indicator(self, animate: bool):
        if not self._buttons:
            return
        btn = self._buttons[self._active_index]
        target_rect = btn.geometry()
        # Adjust margins
        target_rect.adjust(2, 2, -2, -2)

        if animate:
            config = ThemeManager().get_anim_config()
            self._anim.stop()
            self._anim.setDuration(config["duration"])
            self._anim.setStartValue(self.indicator.geometry())
            self._anim.setEndValue(target_rect)
            self._anim.start()
        else:
            self.indicator.setGeometry(target_rect)


class SidePanel(QWidget):
    preprocessing_params_changed = pyqtSignal(object)
    inference_params_changed = pyqtSignal(object)
    camera_selected = pyqtSignal(int)
    theme_changed = pyqtSignal(object)
    panel_toggled = pyqtSignal(bool, str)  # expanded, active_key
    model_selected = pyqtSignal(str)
    recommend_model_requested = pyqtSignal()
    reset_onboarding_requested = pyqtSignal()

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

        # Header (retained for tests compatibility, hidden visually)
        self.title_label = QLabel()
        self.title_label.hide()

        # Custom Segmented Control to switch pages
        self.segmented_control = SegmentedControl(["OPTIMISE", "PREFERENCES"])
        self.root_layout.addWidget(self.segmented_control)

        self.stack = QTabWidget()
        self.stack.setDocumentMode(True)
        self.stack.tabBar().hide() # Hide default tab bar!

        # Optimisation Page
        opt_page = QWidget()
        ol = QVBoxLayout(opt_page)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(SPACING_MD)

        # 1. Vision Source Card
        source_card = QFrame()
        source_card.setObjectName("SidebarGroupCard")
        s_lay = QVBoxLayout(source_card)
        s_lay.setContentsMargins(12, 10, 12, 12)
        s_lay.setSpacing(8)
        s_lay.addWidget(self._section_header("VISION SOURCE"))
        self.camera_combo = QComboBox()
        s_lay.addWidget(self.camera_combo)
        ol.addWidget(source_card)

        # 2. Image Optimization Card
        filter_card = QFrame()
        filter_card.setObjectName("SidebarGroupCard")
        f_lay = QVBoxLayout(filter_card)
        f_lay.setContentsMargins(12, 10, 12, 12)
        f_lay.setSpacing(8)
        f_lay.addWidget(self._section_header("IMAGE OPTIMISATION"))
        self.auto_opt_check = QCheckBox("Auto-Optimise Parameters")
        self.auto_opt_check.setChecked(True)
        self.denoise_check = QCheckBox("Reduce Camera Noise")
        self.denoise_check.setChecked(True)
        self.clahe_slider = PreprocessingSlider("contrast", "Contrast", 1.0, 8.0, 2.0)
        self.gamma_slider = PreprocessingSlider("sun", "Brightness", 0.5, 2.5, 1.0)
        self.clahe_slider.setEnabled(False)
        self.gamma_slider.setEnabled(False)
        
        f_lay.addWidget(self.auto_opt_check)
        f_lay.addWidget(self.denoise_check)
        f_lay.addWidget(self.clahe_slider)
        f_lay.addWidget(self.gamma_slider)
        ol.addWidget(filter_card)
        
        # 3. Model Engine Card
        model_card = QFrame()
        model_card.setObjectName("SidebarGroupCard")
        m_lay = QVBoxLayout(model_card)
        m_lay.setContentsMargins(12, 10, 12, 12)
        m_lay.setSpacing(8)
        m_lay.addWidget(self._section_header("AI INTELLIGENCE"))
        self.confidence_slider = PreprocessingSlider("brain", "Sensitivity", 10.0, 95.0, 50.0, 0, "%")
        m_lay.addWidget(self.confidence_slider)
        
        m_lay.addWidget(self._section_header("MODEL SELECTION"))
        self.model_combo = QComboBox()
        self._populate_models()
        m_lay.addWidget(self.model_combo)
        
        self.recommend_model_btn = QPushButton("Auto-Detect Best Model")
        self.recommend_model_btn.setObjectName("PrimaryButton")
        self.recommend_model_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        m_lay.addWidget(self.recommend_model_btn)
        ol.addWidget(model_card)
        
        ol.addStretch(1)

        # Preferences Page
        prefs_page = QWidget()
        pl = QVBoxLayout(prefs_page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(SPACING_MD)

        pref_card = QFrame()
        pref_card.setObjectName("SidebarGroupCard")
        p_lay = QVBoxLayout(pref_card)
        p_lay.setContentsMargins(12, 10, 12, 12)
        p_lay.setSpacing(8)
        p_lay.addWidget(self._section_header("THEME"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Mode", "Light Mode", "System Sync"])
        p_lay.addWidget(self.theme_combo)
        p_lay.addWidget(self._section_header("UI DYNAMICS"))
        self.anim_combo = QComboBox()
        self.anim_combo.addItems(["Fluid Physics", "Snappy"])
        p_lay.addWidget(self.anim_combo)
        p_lay.addWidget(self._section_header("SYSTEM & HELP"))
        self.reset_onboarding_btn = QPushButton("Reset Tour")
        self.reset_onboarding_btn.setObjectName("ResetOnboardingButton")
        self.reset_onboarding_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        p_lay.addWidget(self.reset_onboarding_btn)
        pl.addWidget(pref_card)

        pl.addStretch(1)

        # Glossary Page
        glossary_page = QWidget()
        gl = QVBoxLayout(glossary_page)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(SPACING_MD)

        glossary_card = QFrame()
        glossary_card.setObjectName("SidebarGroupCard")
        g_lay = QVBoxLayout(glossary_card)
        g_lay.setContentsMargins(12, 10, 12, 12)
        g_lay.setSpacing(10)
        g_lay.addWidget(self._section_header("DEFECT GLOSSARY"))

        # Scroll area for defect list
        g_scroll = QScrollArea()
        g_scroll.setWidgetResizable(True)
        g_scroll.setStyleSheet("background: transparent; border: none;")
        
        g_scroll_content = QWidget()
        g_scroll_lay = QVBoxLayout(g_scroll_content)
        g_scroll_lay.setContentsMargins(0, 0, 0, 0)
        g_scroll_lay.setSpacing(12)

        palette = ThemeManager().get_palette()
        defects_info = [
            ("Missing Hole", "A required drill hole is absent. Typically a drilling toolpath or process fault."),
            ("Mouse Bite", "An edge erosion on a copper trace. Decreases trace cross-section and changes impedance."),
            ("Open Circuit", "A physical break/cut in a copper trace that completely cuts off electrical signal flow."),
            ("Short Circuit", "An accidental solder/copper bridge between traces, causing current leakage or short."),
            ("Excess Solder", "Too much solder deposited, risking solder bridges and masking structural faults."),
            ("Insufficient Solder", "Too little solder deposited, causing mechanically weak joints prone to cracking."),
            ("Cold Solder", "Granular, dull joint due to inadequate melting or quick cooling; poor electrical connection.")
        ]

        for title, desc in defects_info:
            item_frame = QFrame()
            item_frame.setStyleSheet(f"border-bottom: 1px solid {palette['BORDER']}44; padding-bottom: 6px;")
            if title == defects_info[-1][0]:
                item_frame.setStyleSheet("border: none; padding-bottom: 0px;")
            item_lay = QVBoxLayout(item_frame)
            item_lay.setContentsMargins(0, 0, 0, 0)
            item_lay.setSpacing(2)
            
            title_lbl = QLabel(title)
            title_lbl.setFont(QFont(FONT_UI, 10, QFont.Weight.Bold))
            title_lbl.setStyleSheet(f"color: {COLOR_ACCENT_CYAN};")
            
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setFont(QFont(FONT_UI, 9))
            desc_lbl.setProperty("secondary", "true")
            
            item_lay.addWidget(title_lbl)
            item_lay.addWidget(desc_lbl)
            g_scroll_lay.addWidget(item_frame)

        g_scroll_lay.addStretch(1)
        g_scroll.setWidget(g_scroll_content)
        g_lay.addWidget(g_scroll)
        gl.addWidget(glossary_card)

        self.stack.addTab(opt_page, "Inference Options")
        self.stack.addTab(prefs_page, "Preferences")
        self.stack.addTab(glossary_page, "Glossary")
        self.root_layout.addWidget(self.stack)

    def _setup_animations(self):
        self._anim = QPropertyAnimation(self, b"maximumWidth")
        self._anim.setEasingCurve(QEasingCurve.Type.OutExpo)

    def _wire_signals(self):
        self.segmented_control.index_changed.connect(self.stack.setCurrentIndex)
        self.clahe_slider.value_changed.connect(self._on_slider_manual_changed)
        self.gamma_slider.value_changed.connect(self._on_slider_manual_changed)
        self.confidence_slider.value_changed.connect(self._emit_inference_params)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.anim_combo.currentIndexChanged.connect(self._on_anim_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.auto_opt_check.toggled.connect(self._on_auto_opt_toggled)
        self.denoise_check.toggled.connect(self._emit_preprocessing_params)
        self.recommend_model_btn.clicked.connect(self._on_recommend_model_clicked)
        self.reset_onboarding_btn.clicked.connect(self.resetOnboardingTour)

    def resetOnboardingTour(self):
        from PyQt6.QtCore import QSettings
        settings = QSettings("CIRCA", "VisionStudio")
        settings.setValue("hasRunOnboarding", False)
        self.reset_onboarding_requested.emit()

    def set_tab(self, key: str):
        if key == "optimization":
            idx = 0
        elif key == "preferences" or key == "prefs":
            idx = 1
        elif key == "glossary":
            idx = 2
        else:
            idx = 0

        if self._expanded and self.stack.currentIndex() == idx:
            self.toggle()
        else:
            self.stack.setCurrentIndex(idx)
            if idx <= 1:
                self.segmented_control.set_index(idx, animate=False)
                self.segmented_control.show()
            else:
                self.segmented_control.hide()
            self.title_label.setText(key.upper() if key != "prefs" else "SETTINGS")
            if not self._expanded:
                self.toggle()
        self.panel_toggled.emit(self._expanded, key)

    def show_panel(self):
        if not self._expanded:
            self.toggle()

    def hide_panel(self):
        if self._expanded:
            self.toggle()

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

    def _on_model_changed(self, idx):
        if idx >= 0:
            path = self.model_combo.itemData(idx)
            if path:
                self.model_selected.emit(path)

    def _populate_models(self):
        import os
        candidates = [
            ("YOLOv12-Nano (FP16 - Production)", ["models/yolov12_int8.xml"]),
            ("YOLOv12-Nano (Untuned FP16)", [
                "runs/detect/CIRCA_V12N_004_TRAIN_Phase4_Nano/weights/best_int8_openvino_model/best.xml",
                "runs/detect/CIRCA_V12N_004_TRAIN_Phase4_Nano/weights/best_openvino_model/best.xml"
            ]),
            ("YOLOv12-Nano (Tuned FP16)", [
                "runs/detect/CIRCA_V12N_007_TRAIN_copypaste_exhibition/weights/best_int8_openvino_model/best.xml",
                "runs/detect/CIRCA_V12N_007_TRAIN_copypaste_exhibition/weights/best_openvino_model/best.xml"
            ]),
            ("YOLOv12-Small (Untuned FP16)", [
                "runs/detect/CIRCA_V12S_005_TRAIN_Phase4_Small/weights/best_int8_openvino_model/best.xml",
                "runs/detect/CIRCA_V12S_005_TRAIN_Phase4_Small/weights/best_openvino_model/best.xml"
            ]),
            ("YOLOv12-Small (Tuned FP16)", [
                "runs/detect/CIRCA_V12S_008_TRAIN_copypaste_small/weights/best_int8_openvino_model/best.xml",
                "runs/detect/CIRCA_V12S_008_TRAIN_copypaste_small/weights/best_openvino_model/best.xml"
            ]),
            ("YOLOv12-Medium (Untuned FP16)", [
                "runs/detect/CIRCA_V12M_006_TRAIN_Phase4_Medium/weights/best_int8_openvino_model/best.xml",
                "runs/detect/CIRCA_V12M_006_TRAIN_Phase4_Medium/weights/best_openvino_model/best.xml"
            ]),
            ("YOLOv12-Medium (Tuned FP16)", [
                "runs/detect/CIRCA_V12M_009_TRAIN_copypaste_medium/weights/best_int8_openvino_model/best.xml",
                "runs/detect/CIRCA_V12M_009_TRAIN_copypaste_medium/weights/best_openvino_model/best.xml"
            ]),
        ]
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        
        for name, rel_paths in candidates:
            resolved_path = None
            for rel_path in rel_paths:
                abs_path = os.path.join(project_root, rel_path.replace("/", os.sep))
                if os.path.isfile(abs_path):
                    resolved_path = abs_path
                    break
            
            if resolved_path:
                self.model_combo.addItem(name, resolved_path)
            elif name == "YOLOv12-Nano (FP16 - Production)":
                # Fallback to general best XML if production path doesn't exist
                alt_path = os.path.join(project_root, "models", "yolov12_best.xml")
                if os.path.isfile(alt_path):
                    self.model_combo.addItem(name, alt_path)
                    
        self.model_combo.blockSignals(False)

    def _emit_preprocessing_params(self, _=0.0):
        self.preprocessing_params_changed.emit(PreprocessParams(
            clahe_clip_limit=self.clahe_slider.value(),
            gamma=self.gamma_slider.value(),
            blur_threshold=0.0,
            auto_optimize=self.auto_opt_check.isChecked(),
            denoise=self.denoise_check.isChecked(),
        ))

    def _on_slider_manual_changed(self, value):
        if self.auto_opt_check.isChecked():
            self.auto_opt_check.blockSignals(True)
            self.auto_opt_check.setChecked(False)
            self.auto_opt_check.blockSignals(False)
        self._emit_preprocessing_params()

    def _on_auto_opt_toggled(self, checked: bool):
        self.clahe_slider.setEnabled(not checked)
        self.gamma_slider.setEnabled(not checked)
        self._emit_preprocessing_params()

    def _on_recommend_model_clicked(self):
        self.recommend_model_requested.emit()

    @pyqtSlot(float, float)
    def update_auto_params_ui(self, clahe: float, gamma: float):
        self.clahe_slider.blockSignals(True)
        self.gamma_slider.blockSignals(True)
        self.clahe_slider.set_value(clahe)
        self.gamma_slider.set_value(gamma)
        self.clahe_slider.blockSignals(False)
        self.gamma_slider.blockSignals(False)

    def _emit_inference_params(self, _=0.0):
        self.inference_params_changed.emit(InferenceParams(
            confidence_threshold=(100.0 - self.confidence_slider.value()) / 100.0
        ))

    @staticmethod
    def _section_header(text):
        label = QLabel(text)
        label.setFont(QFont(FONT_UI, 9, QFont.Weight.Bold))
        label.setProperty("secondary", "true")
        return label