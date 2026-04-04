"""
ui/control_panel.py
-------------------
ControlPanel — collapsible right-sidebar with four PreprocessingSlider controls
and a camera source selector.

UX Spec: §ControlPanel (collapsible, 280px ↔ 28px, no animation)
         §PreprocessingSlider (composite slider with FONT_MONO value label)
         §Slider Interaction Pattern (live update on every drag tick)

Architecture: QWidget subclasses — no workers, no core imports except models
              for typing. All signals wired in MainWindow, never here.

Module import boundary:
  ALLOWED:   PyQt6.*, ui/theme.py, core/models.py (dataclasses only)
  FORBIDDEN: core/preprocessor.py, core/inference_engine.py, workers/*, cv2
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.models import InferenceParams, PreprocessParams
from ui.theme import (
    COLOR_ACCENT_CYAN,
    COLOR_BG_SURFACE,
    COLOR_BORDER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    CONTROL_PANEL_WIDTH_COLLAPSED,
    CONTROL_PANEL_WIDTH_EXPANDED,
    FONT_MONO,
    FONT_MONO_FALLBACKS,
    FONT_SIZE_BODY,
    FONT_SIZE_LABEL,
    FONT_SIZE_MONO_LIVE,
    FONT_UI,
    FONT_UI_FALLBACKS,
    SPACING_SM,
    SPACING_XS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PreprocessingSlider
# ---------------------------------------------------------------------------


class PreprocessingSlider(QWidget):
    """
    Composite slider widget: name label + FONT_MONO value readout + QSlider.

    Anatomy (from UX spec §PreprocessingSlider):
        [Slider Name]          [Value]   ← Inter 500 12px + JetBrains Mono 12px cyan
        [Shop-floor sub-label]           ← Inter 400 11px COLOR_TEXT_SECONDARY
        [━━━━━━●━━━━━━━━━━━━━]          ← QSlider horizontal

    The value label has a FIXED MINIMUM WIDTH calculated from the widest
    possible value string at FONT_MONO metrics. This prevents the slider row
    from reflowing as digit widths vary at ≥15 FPS update rate (UX spec rule).

    Float ↔ int conversion:
        QSlider works with integers internally. The `_scale` factor converts:
            float_value = slider_int / _scale
            slider_int  = round(float_value * _scale)
        For 1-decimal precision (CLAHE, Gamma): _scale = 10
        For 0-decimal precision (Blur, Confidence): _scale = 1

    Args:
        name:        Display name — e.g. "Glare Control"
        sublabel:    Shop-floor description — e.g. "CLAHE clip limit"
        min_val:     Minimum float value
        max_val:     Maximum float value
        default_val: Initial float value (must be within [min_val, max_val])
        precision:   Decimal digits for display (0 or 1); determines _scale
        suffix:      Appended to value string — e.g. "%" for confidence
        parent:      Qt parent widget
    """

    # Emits the current float value on every drag tick (valueChanged from QSlider)
    value_changed: pyqtSignal = pyqtSignal(float)

    def __init__(
        self,
        name: str,
        sublabel: str,
        min_val: float,
        max_val: float,
        default_val: float,
        precision: int = 1,
        suffix: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._precision = precision
        self._scale = 10**precision
        self._suffix = suffix
        self._min_val = min_val
        self._max_val = max_val

        self._build_ui(name, sublabel, min_val, max_val, default_val)
        self._wire_signals()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(
        self,
        name: str,
        sublabel: str,
        min_val: float,
        max_val: float,
        default_val: float,
    ) -> None:
        """Build and lay out all child widgets."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, SPACING_SM)
        root.setSpacing(SPACING_XS)

        # ── Row 1: name (left) + value readout (right) ──────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(SPACING_XS)

        self._name_label = QLabel(name)
        name_font = QFont()
        name_font.setFamilies([FONT_UI] + FONT_UI_FALLBACKS)
        name_font.setPixelSize(FONT_SIZE_LABEL)
        name_font.setWeight(QFont.Weight.Medium)
        self._name_label.setFont(name_font)
        self._name_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")

        # Value label — FONT_MONO mandatory (UX spec typography rule)
        self._value_label = QLabel()
        mono_font = QFont()
        mono_font.setFamilies([FONT_MONO] + FONT_MONO_FALLBACKS)
        mono_font.setPixelSize(FONT_SIZE_MONO_LIVE)
        self._value_label.setFont(mono_font)
        self._value_label.setStyleSheet(f"color: {COLOR_ACCENT_CYAN};")
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        # Fixed minimum width: prevents reflow as digit widths vary.
        # Computed from the widest possible value string at FONT_MONO metrics.
        max_text = self._format_value(max_val)
        fm = QFontMetrics(mono_font)
        self._value_label.setMinimumWidth(fm.horizontalAdvance(max_text) + 8)

        header_row.addWidget(self._name_label, stretch=1)
        header_row.addWidget(self._value_label)
        root.addLayout(header_row)

        # ── Row 2: shop-floor sub-label ──────────────────────────────
        self._sub_label = QLabel(sublabel)
        sub_font = QFont()
        sub_font.setFamilies([FONT_UI] + FONT_UI_FALLBACKS)
        sub_font.setPixelSize(FONT_SIZE_BODY)
        self._sub_label.setFont(sub_font)
        self._sub_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        root.addWidget(self._sub_label)

        # ── Row 3: QSlider ───────────────────────────────────────────
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(self._to_int(min_val))
        self._slider.setMaximum(self._to_int(max_val))
        self._slider.setValue(self._to_int(default_val))

        # Accessible name for QSS focus ring and screen readers (UX spec §Accessibility)
        self._slider.setAccessibleName(f"{self._name_label.text()} slider")

        root.addWidget(self._slider)

        # Initialise the value label text
        self._refresh_label()

    def _wire_signals(self) -> None:
        """Connect QSlider.valueChanged to internal handler."""
        self._slider.valueChanged.connect(self._on_slider_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def value(self) -> float:
        """Return the current float value."""
        return self._to_float(self._slider.value())

    def set_value(self, value: float) -> None:
        """Programmatically set the slider position (clamps to [min, max])."""
        clamped = max(self._min_val, min(self._max_val, value))
        self._slider.setValue(self._to_int(clamped))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_slider_changed(self, int_val: int) -> None:
        """Fired on every QSlider.valueChanged tick (every drag pixel)."""
        self._refresh_label()
        self.value_changed.emit(self._to_float(int_val))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_int(self, value: float) -> int:
        """Convert float value → integer slider position."""
        return int(round(value * self._scale))

    def _to_float(self, int_val: int) -> float:
        """Convert integer slider position → float value."""
        return int_val / self._scale

    def _format_value(self, value: float) -> str:
        """Format a float value for display, applying precision and suffix."""
        return f"{value:.{self._precision}f}{self._suffix}"

    def _refresh_label(self) -> None:
        """Sync the value label text with the current slider position."""
        self._value_label.setText(self._format_value(self.value()))


# ---------------------------------------------------------------------------
# ControlPanel
# ---------------------------------------------------------------------------


class ControlPanel(QWidget):
    """
    Collapsible right-sidebar housing all camera and preprocessing controls.

    States (UX spec §ControlPanel):
        EXPANDED  (280px) — All controls visible; default launch state.
        COLLAPSED (28px)  — Toggle arrow `›` only; viewport takes remaining space.

    Collapse transition: `setFixedWidth()` + `_content.setVisible()`.
    No animation — snaps for zero-latency feel (UX spec §Interaction).

    Signals:
        preprocessing_params_changed(object: PreprocessParams)
            — Emitted on every CLAHE, Gamma, or Blur slider tick.
              MainWindow connects this → CameraWorker.update_params().

        inference_params_changed(object: InferenceParams)
            — Emitted on every Confidence slider tick.
              MainWindow connects this → InferenceWorker.update_params().

        camera_selected(int)
            — Emitted when the user changes the camera dropdown selection.
              Carries the UVC device index. MainWindow connects this to restart
              CameraWorker with the selected index.
    """

    preprocessing_params_changed: pyqtSignal = pyqtSignal(object)
    inference_params_changed: pyqtSignal = pyqtSignal(object)
    camera_selected: pyqtSignal = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._expanded: bool = True

        self.setObjectName("ControlPanel")
        self.setFixedWidth(CONTROL_PANEL_WIDTH_EXPANDED)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )

        self._build_ui()
        self._wire_signals()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the full panel layout.

        Layout (QHBoxLayout, margin=0, spacing=0):
            [toggle_btn 28px] | [_content — fills remainder]

        The toggle button is ALWAYS visible at the left edge. It never
        moves to the top because it lives in a horizontal layout, not a
        vertical one. When collapsed, _content is hidden and the panel's
        total fixed width shrinks to 28px (toggle button only).
        """
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toggle button (permanently left-edge, full height) ───────
        # Uses ASCII < > (not ‹ ›) for maximum cross-platform glyph weight.
        # font-weight: 900 (Black) makes the arrow visually thick at 16px.
        # border: none removes the default QPushButton frame so the button
        # blends with the panel edge without a visual box.
        self._toggle_btn = QPushButton("<")
        self._toggle_btn.setFixedWidth(CONTROL_PANEL_WIDTH_COLLAPSED)
        self._toggle_btn.setStyleSheet(
            f"color: {COLOR_ACCENT_CYAN};"
            f" background: {COLOR_BG_SURFACE};"
            f" font-weight: 900;"
            f" font-size: 16px;"
            f" border: none;"
        )
        self._toggle_btn.setAccessibleName("Collapse control panel")
        self._toggle_btn.setToolTip("Collapse / expand control panel")
        root.addWidget(self._toggle_btn)

        # ── Content widget (hidden when collapsed) ────────────────────
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(SPACING_SM, 0, SPACING_SM, SPACING_SM)
        content_layout.setSpacing(SPACING_SM)

        # ── Camera section ────────────────────────────────────────────
        content_layout.addWidget(self._section_header("CAMERA SOURCE"))
        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("Camera source selector")
        self.camera_combo.setToolTip("Select USB camera device")

        # Initial state: disabled until first hardware scan completes.
        self.camera_combo.addItem("No cameras found", -1)
        self.camera_combo.setEnabled(False)

        content_layout.addWidget(self.camera_combo)
        content_layout.addWidget(self._divider())

        # ── Preprocessing section ─────────────────────────────────────
        content_layout.addWidget(self._section_header("PREPROCESSING"))

        self.clahe_slider = PreprocessingSlider(
            name="Glare Control",
            sublabel="CLAHE clip limit",
            min_val=1.0,
            max_val=8.0,
            default_val=2.0,
            precision=1,
        )
        self.clahe_slider.setToolTip("Glare Control — adjusts CLAHE clip limit")
        content_layout.addWidget(self.clahe_slider)

        self.gamma_slider = PreprocessingSlider(
            name="Shadow Lift",
            sublabel="Gamma correction",
            min_val=0.5,
            max_val=2.5,
            default_val=1.0,
            precision=1,
        )
        self.gamma_slider.setToolTip("Shadow Lift — adjusts gamma correction")
        content_layout.addWidget(self.gamma_slider)

        self.blur_slider = PreprocessingSlider(
            name="Motion Sensitivity",
            sublabel="Blur variance threshold",
            min_val=20.0,
            max_val=300.0,
            default_val=100.0,
            precision=0,
        )
        self.blur_slider.setToolTip(
            "Motion Sensitivity — adjusts blur variance threshold"
        )
        content_layout.addWidget(self.blur_slider)
        content_layout.addWidget(self._divider())

        # ── Inference section ─────────────────────────────────────────
        content_layout.addWidget(self._section_header("DETECTION"))

        self.confidence_slider = PreprocessingSlider(
            name="Min Confidence",
            sublabel="Detection threshold",
            min_val=10.0,
            max_val=95.0,
            default_val=50.0,
            precision=0,
            suffix="%",
        )
        self.confidence_slider.setToolTip(
            "Min Confidence — minimum confidence threshold for detection"
        )
        content_layout.addWidget(self.confidence_slider)
        content_layout.addStretch(1)  # Push controls to top

        root.addWidget(self._content)

    def _wire_signals(self) -> None:
        """Connect child widget signals to panel-level handlers."""
        self._toggle_btn.clicked.connect(self._on_toggle)

        self.clahe_slider.value_changed.connect(self._emit_preprocessing_params)
        self.gamma_slider.value_changed.connect(self._emit_preprocessing_params)
        self.blur_slider.value_changed.connect(self._emit_preprocessing_params)
        self.confidence_slider.value_changed.connect(self._emit_inference_params)

        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    @pyqtSlot(list)
    def populate_cameras(self, cameras: list) -> None:
        """
        Populate the camera dropdown from enumerate_cameras() output.

        Called by MainWindow after CameraWorker.enumerate_cameras() returns.
        Blocks signals during population to avoid spurious camera_selected
        emissions for each addItem() call.

        Args:
            cameras: List of (device_index: int, name: str) tuples from
                     core.utils.enumerate_cameras().
        """
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        if cameras:
            for device_index, name in cameras:
                self.camera_combo.addItem(name, device_index)
            self.camera_combo.setEnabled(True)
        else:
            self.camera_combo.addItem("No cameras found", -1)
            self.camera_combo.setEnabled(False)

        self.camera_combo.blockSignals(False)

        # Removed redundant camera_selected.emit(cameras[0][0])
        # MainWindow handles recovery/initialisation logic explicitly.

    @pyqtSlot(object)
    def apply_preprocessing_params(self, params: PreprocessParams) -> None:
        """
        Programmatically set all preprocessing slider positions from a
        PreprocessParams instance (e.g. for loading saved settings).

        Does NOT emit preprocessing_params_changed — caller drives this.
        """
        self.clahe_slider.set_value(params.clahe_clip_limit)
        self.gamma_slider.set_value(params.gamma)
        self.blur_slider.set_value(params.blur_threshold)

    # ------------------------------------------------------------------
    # Collapse / expand
    # ------------------------------------------------------------------

    def toggle(self) -> None:
        """Toggle panel between expanded and collapsed state."""
        self._on_toggle()

    @property
    def is_expanded(self) -> bool:
        """True if the panel is currently in expanded state."""
        return self._expanded

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        """Snap the panel to collapsed or expanded state (no animation)."""
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        if self._expanded:
            self.setFixedWidth(CONTROL_PANEL_WIDTH_EXPANDED)
            self._toggle_btn.setText("<")
            self._toggle_btn.setAccessibleName("Collapse control panel")
        else:
            self.setFixedWidth(CONTROL_PANEL_WIDTH_COLLAPSED)
            self._toggle_btn.setText(">")
            self._toggle_btn.setAccessibleName("Expand control panel")

    def _on_camera_changed(self, index: int) -> None:
        """Emit camera_selected with the device index stored in the combo item."""
        if index < 0:
            return
        device_index = self.camera_combo.itemData(index)
        if device_index is not None and device_index >= 0:
            self.camera_selected.emit(device_index)

    def _emit_preprocessing_params(self, _: float = 0.0) -> None:
        """Build and emit PreprocessParams from current slider values."""
        params = PreprocessParams(
            clahe_clip_limit=self.clahe_slider.value(),
            gamma=self.gamma_slider.value(),
            blur_threshold=self.blur_slider.value(),
        )
        self.preprocessing_params_changed.emit(params)

    def _emit_inference_params(self, _: float = 0.0) -> None:
        """Build and emit InferenceParams from confidence slider value.

        Confidence slider emits 10.0–95.0 (percentage display).
        InferenceParams.confidence_threshold expects 0.0–1.0.
        """
        params = InferenceParams(
            confidence_threshold=self.confidence_slider.value() / 100.0
        )
        self.inference_params_changed.emit(params)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_header(text: str) -> QLabel:
        """Create a styled uppercase section header label."""
        label = QLabel(text)
        font = QFont()
        font.setFamilies([FONT_UI] + FONT_UI_FALLBACKS)
        font.setPixelSize(10)
        label.setFont(font)
        label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; "
            f"padding-top: {SPACING_SM}px; "
            f"padding-bottom: {SPACING_XS}px;"
        )
        return label

    @staticmethod
    def _divider() -> QFrame:
        """Create a horizontal section divider line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(
            f"color: {COLOR_BORDER}; background-color: {COLOR_BORDER}; "
            f"border: none; max-height: 1px;"
        )
        return line
