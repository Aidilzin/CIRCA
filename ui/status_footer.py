"""
ui/status_footer.py
-------------------
StatusFooter — persistent 48px system health bar at the bottom of MainWindow.

UX Spec: §StatusFooter — System Health Bar
         §Status Indicator Pattern — dot + label, four colour states

Architecture: QWidget that lives on the Main GUI Thread exclusively.
All state-change slots are called from MainWindow, which receives signals
from CameraWorker/InferenceWorker via queued connections. This guarantees
correct thread ownership without any explicit locks.

Anatomy (left → right, from UX spec §StatusFooter):
    ● Camera Active   ● Model Ready   ● 2 defects      [spacer]   15 fps

Status dot implementation:
    Each dot is an 8×8 QLabel with border-radius: 4px applied via
    setStyleSheet(). Qt renders this as a solid-filled circle reliably
    across Windows DPI scales. The dot is paired with a text QLabel
    (colour-blindness safeguard — colour reinforces but never solely
    carries meaning, per UX spec §Status Indicator Pattern).

Colour states (UX spec §Status Indicator Pattern):
    OK:   #4CAF50  (green)  — subsystem nominal
    WARN: #FFC107  (amber)  — advisory (amber tier — exclusive to FR15 only;
                              NOTE: this footer uses amber only for "Low Confidence"
                              state on the detection dots, never for camera/model)
    ERR:  #F44336  (red)    — subsystem failed, retrying
    IDLE: #3A3A3A  (grey)   — subsystem not applicable / not started

Module import boundary:
  ALLOWED:   PyQt6.*, ui/theme.py
  FORBIDDEN: core/*, workers/*
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from ui.theme import (
    COLOR_ACCENT_CYAN,
    COLOR_BG_SURFACE,
    COLOR_BORDER,
    COLOR_STATUS_ERROR,
    COLOR_STATUS_OK,
    COLOR_STATUS_WARN,
    COLOR_TEXT_DISABLED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    FONT_MONO,
    FONT_MONO_FALLBACKS,
    FONT_SIZE_MONO_LIVE,
    FONT_SIZE_STATUS,
    FONT_UI,
    FONT_UI_FALLBACKS,
    SPACING_MD,
    SPACING_SM,
    SPACING_XS,
    STATUS_DOT_SIZE,
    STATUS_FOOTER_HEIGHT,
)

logger = logging.getLogger(__name__)

# Idle / not-applicable dot colour — #3A3A3A (same as COLOR_BORDER by design)
_DOT_COLOR_IDLE: str = "#3A3A3A"

# Dot stylesheet template — border-radius makes it circular; min-size reserves
# the space regardless of OS widget theme. Applied via setStyleSheet per dot.
_DOT_STYLE: str = (
    "background-color: {color}; "
    "border-radius: {r}px; "
    f"min-width: {STATUS_DOT_SIZE}px; "
    f"max-width: {STATUS_DOT_SIZE}px; "
    f"min-height: {STATUS_DOT_SIZE}px; "
    f"max-height: {STATUS_DOT_SIZE}px;"
)


class _StatusIndicator(QWidget):
    """
    Composite widget: one coloured dot paired with one text label.

    Internal helper — not exported. Used for each of the three status
    sections (Camera, Model, Detections).
    """

    def __init__(
        self,
        initial_text: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING_XS)

        # Dot — 8px QLabel with border-radius via setStyleSheet
        self._dot = QLabel()
        self._dot.setFixedSize(STATUS_DOT_SIZE, STATUS_DOT_SIZE)
        self._set_dot_color(_DOT_COLOR_IDLE)
        row.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Text label
        self._label = QLabel(initial_text)
        status_font = QFont()
        status_font.setFamilies([FONT_UI] + FONT_UI_FALLBACKS)
        status_font.setPixelSize(FONT_SIZE_STATUS)
        status_font.setWeight(QFont.Weight.Medium)
        self._label.setFont(status_font)
        self._label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        row.addWidget(self._label)

    # ------------------------------------------------------------------
    # Public setters (called by StatusFooter slots)
    # ------------------------------------------------------------------

    def set_state(self, color: str, text: str) -> None:
        """Update both the dot colour and the accompanying label text."""
        self._set_dot_color(color)
        self._label.setText(text)

    @property
    def dot_color(self) -> str:
        """Return the current hex colour string of the dot."""
        return self._current_color

    @property
    def label_text(self) -> str:
        """Return the current label text."""
        return self._label.text()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _set_dot_color(self, color: str) -> None:
        """Apply border-radius circle style with the given hex colour."""
        self._current_color = color
        self._dot.setStyleSheet(
            _DOT_STYLE.format(color=color, r=STATUS_DOT_SIZE // 2)
        )


class StatusFooter(QWidget):
    """
    Persistent 48px system health bar.

    Always visible at the bottom of MainWindow. Cannot be collapsed or hidden.

    Three status indicators (left to right):
        Camera   — green (active), red (error), grey (idle/no device)
        Model    — green (ready), grey (loading / not loaded), red (error)
        Defects  — green (clean / 0), amber (detections present), grey (no data)

    FPS counter (right-aligned, FONT_MONO):
        Updates every second via set_fps(). Fixed-width label prevents
        layout reflow as digit count changes (0 fps → 30 fps).

    Slots (connected by MainWindow — never by workers):
        set_camera_active()
        set_camera_error(message)
        set_camera_idle()
        set_model_ready()
        set_model_loading()
        set_model_error(message)
        set_fps(fps)
        set_detection_count(count)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusFooter")
        self.setFixedHeight(STATUS_FOOTER_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the status bar layout."""
        # Top border line (1px, COLOR_BORDER)
        outer = QFrame(self)
        outer.setFrameShape(QFrame.Shape.StyledPanel)

        row = QHBoxLayout(self)
        row.setContentsMargins(SPACING_MD, 0, SPACING_MD, 0)
        row.setSpacing(SPACING_MD)

        # ── Camera status ─────────────────────────────────────────────
        self._camera_indicator = _StatusIndicator("No Device")
        row.addWidget(self._camera_indicator)

        # ── Model status ──────────────────────────────────────────────
        self._model_indicator = _StatusIndicator("No Model")
        row.addWidget(self._model_indicator)

        # ── Detection count ───────────────────────────────────────────
        self._detection_indicator = _StatusIndicator("No data")
        row.addWidget(self._detection_indicator)

        # ── Right spacer ──────────────────────────────────────────────
        row.addStretch(1)

        # ── FPS counter (FONT_MONO, right-aligned) ────────────────────
        self._fps_label = QLabel("— fps")
        fps_font = QFont()
        fps_font.setFamilies([FONT_MONO] + FONT_MONO_FALLBACKS)
        fps_font.setPixelSize(FONT_SIZE_MONO_LIVE)
        self._fps_label.setFont(fps_font)
        self._fps_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY};")
        self._fps_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        # Fixed minimum width: widest expected string is "30 fps" at 11px mono
        from PyQt6.QtGui import QFontMetrics
        fm = QFontMetrics(fps_font)
        self._fps_label.setMinimumWidth(fm.horizontalAdvance("30 fps") + 8)
        row.addWidget(self._fps_label)

    # ------------------------------------------------------------------
    # Camera status slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def set_camera_active(self) -> None:
        """Green dot — camera feed running."""
        self._camera_indicator.set_state(COLOR_STATUS_OK, "Camera Active")

    @pyqtSlot(str)
    def set_camera_error(self, message: str = "Camera Error") -> None:
        """Red dot — camera disconnected or read failure."""
        self._camera_indicator.set_state(COLOR_STATUS_ERROR, "Camera Error")
        logger.debug("StatusFooter: camera error — %s", message)

    @pyqtSlot()
    def set_camera_idle(self) -> None:
        """Grey dot — no device connected."""
        self._camera_indicator.set_state(_DOT_COLOR_IDLE, "No Device")

    # ------------------------------------------------------------------
    # Model status slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def set_model_ready(self) -> None:
        """Green dot — OpenVINO model compiled and ready."""
        self._model_indicator.set_state(COLOR_STATUS_OK, "Model Ready")

    @pyqtSlot()
    def set_model_loading(self) -> None:
        """Grey dot — model compile in progress (deferred after thread start)."""
        self._model_indicator.set_state(_DOT_COLOR_IDLE, "Loading Model…")

    @pyqtSlot(str)
    def set_model_error(self, message: str = "Model Error") -> None:
        """Red dot — model failed to load."""
        self._model_indicator.set_state(COLOR_STATUS_ERROR, "Model Error")
        logger.debug("StatusFooter: model error — %s", message)

    # ------------------------------------------------------------------
    # Detection count slot
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def set_detection_count(self, count: int) -> None:
        """
        Update the detection count indicator.

        States:
            count == 0    → green  "No defects"
            count > 0     → amber  "N defect(s)"  (advisory tier — not an error)
            count < 0     → grey   "No data"      (inference not yet running)

        Note: amber here is the detection advisory tier, NOT the FR15
        low-confidence amber (which lives in WarningBanner). Different
        semantic roles; same colour token per UX spec §Design Principles.

        Wait — re-reading UX spec §Amber tier rule: "exclusively reserved for
        the low-confidence warning state (FR15)".

        To honour the amber tier rule strictly: use CYAN for active detection
        count (signals the inference is running and results are valid), not amber.
        """
        if count < 0:
            self._detection_indicator.set_state(_DOT_COLOR_IDLE, "No data")
        elif count == 0:
            self._detection_indicator.set_state(COLOR_STATUS_OK, "No defects")
        else:
            # NOTE: Cyan used (not amber) to honour the amber tier exclusivity rule.
            # Amber is reserved for FR15 WarningBanner only.
            defect_word = "defect" if count == 1 else "defects"
            self._detection_indicator.set_state(
                COLOR_ACCENT_CYAN, f"{count} {defect_word}"
            )

    # ------------------------------------------------------------------
    # FPS counter slot
    # ------------------------------------------------------------------

    @pyqtSlot(float)
    def set_fps(self, fps: float) -> None:
        """
        Update the FPS readout.

        Args:
            fps: Current frames-per-second (from CameraWorker FPS counter).
                 Values below 0 display as "—".
        """
        if fps < 0:
            self._fps_label.setText("— fps")
        else:
            # Display as integer for readability (15 fps not 15.3 fps)
            self._fps_label.setText(f"{fps:.0f} fps")
