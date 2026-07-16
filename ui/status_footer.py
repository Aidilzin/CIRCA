"""
ui/status_footer.py
-------------------
StatusFooter — persistent 48px system health bar.
Updated for 2026 UX: Dynamic theming via QSS.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from ui.theme import (
    COLOR_ACCENT_CYAN,
    COLOR_STATUS_ERROR,
    COLOR_STATUS_OK,
    FONT_MONO,
    FONT_SIZE_MONO_LIVE,
    FONT_SIZE_STATUS,
    FONT_UI,
    SPACING_MD,
    SPACING_XS,
    STATUS_DOT_SIZE,
    STATUS_FOOTER_HEIGHT,
)

logger = logging.getLogger(__name__)

# Public constant: the idle/off-state dot colour used by all three indicators.
# Exported so tests can import it instead of hardcoding the hex string.
DOT_COLOR_IDLE: str = "#3A3A3A"

class _StatusIndicator(QWidget):
    def __init__(self, initial_text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusIndicator")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 2, 10, 2)
        row.setSpacing(6)

        self._dot = QLabel()
        self._dot.setFixedSize(STATUS_DOT_SIZE, STATUS_DOT_SIZE)
        row.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._label = QLabel(initial_text)
        self._label.setFont(QFont(FONT_UI, FONT_SIZE_STATUS, QFont.Weight.Medium))
        row.addWidget(self._label)

        self.dot_color = DOT_COLOR_IDLE
        self.label_text = initial_text
        self.set_state(self.dot_color, self.label_text)

    def set_state(self, color: str, text: str) -> None:
        self.dot_color = color
        self.label_text = text
        self._dot.setStyleSheet(f"background-color: {color}; border-radius: {STATUS_DOT_SIZE // 2}px;")
        self._label.setText(text)

class StatusFooter(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusFooter")
        self.setFixedHeight(STATUS_FOOTER_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui()

    def _build_ui(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(SPACING_MD, 0, SPACING_MD, 0)
        row.setSpacing(SPACING_MD)

        self._camera_indicator = _StatusIndicator("No Camera")
        self._model_indicator = _StatusIndicator("No Model")
        self._detection_indicator = _StatusIndicator("No data")

        row.addWidget(self._camera_indicator)
        row.addWidget(self._model_indicator)
        row.addWidget(self._detection_indicator)
        row.addStretch(1)

    @pyqtSlot()
    def set_camera_active(self): self._camera_indicator.set_state(COLOR_STATUS_OK, "Camera Connected")
    @pyqtSlot(str)
    def set_camera_error(self, msg="Camera Error"): self._camera_indicator.set_state(COLOR_STATUS_ERROR, msg)
    @pyqtSlot()
    def set_camera_idle(self): self._camera_indicator.set_state(DOT_COLOR_IDLE, "No Camera")
    @pyqtSlot()
    def set_model_ready(self): self._model_indicator.set_state(COLOR_STATUS_OK, "Model Ready")
    @pyqtSlot()
    def set_model_loading(self): self._model_indicator.set_state(DOT_COLOR_IDLE, "Loading Model…")
    @pyqtSlot(str)
    def set_model_error(self, msg="Model Error"): self._model_indicator.set_state(COLOR_STATUS_ERROR, msg)

    @pyqtSlot(str)
    def set_status_text(self, msg: str) -> None:
        self._model_indicator.set_state(COLOR_ACCENT_CYAN, msg)

    @pyqtSlot(int)
    def set_detection_count(self, count: int) -> None:
        if count < 0: self._detection_indicator.set_state(DOT_COLOR_IDLE, "No data")
        elif count == 0: self._detection_indicator.set_state(COLOR_STATUS_OK, "No defects")
        else: self._detection_indicator.set_state(COLOR_ACCENT_CYAN, f"{count} {'defect' if count==1 else 'defects'}")