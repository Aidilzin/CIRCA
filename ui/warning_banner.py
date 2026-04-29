"""
ui/warning_banner.py
--------------------
WarningBanner — FR15 full-width amber advisory bar.
Updated for 2026 UX: Glassmorphic background and dynamic theming.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ui.theme import (
    COLOR_STATUS_WARN,
    FONT_SIZE_LABEL,
    FONT_UI,
    SPACING_MD,
    SPACING_SM,
    WARNING_BANNER_HEIGHT,
)

logger = logging.getLogger(__name__)

COOL_DOWN_FRAMES: int = 30

class WarningBanner(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dismissed_for_current_board: bool = False
        self._good_frame_count: int = 0

        self._build_ui()
        self.setVisible(False)
        self.setFixedHeight(WARNING_BANNER_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Low confidence warning banner")

    def _build_ui(self) -> None:
        # Glassmorphic amber background - slightly more opaque for readability
        self.setStyleSheet(
            "WarningBanner {"
            "  background-color: rgba(255, 193, 7, 60);"
            "  border-bottom: 1px solid rgba(255, 193, 7, 120);"
            "}"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(SPACING_MD, 0, SPACING_MD, 0)
        row.setSpacing(SPACING_SM)

        self._text_label = QLabel("⚠  Low Confidence — Please Verify Manually")
        self._text_label.setFont(QFont(FONT_UI, FONT_SIZE_LABEL, QFont.Weight.Bold))
        # High contrast amber/black depending on theme handled by p['TEXT_PRIMARY'] if we don't override
        # But amber is the 'standard' warning color. Let's use a darker amber for light mode if needed.
        self._text_label.setStyleSheet(f"color: #856404; background: transparent;") # Dark amber for readability
        row.addWidget(self._text_label, stretch=1)

        self._dismiss_btn = QPushButton("×")
        self._dismiss_btn.setFixedSize(24, 24)
        self._dismiss_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dismiss_btn.setObjectName("WarningDismissBtn")
        self._dismiss_btn.setStyleSheet(
            "QPushButton { color: #856404; background: transparent; border: none; font-size: 20px; font-weight: bold; }"
            "QPushButton:hover { color: #000000; }"
        )
        self._dismiss_btn.clicked.connect(self._on_dismiss_clicked)
        row.addWidget(self._dismiss_btn)

    @pyqtSlot(float, float)
    def update_confidence(self, avg_confidence: float, threshold: float) -> None:
        if avg_confidence < threshold:
            self._good_frame_count = 0
            if not self._dismissed_for_current_board:
                self.setVisible(True)
        else:
            self._good_frame_count += 1
            if self._good_frame_count >= COOL_DOWN_FRAMES:
                self._good_frame_count = 0
                self._dismissed_for_current_board = False
                self.setVisible(False)

    @pyqtSlot()
    def reset_board_state(self) -> None:
        self._dismissed_for_current_board = False
        self._good_frame_count = 0

    def _on_dismiss_clicked(self) -> None:
        self._dismissed_for_current_board = True
        self.setVisible(False)