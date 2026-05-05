"""
ui/video_widget.py
------------------
VideoWidget — the central live-feed viewport with QPainter bounding box overlay.
Updated for 2026 UX: Theme-aware rendering and modern status panels.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import QWidget

from core.models import DetectionResult
from ui.theme import (
    BBOX_INNER_WIDTH,
    BBOX_OUTER_ALPHA,
    BBOX_OUTER_WIDTH,
    CHIP_PADDING_H,
    CHIP_PADDING_V,
    DEFECT_CHIP_ALPHA,
    DEFECT_CHIP_TEXT_COLOR,
    DEFECT_CLASS_COLOR_FALLBACK,
    DEFECT_CLASS_COLORS,
    FONT_MONO,
    FONT_MONO_FALLBACKS,
    FONT_SIZE_MONO_CHIP,
    FONT_SIZE_LABEL,
    FONT_UI,
    FONT_UI_FALLBACKS,
    ThemeManager
)
from PyQt6.QtCore import QRect

logger = logging.getLogger(__name__)

class VideoWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_frame: Optional[QImage] = None
        self._detections: Optional[DetectionResult] = None
        self._status_text: str = "Please connect a camera"
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumSize(320, 240)

    @pyqtSlot(QImage)
    def set_frame(self, image: QImage) -> None:
        self._current_frame = image
        self.update()

    @pyqtSlot(object)
    def set_detections(self, result: object) -> None:
        if isinstance(result, DetectionResult):
            self._detections = result
        self.update()

    @pyqtSlot(str)
    def set_status_text(self, text: str) -> None:
        self._status_text = text
        if self._current_frame is None:
            self.update()

    def clear_feed(self, status_text: str = "") -> None:
        if status_text:
            self._status_text = status_text
        self._current_frame = None
        self._detections = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt C++ API override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        try:
            palette = ThemeManager().get_palette()
            painter.fillRect(self.rect(), QColor(palette["BG_BASE"]))

            if self._current_frame is None or self._current_frame.isNull():
                self._draw_status_label(painter, palette)
                return

            letterbox = self._compute_letterbox_rect(
                self._current_frame.width(),
                self._current_frame.height(),
            )
            painter.drawImage(letterbox, self._current_frame)

            if self._detections is not None and self._detections.boxes:
                self._draw_detections(painter, letterbox)

        finally:
            painter.end()

    def _compute_letterbox_rect(self, frame_w: int, frame_h: int) -> QRect:
        widget_w, widget_h = self.width(), self.height()
        if frame_w == 0 or frame_h == 0:
            return QRect(0, 0, widget_w, widget_h)
        scale = min(widget_w / frame_w, widget_h / frame_h)
        drawn_w, drawn_h = int(frame_w * scale), int(frame_h * scale)
        return QRect((widget_w - drawn_w) // 2, (widget_h - drawn_h) // 2, drawn_w, drawn_h)

    def _draw_detections(self, painter: QPainter, letterbox: QRect) -> None:
        frame_w, frame_h = self._current_frame.width(), self._current_frame.height()
        if frame_w == 0 or frame_h == 0:
            return
        scale_x, scale_y = letterbox.width() / frame_w, letterbox.height() / frame_h
        for box in self._detections.boxes:
            self._draw_single_box(painter, box, letterbox, scale_x, scale_y)

    def _draw_single_box(self, painter, box, letterbox, scale_x, scale_y):
        wx = letterbox.x() + int(round(box.x * scale_x))
        wy = letterbox.y() + int(round(box.y * scale_y))
        ww, wh = max(2, int(round(box.width * scale_x))), max(2, int(round(box.height * scale_y)))

        hex_color = DEFECT_CLASS_COLORS.get(box.class_name, DEFECT_CLASS_COLOR_FALLBACK)
        class_color = QColor(hex_color)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, BBOX_OUTER_ALPHA), BBOX_OUTER_WIDTH, Qt.PenJoinStyle.MiterJoin))
        painter.drawRect(wx, wy, ww, wh)
        painter.setPen(QPen(class_color, BBOX_INNER_WIDTH, Qt.PenJoinStyle.MiterJoin))
        painter.drawRect(wx, wy, ww, wh)
        self._draw_label_chip(painter, wx, wy, box, class_color)

    def _draw_label_chip(self, painter, box_x, box_y, box, class_color):
        label = box.class_name.replace("_", " ").upper()
        chip_text = f"{label}  {box.confidence * 100:.0f}%"
        chip_font = QFont(FONT_MONO, FONT_SIZE_MONO_CHIP, QFont.Weight.Medium)
        metrics = QFontMetrics(chip_font)
        chip_w, chip_h = metrics.horizontalAdvance(chip_text) + CHIP_PADDING_H * 2, metrics.height() + CHIP_PADDING_V * 2
        chip_x, chip_y = box_x, box_y - chip_h
        if chip_y < 0: chip_y = box_y

        chip_bg = QColor(class_color)
        chip_bg.setAlpha(DEFECT_CHIP_ALPHA)
        painter.fillRect(chip_x, chip_y, chip_w, chip_h, chip_bg)
        painter.setFont(chip_font)
        painter.setPen(QColor(DEFECT_CHIP_TEXT_COLOR))
        painter.drawText(chip_x + CHIP_PADDING_H, chip_y + CHIP_PADDING_V + metrics.ascent(), chip_text)

    def _draw_status_label(self, painter: QPainter, palette: dict) -> None:
        if not self._status_text:
            return
        self._draw_status_panel(painter, palette)

    def _draw_status_panel(self, painter: QPainter, palette: dict) -> None:
        """Render a centered, rounded-rect panel with the current status text."""
        pw, ph = 300, 72
        px = (self.width() - pw) // 2
        py = (self.height() - ph) // 2
        panel_rect = QRect(px, py, pw, ph)

        painter.setBrush(QBrush(QColor(palette["BG_SURFACE"])))
        painter.setPen(QPen(QColor(palette["BORDER"]), 1))
        painter.drawRoundedRect(panel_rect, 12, 12)

        painter.setFont(QFont(FONT_UI, FONT_SIZE_LABEL, QFont.Weight.Bold))
        painter.setPen(QColor(palette["TEXT_PRIMARY"]))
        painter.drawText(panel_rect, Qt.AlignmentFlag.AlignCenter, self._status_text)