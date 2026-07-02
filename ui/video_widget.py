"""
ui/video_widget.py
------------------
VideoWidget — the central live-feed viewport with QPainter bounding box overlay.
Updated for 2026 UX: Theme-aware rendering and modern status panels.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QRect, QRectF
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
    FONT_SIZE_BODY,
    FONT_UI,
    FONT_UI_FALLBACKS,
    ThemeManager
)
from PyQt6.QtCore import QRect, QRectF
from PyQt6.QtSvg import QSvgRenderer

logger = logging.getLogger(__name__)

class VideoWidget(QWidget):
    fps_updated: pyqtSignal = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_frame: Optional[QImage] = None
        self._detections: Optional[DetectionResult] = None
        self._status_text: str = "Please connect a camera"
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumSize(320, 240)

        # FPS tracking variables
        import time
        self._last_fps_time = time.perf_counter()
        self._frame_count_fps = 0
        self._current_fps = 0.0
        self._highlight_index = -1

    @pyqtSlot(QImage)
    def set_frame(self, image: QImage) -> None:
        self._current_frame = image

        # Calculate live FPS
        import time
        self._frame_count_fps += 1
        now = time.perf_counter()
        dt = now - self._last_fps_time
        if dt >= 1.0:
            self._current_fps = self._frame_count_fps / dt
            self.fps_updated.emit(self._current_fps)
            self._frame_count_fps = 0
            self._last_fps_time = now

        self.update()

    @pyqtSlot(int)
    def set_highlight_index(self, idx: int) -> None:
        self._highlight_index = idx
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

            # Draw Microscope HUD overlay inside the active video letterbox
            self._draw_hud(painter, letterbox, palette)

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

    def _draw_hud(self, painter: QPainter, rect: QRect, palette: dict) -> None:
        if rect.width() < 100 or rect.height() < 100:
            return

        # Top & Bottom Visor Line overlay
        painter.setPen(QPen(QColor(0, 210, 255, 40), 1))
        painter.drawLine(rect.left(), rect.top() + 32, rect.right(), rect.top() + 32)
        painter.drawLine(rect.left(), rect.bottom() - 32, rect.right(), rect.bottom() - 32)

        # 1. Top-left: pulsing indicator + "LIVE MICROSCOPE FEED"
        import time
        import math
        pulse_alpha = int(127 + 128 * abs(math.sin(time.perf_counter() * 3.0)))
        
        pulse_color = QColor(244, 67, 54) # Red
        pulse_color.setAlpha(pulse_alpha)
        painter.setBrush(QBrush(pulse_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(rect.left() + 12, rect.top() + 12, 8, 8)

        painter.setFont(QFont(FONT_UI, 8, QFont.Weight.Bold))
        painter.setPen(QColor(250, 250, 250, 200))
        painter.drawText(rect.left() + 26, rect.top() + 20, "LIVE MICROSCOPE FEED")

        # 2. Top-right: active resolution pill + live FPS readout
        res_text = f"{self._current_frame.width()}x{self._current_frame.height()}"
        fps_text = f"{self._current_fps:.0f} FPS"
        hud_text = f"{res_text}  |  {fps_text}"
        
        metrics = QFontMetrics(painter.font())
        txt_w = metrics.horizontalAdvance(hud_text)
        
        # Readability badge
        badge_bg = QColor(24, 24, 27, 160)
        painter.setBrush(QBrush(badge_bg))
        painter.setPen(QPen(QColor(63, 63, 70, 100), 1))
        painter.drawRoundedRect(rect.right() - txt_w - 24, rect.top() + 6, txt_w + 16, 20, 4, 4)
        
        painter.setPen(QColor(0, 210, 255, 220)) # Cyan
        painter.drawText(rect.right() - txt_w - 16, rect.top() + 20, hud_text)

        # 3. Bottom-left: SHARPNESS status badge
        sharpness_val = 0.0
        if self._detections is not None:
            sharpness_val = getattr(self._detections, "sharpness_variance", 0.0)
        
        sharpness_text = f"SHARPNESS: {sharpness_val:.1f}"
        s_w = metrics.horizontalAdvance(sharpness_text)
        
        painter.setBrush(QBrush(badge_bg))
        painter.setPen(QPen(QColor(63, 63, 70, 100), 1))
        painter.drawRoundedRect(rect.left() + 12, rect.bottom() - 26, s_w + 16, 20, 4, 4)
        
        text_color = QColor(0, 210, 255) if sharpness_val >= 12.5 else QColor(255, 193, 7)
        painter.setPen(text_color)
        painter.drawText(rect.left() + 20, rect.bottom() - 12, sharpness_text)

    def _draw_detections(self, painter: QPainter, letterbox: QRect) -> None:
        frame_w, frame_h = self._current_frame.width(), self._current_frame.height()
        if frame_w == 0 or frame_h == 0:
            return
        scale_x, scale_y = letterbox.width() / frame_w, letterbox.height() / frame_h
        for idx, box in enumerate(self._detections.boxes):
            is_highlighted = (idx == self._highlight_index)
            self._draw_single_box(painter, box, letterbox, scale_x, scale_y, is_highlighted)

    def _draw_single_box(self, painter, box, letterbox, scale_x, scale_y, is_highlighted=False):
        wx = letterbox.x() + int(round(box.x * scale_x))
        wy = letterbox.y() + int(round(box.y * scale_y))
        ww, wh = max(2, int(round(box.width * scale_x))), max(2, int(round(box.height * scale_y)))

        hex_color = DEFECT_CLASS_COLORS.get(box.class_name, DEFECT_CLASS_COLOR_FALLBACK)
        class_color = QColor(hex_color)

        # 1. Subtle translucent fill
        fill_color = QColor(class_color)
        fill_color.setAlpha(40 if is_highlighted else 15)
        painter.setBrush(QBrush(fill_color))
        
        # Faint dashed boundary (alpha 100 if highlighted) for visual completeness
        faint_alpha = 100 if is_highlighted else 30
        faint_pen = QPen(QColor(class_color.red(), class_color.green(), class_color.blue(), faint_alpha), 1, Qt.PenStyle.DashLine)
        painter.setPen(faint_pen)
        painter.drawRoundedRect(wx, wy, ww, wh, 2, 2)

        # 2. Figma-style glowing corner brackets
        len_px = min(12, ww // 3, wh // 3)
        if len_px < 4: len_px = 4

        # Outer high-contrast black corners
        outer_pen = QPen(QColor(0, 0, 0, 200), 5 if is_highlighted else 4, Qt.PenStyle.SolidLine)
        painter.setPen(outer_pen)
        self._draw_corners(painter, wx, wy, ww, wh, len_px)

        # Inner colored corners (Highlight: thicker neon lines)
        inner_pen = QPen(class_color, 3 if is_highlighted else 2, Qt.PenStyle.SolidLine)
        painter.setPen(inner_pen)
        self._draw_corners(painter, wx, wy, ww, wh, len_px)

        # 3. Label chip
        self._draw_label_chip(painter, wx, wy, box, class_color, is_highlighted)

    def _draw_corners(self, painter: QPainter, x: int, y: int, w: int, h: int, l: int) -> None:
        # Top-Left
        painter.drawLine(x, y, x + l, y)
        painter.drawLine(x, y, x, y + l)
        # Top-Right
        painter.drawLine(x + w, y, x + w - l, y)
        painter.drawLine(x + w, y, x + w, y + l)
        # Bottom-Left
        painter.drawLine(x, y + h, x + l, y + h)
        painter.drawLine(x, y + h, x, y + h - l)
        # Bottom-Right
        painter.drawLine(x + w, y + h, x + w - l, y + h)
        painter.drawLine(x + w, y + h, x + w, y + h - l)

    def _draw_label_chip(self, painter, box_x, box_y, box, class_color, is_highlighted=False):
        label = box.class_name.replace("_", " ").upper()
        chip_text = f"{label}  {box.confidence * 100:.0f}%"
        chip_font = QFont(FONT_MONO, FONT_SIZE_MONO_CHIP, QFont.Weight.Bold if is_highlighted else QFont.Weight.Medium)
        metrics = QFontMetrics(chip_font)
        chip_w, chip_h = metrics.horizontalAdvance(chip_text) + CHIP_PADDING_H * 2, metrics.height() + CHIP_PADDING_V * 2
        chip_x, chip_y = box_x, box_y - chip_h
        if chip_y < 0: chip_y = box_y

        chip_bg = QColor(class_color)
        chip_bg.setAlpha(220 if is_highlighted else DEFECT_CHIP_ALPHA)
        
        # Draw rounded chip
        painter.setBrush(QBrush(chip_bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(chip_x, chip_y, chip_w, chip_h, 3, 3)

        painter.setFont(chip_font)
        painter.setPen(QColor("#09090B" if is_highlighted else DEFECT_CHIP_TEXT_COLOR))
        painter.drawText(chip_x + CHIP_PADDING_H, chip_y + CHIP_PADDING_V + metrics.ascent(), chip_text)

    def _draw_status_label(self, painter: QPainter, palette: dict) -> None:
        if not self._status_text:
            return
        self._draw_status_panel(painter, palette)

    def _draw_status_panel(self, painter: QPainter, palette: dict) -> None:
        """Render a centered, modern card with status text and matching icon."""
        pw, ph = 360, 160
        px = (self.width() - pw) // 2
        py = (self.height() - ph) // 2
        panel_rect = QRect(px, py, pw, ph)

        # 1. Glassmorphic card background
        card_bg = QColor(palette["BG_SURFACE"])
        card_bg.setAlpha(240)
        painter.setBrush(QBrush(card_bg))
        painter.setPen(QPen(QColor(palette["BORDER"]), 1.5))
        painter.drawRoundedRect(panel_rect, 16, 16)

        # 2. Determine which icon to draw
        icon_name = "camera"
        subtitle = "Select a source from the sidebar to start scanning"
        title = self._status_text
        
        if "Model" in title or "model" in title.lower():
            icon_name = "brain"
            subtitle = "Initializing YOLOv12 detection core..."
        elif "Benchmarking" in title:
            icon_name = "activity"
            subtitle = "Measuring device latency..."
        elif "Connecting" in title:
            icon_name = "focus"
            subtitle = "Starting video stream capture..."
        elif "Please connect a camera" in title:
            icon_name = "camera"
            subtitle = "Connect a USB camera or select an active source"

        # 3. Draw SVG icon in the upper half of the card
        from ui.theme import ICONS_SVG
        
        svg_xml = ICONS_SVG.get(icon_name, "").format(color=palette["TEXT_SECONDARY"])
        if svg_xml:
            renderer = QSvgRenderer()
            renderer.load(svg_xml.encode('utf-8'))
            icon_size = 32
            icon_x = px + (pw - icon_size) // 2
            icon_y = py + 24
            renderer.render(painter, QRectF(float(icon_x), float(icon_y), float(icon_size), float(icon_size)))

        # 4. Draw Title
        painter.setFont(QFont(FONT_UI, FONT_SIZE_LABEL + 1, QFont.Weight.Bold))
        painter.setPen(QColor(palette["TEXT_PRIMARY"]))
        title_rect = QRect(px + 16, py + 72, pw - 32, 28)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, title)

        # 5. Draw Subtitle
        painter.setFont(QFont(FONT_UI, FONT_SIZE_BODY, QFont.Weight.Normal))
        painter.setPen(QColor(palette["TEXT_SECONDARY"]))
        subtitle_rect = QRect(px + 16, py + 104, pw - 32, 36)
        painter.drawText(subtitle_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, subtitle)