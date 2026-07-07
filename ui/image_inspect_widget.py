"""
ui/image_inspect_widget.py
--------------------------
ImageInspectWidget — the primary inspection workspace for CIRCA.

Supports three states:

  EMPTY   → Drag-and-drop zone prompting the user to load a PCB image.
  LOADING → Spinner/status while inference is running.
  RESULT  → Static image with tiled bounding-box overlay (indefinite display).

No live video stream is shown in this widget. The camera is accessed via the
top-bar "Capture" button in MainWindow, which grabs a single frame and passes
it here as a static image.

Design principles:
  - Drag-and-drop: accepts .jpg, .jpeg, .png, .bmp, .tif, .tiff
  - Built-in drawing helpers (corner brackets, label chips)
  - Emits image_loaded(np.ndarray) when a new image is ready for inference
  - Emits clear_requested() when user wants to reset
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QRect, QRectF, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QBrush, QColor, QDragEnterEvent, QDropEvent, QFont, QFontMetrics,
    QImage, QPainter, QPen,
)
from PyQt6.QtWidgets import QWidget
from PyQt6.QtSvg import QSvgRenderer

from core.models import DetectionResult
from ui.theme import (
    BBOX_INNER_WIDTH, BBOX_OUTER_ALPHA, BBOX_OUTER_WIDTH,
    CHIP_PADDING_H, CHIP_PADDING_V,
    DEFECT_CHIP_ALPHA, DEFECT_CHIP_TEXT_COLOR,
    DEFECT_CLASS_COLOR_FALLBACK, DEFECT_CLASS_COLORS,
    FONT_MONO, FONT_SIZE_MONO_CHIP, FONT_SIZE_LABEL, FONT_SIZE_BODY,
    FONT_UI, ICONS_SVG, ThemeManager,
)

logger = logging.getLogger(__name__)

_ACCEPTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageInspectWidget(QWidget):
    """
    Central inspection workspace. Shows drag-drop zone when empty, static
    analysed image with bounding boxes when a result is available.
    """

    # Emitted when a new image (BGR ndarray) is ready for inference
    image_loaded: pyqtSignal = pyqtSignal(object)
    # Emitted whenever the FPS placeholder (now "analysis time") updates
    fps_updated: pyqtSignal = pyqtSignal(float)

    # ──────────────────────────────────────────────────────────────────────
    class _State:
        EMPTY = "empty"
        ANALYZING = "analyzing"
        RESULT = "result"
        REJECTED = "rejected"   # PCB guard rejected the image

    # ──────────────────────────────────────────────────────────────────────

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._state: str = self._State.EMPTY
        self._qimage: Optional[QImage] = None          # current display image
        self._bgr_frame: Optional[np.ndarray] = None   # raw numpy array
        self._detections: Optional[DetectionResult] = None
        self._status_text: str = ""
        self._guard_reason: str = ""
        self._highlight_index: int = -1
        self._tile_count: int = 0
        self._inference_ms: float = 0.0

        # Drag-and-drop pulsing animation
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(50)
        self._pulse_timer.timeout.connect(self.update)
        self._pulse_timer.start()

        self.setAcceptDrops(True)
        self.setMinimumSize(320, 240)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ──────────────────────────────────────────────────────────────────────
    # Public API (called by MainWindow)
    # ──────────────────────────────────────────────────────────────────────

    def load_image_from_path(self, path: str) -> None:
        """Load a PCB image from a file path and emit image_loaded."""
        bgr = cv2.imread(path)
        if bgr is None:
            logger.error("ImageInspectWidget: could not read '%s'", path)
            return
        self._load_bgr(bgr)

    def load_image_from_array(self, bgr_frame: np.ndarray) -> None:
        """Load a PCB image from a BGR numpy array (e.g., camera capture)."""
        if bgr_frame is None or bgr_frame.size == 0:
            return
        self._load_bgr(bgr_frame.copy())

    def set_analyzing(self) -> None:
        """Switch to analyzing state while inference runs."""
        self._state = self._State.ANALYZING
        self._detections = None
        self.update()

    def set_rejected(self, reason: str) -> None:
        """Show PCB guard rejection message instead of result."""
        self._state = self._State.REJECTED
        self._guard_reason = reason
        self._detections = None
        self.update()

    @pyqtSlot(object)
    def set_detections(self, result: object) -> None:
        """Receive inference result and switch to RESULT state."""
        if not isinstance(result, DetectionResult):
            return
        self._detections = result
        self._state = self._State.RESULT
        self._inference_ms = getattr(result, "inference_time_ms", 0.0)
        self._tile_count = getattr(result, "tile_count", 1)
        self.fps_updated.emit(self._inference_ms)
        self.update()

    @pyqtSlot(str)
    def set_status_text(self, text: str) -> None:
        self._status_text = text
        self.update()

    @pyqtSlot(int)
    def set_highlight_index(self, idx: int) -> None:
        self._highlight_index = idx
        self.update()

    def clear_feed(self, status_text: str = "") -> None:
        """Reset to empty state."""
        self._state = self._State.EMPTY
        self._qimage = None
        self._bgr_frame = None
        self._detections = None
        self._status_text = status_text
        self._guard_reason = ""
        self._highlight_index = -1
        self.update()

    # Compatibility shim — MainWindow may call set_frame for camera preview.
    # In image-inspect mode we ignore streaming frames; only explicit loads matter.
    @pyqtSlot(QImage)
    def set_frame(self, image: QImage) -> None:
        pass  # no-op: camera stream not displayed here

    # ──────────────────────────────────────────────────────────────────────
    # Drag-and-drop
    # ──────────────────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in _ACCEPTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in _ACCEPTED_EXTENSIONS and os.path.isfile(path):
                self.load_image_from_path(path)
                event.acceptProposedAction()
                return

    # ──────────────────────────────────────────────────────────────────────
    # Paint
    # ──────────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        try:
            palette = ThemeManager().get_palette()
            painter.fillRect(self.rect(), QColor(palette["BG_BASE"]))

            if self._state == self._State.EMPTY:
                self._draw_drop_zone(painter, palette)
            elif self._state == self._State.ANALYZING:
                self._draw_analyzing(painter, palette)
            elif self._state == self._State.REJECTED:
                self._draw_rejected(painter, palette)
            elif self._state == self._State.RESULT:
                if self._qimage and not self._qimage.isNull():
                    letterbox = self._compute_letterbox(
                        self._qimage.width(), self._qimage.height()
                    )
                    painter.drawImage(letterbox, self._qimage)
                    self._draw_result_hud(painter, letterbox, palette)
                    if self._detections and self._detections.boxes:
                        self._draw_detections(painter, letterbox)
                else:
                    self._draw_drop_zone(painter, palette)
        finally:
            painter.end()

    # ──────────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_letterbox(self, img_w: int, img_h: int) -> QRect:
        ww, wh = self.width(), self.height()
        if img_w == 0 or img_h == 0:
            return QRect(0, 0, ww, wh)
        scale = min(ww / img_w, wh / img_h)
        dw, dh = int(img_w * scale), int(img_h * scale)
        return QRect((ww - dw) // 2, (wh - dh) // 2, dw, dh)

    def _draw_drop_zone(self, painter: QPainter, palette: dict) -> None:
        import math, time

        cw, ch = self.width(), self.height()
        box_w, box_h = min(480, cw - 40), min(280, ch - 40)
        bx = (cw - box_w) // 2
        by = (ch - box_h) // 2

        # Pulsing dashed border
        t = time.perf_counter()
        alpha = int(80 + 60 * abs(math.sin(t * 1.5)))
        border_color = QColor(0, 200, 255, alpha)
        pen = QPen(border_color, 2, Qt.PenStyle.DashLine)
        pen.setDashPattern([8, 6])
        painter.setPen(pen)
        painter.setBrush(QColor(palette["BG_SURFACE"]))
        painter.drawRoundedRect(bx, by, box_w, box_h, 16, 16)

        # Upload icon
        svg_xml = ICONS_SVG.get("upload", "").format(color=palette["TEXT_SECONDARY"])
        if svg_xml:
            renderer = QSvgRenderer()
            renderer.load(svg_xml.encode("utf-8"))
            icon_size = 48
            ix = (cw - icon_size) // 2
            iy = by + 36
            renderer.render(painter, QRectF(float(ix), float(iy), float(icon_size), float(icon_size)))

        # Primary text
        painter.setFont(QFont(FONT_UI, FONT_SIZE_LABEL + 2, QFont.Weight.Bold))
        painter.setPen(QColor(palette["TEXT_PRIMARY"]))
        title_rect = QRect(bx + 16, by + 100, box_w - 32, 32)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "Drop a PCB Image Here")

        # Sub text
        painter.setFont(QFont(FONT_UI, FONT_SIZE_BODY))
        painter.setPen(QColor(palette["TEXT_SECONDARY"]))
        sub_rect = QRect(bx + 16, by + 138, box_w - 32, 24)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter,
                         "or click  📁 Load Image  in the toolbar")

        # Supported formats hint
        painter.setFont(QFont(FONT_UI, 8))
        painter.setPen(QColor(palette["TEXT_SECONDARY"]))
        hint_rect = QRect(bx + 16, by + box_h - 36, box_w - 32, 20)
        painter.drawText(hint_rect, Qt.AlignmentFlag.AlignCenter,
                         "Accepts: JPG · PNG · BMP · TIFF")

    def _draw_analyzing(self, painter: QPainter, palette: dict) -> None:
        import math, time

        if self._qimage and not self._qimage.isNull():
            letterbox = self._compute_letterbox(self._qimage.width(), self._qimage.height())
            painter.setOpacity(0.4)
            painter.drawImage(letterbox, self._qimage)
            painter.setOpacity(1.0)

        cw, ch = self.width(), self.height()
        # Spinning arc
        t = time.perf_counter()
        cx, cy = cw // 2, ch // 2
        r = 30
        span = int(270 + 60 * math.sin(t * 2))
        start = int((t * 200) % 360) * 16
        painter.setPen(QPen(QColor(0, 200, 255), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(cx - r, cy - r - 30, r * 2, r * 2, start, span * 16)

        painter.setFont(QFont(FONT_UI, FONT_SIZE_LABEL + 1, QFont.Weight.Bold))
        painter.setPen(QColor(palette["TEXT_PRIMARY"]))
        painter.drawText(QRect(0, cy + 14, cw, 32), Qt.AlignmentFlag.AlignCenter,
                         "Analysing PCB…")
        painter.setFont(QFont(FONT_UI, FONT_SIZE_BODY))
        painter.setPen(QColor(palette["TEXT_SECONDARY"]))
        painter.drawText(QRect(0, cy + 46, cw, 24), Qt.AlignmentFlag.AlignCenter,
                         "Tiled inference running — please wait")

    def _draw_rejected(self, painter: QPainter, palette: dict) -> None:
        cw, ch = self.width(), self.height()
        pw, ph = 400, 180
        px = (cw - pw) // 2
        py = (ch - ph) // 2

        card_bg = QColor(palette["BG_SURFACE"])
        card_bg.setAlpha(240)
        painter.setBrush(QBrush(card_bg))
        painter.setPen(QPen(QColor(255, 80, 80, 160), 1.5))
        painter.drawRoundedRect(px, py, pw, ph, 16, 16)

        svg_xml = ICONS_SVG.get("alert", "").format(color="#FF5050")
        if svg_xml:
            renderer = QSvgRenderer()
            renderer.load(svg_xml.encode("utf-8"))
            renderer.render(painter, QRectF(float((cw - 32) // 2), float(py + 20), 32.0, 32.0))

        painter.setFont(QFont(FONT_UI, FONT_SIZE_LABEL + 1, QFont.Weight.Bold))
        painter.setPen(QColor("#FF5050"))
        painter.drawText(QRect(px + 16, py + 64, pw - 32, 28),
                         Qt.AlignmentFlag.AlignCenter, "Not a PCB Image")

        painter.setFont(QFont(FONT_UI, FONT_SIZE_BODY))
        painter.setPen(QColor(palette["TEXT_SECONDARY"]))
        painter.drawText(QRect(px + 16, py + 100, pw - 32, 56),
                         Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                         self._guard_reason or "Please load a PCB photograph.")

    def _draw_result_hud(self, painter: QPainter, rect: QRect, palette: dict) -> None:
        """Top-left: ANALYSIS COMPLETE badge. Top-right: resolution + timing."""
        painter.setPen(QPen(QColor(0, 210, 255, 35), 1))
        painter.drawLine(rect.left(), rect.top() + 30, rect.right(), rect.top() + 30)

        badge_bg = QColor(24, 24, 27, 170)
        painter.setFont(QFont(FONT_UI, 8, QFont.Weight.Bold))

        # Top-left: ANALYSED badge
        painter.setBrush(QBrush(badge_bg))
        painter.setPen(QPen(QColor(63, 63, 70, 100), 1))
        painter.drawRoundedRect(rect.left() + 8, rect.top() + 6, 140, 20, 4, 4)
        painter.setPen(QColor(0, 210, 255, 220))
        n_det = len(self._detections.boxes) if self._detections else 0
        det_label = f"ANALYSED  |  {n_det} defect{'s' if n_det != 1 else ''}"
        painter.drawText(rect.left() + 16, rect.top() + 20, det_label)

        # Top-right: resolution + inference time
        if self._qimage:
            res = f"{self._qimage.width()}x{self._qimage.height()}"
            if self._tile_count > 1:
                timing = f"{res}  |  {self._tile_count} tiles  |  {self._inference_ms:.0f}ms"
            else:
                timing = f"{res}  |  {self._inference_ms:.0f}ms"
            metrics = QFontMetrics(painter.font())
            tw = metrics.horizontalAdvance(timing)
            painter.setBrush(QBrush(badge_bg))
            painter.setPen(QPen(QColor(63, 63, 70, 100), 1))
            painter.drawRoundedRect(rect.right() - tw - 24, rect.top() + 6, tw + 16, 20, 4, 4)
            painter.setPen(QColor(0, 210, 255, 220))
            painter.drawText(rect.right() - tw - 16, rect.top() + 20, timing)

    def _draw_detections(self, painter: QPainter, letterbox: QRect) -> None:
        if self._qimage is None:
            return
        fw, fh = self._qimage.width(), self._qimage.height()
        if fw == 0 or fh == 0:
            return
        sx = letterbox.width() / fw
        sy = letterbox.height() / fh
        for idx, box in enumerate(self._detections.boxes):
            self._draw_single_box(painter, box, letterbox, sx, sy,
                                  idx == self._highlight_index)

    def _draw_single_box(self, painter, box, letterbox, sx, sy, highlighted=False):
        wx = letterbox.x() + int(round(box.x * sx))
        wy = letterbox.y() + int(round(box.y * sy))
        ww = max(2, int(round(box.width * sx)))
        wh = max(2, int(round(box.height * sy)))

        hex_color = DEFECT_CLASS_COLORS.get(box.class_name, DEFECT_CLASS_COLOR_FALLBACK)
        cc = QColor(hex_color)

        fill = QColor(cc)
        fill.setAlpha(40 if highlighted else 15)
        painter.setBrush(QBrush(fill))
        faint_pen = QPen(QColor(cc.red(), cc.green(), cc.blue(),
                                100 if highlighted else 30), 1, Qt.PenStyle.DashLine)
        painter.setPen(faint_pen)
        painter.drawRoundedRect(wx, wy, ww, wh, 2, 2)

        l = min(12, ww // 3, wh // 3)
        if l < 4:
            l = 4

        painter.setPen(QPen(QColor(0, 0, 0, 200), 5 if highlighted else 4))
        self._draw_corners(painter, wx, wy, ww, wh, l)

        painter.setPen(QPen(cc, 3 if highlighted else 2))
        self._draw_corners(painter, wx, wy, ww, wh, l)

        self._draw_label_chip(painter, wx, wy, box, cc, highlighted)

    def _draw_corners(self, painter, x, y, w, h, l):
        painter.drawLine(x, y, x + l, y)
        painter.drawLine(x, y, x, y + l)
        painter.drawLine(x + w, y, x + w - l, y)
        painter.drawLine(x + w, y, x + w, y + l)
        painter.drawLine(x, y + h, x + l, y + h)
        painter.drawLine(x, y + h, x, y + h - l)
        painter.drawLine(x + w, y + h, x + w - l, y + h)
        painter.drawLine(x + w, y + h, x + w, y + h - l)

    def _draw_label_chip(self, painter, bx, by, box, cc, highlighted=False):
        label = box.class_name.replace("_", " ").upper()
        text = f"{label}  {box.confidence * 100:.0f}%"
        font = QFont(FONT_MONO, FONT_SIZE_MONO_CHIP,
                     QFont.Weight.Bold if highlighted else QFont.Weight.Medium)
        metrics = QFontMetrics(font)
        cw = metrics.horizontalAdvance(text) + CHIP_PADDING_H * 2
        ch = metrics.height() + CHIP_PADDING_V * 2
        cx = bx
        cy = by - ch if by - ch >= 0 else by

        chip_bg = QColor(cc)
        chip_bg.setAlpha(220 if highlighted else DEFECT_CHIP_ALPHA)
        painter.setBrush(QBrush(chip_bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(cx, cy, cw, ch, 3, 3)

        painter.setFont(font)
        painter.setPen(QColor("#09090B" if highlighted else DEFECT_CHIP_TEXT_COLOR))
        painter.drawText(cx + CHIP_PADDING_H, cy + CHIP_PADDING_V + metrics.ascent(), text)

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_bgr(self, bgr: np.ndarray) -> None:
        """Store BGR array, convert to QImage for display, emit image_loaded."""
        self._bgr_frame = bgr
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._qimage = QImage(
            rgb.tobytes(), w, h,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()  # .copy() so the buffer is owned by QImage
        self._state = self._State.ANALYZING
        self._detections = None
        self.update()
        # Emit so MainWindow can route to inference worker
        self.image_loaded.emit(bgr)
