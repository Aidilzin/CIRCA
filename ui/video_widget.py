"""
ui/video_widget.py
------------------
VideoWidget — the central live-feed viewport with QPainter bounding box overlay.

Architecture: QWidget subclass that lives exclusively on the Main GUI Thread.
Thread safety is guaranteed by Qt's queued signal/slot mechanism: slots
(set_frame, set_detections, set_status_text) are called by the event loop
when signals from worker threads are delivered, always on the GUI thread.

UX Spec reference:
  §Component Strategy — VideoWidget (Phase 1 Core Render)
  §Bounding Box Visual Specification — dual-stroke + label chip
  §Empty / Loading State Pattern — idle, initialising, error states

Rendering pipeline (per paintEvent call):
  1. Fill widget background with COLOR_BG_BASE (#121212)
  2. If frame exists: scale-to-fit into widget rect (letterbox, aspect-correct)
  3. If detections exist and frame exists: draw dual-stroke bounding boxes
                                           + label chips over the scaled image
  4. If no frame: draw centred status label (idle / error state text)

Module import boundary (architecture §Boundaries):
  ALLOWED:   PyQt6.*, ui/theme.py, core/models.py (type hints only), logging
  FORBIDDEN: core/preprocessor.py, core/inference_engine.py, cv2, openvino,
             workers/* (VideoWidget must never touch worker internals)
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QRect, QRectF, Qt, pyqtSlot
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

from core.models import BoundingBox, DetectionResult
from ui.theme import (
    BBOX_INNER_WIDTH,
    BBOX_OUTER_ALPHA,
    BBOX_OUTER_WIDTH,
    CHIP_PADDING_H,
    CHIP_PADDING_V,
    COLOR_BG_BASE,
    COLOR_BG_SURFACE,
    COLOR_BORDER,
    COLOR_STATUS_ERROR,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
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
)

logger = logging.getLogger(__name__)


class VideoWidget(QWidget):
    """
    Central live-feed viewport widget.

    Displays a QImage camera feed scaled to fit the widget with letterboxing,
    and overlays bounding box detections rendered directly by QPainter
    (no intermediate pixmap — zero-copy, zero-latency rendering).

    State model (UX spec §Empty / Loading State Pattern):
      IDLE        — No frame received yet. Renders _status_text centred.
      ACTIVE      — Frames arriving. Renders live feed + detection overlays.
      ERROR       — Camera/inference error. Renders _status_text centred.

    Connected signals (set up in MainWindow.__init__, never inside this file):
      CameraWorker.new_frame          → VideoWidget.set_frame
      InferenceWorker.new_detections  → VideoWidget.set_detections
      CameraWorker.camera_error       → VideoWidget.set_status_text  (optional)

    paintEvent performance note:
      QPainter drawing in paintEvent is synchronous on the GUI thread.
      To maintain ≥15 FPS (NFR3), keep paintEvent fast:
        - No I/O, no allocation of large buffers.
        - QImage is stored by value (Qt smart-pointers internally).
        - Bounding box list is typically ≤10 elements — iteration is O(1) effective.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Current camera frame — None until first set_frame() call.
        self._current_frame: Optional[QImage] = None

        # Current detection result — None until first set_detections() call.
        self._detections: Optional[DetectionResult] = None

        # Status text shown when no frame is available (idle / error states).
        # UX spec §Empty / Loading State Pattern.
        self._status_text: str = "Please connect a camera"

        # Solid #121212 background (prevent flicker on resize).
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        # Minimum reasonable size so letterbox maths never divides by zero.
        self.setMinimumSize(320, 240)

    # ------------------------------------------------------------------
    # Public slots (connected by MainWindow — never by workers)
    # ------------------------------------------------------------------

    @pyqtSlot(QImage)
    def set_frame(self, image: QImage) -> None:
        """
        Receive a new camera frame and schedule a repaint.

        Called at ≥15 FPS from CameraWorker via queued signal/slot.
        Qt ensures this slot is always invoked on the GUI thread.

        Args:
            image: RGB888 QImage from core.utils.bgr_frame_to_qimage().
        """
        self._current_frame = image
        self.update()   # Schedules paintEvent — never triggers it directly.

    @pyqtSlot(object)
    def set_detections(self, result: object) -> None:
        """
        Receive a new DetectionResult and schedule a repaint.

        Called once per inference pass from InferenceWorker via queued signal.
        The result is typed as object to match pyqtSignal(object) declaration.

        The detection result is rendered on the NEXT paintEvent cycle — the
        frame it corresponds to may already have advanced if inference is
        slower than the camera FPS (NFR3 — boxes snap to new position).

        Args:
            result: DetectionResult instance (zero or more BoundingBox objects).
        """
        if isinstance(result, DetectionResult):
            self._detections = result
        else:
            logger.warning(
                "VideoWidget.set_detections received unexpected type: %s",
                type(result).__name__,
            )
            self._detections = None
        self.update()

    @pyqtSlot(str)
    def set_status_text(self, text: str) -> None:
        """
        Update the centred status label shown when no live frame is available.

        Used by MainWindow to display:
          "Please connect a camera" — on launch or when active device removed
          "Connecting…"            — while CameraWorker is starting
          "camera unavailable"     — on camera_error signal (stylized panel)

        The status text is only visible when _current_frame is None.
        """
        self._status_text = text
        if self._current_frame is None:
            self.update()

    def clear_frame(self) -> None:
        """
        Drop the current frame (e.g. on camera disconnect) and return to
        the idle/error state showing the status text.
        """
        self._current_frame = None
        self._detections = None
        self.update()

    # ------------------------------------------------------------------
    # Qt paint event (Main GUI Thread only — always)
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """
        Render the live feed and detection overlays.

        Called by Qt after self.update() — always on the GUI thread.
        Do not call this method directly. Use self.update() to schedule it.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        try:
            # Stage 1: Fill viewport background (#121212 letterbox)
            painter.fillRect(self.rect(), QColor(COLOR_BG_BASE))

            if self._current_frame is None or self._current_frame.isNull():
                # ── Idle / Error state ───────────────────────────────────
                self._draw_status_label(painter)
                return

            # Stage 2: Draw camera frame (letterboxed, aspect-correct)
            letterbox = self._compute_letterbox_rect(
                self._current_frame.width(),
                self._current_frame.height(),
            )
            painter.drawImage(letterbox, self._current_frame)

            # Stage 3: Draw bounding box overlays (if any detections)
            if (
                self._detections is not None
                and self._detections.boxes
            ):
                self._draw_detections(painter, letterbox)

        finally:
            painter.end()

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _compute_letterbox_rect(
        self, frame_w: int, frame_h: int
    ) -> QRect:
        """
        Compute the rect within the widget where the frame should be drawn,
        maintaining the frame's original aspect ratio (letterboxing).

        The image is centred: excess space is filled by the #121212 background
        (already drawn before this call).

        Args:
            frame_w / frame_h: Pixel dimensions of the source QImage.

        Returns:
            QRect specifying where to draw the image within the widget.
        """
        widget_w = self.width()
        widget_h = self.height()

        # Guard: avoid division by zero on degenerate frames.
        if frame_w == 0 or frame_h == 0:
            return QRect(0, 0, widget_w, widget_h)

        # Uniform scale — fit within widget, never stretch.
        scale = min(widget_w / frame_w, widget_h / frame_h)
        drawn_w = int(frame_w * scale)
        drawn_h = int(frame_h * scale)

        # Centre the image in the available space.
        offset_x = (widget_w - drawn_w) // 2
        offset_y = (widget_h - drawn_h) // 2

        return QRect(offset_x, offset_y, drawn_w, drawn_h)

    def _draw_detections(
        self, painter: QPainter, letterbox: QRect
    ) -> None:
        """
        Draw all bounding boxes and label chips for self._detections.

        Coordinate transform:
          BoundingBox coordinates are in the original camera frame space
          (e.g. 640×480). They must be scaled to letterbox pixel space:
              box_x_widget = letterbox.x() + box.x * (letterbox.w / frame.w)

        Args:
            painter:   Active QPainter (rendering in progress).
            letterbox: The rect where the frame is drawn.
        """
        frame_w = self._current_frame.width()   # type: ignore[union-attr]
        frame_h = self._current_frame.height()  # type: ignore[union-attr]

        if frame_w == 0 or frame_h == 0:
            return

        scale_x = letterbox.width()  / frame_w
        scale_y = letterbox.height() / frame_h

        for box in self._detections.boxes:  # type: ignore[union-attr]
            self._draw_single_box(painter, box, letterbox, scale_x, scale_y)

    def _draw_single_box(
        self,
        painter: QPainter,
        box: BoundingBox,
        letterbox: QRect,
        scale_x: float,
        scale_y: float,
    ) -> None:
        """
        Draw one detection box with dual-stroke rendering + label chip.

        Dual-stroke pattern (UX spec §Bounding Box Visual Specification):
          Step 1: Draw rect with BBOX_OUTER_WIDTH pen (black, 70% opacity)
                  This creates the shadow/separation layer.
          Step 2: Draw SAME rect with BBOX_INNER_WIDTH pen (class colour)
                  The inner coloured stroke sits centred on the edge, with
                  the black stroke visible as a halo on both sides.

        Args:
            box:        BoundingBox with original-frame-space coordinates.
            letterbox:  Widget-space rect where the image is drawn.
            scale_x/y:  Scale factors from frame space → widget space.
        """
        # Transform bounding box into letterbox (widget) space.
        wx = letterbox.x() + int(round(box.x * scale_x))
        wy = letterbox.y() + int(round(box.y * scale_y))
        ww = max(2, int(round(box.width  * scale_x)))
        wh = max(2, int(round(box.height * scale_y)))

        # Resolve class colour.
        hex_color = DEFECT_CLASS_COLORS.get(
            box.class_name, DEFECT_CLASS_COLOR_FALLBACK
        )
        class_color = QColor(hex_color)

        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        # ── Step 1: Outer shadow stroke ──────────────────────────────
        outer_color = QColor(0, 0, 0, BBOX_OUTER_ALPHA)   # black, 70% opacity
        outer_pen = QPen(outer_color, BBOX_OUTER_WIDTH)
        outer_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(outer_pen)
        painter.drawRect(wx, wy, ww, wh)

        # ── Step 2: Inner class-colour stroke ────────────────────────
        inner_pen = QPen(class_color, BBOX_INNER_WIDTH)
        inner_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(inner_pen)
        painter.drawRect(wx, wy, ww, wh)

        # ── Step 3: Label chip at top-left corner ────────────────────
        self._draw_label_chip(painter, wx, wy, box, class_color)

    def _draw_label_chip(
        self,
        painter: QPainter,
        box_x: int,
        box_y: int,
        box: BoundingBox,
        class_color: QColor,
    ) -> None:
        """
        Draw the label chip at the top-left corner of the bounding box.

        Chip format: "SOLDER BRIDGE  94%"
        Rendered using FONT_MONO (JetBrains Mono) at FONT_SIZE_MONO_CHIP (11px)
        to prevent horizontal layout jitter at ≥15 FPS (UX spec typography rule).

        Chip anatomy:
          [class colour background at 80% opacity]
          [FONT_MONO text in #E8E8E8]
          Chip sits above the box top edge; falls inside if box is at top.

        Args:
            box_x / box_y:  Top-left corner of the bounding box in widget space.
            box:            BoundingBox carrying class_name and confidence.
            class_color:    QColor for this defect class (no alpha yet).
        """
        # Format chip text: class name + confidence percentage.
        # Replace underscores with spaces for readability; use UPPER case.
        label = box.class_name.replace("_", " ").upper()
        confidence_pct = f"{box.confidence * 100:.0f}%"
        chip_text = f"{label}  {confidence_pct}"

        # Build the monospace chip font (JetBrains Mono with fallbacks).
        chip_font = QFont()
        chip_font.setFamilies([FONT_MONO] + FONT_MONO_FALLBACKS)
        chip_font.setPixelSize(FONT_SIZE_MONO_CHIP)
        chip_font.setWeight(QFont.Weight.Medium)

        metrics = QFontMetrics(chip_font)
        text_w = metrics.horizontalAdvance(chip_text)
        text_h = metrics.height()

        chip_w = text_w + CHIP_PADDING_H * 2
        chip_h = text_h + CHIP_PADDING_V * 2

        # Position: above the box top edge; fall inside if near top of widget.
        chip_x = box_x
        chip_y = box_y - chip_h
        if chip_y < 0:
            chip_y = box_y  # Fall inside the box to stay within the viewport.

        # Draw chip background: class colour at 80% (DEFECT_CHIP_ALPHA) opacity.
        chip_bg = QColor(class_color)
        chip_bg.setAlpha(DEFECT_CHIP_ALPHA)
        painter.fillRect(chip_x, chip_y, chip_w, chip_h, chip_bg)

        # Draw chip text (#E8E8E8 — primary text on semi-transparent colour).
        painter.setFont(chip_font)
        painter.setPen(QColor(DEFECT_CHIP_TEXT_COLOR))
        painter.drawText(
            chip_x + CHIP_PADDING_H,
            chip_y + CHIP_PADDING_V + metrics.ascent(),
            chip_text,
        )

    def _draw_status_label(self, painter: QPainter) -> None:
        """
        Draw the centred status label for idle/error states (no camera feed).

        If the status is "camera unavailable", it renders as a stylized
        panel per user request. Otherwise, it renders as a standard
        secondary status label.
        """
        if not self._status_text:
            return

        if self._status_text == "camera unavailable":
            self._draw_unavailable_panel(painter)
            return

        status_font = QFont()
        status_font.setFamilies([FONT_MONO] + FONT_MONO_FALLBACKS)
        status_font.setPixelSize(FONT_SIZE_LABEL)

        painter.setFont(status_font)
        painter.setPen(QColor(COLOR_TEXT_SECONDARY))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            self._status_text,
        )

    def _draw_unavailable_panel(self, painter: QPainter) -> None:
        """
        Draw a prominent 'camera unavailable' panel in the centre of the viewport.
        """
        # Panel dimensions
        pw, ph = 240, 60
        px = (self.width() - pw) // 2
        py = (self.height() - ph) // 2
        panel_rect = QRect(px, py, pw, ph)

        # Draw panel background (Elevated surface with border)
        painter.setBrush(QBrush(QColor(COLOR_BG_SURFACE)))
        painter.setPen(QPen(QColor(COLOR_BORDER), 1))
        painter.drawRoundedRect(panel_rect, 4, 4)

        # Draw accent bar on the left (Error red)
        accent_w = 4
        accent_rect = QRect(px, py, accent_w, ph)
        painter.fillRect(accent_rect, QColor(COLOR_STATUS_ERROR))

        # Draw text
        font = QFont()
        font.setFamilies([FONT_UI] + FONT_UI_FALLBACKS)
        font.setPixelSize(FONT_SIZE_LABEL)
        font.setWeight(QFont.Weight.DemiBold)
        font.setCapitalization(QFont.Capitalization.AllUppercase)

        painter.setFont(font)
        painter.setPen(QColor(COLOR_TEXT_PRIMARY))
        
        # Center text within the panel (accounting for the accent bar)
        text_rect = QRect(px + accent_w, py, pw - accent_w, ph)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter,
            "camera unavailable"
        )
