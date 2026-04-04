"""
ui/warning_banner.py
--------------------
WarningBanner — FR15 full-width amber advisory bar.

UX Spec: §WarningBanner — FR15 Low-Confidence Advisory (Dual-Dismissal)
         §Amber tier rule — #FFC107 EXCLUSIVELY reserved for this component

Purpose:
    Displayed above VideoWidget when InferenceWorker.new_detections carries
    an average confidence below the configured threshold. Implements two
    independent dismissal paths so technicians are never trapped by a banner
    they have already acted on:

    Path 1  — Manual:  Leo presses × → banner hides for the current board.
    Path 2  — Auto:    Average confidence recovers for 2 consecutive seconds
                       (30 frames at ~15 FPS) → banner resets automatically.

Architecture:
    WarningBanner is a passive QWidget — it does NOT connect to any worker
    signals directly. MainWindow mediates: on each new_detections signal from
    InferenceWorker, MainWindow calls banner.update_confidence(). This keeps
    the signal graph clean and the banner unit-testable without mocks.

Cool-down implementation:
    The user specified a frame-counter implementation (~30 consecutive good
    frames ≈ 2 seconds at ~15 FPS) rather than a QTimer. This avoids event-
    loop dependency in unit tests (no need for QEventLoop or QTimer.singleShot
    tricks) and works correctly even if inference runs faster/slower than
    15 FPS within typical ranges.

    The counter is reset to 0 on any bad frame, so the 2-second window is
    CONTINUOUS — a single dip below threshold restarts the countdown.

State machine (from UX spec §WarningBanner — Internal State Machine):

           avg_conf >= threshold
           for COOL_DOWN_FRAMES consecutive frames
   HIDDEN ◄──────────────────────── VISIBLE (pulsing amber)
     │                                       ▲
     │  avg_conf < threshold                  │  avg_conf < threshold
     │  AND NOT _dismissed_for_current_board  │  (flag cleared by auto-reset)
     └─────────────────────────────► DISMISSED (hidden)
                                               │
                               × button ───────┘

Amber tier rule (UX spec §Colour Palette):
    COLOR_STATUS_WARN (#FFC107) is EXCLUSIVELY reserved for this component.
    It must not appear anywhere else in the UI as a decorative or accent colour.
    Verified by test_ui_theme_and_video_widget.py::test_defect_colors_do_not_use_status_warn.

Layout impact:
    WarningBanner lives in a QVBoxLayout above VideoWidget. When visible, it
    occupies WARNING_BANNER_HEIGHT (32px); VideoWidget shrinks accordingly.
    The banner never overlaps the video feed.

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
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ui.theme import (
    COLOR_STATUS_WARN,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    FONT_SIZE_LABEL,
    FONT_UI,
    FONT_UI_FALLBACKS,
    SPACING_SM,
    SPACING_XS,
    WARNING_BANNER_HEIGHT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constant — exposed so tests can inspect the threshold and drive
# exactly COOL_DOWN_FRAMES good frames without sleeping.
# ---------------------------------------------------------------------------

#: Number of consecutive frames at or above threshold for auto-reset.
#: At ~15 FPS × 30 frames = 2 seconds (UX spec §WarningBanner cool-down period).
COOL_DOWN_FRAMES: int = 30


class WarningBanner(QWidget):
    """
    FR15 full-width amber advisory bar (32px fixed height).

    Anatomy:
        [⚠  Low Confidence — Please Verify Manually         ×]

    Starts HIDDEN. Becomes visible when update_confidence() receives
    avg_confidence below threshold AND the board has not been manually dismissed.

    Slots (connected by MainWindow — never by workers directly):
        update_confidence(avg_confidence, threshold)
            — Called on every new_detections signal from InferenceWorker.
              Drives the entire visible/hidden state machine.

        reset_board_state()
            — Called by MainWindow when a new PCB board is placed under the
              camera. Clears the manual-dismiss flag so the banner can re-appear
              if confidence is low on the next board.

    Internal state:
        _dismissed_for_current_board : bool
            Set True when × is clicked. Prevents re-show until auto-reset fires
            (30 consecutive good frames) or reset_board_state() is called.

        _good_frame_count : int
            Counts consecutive frames where avg_confidence >= threshold.
            Reset to 0 on any bad frame. When it reaches COOL_DOWN_FRAMES,
            the auto-reset fires.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Internal state
        self._dismissed_for_current_board: bool = False
        self._good_frame_count: int = 0

        # Build UI
        self._build_ui()

        # Starts HIDDEN — appears only when update_confidence() triggers it.
        self.setVisible(False)

        # Fixed height — layout impact: VideoWidget shrinks by 32px when visible.
        self.setFixedHeight(WARNING_BANNER_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setAccessibleName("Low confidence warning banner")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the banner layout: warning text (left) + dismiss button (right)."""
        # Background: rgba(255, 193, 7, 30) — amber at ~12% opacity.
        # Border-bottom: 1px rgba(255, 193, 7, 90) — amber at ~35% opacity.
        # Applied via setStyleSheet so it doesn't interfere with child styles.
        #
        # TODO (Phase 3 polish): Add QPropertyAnimation pulsing the background
        # opacity between 8% (alpha=20) and 18% (alpha=46) at 1 Hz.
        # Deferred — requires QGraphicsOpacityEffect or custom paintEvent.
        self.setStyleSheet(
            "WarningBanner {"
            "  background-color: rgba(255, 193, 7, 30);"
            "  border-bottom: 1px solid rgba(255, 193, 7, 90);"
            "}"
        )

        row = QHBoxLayout(self)
        row.setContentsMargins(SPACING_SM, 0, SPACING_SM, 0)
        row.setSpacing(SPACING_SM)

        # ── Warning text ──────────────────────────────────────────────
        self._text_label = QLabel("⚠  Low Confidence — Please Verify Manually")
        text_font = QFont()
        text_font.setFamilies([FONT_UI] + FONT_UI_FALLBACKS)
        text_font.setPixelSize(FONT_SIZE_LABEL)
        text_font.setWeight(QFont.Weight.Medium)
        self._text_label.setFont(text_font)
        self._text_label.setStyleSheet(f"color: {COLOR_STATUS_WARN}; background: transparent;")
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._text_label, stretch=1)

        # ── Dismiss button ────────────────────────────────────────────
        # 20×20px per UX spec. Uses × (U+00D7 multiplication sign — visually
        # distinct from the ASCII 'x' for accessibility and aesthetics).
        self._dismiss_btn = QPushButton("×")
        self._dismiss_btn.setFixedSize(20, 20)
        self._dismiss_btn.setToolTip("Dismiss warning (for this board)")
        self._dismiss_btn.setAccessibleName("Dismiss low confidence warning")
        self._dismiss_btn.setStyleSheet(
            f"QPushButton {{"
            f"  color: {COLOR_TEXT_SECONDARY};"
            f"  background: transparent;"
            f"  border: none;"
            f"  font-size: 16px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  color: {COLOR_TEXT_PRIMARY};"
            f"}}"
        )
        self._dismiss_btn.clicked.connect(self._on_dismiss_clicked)
        row.addWidget(self._dismiss_btn, alignment=Qt.AlignmentFlag.AlignVCenter)

    # ------------------------------------------------------------------
    # Public slots
    # ------------------------------------------------------------------

    @pyqtSlot(float, float)
    def update_confidence(self, avg_confidence: float, threshold: float) -> None:
        """
        Drive the visible/hidden state machine from the latest inference result.

        Called by MainWindow on EVERY new_detections signal — approximately
        once per frame (15+ times per second while inference is running).

        State transitions:
            avg_confidence < threshold:
                → Reset _good_frame_count to 0 (restart cool-down window)
                → Show banner UNLESS _dismissed_for_current_board is True

            avg_confidence >= threshold (good frame):
                → Increment _good_frame_count
                → If _good_frame_count reaches COOL_DOWN_FRAMES (30 frames):
                    - Auto-reset: clear _dismissed_for_current_board
                    - Hide banner
                    - Reset counter to 0

        Args:
            avg_confidence: Current average confidence from DetectionResult.
                            Should be in [0.0, 1.0]. Values >= 1.0 (e.g. the
                            clean board sentinel value 1.0) are treated as good.
            threshold:      Current confidence_threshold from InferenceParams.
                            Passed at runtime so the slider can update it live
                            without requiring a re-connect.
        """
        if avg_confidence < threshold:
            # ── BAD FRAME ────────────────────────────────────────────────
            # Any dip below threshold restarts the full 2-second cool-down.
            self._good_frame_count = 0

            if not self._dismissed_for_current_board:
                # Show the amber advisory (Path 2 → Path 1 entry point).
                self.setVisible(True)
            # If dismissed: stay hidden. Leo has already noted the warning.

        else:
            # ── GOOD FRAME ───────────────────────────────────────────────
            self._good_frame_count += 1

            if self._good_frame_count >= COOL_DOWN_FRAMES:
                # 2 consecutive seconds of high confidence → auto-reset.
                logger.debug(
                    "WarningBanner: auto-reset after %d good frames.",
                    self._good_frame_count,
                )
                self._good_frame_count = 0
                self._dismissed_for_current_board = False  # Ready for next board.
                self.setVisible(False)

    @pyqtSlot()
    def reset_board_state(self) -> None:
        """
        Signal that a new PCB board has been placed under the camera.

        Clears the manual-dismiss flag (_dismissed_for_current_board) and
        resets the good-frame counter so the banner can appear again if the new
        board also has low confidence.

        Does NOT immediately show or hide the banner — the next
        update_confidence() call determines visibility based on the new board's
        confidence readings.

        Connected by MainWindow to a "new board" trigger (e.g. a manual button
        or a future board-change detector).
        """
        self._dismissed_for_current_board = False
        self._good_frame_count = 0
        logger.debug("WarningBanner: board state reset.")

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_dismiss_clicked(self) -> None:
        """
        Manual dismissal (Path 1).

        Hides the banner and sets the dismiss flag. The banner will not
        re-appear even if confidence stays low — until:
          - 30 consecutive good frames fire the auto-reset, OR
          - MainWindow calls reset_board_state() for a new board placement.
        """
        logger.debug("WarningBanner: manually dismissed for current board.")
        self._dismissed_for_current_board = True
        self.setVisible(False)
