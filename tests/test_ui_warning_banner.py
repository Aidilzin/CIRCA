"""
tests/test_ui_warning_banner.py
--------------------------------
Unit tests for ui/warning_banner.py (FR15 WarningBanner).

Testing strategy:
  WarningBanner is driven entirely through update_confidence() and
  reset_board_state() — no QTimer, no QEventLoop tricks needed. The
  frame-counter design makes the state machine fully synchronous and
  deterministic in tests.

  Key contract to verify:
    1. Starts hidden.
    2. Low confidence → show (unless dismissed).
    3. Low confidence while dismissed → stay hidden.
    4. Good frame → increment counter; banner stays visible unless COOL_DOWN_FRAMES hit.
    5. Bad frame resets good-frame counter to 0, restarting the 2-second window.
    6. Exactly COOL_DOWN_FRAMES consecutive good frames → auto-reset
       (hide, clear dismissed flag, reset counter).
    7. COOL_DOWN_FRAMES - 1 consecutive good frames → NOT yet auto-reset.
    8. × button click → hide + set dismissed flag.
    9. reset_board_state() → clear dismissed + reset counter (no visibility change).
   10. After reset_board_state(), low confidence re-shows the banner.
   11. Amber tier rule: widget uses COLOR_STATUS_WARN for text, not for background.
   12. Fixed height is 32px.
   13. COOL_DOWN_FRAMES is the public constant (testable).

isHidden() is used throughout instead of isVisible() because WarningBanner
explicitly calls setVisible(False) — isHidden() reflects explicit hide state
correctly even for unshown widgets (confirmed in TestControlPanelInit).

QApplication provided by tests/conftest.py.
"""

from __future__ import annotations

import pytest

from ui.theme import (
    COLOR_STATUS_WARN,
    WARNING_BANNER_HEIGHT,
)
from ui.warning_banner import COOL_DOWN_FRAMES, WarningBanner


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def banner() -> WarningBanner:
    """Fresh WarningBanner for each test."""
    return WarningBanner()


# Convenience thresholds used across tests
LOW_CONF = 0.40  # avg_confidence BELOW threshold → bad frame
HIGH_CONF = 0.90  # avg_confidence ABOVE threshold → good frame
THRESHOLD = 0.50  # The configured threshold


# ===========================================================================
# Initialisation
# ===========================================================================


class TestWarningBannerInit:
    def test_starts_hidden(self, banner: WarningBanner) -> None:
        """FR15: banner must not be visible at startup."""
        assert banner.isHidden()

    def test_fixed_height_is_32(self, banner: WarningBanner) -> None:
        """UX spec: fixed height 32px."""
        assert banner.height() == WARNING_BANNER_HEIGHT

    def test_dismissed_flag_starts_false(self, banner: WarningBanner) -> None:
        assert banner._dismissed_for_current_board is False

    def test_good_frame_count_starts_zero(self, banner: WarningBanner) -> None:
        assert banner._good_frame_count == 0

    def test_cool_down_frames_constant_is_30(self) -> None:
        """
        Module-level constant COOL_DOWN_FRAMES must equal 30.
        At ~15 FPS × 30 frames = 2 seconds (UX spec §WarningBanner cool-down).
        """
        assert COOL_DOWN_FRAMES == 30

    def test_has_warning_text_label(self, banner: WarningBanner) -> None:
        assert hasattr(banner, "_text_label")
        assert "Low Confidence" in banner._text_label.text()

    def test_has_dismiss_button(self, banner: WarningBanner) -> None:
        assert hasattr(banner, "_dismiss_btn")

    def test_dismiss_button_text_is_multiplication_sign(
        self, banner: WarningBanner
    ) -> None:
        """
        UX spec: × character (U+00D7) — not ASCII 'x'. Visually distinct
        and more accessible for screen readers.
        """
        assert "×" in banner._dismiss_btn.text()

    def test_warning_text_uses_status_warn_color(self, banner: WarningBanner) -> None:
        """Text label QSS must reference COLOR_STATUS_WARN (#FFC107 — amber tier)."""
        assert COLOR_STATUS_WARN in banner._text_label.styleSheet()


# ===========================================================================
# update_confidence — bad frame (avg_confidence < threshold)
# ===========================================================================


class TestUpdateConfidenceBadFrame:
    def test_low_confidence_shows_banner(self, banner: WarningBanner) -> None:
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert not banner.isHidden()

    def test_low_confidence_does_not_show_if_dismissed(
        self, banner: WarningBanner
    ) -> None:
        """Path 1 dismissed: banner must stay hidden even on bad frames."""
        banner._dismissed_for_current_board = True
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert banner.isHidden()

    def test_low_confidence_resets_good_frame_counter(
        self, banner: WarningBanner
    ) -> None:
        """Bad frame must reset the continuous cool-down counter to 0."""
        banner._good_frame_count = 15  # Simulate halfway through cool-down
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert banner._good_frame_count == 0

    def test_low_confidence_counter_reset_on_any_bad_frame(
        self, banner: WarningBanner
    ) -> None:
        """Counter resets on ANY bad frame — even during an ongoing cool-down."""
        # Build up 29 good frames then one bad frame
        for _ in range(COOL_DOWN_FRAMES - 1):
            banner.update_confidence(HIGH_CONF, THRESHOLD)
        assert banner._good_frame_count == COOL_DOWN_FRAMES - 1

        # One bad frame → counter back to 0
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert banner._good_frame_count == 0

    def test_exactly_at_threshold_is_good_frame(self, banner: WarningBanner) -> None:
        """avg_confidence == threshold is a good frame (>= comparison)."""
        banner.update_confidence(THRESHOLD, THRESHOLD)
        assert banner._good_frame_count == 1

    def test_just_below_threshold_is_bad_frame(self, banner: WarningBanner) -> None:
        epsilon = 1e-9
        banner.update_confidence(THRESHOLD - epsilon, THRESHOLD)
        assert banner._good_frame_count == 0

    def test_low_confidence_shows_banner_repeatedly(
        self, banner: WarningBanner
    ) -> None:
        """Multiple bad frames: banner stays visible (already shown)."""
        banner.update_confidence(LOW_CONF, THRESHOLD)
        banner.update_confidence(LOW_CONF, THRESHOLD)
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert not banner.isHidden()


# ===========================================================================
# update_confidence — good frame (avg_confidence >= threshold)
# ===========================================================================


class TestUpdateConfidenceGoodFrame:
    def test_single_good_frame_increments_counter(self, banner: WarningBanner) -> None:
        banner.update_confidence(HIGH_CONF, THRESHOLD)
        assert banner._good_frame_count == 1

    def test_good_frames_do_not_hide_banner_below_threshold(
        self, banner: WarningBanner
    ) -> None:
        """Banner visible from bad frame → stays visible during cool-down period."""
        banner.update_confidence(LOW_CONF, THRESHOLD)  # show banner
        for _ in range(COOL_DOWN_FRAMES - 1):
            banner.update_confidence(HIGH_CONF, THRESHOLD)
        # Not yet reached COOL_DOWN_FRAMES → still visible
        assert not banner.isHidden()

    def test_good_frames_up_to_threshold_minus_1_does_not_auto_reset(
        self, banner: WarningBanner
    ) -> None:
        """29 good frames is NOT enough to trigger auto-reset (need 30)."""
        for _ in range(COOL_DOWN_FRAMES - 1):
            banner.update_confidence(HIGH_CONF, THRESHOLD)
        assert banner._good_frame_count == COOL_DOWN_FRAMES - 1
        assert banner._dismissed_for_current_board is False  # Unchanged

    def test_good_frames_start_from_hidden_banner(self, banner: WarningBanner) -> None:
        """
        From hidden → good frames should not change dismissed flag until cool-down.
        This tests the start-of-session case (no bad frames yet).
        """
        assert banner.isHidden()
        for _ in range(5):
            banner.update_confidence(HIGH_CONF, THRESHOLD)
        assert banner.isHidden()
        assert banner._dismissed_for_current_board is False


# ===========================================================================
# Auto-reset path (Path 2) — COOL_DOWN_FRAMES consecutive good frames
# ===========================================================================


class TestAutoReset:
    def _trigger_banner(self, banner: WarningBanner) -> None:
        """Helper: make banner visible with a bad frame."""
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert not banner.isHidden()

    def _drive_good_frames(self, banner: WarningBanner, n: int) -> None:
        """Helper: send n consecutive good frames."""
        for _ in range(n):
            banner.update_confidence(HIGH_CONF, THRESHOLD)

    def test_exactly_cool_down_frames_triggers_auto_reset(
        self, banner: WarningBanner
    ) -> None:
        """
        Critical: EXACTLY COOL_DOWN_FRAMES (30) consecutive good frames must
        trigger the auto-reset — no more, no less.
        """
        self._trigger_banner(banner)
        self._drive_good_frames(banner, COOL_DOWN_FRAMES)
        assert banner.isHidden()

    def test_auto_reset_hides_banner(self, banner: WarningBanner) -> None:
        self._trigger_banner(banner)
        self._drive_good_frames(banner, COOL_DOWN_FRAMES)
        assert banner.isHidden()

    def test_auto_reset_clears_dismissed_flag(self, banner: WarningBanner) -> None:
        """
        After auto-reset, _dismissed_for_current_board must be False so the
        banner can re-appear if the next board has low confidence.
        """
        banner._dismissed_for_current_board = True  # Pre-set dismissed state
        self._drive_good_frames(banner, COOL_DOWN_FRAMES)
        assert banner._dismissed_for_current_board is False

    def test_auto_reset_resets_good_frame_counter(self, banner: WarningBanner) -> None:
        """Counter resets to 0 after auto-reset — ready for the next board."""
        self._drive_good_frames(banner, COOL_DOWN_FRAMES)
        assert banner._good_frame_count == 0

    def test_banner_can_reappear_after_auto_reset(self, banner: WarningBanner) -> None:
        """
        After auto-reset, a subsequent bad frame must be able to re-show the
        banner (flag cleared → not dismissed).
        """
        self._drive_good_frames(banner, COOL_DOWN_FRAMES)
        # Next board has low confidence
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert not banner.isHidden()

    def test_interrupted_cool_down_does_not_auto_reset(
        self, banner: WarningBanner
    ) -> None:
        """
        A bad frame during the cool-down window restarts the counter.
        29 good → 1 bad → 29 good must NOT trigger auto-reset.
        """
        self._trigger_banner(banner)
        self._drive_good_frames(banner, COOL_DOWN_FRAMES - 1)  # 29 good
        banner.update_confidence(LOW_CONF, THRESHOLD)  # 1 bad — resets
        self._drive_good_frames(banner, COOL_DOWN_FRAMES - 1)  # 29 more

        # Total good frames since last reset = 29, not 30 → no auto-reset
        assert not banner.isHidden()
        assert banner._good_frame_count == COOL_DOWN_FRAMES - 1

    def test_exactly_30_good_frames_after_interruption(
        self, banner: WarningBanner
    ) -> None:
        """After interruption: need a FULL 30 new consecutive good frames."""
        self._trigger_banner(banner)
        self._drive_good_frames(banner, COOL_DOWN_FRAMES - 1)  # 29 good
        banner.update_confidence(LOW_CONF, THRESHOLD)  # Reset counter
        self._drive_good_frames(banner, COOL_DOWN_FRAMES)  # 30 fresh good
        assert banner.isHidden()


# ===========================================================================
# Manual dismissal path (Path 1) — × button
# ===========================================================================


class TestManualDismissal:
    def test_dismiss_button_hides_banner(self, banner: WarningBanner) -> None:
        banner.update_confidence(LOW_CONF, THRESHOLD)  # show
        assert not banner.isHidden()
        banner._dismiss_btn.click()
        assert banner.isHidden()

    def test_dismiss_button_sets_dismissed_flag(self, banner: WarningBanner) -> None:
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert banner._dismissed_for_current_board is False
        banner._dismiss_btn.click()
        assert banner._dismissed_for_current_board is True

    def test_dismissed_banner_suppresses_on_subsequent_bad_frames(
        self, banner: WarningBanner
    ) -> None:
        """After × click, bad frames must not re-show the banner."""
        banner.update_confidence(LOW_CONF, THRESHOLD)
        banner._dismiss_btn.click()
        # More bad frames — banner must stay dismissed
        for _ in range(10):
            banner.update_confidence(LOW_CONF, THRESHOLD)
        assert banner.isHidden()

    def test_dismissed_flag_persists_across_bad_frames(
        self, banner: WarningBanner
    ) -> None:
        banner._on_dismiss_clicked()  # Direct call (internal method)
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert banner._dismissed_for_current_board is True

    def test_dismiss_before_banner_shown_sets_flag(self, banner: WarningBanner) -> None:
        """Clicking × before banner is shown: sets flag, stays hidden."""
        assert banner.isHidden()
        banner._on_dismiss_clicked()
        assert banner.isHidden()
        assert banner._dismissed_for_current_board is True

    def test_dismissal_does_not_affect_good_frame_counter(
        self, banner: WarningBanner
    ) -> None:
        """× click must not touch _good_frame_count."""
        banner._good_frame_count = 15
        banner._on_dismiss_clicked()
        assert banner._good_frame_count == 15


# ===========================================================================
# reset_board_state()
# ===========================================================================


class TestResetBoardState:
    def test_reset_clears_dismissed_flag(self, banner: WarningBanner) -> None:
        banner._dismissed_for_current_board = True
        banner.reset_board_state()
        assert banner._dismissed_for_current_board is False

    def test_reset_clears_good_frame_counter(self, banner: WarningBanner) -> None:
        banner._good_frame_count = 22
        banner.reset_board_state()
        assert banner._good_frame_count == 0

    def test_reset_does_not_change_visibility_when_hidden(
        self, banner: WarningBanner
    ) -> None:
        """reset_board_state() must not alter visibility — next update_confidence decides."""
        assert banner.isHidden()
        banner.reset_board_state()
        assert banner.isHidden()

    def test_reset_does_not_hide_visible_banner(self, banner: WarningBanner) -> None:
        """If the banner is visible when reset_board_state() fires, it stays visible."""
        banner.update_confidence(LOW_CONF, THRESHOLD)
        assert not banner.isHidden()
        banner.reset_board_state()
        assert not banner.isHidden()

    def test_reset_allows_banner_to_reappear_after_manual_dismiss(
        self, banner: WarningBanner
    ) -> None:
        """
        Flow: low conf → banner shows → Leo clicks × → new board placed
        → reset_board_state() → low conf again → banner re-shows.
        """
        banner.update_confidence(LOW_CONF, THRESHOLD)  # Step 1: show
        banner._dismiss_btn.click()  # Step 2: dismiss
        assert banner.isHidden()
        banner.reset_board_state()  # Step 3: new board
        banner.update_confidence(LOW_CONF, THRESHOLD)  # Step 4: low conf on new board
        assert not banner.isHidden()

    def test_reset_allows_auto_reset_to_trigger_after_dismissal(
        self, banner: WarningBanner
    ) -> None:
        """
        After reset_board_state() the auto-reset path still works correctly.
        reset clears dismissed flag, so 30 good frames can still trigger.
        """
        banner._dismissed_for_current_board = True
        banner.reset_board_state()
        # 30 consecutive good frames after reset — auto-reset fires
        for _ in range(COOL_DOWN_FRAMES):
            banner.update_confidence(HIGH_CONF, THRESHOLD)
        assert banner._dismissed_for_current_board is False
        assert banner._good_frame_count == 0


# ===========================================================================
# Full end-to-end state machine flows
# ===========================================================================


class TestStateMachineFlows:
    def test_full_manual_dismiss_and_auto_recover_flow(
        self, banner: WarningBanner
    ) -> None:
        """
        Full UX flow:
          1. Board A: low confidence → banner shows
          2. Leo clicks × → banner dismissed for Board A
          3. Board A stays: bad frames → banner stays dismissed
          4. Board A removed: 30 good frames → auto-reset fires
          5. Board B: low confidence → banner shows again
        """
        # 1. Low confidence on Board A
        banner.update_confidence(0.30, THRESHOLD)
        assert not banner.isHidden()

        # 2. Leo dismisses
        banner._dismiss_btn.click()
        assert banner.isHidden()

        # 3. More bad frames — still dismissed
        for _ in range(10):
            banner.update_confidence(0.30, THRESHOLD)
        assert banner.isHidden()

        # 4. Board A removed; confidence soars for 30 frames
        for _ in range(COOL_DOWN_FRAMES):
            banner.update_confidence(0.95, THRESHOLD)
        assert banner.isHidden()
        assert banner._dismissed_for_current_board is False  # Ready for Board B

        # 5. Board B: low confidence → banner re-shows
        banner.update_confidence(0.25, THRESHOLD)
        assert not banner.isHidden()

    def test_threshold_change_affects_next_update(self, banner: WarningBanner) -> None:
        """
        The threshold is a runtime parameter — MainWindow passes the current
        value on every call. Verify the state machine respects a changed threshold.
        """
        # With threshold=0.50, 0.45 is a bad frame
        banner.update_confidence(0.45, 0.50)
        assert not banner.isHidden()

        # After × dismiss
        banner._dismiss_btn.click()

        # With threshold=0.40, 0.45 is now a GOOD frame (0.45 >= 0.40)
        for _ in range(COOL_DOWN_FRAMES):
            banner.update_confidence(0.45, 0.40)
        # Auto-reset fired because 0.45 was consistently above 0.40
        assert banner.isHidden()
        assert banner._dismissed_for_current_board is False

    def test_zero_confidence_always_bad(self, banner: WarningBanner) -> None:
        """avg_confidence=0.0 (no detections) must show the banner."""
        banner.update_confidence(0.0, THRESHOLD)
        assert not banner.isHidden()

    def test_full_confidence_always_good_and_auto_resets(
        self, banner: WarningBanner
    ) -> None:
        """avg_confidence=1.0 (DetectionResult.average_confidence clean-board sentinel)."""
        for _ in range(COOL_DOWN_FRAMES):
            banner.update_confidence(1.0, THRESHOLD)
        assert banner.isHidden()
        assert banner._good_frame_count == 0
