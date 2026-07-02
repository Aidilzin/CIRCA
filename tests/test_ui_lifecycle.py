"""
tests/test_ui_lifecycle.py
---------------------------
Unit tests for MainWindow layout, lifecycle, and event handling.

Focus:
  - Layout structure and component presence
  - closeEvent safety and worker teardown
  - SidePanel / ActivityBar presence (replaces old ControlPanel)
  - USB hotplug debouncing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QMainWindow

from ui.main_window import MainWindow
from ui.sidebar import SidePanel
from ui.status_footer import StatusFooter
from ui.video_widget import VideoWidget
from ui.warning_banner import WarningBanner
from ui.theme import (
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
)
# Import the shared fixture
from tests.ui_test_utils import win


class TestMainWindowLayout:
    def test_is_qmainwindow(self, win):
        assert isinstance(win, QMainWindow)

    def test_has_video_widget(self, win):
        assert isinstance(win.video_widget, VideoWidget)

    def test_has_warning_banner(self, win):
        assert isinstance(win.warning_banner, WarningBanner)

    def test_warning_banner_starts_hidden(self, win):
        assert win.warning_banner.isHidden()

    def test_has_status_footer(self, win):
        assert isinstance(win.status_footer, StatusFooter)

    def test_has_top_settings_btn(self, win):
        """MainWindow must have the settings toggle configuration button in the top bar."""
        from PyQt6.QtWidgets import QPushButton
        assert hasattr(win, "top_settings_btn")
        assert isinstance(win.top_settings_btn, QPushButton)

    def test_has_side_panel(self, win):
        """MainWindow must have the SidePanel for settings and optimisation controls."""
        assert isinstance(win.side_panel, SidePanel)

    def test_minimum_width(self, win):
        assert win.minimumWidth() == WINDOW_MIN_WIDTH

    def test_minimum_height(self, win):
        assert win.minimumHeight() == WINDOW_MIN_HEIGHT

    def test_window_title_contains_circa(self, win):
        assert "CIRCA" in win.windowTitle()


class TestCloseEvent:
    def test_close_event_calls_camera_worker_stop(self, win):
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        assert win.camera_worker.stop_called is True

    def test_close_event_calls_camera_thread_quit(self, win):
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        win.camera_thread.quit.assert_called_once()

    def test_close_event_accepts_event(self, win):
        from PyQt6.QtGui import QCloseEvent
        event = QCloseEvent()
        win.closeEvent(event)
        assert event.isAccepted()

    def test_stop_called_before_quit(self, win):
        from PyQt6.QtGui import QCloseEvent
        call_log: list[str] = []
        win.camera_worker.stop = lambda: call_log.append("stop")
        win.camera_thread.quit = MagicMock(side_effect=lambda: call_log.append("quit"))
        win.camera_thread.wait = MagicMock(return_value=True)

        event = QCloseEvent()
        win.closeEvent(event)
        assert call_log.index("stop") < call_log.index("quit")


class TestMainWindowDefaults:
    def test_video_widget_initial_status_has_text(self, win):
        assert len(win.video_widget._status_text) > 0

    def test_side_panel_clahe_default(self, win):
        """SidePanel CLAHE slider must default to 2.0 (UX spec)."""
        assert abs(win.side_panel.clahe_slider.value() - 2.0) < 0.05

    def test_side_panel_confidence_default(self, win):
        """SidePanel confidence slider must default to 50% (UX spec)."""
        assert abs(win.side_panel.confidence_slider.value() - 50.0) < 0.5

    def test_side_panel_starts_collapsed(self, win):
        """SidePanel starts collapsed (maximumWidth == 0) per VS Code UX pattern."""
        assert win.side_panel.maximumWidth() == 0


class TestSidePanelIntegration:
    def test_settings_btn_click_opens_side_panel(self, win):
        """Clicking the settings button should expand the side panel."""
        win.top_settings_btn.click()
        # After click, panel should be expanded (maximumWidth > 0)
        # Note: animation sets maximumWidth, so check it changed from 0
        assert win.side_panel.maximumWidth() > 0 or win.side_panel._expanded is True

    def test_side_panel_has_camera_combo(self, win):
        assert hasattr(win.side_panel, "camera_combo")

    def test_side_panel_has_clahe_slider(self, win):
        assert hasattr(win.side_panel, "clahe_slider")

    def test_side_panel_has_confidence_slider(self, win):
        assert hasattr(win.side_panel, "confidence_slider")


class TestUsbHotplug:
    def test_usb_debounce_timer_exists_and_is_single_shot(self, win):
        from PyQt6.QtCore import QTimer
        assert hasattr(win, "_usb_debounce_timer")
        assert isinstance(win._usb_debounce_timer, QTimer)
        assert win._usb_debounce_timer.isSingleShot()

    def test_on_usb_timer_timeout_launches_scan(self, win):
        with patch.object(win, "_start_async_scan") as mock_scan:
            win._on_usb_timer_timeout()
        mock_scan.assert_called_once()

    def test_on_cameras_found_hotplug_updates_dropdown(self, win):
        new_cameras = [(0, "New Camera")]
        win._on_cameras_found_hotplug(new_cameras)
        assert win.side_panel.camera_combo.count() == 1
        assert win.side_panel.camera_combo.itemText(0) == "New Camera"

    def test_on_cameras_found_hotplug_empty_clears_status(self, win):
        """When no cameras found after hotplug removal, status reverts to idle."""
        win._on_cameras_found_hotplug([])
        assert "connect" in win.video_widget._status_text.lower()
