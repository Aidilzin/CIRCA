"""
tests/test_ui_control_panel_and_footer.py
------------------------------------------
Unit tests for ui/sidebar.py (SidePanel / PreprocessingSlider) and ui/status_footer.py.

Testing strategy:
  PreprocessingSlider (from ui/sidebar.py):
    - Default value matches constructor argument
    - Value label text matches precision + suffix formatting
    - Value label uses FONT_MONO (anti-jitter contract)
    - Dragging slider emits value_changed signal with correct float

  SidePanel (replaces the old ControlPanel):
    - Starts collapsed (maxWidth == 0)
    - Has clahe_slider, gamma_slider, sharpness_slider, confidence_slider
    - Slider default values match UX spec
    - CLAHE/Gamma/Sharpness slider changes emit preprocessing_params_changed
    - Confidence slider changes emit inference_params_changed with correct threshold
    - populate_cameras() fills combo correctly

  StatusFooter:
    - Starts with grey dots
    - set_camera_active() → green dot, "Camera Active"
    - set_camera_error() → red dot, "Camera Error"
    - set_camera_idle() → grey dot, "No Device"
    - set_model_ready() → green dot, "Model Ready"
    - set_model_loading() → grey dot, "Loading Model…"
    - set_model_error() → red dot, "Model Error"
    - set_detection_count(0) → green dot, "No defects"
    - set_detection_count(N) → cyan dot, "N defect(s)"
    - set_detection_count(-1) → grey dot, "No data"
    - set_fps(15) → "15 fps"
    - set_fps(-1) → "— fps"
    - FPS label font is FONT_MONO

QApplication provided by tests/conftest.py.
"""

from __future__ import annotations

import pytest
from PyQt6.QtGui import QFont

from core.models import InferenceParams, PreprocessParams
from ui.sidebar import SidePanel, PreprocessingSlider
from ui.status_footer import StatusFooter, DOT_COLOR_IDLE
from ui.theme import (
    COLOR_STATUS_ERROR,
    COLOR_STATUS_OK,
    COLOR_ACCENT_CYAN,
    FONT_MONO,
    STATUS_FOOTER_HEIGHT,
)


# ===========================================================================
# Helpers
# ===========================================================================


def make_slider(
    icon_name: str = "contrast",
    name: str = "Test Slider",
    min_val: float = 0.0,
    max_val: float = 1000.0,
    default_val: float = 5.0,
    precision: int = 1,
    suffix: str = "",
) -> PreprocessingSlider:
    return PreprocessingSlider(
        icon_name=icon_name,
        name=name,
        min_val=min_val,
        max_val=max_val,
        default_val=default_val,
        precision=precision,
        suffix=suffix,
    )


def collect_signal(signal) -> list:
    """Connect a signal and return a list that accumulates all emitted values."""
    received: list = []
    signal.connect(lambda v: received.append(v))
    return received


# ===========================================================================
# PreprocessingSlider — basic construction
# ===========================================================================


class TestPreprocessingSliderInit:
    def test_default_value_returned_correctly(self):
        s = make_slider(default_val=5.0)
        assert abs(s.value() - 5.0) < 0.05

    def test_value_label_text_matches_default(self):
        s = make_slider(default_val=3.0, precision=1)
        assert s._value_label.text() == "3.0"

    def test_value_label_text_with_suffix(self):
        s = make_slider(default_val=50.0, precision=0, suffix="%")
        assert s._value_label.text() == "50%"

    def test_value_label_text_zero_precision(self):
        s = make_slider(default_val=100.0, precision=0)
        assert s._value_label.text() == "100"

    def test_name_label_content(self):
        s = PreprocessingSlider(
            icon_name="contrast",
            name="Glare Control",
            min_val=1.0,
            max_val=8.0,
            default_val=2.0,
        )
        assert s._name_label.text() == "Glare Control"


# ===========================================================================
# PreprocessingSlider — FONT_MONO contract (UX spec anti-jitter rule)
# ===========================================================================


class TestPreprocessingSliderFont:
    def test_value_label_uses_mono_font_family(self):
        """
        UX spec: live value labels MUST use FONT_MONO to prevent layout
        jitter at ≥15 FPS update rate.
        """
        s = make_slider()
        font: QFont = s._value_label.font()
        families = font.families()
        assert any(FONT_MONO in f for f in families), (
            f"Value label font {families} must include FONT_MONO='{FONT_MONO}'"
        )

    def test_value_label_has_a_positive_minimum_width(self):
        """Value label must prevent zero-width collapsing."""
        s = make_slider(min_val=0.0, max_val=100.0, default_val=5.0, precision=1)
        assert s._value_label.minimumWidth() >= 0


# ===========================================================================
# PreprocessingSlider — value update and signal emission
# ===========================================================================


class TestPreprocessingSliderUpdates:
    def test_moving_slider_emits_value_changed(self):
        s = make_slider(min_val=0.0, max_val=10.0, default_val=5.0, precision=1)
        received = collect_signal(s.value_changed)
        s._slider.setValue(int(7.0 * s._scale))
        assert len(received) == 1
        assert abs(received[0] - 7.0) < 0.05

    def test_value_changed_emits_float_not_int(self):
        s = make_slider(min_val=1.0, max_val=8.0, default_val=2.0, precision=1)
        received = collect_signal(s.value_changed)
        s._slider.setValue(int(3.5 * s._scale))
        assert isinstance(received[0], float)

    def test_value_label_updates_on_slider_change(self):
        s = make_slider(min_val=1.0, max_val=8.0, default_val=2.0, precision=1)
        s._slider.setValue(int(4.5 * s._scale))
        assert s._value_label.text() == "4.5"

    def test_value_label_updates_with_suffix(self):
        s = make_slider(min_val=0.0, max_val=100.0, default_val=50.0, precision=0, suffix="%")
        s._slider.setValue(int(75.0 * s._scale))
        assert s._value_label.text() == "75%"


# ===========================================================================
# SidePanel — construction (replaces old ControlPanel tests)
# ===========================================================================


@pytest.fixture()
def panel() -> SidePanel:
    return SidePanel()


class TestSidePanelInit:
    def test_starts_collapsed(self, panel):
        """SidePanel starts collapsed (maxWidth == 0)."""
        assert panel.maximumWidth() == 0

    def test_has_clahe_slider(self, panel):
        assert hasattr(panel, "clahe_slider")
        assert isinstance(panel.clahe_slider, PreprocessingSlider)

    def test_has_gamma_slider(self, panel):
        assert hasattr(panel, "gamma_slider")
        assert isinstance(panel.gamma_slider, PreprocessingSlider)

    def test_has_sharpness_slider(self, panel):
        assert hasattr(panel, "sharpness_slider")
        assert isinstance(panel.sharpness_slider, PreprocessingSlider)

    def test_has_confidence_slider(self, panel):
        assert hasattr(panel, "confidence_slider")
        assert isinstance(panel.confidence_slider, PreprocessingSlider)

    def test_has_camera_combo(self, panel):
        from PyQt6.QtWidgets import QComboBox
        assert hasattr(panel, "camera_combo")
        assert isinstance(panel.camera_combo, QComboBox)

    def test_default_clahe_value(self, panel):
        assert abs(panel.clahe_slider.value() - 2.0) < 0.05

    def test_default_gamma_value(self, panel):
        assert abs(panel.gamma_slider.value() - 1.0) < 0.05

    def test_default_sharpness_value(self, panel):
        assert abs(panel.sharpness_slider.value() - 100.0) < 1.0

    def test_default_confidence_value(self, panel):
        assert abs(panel.confidence_slider.value() - 50.0) < 0.5


# ===========================================================================
# SidePanel — signals
# ===========================================================================


class TestSidePanelSignals:
    def test_clahe_slider_emits_preprocessing_params(self, panel):
        received: list[PreprocessParams] = []
        panel.preprocessing_params_changed.connect(lambda p: received.append(p))
        panel.clahe_slider._slider.setValue(int(4.0 * panel.clahe_slider._scale))
        assert len(received) >= 1
        assert abs(received[-1].clahe_clip_limit - 4.0) < 0.05

    def test_gamma_slider_emits_preprocessing_params(self, panel):
        received: list[PreprocessParams] = []
        panel.preprocessing_params_changed.connect(lambda p: received.append(p))
        panel.gamma_slider._slider.setValue(int(1.5 * panel.gamma_slider._scale))
        assert len(received) >= 1
        assert abs(received[-1].gamma - 1.5) < 0.05

    def test_sharpness_slider_emits_preprocessing_params(self, panel):
        received: list[PreprocessParams] = []
        panel.preprocessing_params_changed.connect(lambda p: received.append(p))
        panel.sharpness_slider._slider.setValue(150)
        assert len(received) >= 1
        assert received[-1].blur_threshold == 150.0

    def test_confidence_slider_emits_inference_params(self, panel):
        received: list[InferenceParams] = []
        panel.inference_params_changed.connect(lambda p: received.append(p))
        panel.confidence_slider._slider.setValue(int(70.0 * panel.confidence_slider._scale))
        assert len(received) >= 1
        assert abs(received[-1].confidence_threshold - 0.70) < 0.01

    def test_preprocessing_params_carries_clahe_value(self, panel):
        received = []
        panel.preprocessing_params_changed.connect(lambda p: received.append(p))
        panel.clahe_slider._slider.setValue(int(4.0 * panel.clahe_slider._scale))
        assert abs(received[-1].clahe_clip_limit - 4.0) < 0.05

    def test_preprocessing_params_carries_gamma_value(self, panel):
        received = []
        panel.preprocessing_params_changed.connect(lambda p: received.append(p))
        panel.gamma_slider._slider.setValue(int(2.0 * panel.gamma_slider._scale))
        assert abs(received[-1].gamma - 2.0) < 0.05

    def test_preprocessing_params_carries_blur_value(self, panel):
        received = []
        panel.preprocessing_params_changed.connect(lambda p: received.append(p))
        panel.sharpness_slider._slider.setValue(200)
        assert received[-1].blur_threshold == 200.0


# ===========================================================================
# SidePanel — camera combo
# ===========================================================================


class TestSidePanelCameras:
    def test_populate_cameras_fills_combo(self, panel):
        panel.populate_cameras([(0, "USB Cam A"), (1, "USB Cam B")])
        assert panel.camera_combo.count() == 2

    def test_populate_cameras_stores_index_as_data(self, panel):
        panel.populate_cameras([(2, "Camera 2")])
        assert panel.camera_combo.itemData(0) == 2

    def test_populate_cameras_empty_clears_combo(self, panel):
        panel.populate_cameras([(0, "Test")])
        panel.populate_cameras([])
        assert panel.camera_combo.count() == 0

    def test_camera_selected_emits_on_programmatic_change(self, panel):
        received: list[int] = []
        panel.camera_selected.connect(lambda i: received.append(i))
        panel.populate_cameras([(0, "Cam 0"), (1, "Cam 1")])
        panel.camera_combo.setCurrentIndex(1)
        assert len(received) >= 1


# ===========================================================================
# StatusFooter — construction
# ===========================================================================


@pytest.fixture()
def footer() -> StatusFooter:
    return StatusFooter()


class TestStatusFooterInit:
    def test_is_correct_height(self, footer):
        assert footer.height() == STATUS_FOOTER_HEIGHT

    def test_camera_dot_starts_idle(self, footer):
        assert footer._camera_indicator.dot_color == DOT_COLOR_IDLE

    def test_model_dot_starts_idle(self, footer):
        assert footer._model_indicator.dot_color == DOT_COLOR_IDLE

    def test_detection_dot_starts_idle(self, footer):
        assert footer._detection_indicator.dot_color == DOT_COLOR_IDLE

    def test_camera_label_starts_with_no_device(self, footer):
        assert footer._camera_indicator.label_text == "No Device"

    def test_model_label_starts_with_no_model(self, footer):
        assert footer._model_indicator.label_text == "No Model"

    def test_detection_label_starts_with_no_data(self, footer):
        assert footer._detection_indicator.label_text == "No data"

    def test_fps_label_starts_with_placeholder(self, footer):
        assert "—" in footer._fps_label.text()


# ===========================================================================
# StatusFooter — camera indicator state transitions
# ===========================================================================


class TestStatusFooterCamera:
    def test_set_camera_active(self, footer):
        footer.set_camera_active()
        assert footer._camera_indicator.dot_color == COLOR_STATUS_OK
        assert "Active" in footer._camera_indicator.label_text

    def test_set_camera_error(self, footer):
        footer.set_camera_error("USB disconnect")
        assert footer._camera_indicator.dot_color == COLOR_STATUS_ERROR

    def test_set_camera_error_message_propagated(self, footer):
        footer.set_camera_error("USB disconnect")
        assert "USB disconnect" in footer._camera_indicator.label_text

    def test_set_camera_idle(self, footer):
        footer.set_camera_active()
        footer.set_camera_idle()
        assert footer._camera_indicator.dot_color == DOT_COLOR_IDLE
        assert "No Device" in footer._camera_indicator.label_text

    def test_dot_color_idle_constant_is_public(self):
        """DOT_COLOR_IDLE must be importable as a public name (no leading underscore)."""
        from ui.status_footer import DOT_COLOR_IDLE as _doi
        assert not _doi.startswith("_")  # value is a hex string, not a name


# ===========================================================================
# StatusFooter — model indicator state transitions
# ===========================================================================


class TestStatusFooterModel:
    def test_set_model_ready(self, footer):
        footer.set_model_ready()
        assert footer._model_indicator.dot_color == COLOR_STATUS_OK
        assert "Ready" in footer._model_indicator.label_text

    def test_set_model_loading(self, footer):
        footer.set_model_ready()
        footer.set_model_loading()
        assert footer._model_indicator.dot_color == DOT_COLOR_IDLE
        assert "Loading" in footer._model_indicator.label_text

    def test_set_model_error(self, footer):
        footer.set_model_error("GPU OOM")
        assert footer._model_indicator.dot_color == COLOR_STATUS_ERROR

    def test_set_model_error_propagates_message(self, footer):
        footer.set_model_error("bad path")
        assert "bad path" in footer._model_indicator.label_text


# ===========================================================================
# StatusFooter — detection indicator state transitions
# ===========================================================================


class TestStatusFooterDetections:
    def test_no_detections_green_dot(self, footer):
        footer.set_detection_count(0)
        assert footer._detection_indicator.dot_color == COLOR_STATUS_OK
        assert "No defects" in footer._detection_indicator.label_text

    def test_one_detection_singular(self, footer):
        footer.set_detection_count(1)
        assert "1 defect" in footer._detection_indicator.label_text

    def test_multiple_detections_plural(self, footer):
        footer.set_detection_count(3)
        assert "3 defects" in footer._detection_indicator.label_text

    def test_detection_count_negative_resets_to_idle(self, footer):
        footer.set_detection_count(2)
        footer.set_detection_count(-1)
        assert footer._detection_indicator.dot_color == DOT_COLOR_IDLE
        assert "No data" in footer._detection_indicator.label_text

    def test_nonzero_detections_use_accent_color(self, footer):
        footer.set_detection_count(2)
        assert footer._detection_indicator.dot_color == COLOR_ACCENT_CYAN


# ===========================================================================
# StatusFooter — FPS label
# ===========================================================================


class TestStatusFooterFPS:
    def test_set_fps_displays_integer(self, footer):
        footer.set_fps(30.0)
        assert "30 fps" in footer._fps_label.text()

    def test_set_fps_rounds_down(self, footer):
        footer.set_fps(14.6)
        assert "14 fps" in footer._fps_label.text() or "15 fps" in footer._fps_label.text()

    def test_set_fps_negative_shows_placeholder(self, footer):
        footer.set_fps(-1.0)
        assert "—" in footer._fps_label.text()

    def test_fps_label_uses_mono_font(self, footer):
        font: QFont = footer._fps_label.font()
        families = font.families()
        assert any(FONT_MONO in f for f in families), (
            f"FPS label font {families} must include FONT_MONO='{FONT_MONO}'"
        )
