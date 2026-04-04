"""
tests/test_ui_control_panel_and_footer.py
------------------------------------------
Unit tests for ui/control_panel.py and ui/status_footer.py.

Testing strategy:
  PreprocessingSlider:
    - Default value matches constructor argument
    - Value label text matches precision + suffix formatting
    - Value label uses FONT_MONO (anti-jitter contract)
    - Value label has fixed minimum width (wider than the text at current value)
    - Dragging slider emits value_changed signal with correct float
    - set_value() clamps to [min_val, max_val]
    - Scale conversion round-trip (float → int → float) is within precision

  ControlPanel:
    - Starts expanded (width == 280, content visible)
    - Toggle collapses to 28px, content hidden
    - Toggle again expands back; content visible
    - After toggle, button text changes (‹ / ›)
    - CLAHE/Gamma/Blur slider changes emit preprocessing_params_changed
    - Confidence slider changes emit inference_params_changed
    - Confidence → InferenceParams conversion: 50% → confidence_threshold = 0.50
    - Default slider values match UX spec
    - populate_cameras() fills combo and emits camera_selected

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

All tests run headlessly — QWidget rendering is not exercised (no QPainter
calls). State management and signal correctness are verified directly.
QApplication provided by tests/conftest.py.
"""

from __future__ import annotations


from PyQt6.QtGui import QFont

from core.models import InferenceParams, PreprocessParams
from ui.control_panel import ControlPanel, PreprocessingSlider
from ui.status_footer import StatusFooter, _DOT_COLOR_IDLE
from ui.theme import (
    COLOR_STATUS_ERROR,
    COLOR_STATUS_OK,
    COLOR_STATUS_WARN,
    CONTROL_PANEL_WIDTH_COLLAPSED,
    CONTROL_PANEL_WIDTH_EXPANDED,
    FONT_MONO,
    STATUS_FOOTER_HEIGHT,
)


# ===========================================================================
# Helpers
# ===========================================================================


def make_slider(
    name: str = "Test Slider",
    sublabel: str = "test sub",
    min_val: float = 0.0,
    max_val: float = 1000.0,  # Wide range so all default_val values are in range
    default_val: float = 5.0,
    precision: int = 1,
    suffix: str = "",
) -> PreprocessingSlider:
    return PreprocessingSlider(
        name=name,
        sublabel=sublabel,
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
            name="Glare Control",
            sublabel="sub",
            min_val=1.0,
            max_val=8.0,
            default_val=2.0,
        )
        assert s._name_label.text() == "Glare Control"

    def test_sub_label_content(self):
        s = PreprocessingSlider(
            name="name",
            sublabel="CLAHE clip limit",
            min_val=1.0,
            max_val=8.0,
            default_val=2.0,
        )
        assert s._sub_label.text() == "CLAHE clip limit"


# ===========================================================================
# PreprocessingSlider — FONT_MONO contract (UX spec anti-jitter rule)
# ===========================================================================


class TestPreprocessingSliderFont:
    def test_value_label_uses_mono_font_family(self):
        """
        UX spec: live value labels MUST use FONT_MONO to prevent layout
        jitter at ≥15 FPS update rate. Verify the value label's font family
        list contains the FONT_MONO token ("JetBrains Mono").
        """
        s = make_slider()
        font: QFont = s._value_label.font()
        families = font.families()
        assert any(FONT_MONO in f for f in families), (
            f"Value label font {families} must include FONT_MONO='{FONT_MONO}'"
        )

    def test_value_label_has_fixed_minimum_width(self):
        """
        The value label must have a fixed minimum width (computed from the
        widest possible value string). This prevents QLabel from resizing the
        slider row on every drag tick.
        """
        s = make_slider(min_val=0.0, max_val=100.0, default_val=5.0, precision=1)
        # The min width should be at least as wide as the max value text "100.0"
        assert s._value_label.minimumWidth() > 0

    def test_value_label_minimum_width_covers_max_value(self):
        """Min width must be >= horizontal advance of the maximum value string."""
        from PyQt6.QtGui import QFontMetrics

        s = make_slider(min_val=0.0, max_val=999.9, default_val=1.0, precision=1)
        fm = QFontMetrics(s._value_label.font())
        max_text_width = fm.horizontalAdvance("999.9")
        assert s._value_label.minimumWidth() >= max_text_width


# ===========================================================================
# PreprocessingSlider — value update and signal emission
# ===========================================================================


class TestPreprocessingSliderUpdates:
    def test_moving_slider_emits_value_changed(self):
        s = make_slider(min_val=0.0, max_val=10.0, default_val=5.0, precision=1)
        received = collect_signal(s.value_changed)
        s._slider.setValue(s._to_int(7.0))
        assert len(received) == 1
        assert abs(received[0] - 7.0) < 0.05

    def test_value_changed_emits_float_not_int(self):
        s = make_slider(min_val=1.0, max_val=8.0, default_val=2.0, precision=1)
        received = collect_signal(s.value_changed)
        s._slider.setValue(s._to_int(3.5))
        assert isinstance(received[0], float)

    def test_value_label_updates_on_slider_change(self):
        s = make_slider(min_val=1.0, max_val=8.0, default_val=2.0, precision=1)
        s._slider.setValue(s._to_int(4.5))
        assert s._value_label.text() == "4.5"

    def test_value_label_updates_with_suffix(self):
        s = make_slider(
            min_val=10.0, max_val=95.0, default_val=50.0, precision=0, suffix="%"
        )
        s._slider.setValue(int(75))
        assert s._value_label.text() == "75%"

    def test_set_value_updates_slider_position(self):
        s = make_slider(min_val=0.0, max_val=10.0, default_val=0.0, precision=1)
        s.set_value(6.5)
        assert abs(s.value() - 6.5) < 0.05

    def test_set_value_clamps_above_max(self):
        s = make_slider(min_val=0.0, max_val=10.0, default_val=0.0, precision=1)
        s.set_value(999.0)
        assert s.value() <= 10.0

    def test_set_value_clamps_below_min(self):
        s = make_slider(min_val=2.0, max_val=10.0, default_val=5.0, precision=1)
        s.set_value(-99.0)
        assert s.value() >= 2.0

    def test_no_emission_when_value_unchanged(self):
        s = make_slider(min_val=0.0, max_val=10.0, default_val=5.0, precision=1)
        received = collect_signal(s.value_changed)
        s.set_value(5.0)  # Same as default — slider int unchanged
        assert len(received) == 0


# ===========================================================================
# PreprocessingSlider — float ↔ int scale conversion
# ===========================================================================


class TestPreprocessingSliderScale:
    def test_to_int_precision_1(self):
        s = make_slider(precision=1)
        assert s._to_int(2.0) == 20
        assert s._to_int(3.5) == 35

    def test_to_int_precision_0(self):
        s = make_slider(precision=0)
        assert s._to_int(50.0) == 50
        assert s._to_int(100.0) == 100

    def test_to_float_precision_1(self):
        s = make_slider(precision=1)
        assert abs(s._to_float(30) - 3.0) < 1e-9
        assert abs(s._to_float(25) - 2.5) < 1e-9

    def test_round_trip_precision_1(self):
        """float → int → float must match (within precision) for all slider values."""
        s = make_slider(min_val=1.0, max_val=8.0, default_val=2.0, precision=1)
        for v in [1.0, 2.0, 3.5, 5.0, 7.5, 8.0]:
            assert abs(s._to_float(s._to_int(v)) - v) < 0.05

    def test_round_trip_precision_0(self):
        s = make_slider(min_val=10.0, max_val=95.0, default_val=50.0, precision=0)
        for v in [10.0, 50.0, 75.0, 95.0]:
            assert abs(s._to_float(s._to_int(v)) - v) < 0.5


# ===========================================================================
# ControlPanel — initialisation and default UX spec values
# ===========================================================================


class TestControlPanelInit:
    def test_starts_expanded(self):
        cp = ControlPanel()
        assert cp.is_expanded is True

    def test_starts_at_expanded_width(self):
        cp = ControlPanel()
        assert cp.maximumWidth() == CONTROL_PANEL_WIDTH_EXPANDED

    def test_content_visible_when_expanded(self):
        cp = ControlPanel()
        # isHidden() reflects the explicit setVisible() state correctly even
        # for unshown top-level widgets (isVisible() requires show() to return True).
        assert not cp._content.isHidden()

    def test_toggle_button_shows_collapse_arrow_when_expanded(self):
        cp = ControlPanel()
        assert cp._toggle_btn.text() == "<"

    def test_default_clahe_value(self):
        """UX spec §Slider instances: CLAHE default = 2.0"""
        cp = ControlPanel()
        assert abs(cp.clahe_slider.value() - 2.0) < 0.05

    def test_default_gamma_value(self):
        """UX spec §Slider instances: Gamma default = 1.0"""
        cp = ControlPanel()
        assert abs(cp.gamma_slider.value() - 1.0) < 0.05

    def test_default_blur_value(self):
        """UX spec §Slider instances: Blur default = 100"""
        cp = ControlPanel()
        assert abs(cp.blur_slider.value() - 100.0) < 0.5

    def test_default_confidence_value(self):
        """UX spec §Slider instances: Confidence default = 50%"""
        cp = ControlPanel()
        assert abs(cp.confidence_slider.value() - 50.0) < 0.5


# ===========================================================================
# ControlPanel — collapse / expand toggle
# ===========================================================================


class TestControlPanelToggle:
    def test_toggle_collapses_panel(self):
        cp = ControlPanel()
        cp.toggle()
        assert cp.is_expanded is False

    def test_toggle_sets_collapsed_width(self):
        cp = ControlPanel()
        cp.toggle()
        assert cp.maximumWidth() == CONTROL_PANEL_WIDTH_COLLAPSED

    def test_toggle_hides_content(self):
        cp = ControlPanel()
        cp.toggle()
        assert not cp._content.isVisible()

    def test_toggle_button_shows_expand_arrow_when_collapsed(self):
        cp = ControlPanel()
        cp.toggle()
        assert cp._toggle_btn.text() == ">"

    def test_double_toggle_restores_expanded_state(self):
        cp = ControlPanel()
        cp.toggle()
        cp.toggle()
        assert cp.is_expanded is True
        assert cp.maximumWidth() == CONTROL_PANEL_WIDTH_EXPANDED
        assert not cp._content.isHidden()  # isHidden() correct for unshown widgets
        assert cp._toggle_btn.text() == "<"

    def test_toggle_via_button_click(self):
        cp = ControlPanel()
        cp._toggle_btn.click()
        assert cp.is_expanded is False
        cp._toggle_btn.click()
        assert cp.is_expanded is True


# ===========================================================================
# ControlPanel — signal emission
# ===========================================================================


class TestControlPanelSignals:
    def test_clahe_slider_emits_preprocessing_params(self):
        cp = ControlPanel()
        received: list[PreprocessParams] = []
        cp.preprocessing_params_changed.connect(lambda p: received.append(p))
        cp.clahe_slider._slider.setValue(cp.clahe_slider._to_int(3.0))
        assert len(received) == 1
        assert isinstance(received[0], PreprocessParams)

    def test_gamma_slider_emits_preprocessing_params(self):
        cp = ControlPanel()
        received: list = []
        cp.preprocessing_params_changed.connect(lambda p: received.append(p))
        cp.gamma_slider._slider.setValue(cp.gamma_slider._to_int(1.5))
        assert len(received) == 1

    def test_blur_slider_emits_preprocessing_params(self):
        cp = ControlPanel()
        received: list = []
        cp.preprocessing_params_changed.connect(lambda p: received.append(p))
        cp.blur_slider._slider.setValue(150)
        assert len(received) == 1

    def test_confidence_slider_emits_inference_params(self):
        cp = ControlPanel()
        received: list[InferenceParams] = []
        cp.inference_params_changed.connect(lambda p: received.append(p))
        cp.confidence_slider._slider.setValue(70)
        assert len(received) == 1
        assert isinstance(received[0], InferenceParams)

    def test_confidence_converted_to_fraction(self):
        """
        Critical: confidence slider emits 70.0 → InferenceParams.confidence_threshold = 0.70
        Not 70.0. Validates the /100.0 conversion in _emit_inference_params().
        """
        cp = ControlPanel()
        received: list[InferenceParams] = []
        cp.inference_params_changed.connect(lambda p: received.append(p))
        cp.confidence_slider._slider.setValue(70)
        assert abs(received[0].confidence_threshold - 0.70) < 1e-9

    def test_preprocessing_params_carries_clahe_value(self):
        cp = ControlPanel()
        received: list[PreprocessParams] = []
        cp.preprocessing_params_changed.connect(lambda p: received.append(p))
        cp.clahe_slider._slider.setValue(cp.clahe_slider._to_int(4.0))
        assert abs(received[0].clahe_clip_limit - 4.0) < 0.05

    def test_preprocessing_params_carries_gamma_value(self):
        cp = ControlPanel()
        received: list[PreprocessParams] = []
        cp.preprocessing_params_changed.connect(lambda p: received.append(p))
        cp.gamma_slider._slider.setValue(cp.gamma_slider._to_int(2.0))
        assert abs(received[0].gamma - 2.0) < 0.05

    def test_preprocessing_params_carries_blur_value(self):
        cp = ControlPanel()
        received: list[PreprocessParams] = []
        cp.preprocessing_params_changed.connect(lambda p: received.append(p))
        cp.blur_slider._slider.setValue(200)
        assert abs(received[0].blur_threshold - 200.0) < 0.5

    def test_no_camera_selected_emitted_for_empty_populate(self):
        cp = ControlPanel()
        received: list = []
        cp.camera_selected.connect(lambda i: received.append(i))
        cp.populate_cameras([])
        assert received == []

    def test_populate_cameras_emits_first_index(self):
        cp = ControlPanel()
        received: list[int] = []
        cp.camera_selected.connect(lambda i: received.append(i))
        cp.populate_cameras([(0, "USB Camera 0"), (1, "USB Camera 1")])
        assert len(received) == 1
        assert received[0] == 0


# ===========================================================================
# ControlPanel — populate_cameras
# ===========================================================================


class TestControlPanelCameras:
    def test_populate_cameras_adds_items(self):
        cp = ControlPanel()
        cp.populate_cameras([(0, "Camera 0"), (2, "Camera 2")])
        assert cp.camera_combo.count() == 2

    def test_populate_cameras_stores_device_index(self):
        cp = ControlPanel()
        cp.populate_cameras([(5, "USB Camera")])
        assert cp.camera_combo.itemData(0) == 5

    def test_populate_cameras_empty_shows_placeholder(self):
        cp = ControlPanel()
        cp.populate_cameras([])
        assert cp.camera_combo.count() == 1
        assert cp.camera_combo.itemText(0) == "No cameras found"
        assert cp.camera_combo.isEnabled() is False

    def test_populate_cameras_clears_previous(self):
        cp = ControlPanel()
        cp.populate_cameras([(0, "A"), (1, "B")])
        cp.populate_cameras([(3, "C")])
        assert cp.camera_combo.count() == 1
        assert cp.camera_combo.itemData(0) == 3

    def test_apply_preprocessing_params_sets_sliders(self):
        cp = ControlPanel()
        params = PreprocessParams(clahe_clip_limit=5.0, gamma=1.5, blur_threshold=200.0)
        cp.apply_preprocessing_params(params)
        assert abs(cp.clahe_slider.value() - 5.0) < 0.05
        assert abs(cp.gamma_slider.value() - 1.5) < 0.05
        assert abs(cp.blur_slider.value() - 200.0) < 0.5


# ===========================================================================
# StatusFooter — initialisation
# ===========================================================================


class TestStatusFooterInit:
    def test_fixed_height_is_48(self):
        footer = StatusFooter()
        assert (
            footer.height() == STATUS_FOOTER_HEIGHT
            or footer.minimumHeight() == STATUS_FOOTER_HEIGHT
        )

    def test_camera_indicator_starts_grey(self):
        footer = StatusFooter()
        assert footer._camera_indicator.dot_color == _DOT_COLOR_IDLE

    def test_model_indicator_starts_grey(self):
        footer = StatusFooter()
        assert footer._model_indicator.dot_color == _DOT_COLOR_IDLE

    def test_detection_indicator_starts_grey(self):
        footer = StatusFooter()
        assert footer._detection_indicator.dot_color == _DOT_COLOR_IDLE

    def test_fps_label_starts_with_dash(self):
        footer = StatusFooter()
        assert "—" in footer._fps_label.text() or "-" in footer._fps_label.text()


# ===========================================================================
# StatusFooter — camera status
# ===========================================================================


class TestStatusFooterCamera:
    def test_set_camera_active_sets_green_dot(self):
        footer = StatusFooter()
        footer.set_camera_active()
        assert footer._camera_indicator.dot_color == COLOR_STATUS_OK

    def test_set_camera_active_label_text(self):
        footer = StatusFooter()
        footer.set_camera_active()
        assert footer._camera_indicator.label_text == "Camera Active"

    def test_set_camera_error_sets_red_dot(self):
        footer = StatusFooter()
        footer.set_camera_error("USB unplugged")
        assert footer._camera_indicator.dot_color == COLOR_STATUS_ERROR

    def test_set_camera_error_label_text(self):
        footer = StatusFooter()
        footer.set_camera_error()
        assert footer._camera_indicator.label_text == "Camera Error"

    def test_set_camera_idle_sets_grey_dot(self):
        footer = StatusFooter()
        footer.set_camera_active()  # start active
        footer.set_camera_idle()  # go idle
        assert footer._camera_indicator.dot_color == _DOT_COLOR_IDLE

    def test_set_camera_idle_label_text(self):
        footer = StatusFooter()
        footer.set_camera_idle()
        assert footer._camera_indicator.label_text == "No Device"

    def test_camera_status_state_machine(self):
        footer = StatusFooter()
        footer.set_camera_active()
        assert footer._camera_indicator.dot_color == COLOR_STATUS_OK
        footer.set_camera_error()
        assert footer._camera_indicator.dot_color == COLOR_STATUS_ERROR
        footer.set_camera_idle()
        assert footer._camera_indicator.dot_color == _DOT_COLOR_IDLE


# ===========================================================================
# StatusFooter — model status
# ===========================================================================


class TestStatusFooterModel:
    def test_set_model_ready_sets_green_dot(self):
        footer = StatusFooter()
        footer.set_model_ready()
        assert footer._model_indicator.dot_color == COLOR_STATUS_OK

    def test_set_model_ready_label_text(self):
        footer = StatusFooter()
        footer.set_model_ready()
        assert footer._model_indicator.label_text == "Model Ready"

    def test_set_model_loading_sets_grey_dot(self):
        footer = StatusFooter()
        footer.set_model_loading()
        assert footer._model_indicator.dot_color == _DOT_COLOR_IDLE

    def test_set_model_loading_label_text(self):
        footer = StatusFooter()
        footer.set_model_loading()
        assert "Loading" in footer._model_indicator.label_text

    def test_set_model_error_sets_red_dot(self):
        footer = StatusFooter()
        footer.set_model_error("file not found")
        assert footer._model_indicator.dot_color == COLOR_STATUS_ERROR

    def test_set_model_error_label_text(self):
        footer = StatusFooter()
        footer.set_model_error()
        assert "Error" in footer._model_indicator.label_text

    def test_model_status_state_machine(self):
        footer = StatusFooter()
        footer.set_model_loading()
        assert footer._model_indicator.dot_color == _DOT_COLOR_IDLE
        footer.set_model_ready()
        assert footer._model_indicator.dot_color == COLOR_STATUS_OK
        footer.set_model_error()
        assert footer._model_indicator.dot_color == COLOR_STATUS_ERROR


# ===========================================================================
# StatusFooter — detection count
# ===========================================================================


class TestStatusFooterDetections:
    def test_zero_detections_green_dot(self):
        footer = StatusFooter()
        footer.set_detection_count(0)
        assert footer._detection_indicator.dot_color == COLOR_STATUS_OK

    def test_zero_detections_label_text(self):
        footer = StatusFooter()
        footer.set_detection_count(0)
        assert footer._detection_indicator.label_text == "No defects"

    def test_one_detection_singular(self):
        """Grammar check: 1 defect (not "1 defects")."""
        footer = StatusFooter()
        footer.set_detection_count(1)
        text = footer._detection_indicator.label_text
        assert "1" in text
        assert "defects" not in text or "defect" in text
        assert "defects" not in text  # strictly singular

    def test_two_detections_plural(self):
        footer = StatusFooter()
        footer.set_detection_count(2)
        text = footer._detection_indicator.label_text
        assert "2" in text
        assert "defects" in text

    def test_no_data_grey_dot(self):
        footer = StatusFooter()
        footer.set_detection_count(-1)
        assert footer._detection_indicator.dot_color == _DOT_COLOR_IDLE

    def test_no_data_label_text(self):
        footer = StatusFooter()
        footer.set_detection_count(-1)
        assert footer._detection_indicator.label_text == "No data"

    def test_detection_dot_not_amber(self):
        """
        Amber tier rule: detection dots must NOT use COLOR_STATUS_WARN (#FFC107).
        That colour is reserved exclusively for FR15 WarningBanner.
        """
        footer = StatusFooter()
        footer.set_detection_count(5)
        assert footer._detection_indicator.dot_color != COLOR_STATUS_WARN


# ===========================================================================
# StatusFooter — FPS counter
# ===========================================================================


class TestStatusFooterFps:
    def test_set_fps_positive_value(self):
        footer = StatusFooter()
        footer.set_fps(15.0)
        assert "15" in footer._fps_label.text()
        assert "fps" in footer._fps_label.text()

    def test_set_fps_zero(self):
        footer = StatusFooter()
        footer.set_fps(0.0)
        assert "0" in footer._fps_label.text()

    def test_set_fps_negative_shows_dash(self):
        footer = StatusFooter()
        footer.set_fps(-1.0)
        text = footer._fps_label.text()
        assert "—" in text or "-" in text

    def test_set_fps_rounds_to_int(self):
        """FPS is displayed as integer (15 not 15.3) per UX spec."""
        footer = StatusFooter()
        footer.set_fps(15.7)
        # "16 fps" or "15 fps" depending on rounding — just not "15.7"
        assert "15.7" not in footer._fps_label.text()
        assert "fps" in footer._fps_label.text()

    def test_fps_label_uses_mono_font(self):
        """Live-updating FPS readout must use FONT_MONO (anti-jitter rule)."""
        footer = StatusFooter()
        font: QFont = footer._fps_label.font()
        families = font.families()
        assert any(FONT_MONO in f for f in families), (
            f"FPS label font {families} must include FONT_MONO='{FONT_MONO}'"
        )

    def test_fps_label_has_minimum_width(self):
        """Fixed minimum width prevents layout reflow as digit count changes."""
        footer = StatusFooter()
        assert footer._fps_label.minimumWidth() > 0
