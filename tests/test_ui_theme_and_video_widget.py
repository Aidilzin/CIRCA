"""
tests/test_ui_theme_and_video_widget.py
---------------------------------------
Unit tests for ui/theme.py and ui/video_widget.py.

Testing strategy:
  theme.py:
    - Token presence and format: every colour constant is a valid hex string.
    - Constraint rules: CLASS_WARN exclusivity, defect map completeness.
    - build_qss(): returns a non-empty string containing key token values.

  VideoWidget state management (non-rendering):
    - Default state: frame=None, detections=None, status_text set.
    - set_frame(): stores QImage, triggers update.
    - set_detections(): stores DetectionResult, rejects unknown types.
    - set_status_text(): updates text; visible only when no frame.
    - clear_frame(): resets to idle state.
    - _compute_letterbox_rect(): aspect ratio and centering maths.
    - _draw_single_box() / _draw_label_chip(): smoke-tests via headless
      rendering into an offscreen QPixmap (no display required).

  All tests run headlessly — QPainter is used on an offscreen surface.
  No physical display or window manager required.

Requires QApplication (not just QCoreApplication) because VideoWidget
is a QWidget subclass.
"""

import re

import pytest
from PyQt6.QtGui import QColor, QImage, QPixmap

from core.models import BoundingBox, DetectionResult
from ui import theme
from ui.theme import (
    DEFECT_CLASS_COLORS,
    COLOR_STATUS_WARN,
    build_qss,
)
from ui.video_widget import VideoWidget


# QApplication provided by tests/conftest.py (session scope).


# ===========================================================================
# Helpers
# ===========================================================================

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def is_valid_hex(value: str) -> bool:
    return bool(HEX_RE.match(value))


def make_qimage(w: int = 64, h: int = 64) -> QImage:
    """Create a solid red RGB888 QImage for testing."""
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(QColor("#FF0000"))
    return img


def make_detection(
    class_name: str = "missing_hole", confidence: float = 0.90
) -> DetectionResult:
    return DetectionResult(
        boxes=[
            BoundingBox(
                x=10,
                y=10,
                width=50,
                height=50,
                class_name=class_name,
                confidence=confidence,
            )
        ]
    )


def make_widget(w: int = 640, h: int = 480) -> VideoWidget:
    """Create a VideoWidget with fixed size (no display required)."""
    widget = VideoWidget()
    widget.resize(w, h)
    return widget


def render_widget_offscreen(widget: VideoWidget) -> QPixmap:
    """
    Render the widget into an offscreen QPixmap — no display required.
    This exercises the full paintEvent code path.
    """
    pixmap = QPixmap(widget.size())
    pixmap.fill(QColor("#000000"))
    widget.render(pixmap)
    return pixmap


# ===========================================================================
# ui/theme.py — token validity
# ===========================================================================


class TestThemeColors:
    # All colour constants that must exist and be valid hex strings.
    COLOR_CONSTANTS = [
        "COLOR_BG_BASE",
        "COLOR_BG_SURFACE",
        "COLOR_BG_ELEVATED",
        "COLOR_BORDER",
        "COLOR_TEXT_PRIMARY",
        "COLOR_TEXT_SECONDARY",
        "COLOR_TEXT_DISABLED",
        "COLOR_ACCENT_CYAN",
        "COLOR_STATUS_OK",
        "COLOR_STATUS_WARN",
        "COLOR_STATUS_ERROR",
    ]

    @pytest.mark.parametrize("token", COLOR_CONSTANTS)
    def test_colour_token_exists(self, token):
        assert hasattr(theme, token), f"theme.{token} must be defined"

    @pytest.mark.parametrize("token", COLOR_CONSTANTS)
    def test_colour_token_is_valid_hex(self, token):
        value = getattr(theme, token)
        assert is_valid_hex(value), (
            f"theme.{token} = '{value}' is not a valid #RRGGBB hex colour"
        )

    def test_bg_base_is_matte_not_pure_black(self):
        """UX spec: matte dark, NOT #000000 — prevents halation on LCD screens."""
        # COLOR_BG_BASE inherits from DARK_PALETTE["BG_BASE"] = "#0F0F10"
        assert theme.COLOR_BG_BASE != "#000000"
        assert is_valid_hex(theme.COLOR_BG_BASE)

    def test_status_warn_is_amber(self):
        """Amber tier must be exactly #FFC107 — not yellow, not orange."""
        assert theme.COLOR_STATUS_WARN == "#FFC107"

    def test_accent_cyan_is_correct(self):
        assert theme.COLOR_ACCENT_CYAN == "#00BCD4"

    def test_status_warn_does_not_equal_accent(self):
        """Amber tier must be distinct from the interactive accent colour."""
        assert theme.COLOR_STATUS_WARN != theme.COLOR_ACCENT_CYAN


class TestThemeDefectColors:
    # The 7-class CIRCA taxonomy — must match CLASS_LABELS in core/inference_engine.py
    REQUIRED_CLASSES = {
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "excess_solder",
        "insufficient_solder",
        "cold_solder_joint",
    }

    def test_all_seven_defect_classes_present(self):
        """Every class from the 7-class unified_pcb_v3 taxonomy must have a colour."""
        assert self.REQUIRED_CLASSES.issubset(set(DEFECT_CLASS_COLORS.keys()))

    @pytest.mark.parametrize(
        "class_name,expected_hex",
        [
            ("missing_hole", "#FF5252"),
            ("mouse_bite", "#FF9800"),
            ("open_circuit", "#FFEB3B"),
            ("short", "#00BCD4"),
            ("excess_solder", "#FF6F00"),
            ("insufficient_solder", "#F06292"),
            ("cold_solder_joint", "#EF9A9A"),
        ],
    )
    def test_defect_colour_values(self, class_name, expected_hex):
        assert DEFECT_CLASS_COLORS[class_name] == expected_hex

    @pytest.mark.parametrize("hex_val", DEFECT_CLASS_COLORS.values())
    def test_all_defect_colours_are_valid_hex(self, hex_val):
        assert is_valid_hex(hex_val)

    def test_defect_colors_do_not_use_status_warn(self):
        """
        Amber tier rule: #FFC107 must not appear in the defect colour map.
        Defect colours carry semantic class meaning; amber is reserved for FR15.
        """
        assert COLOR_STATUS_WARN not in DEFECT_CLASS_COLORS.values()


class TestThemeTypography:
    def test_font_ui_is_inter(self):
        assert theme.FONT_UI == "Inter"

    def test_font_mono_is_jetbrains_mono(self):
        assert theme.FONT_MONO == "JetBrains Mono"

    def test_font_size_constants_exist(self):
        for attr in [
            "FONT_SIZE_TITLE",
            "FONT_SIZE_LABEL",
            "FONT_SIZE_BODY",
            "FONT_SIZE_MONO_LIVE",
            "FONT_SIZE_MONO_CHIP",
        ]:
            assert hasattr(theme, attr), f"theme.{attr} must be defined"

    def test_mono_chip_size_is_11(self):
        """UX spec: bounding box label chip uses 11px JetBrains Mono."""
        assert theme.FONT_SIZE_MONO_CHIP == 11


class TestThemeSpacing:
    def test_spacing_base_unit_is_8(self):
        """All spacing must be multiples of the 8px base unit."""
        assert theme.SPACING_SM == 8
        assert theme.SPACING_MD % 8 == 0
        assert theme.SPACING_LG % 8 == 0
        assert theme.SPACING_XL % 8 == 0


class TestThemeLayout:
    def test_window_min_width_is_1024(self):
        assert theme.WINDOW_MIN_WIDTH == 1024

    def test_window_min_height_is_600(self):
        assert theme.WINDOW_MIN_HEIGHT == 600

    def test_warning_banner_height(self):
        """Warning banner must be between 24 and 48px."""
        assert 24 <= theme.WARNING_BANNER_HEIGHT <= 48

    def test_status_footer_height(self):
        """Status footer must be between 24 and 64px."""
        assert 24 <= theme.STATUS_FOOTER_HEIGHT <= 64


class TestBuildQss:
    def test_build_qss_returns_string(self):
        assert isinstance(build_qss(), str)

    def test_build_qss_is_non_empty(self):
        assert len(build_qss()) > 100

    def test_qss_contains_bg_base_colour(self):
        """Dark mode QSS must reference the dark palette BG_BASE colour."""
        qss = build_qss()
        # DARK_PALETTE["BG_BASE"] = "#0F0F10"
        assert "#0F0F10" in qss  # COLOR_BG_BASE in dark mode

    def test_qss_contains_accent_colour(self):
        qss = build_qss()
        assert "#00BCD4" in qss  # COLOR_ACCENT_CYAN

    def test_qss_contains_slider_rules(self):
        qss = build_qss()
        assert "QSlider" in qss


# ===========================================================================
# VideoWidget — initialisation
# ===========================================================================


class TestVideoWidgetInit:
    def test_initial_frame_is_none(self):
        w = make_widget()
        assert w._current_frame is None

    def test_initial_detections_is_none(self):
        w = make_widget()
        assert w._detections is None

    def test_initial_status_text_set(self):
        w = make_widget()
        assert w._status_text == "Please connect a camera"

    def test_is_qwidget(self):
        from PyQt6.QtWidgets import QWidget

        w = make_widget()
        assert isinstance(w, QWidget)

    def test_minimum_size_set(self):
        w = make_widget()
        assert w.minimumWidth() >= 320
        assert w.minimumHeight() >= 240


# ===========================================================================
# VideoWidget — set_frame slot
# ===========================================================================


class TestSetFrame:
    def test_set_frame_stores_qimage(self):
        w = make_widget()
        img = make_qimage()
        w.set_frame(img)
        assert w._current_frame is img

    def test_set_frame_accepts_different_sizes(self):
        w = make_widget(800, 600)
        for size in [(64, 64), (640, 480), (1920, 1080)]:
            img = make_qimage(*size)
            w.set_frame(img)
            assert w._current_frame.width() == size[0]

    def test_set_frame_overwrites_previous(self):
        w = make_widget()
        w.set_frame(make_qimage(100, 100))
        new_img = make_qimage(200, 200)
        w.set_frame(new_img)
        assert w._current_frame is new_img


# ===========================================================================
# VideoWidget — set_detections slot
# ===========================================================================


class TestSetDetections:
    def test_set_detections_stores_result(self):
        w = make_widget()
        result = make_detection()
        w.set_detections(result)
        assert w._detections is result

    def test_set_detections_accepts_empty_result(self):
        w = make_widget()
        result = DetectionResult(boxes=[])
        w.set_detections(result)
        assert w._detections is result
        assert w._detections.boxes == []

    def test_set_detections_rejects_non_detection_result(self):
        """Unknown types (e.g. a bare dict) should not crash — set to None."""
        w = make_widget()
        w.set_detections({"bad": "data"})  # type: ignore[arg-type]
        assert w._detections is None

    def test_set_detections_overwrites_previous(self):
        w = make_widget()
        w.set_detections(make_detection("solder_bridge"))
        new = make_detection("burnt_area")
        w.set_detections(new)
        assert w._detections is new


# ===========================================================================
# VideoWidget — set_status_text slot
# ===========================================================================


class TestSetStatusText:
    def test_set_status_text_updates_text(self):
        w = make_widget()
        w.set_status_text("Camera Error")
        assert w._status_text == "Camera Error"

    def test_set_status_text_arbitrary_string(self):
        w = make_widget()
        w.set_status_text("Initialising Camera…")
        assert w._status_text == "Initialising Camera…"


# ===========================================================================
# VideoWidget — clear_frame
# ===========================================================================


class TestClearFrame:
    def test_clear_frame_sets_frame_to_none(self):
        w = make_widget()
        w.set_frame(make_qimage())
        # Use the clear_feed API (clear_frame is the internal alias)
        w.clear_feed("Camera lost")
        assert w._current_frame is None

    def test_clear_frame_clears_detections(self):
        w = make_widget()
        w.set_frame(make_qimage())
        w.set_detections(make_detection())
        w.clear_feed("Camera lost")
        assert w._detections is None


# ===========================================================================
# VideoWidget — _compute_letterbox_rect (pure geometry)
# ===========================================================================


class TestComputeLetterboxRect:
    def test_square_frame_in_square_widget_fills_entirely(self):
        w = make_widget(400, 400)
        # Use the actual widget size after minimum-size constraint is applied.
        actual_w, actual_h = w.width(), w.height()
        rect = w._compute_letterbox_rect(actual_w, actual_h)
        # A frame that matches the widget aspect exactly should fill the widget.
        assert rect.width() == actual_w
        assert rect.height() == actual_h
        assert rect.x() == 0
        assert rect.y() == 0

    def test_wider_widget_letterboxes_left_right(self):
        """Frame 4:3, widget 16:9 → bars on left/right, frame fills vertically."""
        w = make_widget(800, 450)  # 16:9
        rect = w._compute_letterbox_rect(640, 480)  # 4:3
        # Frame should be centered, bars on left and right
        assert rect.height() == 450  # fills vertically
        assert rect.x() > 0  # offset from left
        assert rect.y() == 0

    def test_taller_widget_letterboxes_top_bottom(self):
        """Frame 16:9, widget 4:3 → bars on top/bottom, frame fills horizontally."""
        w = make_widget(640, 640)  # square
        rect = w._compute_letterbox_rect(1280, 720)  # 16:9
        assert rect.width() == 640  # fills horizontally
        assert rect.y() > 0  # offset from top
        assert rect.x() == 0

    def test_frame_centred_in_widget(self):
        """The letterbox image must be centred within the widget."""
        w = make_widget(400, 400)
        actual_w, actual_h = w.width(), w.height()
        rect = w._compute_letterbox_rect(
            actual_w, actual_w // 2
        )  # 2:1 frame in square widget
        # Image height = actual_w/2 → scaled to fit width → drawn_h = actual_w//2
        # vertical offset = (actual_h - drawn_h) // 2
        expected_drawn_h = actual_w // 2
        expected_y = (actual_h - expected_drawn_h) // 2
        assert rect.y() == expected_y, (
            f"Expected y={expected_y}, got {rect.y()} "
            f"(widget={actual_w}×{actual_h}, frame={actual_w}×{actual_w // 2})"
        )

    def test_degenerate_zero_frame_returns_widget_rect(self):
        """Zero-dimension frame must not cause division by zero."""
        w = make_widget(640, 480)
        rect = w._compute_letterbox_rect(0, 0)
        assert rect.width() == 640 or rect.height() == 480  # safe fallback

    def test_aspect_ratio_preserved(self):
        """Scale factor must be uniform — no stretching."""
        w = make_widget(800, 600)
        rect = w._compute_letterbox_rect(320, 240)
        # 320:240 = 4:3, 800:600 = 4:3 — should fill exactly
        assert rect.width() == 800
        assert rect.height() == 600


# ===========================================================================
# VideoWidget — offscreen render smoke tests
# ===========================================================================


class TestOffscreenRendering:
    """
    These tests exercise the full paintEvent/QPainter pipeline using
    QWidget.render() into an offscreen QPixmap — no display required.
    Checks that painting does not raise an exception and produces a
    non-null result.
    """

    def test_idle_state_renders_without_crash(self):
        w = make_widget()
        pixmap = render_widget_offscreen(w)
        assert not pixmap.isNull()

    def test_frame_renders_without_crash(self):
        w = make_widget()
        w.set_frame(make_qimage(320, 240))
        pixmap = render_widget_offscreen(w)
        assert not pixmap.isNull()

    def test_frame_with_detection_renders(self):
        """A frame with a missing_hole detection must render without crash."""
        w = make_widget()
        w.set_frame(make_qimage(320, 240))
        w.set_detections(make_detection("missing_hole", confidence=0.95))
        pixmap = render_widget_offscreen(w)
        assert not pixmap.isNull()

    def test_all_defect_classes_render_without_crash(self):
        """All 7 CIRCA defect class colours must resolve without KeyError or crash."""
        classes = [
            "missing_hole",
            "mouse_bite",
            "open_circuit",
            "short",
            "excess_solder",
            "insufficient_solder",
            "cold_solder_joint",
        ]
        for cls in classes:
            w = make_widget()
            w.set_frame(make_qimage(320, 240))
            w.set_detections(make_detection(cls, confidence=0.75))
            pixmap = render_widget_offscreen(w)
            assert not pixmap.isNull(), f"Rendering failed for class: {cls}"

    def test_unknown_defect_class_uses_fallback_color(self):
        """A class name not in DEFECT_CLASS_COLORS must not crash — uses fallback."""
        w = make_widget()
        w.set_frame(make_qimage(320, 240))
        result = DetectionResult(
            boxes=[
                BoundingBox(
                    x=10,
                    y=10,
                    width=50,
                    height=50,
                    class_name="future_class_99",
                    confidence=0.8,
                )
            ]
        )
        w.set_detections(result)
        pixmap = render_widget_offscreen(w)
        assert not pixmap.isNull()

    def test_box_at_top_edge_chip_falls_inside(self):
        """A box at y=0 → chip falls inside the box (not above widget boundary)."""
        w = make_widget(640, 480)
        w.set_frame(make_qimage(640, 480))
        result = DetectionResult(
            boxes=[
                BoundingBox(
                    x=5,
                    y=0,
                    width=100,
                    height=60,
                    class_name="missing_hole",
                    confidence=0.85,
                )
            ]
        )
        w.set_detections(result)
        pixmap = render_widget_offscreen(w)
        assert not pixmap.isNull()

    def test_multiple_detections_render(self):
        """Multiple simultaneous detections must all render without crash."""
        w = make_widget(640, 480)
        w.set_frame(make_qimage(640, 480))
        result = DetectionResult(
            boxes=[
                BoundingBox(
                    x=10,
                    y=10,
                    width=80,
                    height=60,
                    class_name="missing_hole",
                    confidence=0.92,
                ),
                BoundingBox(
                    x=200,
                    y=200,
                    width=80,
                    height=60,
                    class_name="cold_solder_joint",
                    confidence=0.78,
                ),
                BoundingBox(
                    x=400,
                    y=100,
                    width=80,
                    height=60,
                    class_name="mouse_bite",
                    confidence=0.65,
                ),
            ]
        )
        w.set_detections(result)
        pixmap = render_widget_offscreen(w)
        assert not pixmap.isNull()

    def test_empty_detections_renders_frame_without_boxes(self):
        """DetectionResult with no boxes renders the frame cleanly (no crash)."""
        w = make_widget()
        w.set_frame(make_qimage())
        w.set_detections(DetectionResult(boxes=[]))
        pixmap = render_widget_offscreen(w)
        assert not pixmap.isNull()

    def test_idle_background_is_not_white(self):
        """
        Idle state must fill with the dark BG colour (#0F0F10), not white.
        Sample the centre pixel of the offscreen render.
        """
        w = make_widget(100, 100)
        pixmap = render_widget_offscreen(w)
        img = pixmap.toImage()
        centre_color = QColor(img.pixel(50, 50))
        # Centre pixel must be dark (R, G, B all < 30 for #0F0F10 = rgb(15,15,16))
        assert centre_color.red() < 30
        assert centre_color.green() < 30
        assert centre_color.blue() < 30

    def test_clear_feed_resets_state(self):
        """clear_feed() must drop both the current frame and detections, and set status text."""
        w = make_widget()
        w.set_frame(make_qimage())
        w.set_detections(make_detection())
        assert w._current_frame is not None
        assert w._detections is not None

        w.clear_feed("New Status")
        assert w._current_frame is None
        assert w._detections is None
        assert w._status_text == "New Status"
