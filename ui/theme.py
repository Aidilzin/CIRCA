"""
ui/theme.py
-----------
Single source of truth for all CIRCA design tokens.

Usage rule: Every colour, font, spacing value, and layout constant used
anywhere in the ui/ layer MUST be referenced from this file. Never hardcode
hex strings, font names, or pixel values inside widget files.

This file has NO PyQt6 imports — it is pure Python constants so it can be
imported by unit tests and by non-Qt tooling without a display server.

UX Spec reference: §Visual Design Foundation (colour palette, typography,
spacing, bounding box specification, layout constraints).
"""

from __future__ import annotations

# ===========================================================================
# Colour Palette
# UX Spec: §Colour Palette Specification
# ===========================================================================

# Background layers (dark theme hierarchy)
COLOR_BG_BASE     = "#121212"   # Main window BG, viewport letterbox fill
COLOR_BG_SURFACE  = "#1E1E1E"   # Control panel BG, card surfaces
COLOR_BG_ELEVATED = "#2A2A2A"   # Slider tracks, input field BG, hover states
COLOR_BORDER      = "#3A3A3A"   # Panel dividers, slider borders

# Text tiers
COLOR_TEXT_PRIMARY   = "#E8E8E8"  # All primary labels, headings
COLOR_TEXT_SECONDARY = "#9E9E9E"  # Slider sub-labels, status descriptions
COLOR_TEXT_DISABLED  = "#4A4A4A"  # Inactive controls (not read-critical)

# Interactive accent
COLOR_ACCENT_CYAN = "#00BCD4"   # Slider thumb, active state, panel toggle

# Status tier — each maps to exactly one UI state
COLOR_STATUS_OK    = "#4CAF50"  # Camera Active, Model Ready (green dot)
COLOR_STATUS_WARN  = "#FFC107"  # Low-confidence warning (AMBER TIER — EXCLUSIVE)
COLOR_STATUS_ERROR = "#F44336"  # Camera/inference failure (red dot)

# Amber tier rule (enforced at review level):
# COLOR_STATUS_WARN is reserved EXCLUSIVELY for the FR15 low-confidence
# warning state. It must not appear anywhere else — not as an accent,
# not as a hover state, not as a decorative element.

# ===========================================================================
# Defect Class Colour Map
# UX Spec: §Bounding Box Visual Specification — Defect Class Colour Map
# These colours are used for both the inner stroke and the label chip BG.
# ===========================================================================

DEFECT_CLASS_COLORS: dict[str, str] = {
    "solder_bridge":        "#FF5252",   # Red
    "missing_component":    "#FF9800",   # Orange
    "misaligned_component": "#FFEB3B",   # Yellow
    "burnt_area":           "#9C27B0",   # Purple
}

# Fallback colour for any class not in DEFECT_CLASS_COLORS
# (e.g. a future model re-trained with additional classes)
DEFECT_CLASS_COLOR_FALLBACK = COLOR_ACCENT_CYAN

# Label chip text colour — always light on the semi-opaque chip background
DEFECT_CHIP_TEXT_COLOR = COLOR_TEXT_PRIMARY   # "#E8E8E8"

# Label chip background opacity (80% = 204/255)
DEFECT_CHIP_ALPHA = 204

# ===========================================================================
# Bounding Box Stroke Specification
# UX Spec: §Bounding Box Visual Specification — Dual-stroke design
# ===========================================================================

# Outer shadow stroke — universally visible against any PCB substrate colour
BBOX_OUTER_WIDTH   = 4    # px — wider outer draw that bleeds around inner line
BBOX_OUTER_ALPHA   = 179  # 70% of 255 — semi-transparent black shadow

# Inner signal stroke — carries semantic colour coding per defect class
BBOX_INNER_WIDTH   = 2    # px — thinner coloured inner stroke

# ===========================================================================
# Typography
# UX Spec: §Typography Specification
# ===========================================================================

# UI font — Inter for all labels, headings, panel text
FONT_UI   = "Inter"           # Fallback: "Segoe UI"
FONT_UI_FALLBACKS = ["Segoe UI", "Arial"]

# Monospace font — JetBrains Mono for ALL live-updating numerical values
FONT_MONO = "JetBrains Mono"  # Fallback: "Roboto Mono", "Consolas"
FONT_MONO_FALLBACKS = ["Roboto Mono", "Consolas", "Courier New"]

# Typography scale (size in px)
FONT_SIZE_TITLE   = 14   # Inter 600 — window title, section headings
FONT_SIZE_LABEL   = 12   # Inter 500 — slider labels, control group headers
FONT_SIZE_BODY    = 11   # Inter 400 — status descriptions, helper text
FONT_SIZE_STATUS  = 11   # Inter 500 — status dot labels
FONT_SIZE_MONO_LIVE  = 12  # JetBrains Mono 400 — confidence scores, FPS, latency
FONT_SIZE_MONO_CHIP  = 11  # JetBrains Mono 500 — bounding box label chips

# ===========================================================================
# Spacing System (base unit: 8px)
# UX Spec: §Spacing & Layout Foundation
# ===========================================================================

SPACING_XS = 4    # px — icon-to-text gap, status dot margin
SPACING_SM = 8    # px — intra-group padding (label to slider)
SPACING_MD = 16   # px — between control groups within panel
SPACING_LG = 24   # px — panel section dividers
SPACING_XL = 32   # px — major layout region separation

# ===========================================================================
# Layout Constants
# UX Spec: §Spacing & Layout Foundation, §Window Constraint Strategy
# ===========================================================================

WINDOW_MIN_WIDTH      = 1024  # px — below this, detection chips become unreadable
WINDOW_MIN_HEIGHT     = 600   # px — below this, footer + banner occlude viewport
WINDOW_DEFAULT_WIDTH  = 1280  # px — safe for 1366×768 monitors with taskbar
WINDOW_DEFAULT_HEIGHT = 800   # px

CONTROL_PANEL_WIDTH_EXPANDED  = 280  # px — all controls visible
CONTROL_PANEL_WIDTH_COLLAPSED = 28   # px — toggle arrow only

WARNING_BANNER_HEIGHT = 32   # px — FR15 amber advisory bar above viewport
STATUS_FOOTER_HEIGHT  = 48   # px — persistent system health bar
TITLE_BAR_HEIGHT      = 36   # px — application title bar

# Status dot dimensions (8px filled circle for ● indicators)
STATUS_DOT_SIZE = 8   # px — QLabel with border-radius: 4px QSS

# Label chip padding inside bounding box chips
CHIP_PADDING_H = 6   # px — horizontal padding left/right of chip text
CHIP_PADDING_V = 3   # px — vertical padding above/below chip text

# ===========================================================================
# QSS Stylesheet helper — full application dark theme
# Applied once in main.py via QApplication.setStyleSheet(build_qss())
# ===========================================================================

def build_qss() -> str:
    """
    Return the full application QSS stylesheet as a string.

    Applied once at startup in main.py:
        app.setStyleSheet(build_qss())

    All colour references use variables from this module so the theme is
    consistent and can be updated from a single location.
    """
    return f"""
    /* ── Global ── */
    QMainWindow, QWidget {{
        background-color: {COLOR_BG_BASE};
        color: {COLOR_TEXT_PRIMARY};
        font-family: "{FONT_UI}", "Segoe UI", Arial;
        font-size: {FONT_SIZE_BODY}px;
    }}

    /* ── Control panel surface ── */
    #ControlPanel {{
        background-color: {COLOR_BG_SURFACE};
        border-left: 1px solid {COLOR_BORDER};
    }}

    /* ── Section dividers ── */
    QFrame[frameShape="4"],
    QFrame[frameShape="5"] {{
        color: {COLOR_BORDER};
        background-color: {COLOR_BORDER};
        border: none;
        max-height: 1px;
    }}

    /* ── Labels ── */
    QLabel {{
        background: transparent;
        color: {COLOR_TEXT_PRIMARY};
    }}
    QLabel[secondary="true"] {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: {FONT_SIZE_BODY}px;
    }}

    /* ── Sliders ── */
    QSlider::groove:horizontal {{
        background: {COLOR_BG_ELEVATED};
        height: 4px;
        border-radius: 2px;
        border: 1px solid {COLOR_BORDER};
    }}
    QSlider::handle:horizontal {{
        background: {COLOR_ACCENT_CYAN};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: #26C6DA;
    }}
    QSlider::handle:horizontal:focus {{
        border: 2px solid {COLOR_ACCENT_CYAN};
        outline: 2px solid rgba(0, 188, 212, 0.4);
    }}
    QSlider::sub-page:horizontal {{
        background: {COLOR_ACCENT_CYAN};
        border-radius: 2px;
    }}

    /* ── Dropdowns ── */
    QComboBox {{
        background-color: {COLOR_BG_ELEVATED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        color: {COLOR_TEXT_PRIMARY};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLOR_BG_ELEVATED};
        border: 1px solid {COLOR_BORDER};
        selection-background-color: {COLOR_ACCENT_CYAN};
        selection-color: {COLOR_BG_BASE};
    }}

    /* ── Buttons ── */
    QPushButton {{
        background-color: {COLOR_BG_ELEVATED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        padding: 4px 12px;
        color: {COLOR_TEXT_PRIMARY};
    }}
    QPushButton:hover {{
        background-color: {COLOR_BORDER};
    }}
    QPushButton:focus {{
        border: 1px solid {COLOR_ACCENT_CYAN};
    }}

    /* ── Status footer ── */
    #StatusFooter {{
        background-color: {COLOR_BG_SURFACE};
        border-top: 1px solid {COLOR_BORDER};
    }}
    """
