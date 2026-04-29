"""
ui/theme.py
-----------
Single source of truth for all CIRCA design tokens.
Refactored for 2026 Modern AI Studio Aesthetics with SVG Support.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Dict, Any

# ===========================================================================
# Theme & Animation Enums
# ===========================================================================

class ThemeMode(Enum):
    SYSTEM = auto()
    DARK = auto()
    LIGHT = auto()

class AnimationMode(Enum):
    SNAPPY = auto()
    FLUID = auto()

# ===========================================================================
# Design Tokens - Shared
# ===========================================================================

COLOR_ACCENT_CYAN = "#00BCD4"
COLOR_STATUS_OK = "#4CAF50"
COLOR_STATUS_WARN = "#FFC107"
COLOR_STATUS_ERROR = "#F44336"

DEFECT_CLASS_COLORS: dict[str, str] = {
    "missing_hole": "#FF5252",
    "mouse_bite": "#FF9800",
    "open_circuit": "#FFEB3B",
    "short": "#00BCD4",
    "spur": "#4CAF50",
    "spurious_copper": "#9C27B0",
}

FONT_UI = "Inter"
FONT_UI_FALLBACKS = ["Segoe UI", "Arial"]
FONT_MONO = "JetBrains Mono"
FONT_MONO_FALLBACKS = ["Roboto Mono", "Consolas", "Courier New"]

FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_BODY = 11
FONT_SIZE_STATUS = 11
FONT_SIZE_MONO_LIVE = 12
FONT_SIZE_MONO_CHIP = 11

SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 16
SPACING_LG = 24
SPACING_XL = 32

WINDOW_MIN_WIDTH = 1024
WINDOW_MIN_HEIGHT = 600
WINDOW_DEFAULT_WIDTH = 1280
WINDOW_DEFAULT_HEIGHT = 800

ACTIVITY_BAR_WIDTH = 48
SIDE_PANEL_WIDTH_EXPANDED = 280

WARNING_BANNER_HEIGHT = 28
STATUS_FOOTER_HEIGHT = 32
TITLE_BAR_HEIGHT = 32
STATUS_DOT_SIZE = 6

# Lucide Icons (Full SVG XML)
_SVG_TEMPLATE = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'

ICONS_SVG = {
    "settings": _SVG_TEMPLATE.format(color="{color}", path='<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>'),
    "close": _SVG_TEMPLATE.format(color="{color}", path='<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'),
    "camera": _SVG_TEMPLATE.format(color="{color}", path='<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>'),
    "activity": _SVG_TEMPLATE.format(color="{color}", path='<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>'),
    "contrast": _SVG_TEMPLATE.format(color="{color}", path='<circle cx="12" cy="12" r="10"/><path d="M12 18a6 6 0 0 0 0-12v12z"/>'),
    "sun": _SVG_TEMPLATE.format(color="{color}", path='<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>'),
    "brain": _SVG_TEMPLATE.format(color="{color}", path='<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .52 5.862 3 3 0 1 0 5.182 2.227h1.642a3 3 0 1 0 5.182-2.227 4 4 0 0 0 .52-5.862 4 4 0 0 0-2.526-5.77A3 3 0 1 0 12 5z"/><path d="M9 13a4.5 4.5 0 0 0 3-4"/><path d="M15 13a4.5 4.5 0 0 1-3-4"/><path d="M12 13V8"/>'),
    "focus": _SVG_TEMPLATE.format(color="{color}", path='<circle cx="12" cy="12" r="3"/><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'),
    "sidebar": _SVG_TEMPLATE.format(color="{color}", path='<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/>'),
}

# ===========================================================================
# Palettes
# ===========================================================================

DARK_PALETTE = {
    "BG_BASE": "#0F0F10",
    "BG_SURFACE": "#1C1C1E",
    "BG_ELEVATED": "#2C2C2E",
    "BORDER": "#3A3A3C",
    "TEXT_PRIMARY": "#FFFFFF",
    "TEXT_SECONDARY": "#A1A1A6",
    "TEXT_DISABLED": "#48484A",
    "GLASS_BG": "rgba(28, 28, 30, 0.88)",
}

LIGHT_PALETTE = {
    "BG_BASE": "#F2F2F7",
    "BG_SURFACE": "#FFFFFF",
    "BG_ELEVATED": "#E5E5EA",
    "BORDER": "#D1D1D6",
    "TEXT_PRIMARY": "#000000",
    "TEXT_SECONDARY": "#636366",
    "TEXT_DISABLED": "#C7C7CC",
    "GLASS_BG": "rgba(255, 255, 255, 0.94)",
}

# ===========================================================================
# Animation Easing
# ===========================================================================

ANIMATION_CONFIGS = {
    AnimationMode.SNAPPY: {
        "duration": 200,
        "easing": "InOutQuad"
    },
    AnimationMode.FLUID: {
        "duration": 600,
        "easing": "OutExpo"
    }
}

# ===========================================================================
# Rendering Shared
# ===========================================================================

BBOX_INNER_WIDTH = 2
BBOX_OUTER_WIDTH = 4
BBOX_OUTER_ALPHA = 179
CHIP_PADDING_H = 6
CHIP_PADDING_V = 3
DEFECT_CHIP_ALPHA = 204
DEFECT_CHIP_TEXT_COLOR = "#FFFFFF"
DEFECT_CLASS_COLOR_FALLBACK = COLOR_ACCENT_CYAN

# Legacy Layer
COLOR_BG_BASE = DARK_PALETTE["BG_BASE"]
COLOR_BG_SURFACE = DARK_PALETTE["BG_SURFACE"]
COLOR_BG_ELEVATED = DARK_PALETTE["BG_ELEVATED"]
COLOR_BORDER = DARK_PALETTE["BORDER"]
COLOR_TEXT_PRIMARY = DARK_PALETTE["TEXT_PRIMARY"]
COLOR_TEXT_SECONDARY = DARK_PALETTE["TEXT_SECONDARY"]
COLOR_TEXT_DISABLED = DARK_PALETTE["TEXT_DISABLED"]

# ===========================================================================
# Theme Manager
# ===========================================================================

class ThemeManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ThemeManager, cls).__new__(cls)
            cls._instance._current_mode = ThemeMode.DARK
            cls._instance._animation_mode = AnimationMode.FLUID
        return cls._instance

    @property
    def mode(self) -> ThemeMode: return self._current_mode
    @mode.setter
    def mode(self, value: ThemeMode): self._current_mode = value
    @property
    def anim_mode(self) -> AnimationMode: return self._animation_mode
    @anim_mode.setter
    def anim_mode(self, value: AnimationMode): self._animation_mode = value
    def get_palette(self) -> Dict[str, str]:
        return LIGHT_PALETTE if self._current_mode == ThemeMode.LIGHT else DARK_PALETTE
    def get_anim_config(self) -> Dict[str, Any]:
        return ANIMATION_CONFIGS[self._animation_mode]

def build_qss(mode: ThemeMode = ThemeMode.DARK) -> str:
    p = LIGHT_PALETTE if mode == ThemeMode.LIGHT else DARK_PALETTE

    return f"""
    QMainWindow, QWidget {{
        background-color: {p['BG_BASE']};
        color: {p['TEXT_PRIMARY']};
        font-family: "{FONT_UI}", "Segoe UI", Arial;
        font-size: {FONT_SIZE_BODY}px;
    }}

    #ActivityBar {{
        background-color: {p['BG_SURFACE']};
        border-right: 1px solid {p['BORDER']};
        min-width: {ACTIVITY_BAR_WIDTH}px;
        max-width: {ACTIVITY_BAR_WIDTH}px;
    }}

    #SidePanel {{
        background-color: {p['BG_SURFACE']};
        border-right: 1px solid {p['BORDER']};
    }}

    #SidePanel QWidget {{
        background-color: transparent;
    }}

    #ActivityButton {{
        background-color: transparent;
        border: none;
        border-radius: 8px;
        padding: 8px;
        min-width: 40px;
        min-height: 40px;
    }}
    #ActivityButton:hover {{
        background-color: {p['BG_ELEVATED']};
    }}
    #ActivityButton[active="true"] {{
        border-left: 2px solid {COLOR_ACCENT_CYAN};
        border-radius: 0px;
    }}

    QTabWidget::pane {{
        border-top: 1px solid {p['BORDER']};
        background: transparent;
        margin-top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        padding: 12px 24px;
        color: {p['TEXT_SECONDARY']};
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        color: {COLOR_ACCENT_CYAN};
        border-bottom: 2px solid {COLOR_ACCENT_CYAN};
    }}

    QLabel {{ background: transparent; color: {p['TEXT_PRIMARY']}; }}
    QLabel[secondary="true"] {{ color: {p['TEXT_SECONDARY']}; }}

    QSlider::groove:horizontal {{ background: {p['BG_ELEVATED']}; height: 4px; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {p['TEXT_PRIMARY']};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{ background: {COLOR_ACCENT_CYAN}; border-radius: 2px; }}

    QComboBox {{
        background-color: {p['BG_ELEVATED']};
        border: 1px solid {p['BORDER']};
        border-radius: 8px;
        padding: 8px 12px;
        color: {p['TEXT_PRIMARY']};
    }}
    QComboBox::drop-down {{ border: none; }}

    QComboBox QAbstractItemView {{
        background-color: {p['BG_SURFACE']};
        border: 1px solid {p['BORDER']};
        selection-background-color: {COLOR_ACCENT_CYAN};
        selection-color: {p['TEXT_PRIMARY']};
        color: {p['TEXT_PRIMARY']};
        outline: none;
    }}

    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding-left: 10px;
        background-color: {p['BG_SURFACE']};
        color: {p['TEXT_PRIMARY']};
    }}

    QComboBox QAbstractItemView::item:selected {{
        background-color: {COLOR_ACCENT_CYAN};
        color: {p['TEXT_PRIMARY']};
    }}

    QPushButton {{
        background-color: {p['BG_ELEVATED']};
        border: 1px solid {p['BORDER']};
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background-color: {p['BORDER']}; }}

    #StatusFooter {{
        background-color: {p['BG_SURFACE']};
        border-top: 1px solid {p['BORDER']};
    }}
    """