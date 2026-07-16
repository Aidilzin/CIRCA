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

COLOR_ACCENT_CYAN = "#00C7FF"
COLOR_ACCENT_CYAN_HOVER = "#33D2FF"
COLOR_ACCENT_CYAN_PRESSED = "#009ECB"
COLOR_ACCENT_BLUE = "#0068B5"
COLOR_STATUS_OK = "#00FF9D"
COLOR_STATUS_WARN = "#FFD700"
COLOR_STATUS_ERROR = "#FF5050"

# Colour map for unified_pcb_v3 — nc=7 (must match CLASS_LABELS in core/inference_engine.py).
# IDs are the unified taxonomy indices; update here and inference_engine.py in sync.
DEFECT_CLASS_COLORS: dict[str, str] = {
    # IPC-A-600 bare-board defects (classes 0–3)
    "missing_hole": "#FF5252",
    "mouse_bite": "#FF9800",
    "open_circuit": "#FFEB3B",
    "short": "#00BCD4",
    # IPC-A-610H assembly-stage solder defects (classes 4–6)
    "excess_solder": "#FF6F00",
    "insufficient_solder": "#F06292",
    "cold_solder_joint": "#EF9A9A",
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
    "sliders": _SVG_TEMPLATE.format(color="{color}", path='<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="2" y1="14" x2="6" y2="14"/><line x1="10" y1="8" x2="14" y2="8"/><line x1="18" y1="16" x2="22" y2="16"/>'),
    "refresh": _SVG_TEMPLATE.format(color="{color}", path='<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8M16 3h5v5M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16M8 21H3v-5"/>'),
    "export": _SVG_TEMPLATE.format(color="{color}", path='<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>'),
    "log": _SVG_TEMPLATE.format(color="{color}", path='<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>'),
    "info": _SVG_TEMPLATE.format(color="{color}", path='<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'),
    "cpu": _SVG_TEMPLATE.format(color="{color}", path='<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 15h3"/><path d="M1 9h3"/><path d="M1 15h3"/>'),
    "terminal": _SVG_TEMPLATE.format(color="{color}", path='<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>'),
    "check": _SVG_TEMPLATE.format(color="{color}", path='<polyline points="20 6 9 17 4 12"/>'),
    "upload": _SVG_TEMPLATE.format(color="{color}", path='<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>'),
    "image": _SVG_TEMPLATE.format(color="{color}", path='<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>'),
    "alert": _SVG_TEMPLATE.format(color="{color}", path='<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
    "folder_open": _SVG_TEMPLATE.format(color="{color}", path='<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>'),
    "arrow_left": _SVG_TEMPLATE.format(color="{color}", path='<path d="M19 12H5"/><polyline points="12 19 5 12 12 5"/>'),
    "arrow_right": _SVG_TEMPLATE.format(color="{color}", path='<path d="M5 12h14"/><polyline points="12 5 19 12 12 19"/>'),
}

# ===========================================================================
# Palettes
# ===========================================================================

DARK_PALETTE = {
    "BG_BASE": "#0B0F19",
    "BG_SURFACE": "#131B2E",
    "BG_ELEVATED": "#1C2740",
    "BORDER": "rgba(0, 199, 255, 0.15)",
    "TEXT_PRIMARY": "#E2E8F0",
    "TEXT_SECONDARY": "#94A3B8",
    "TEXT_DISABLED": "#475569",
    "GLASS_BG": "rgba(11, 15, 25, 0.85)",
}

LIGHT_PALETTE = {
    "BG_BASE": "#F4F4F5",
    "BG_SURFACE": "#FFFFFF",
    "BG_ELEVATED": "#E4E4E7",
    "BORDER": "#D4D4D8",
    "TEXT_PRIMARY": "#09090B",
    "TEXT_SECONDARY": "#61616A",
    "TEXT_DISABLED": "#A1A1AA",
    "GLASS_BG": "rgba(255, 255, 255, 0.90)",
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
        color: {p['TEXT_PRIMARY']};
    }}

    #SidePanel QTabWidget::pane {{
        border-top: 1px solid {p['BORDER']};
        background-color: {p['BG_SURFACE']};
        background: {p['BG_SURFACE']};
    }}

    #SidePanel QTabWidget > QWidget {{
        background-color: {p['BG_SURFACE']};
    }}

    #SidePanel QTabBar {{
        background-color: {p['BG_SURFACE']};
        border: none;
    }}

    #SidePanel QTabBar::tab {{
        background-color: {p['BG_SURFACE']};
        border: none;
        padding: 8px 16px;
        color: {p['TEXT_SECONDARY']};
        font-weight: bold;
        font-size: 11px;
        letter-spacing: 0.5px;
        border-bottom: 2px solid transparent;
    }}
    #SidePanel QTabBar::tab:selected {{
        background-color: {p['BG_SURFACE']};
        color: {COLOR_ACCENT_CYAN};
        border-bottom: 2px solid {COLOR_ACCENT_CYAN};
    }}

    #SidePanel QLabel[secondary="true"] {{
        border-top: 1px solid {p['BORDER']};
        padding-top: 8px;
        margin-top: 6px;
        color: {p['TEXT_SECONDARY']};
        font-size: 10px;
        font-weight: bold;
        letter-spacing: 0.5px;
    }}

    #SidePanel QComboBox {{
        background-color: {p['BG_ELEVATED']};
        border: 1px solid {p['BORDER']};
        border-radius: 4px;
        padding: 6px 12px;
        color: {p['TEXT_PRIMARY']};
    }}

    #SidePanel QComboBox::drop-down {{
        border: none;
        background: transparent;
    }}

    #SidePanel QPushButton {{
        background-color: {p['BG_ELEVATED']};
        border: 1px solid {p['BORDER']};
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 600;
        color: {p['TEXT_PRIMARY']};
    }}
    #SidePanel QPushButton:hover {{
        background-color: {p['BORDER']};
        border-color: {COLOR_ACCENT_CYAN};
    }}
    #SidePanel QPushButton#PrimaryButton {{
        background-color: {COLOR_ACCENT_CYAN};
        color: {p['BG_BASE']};
        border: none;
        border-radius: 4px;
    }}
    #SidePanel QPushButton#PrimaryButton:hover {{
        background-color: {COLOR_ACCENT_CYAN_HOVER};
    }}

    #ActivityButton {{
        background-color: transparent;
        border: none;
        border-radius: 4px;
        margin: 4px 4px;
        padding: 8px;
        min-width: 40px;
        min-height: 40px;
    }}
    #ActivityButton:hover {{
        background-color: {p['BG_ELEVATED']};
    }}
    #ActivityButton[active="true"] {{
        background-color: {p['BG_ELEVATED']};
        border-left: 3px solid {COLOR_ACCENT_CYAN};
        border-top-left-radius: 0px;
        border-bottom-left-radius: 0px;
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
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

    QSlider::groove:horizontal {{
        background: {p['BG_ELEVATED']};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {COLOR_ACCENT_CYAN};
        width: 14px;
        height: 14px;
        margin: -4px 0;
        border-radius: 7px;
        border: 2px solid {p['BG_SURFACE']};
    }}
    QSlider::handle:horizontal:hover {{
        background: {p['TEXT_PRIMARY']};
    }}
    QSlider::sub-page:horizontal {{
        background: {COLOR_ACCENT_BLUE};
        border-radius: 3px;
    }}

    QComboBox {{
        background-color: {p['BG_ELEVATED']};
        border: 1px solid {p['BORDER']};
        border-radius: 4px;
        padding: 8px 12px;
        color: {p['TEXT_PRIMARY']};
    }}
    QComboBox:hover {{
        border-color: {COLOR_ACCENT_CYAN};
    }}
    QComboBox::drop-down {{
        border: none;
    }}

    QComboBox QAbstractItemView {{
        background-color: {p['BG_SURFACE']};
        border: 1px solid {p['BORDER']};
        selection-background-color: {COLOR_ACCENT_CYAN};
        selection-color: {p['BG_BASE']};
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
        color: {p['BG_BASE']};
    }}

    QPushButton {{
        background-color: {p['BG_ELEVATED']};
        border: 1px solid {p['BORDER']};
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 600;
        color: {p['TEXT_PRIMARY']};
    }}
    QPushButton:hover {{
        background-color: {p['BORDER']};
        border-color: {COLOR_ACCENT_CYAN};
    }}
    QPushButton:pressed {{
        background-color: {p['BG_SURFACE']};
    }}

    QPushButton#PrimaryButton {{
        background-color: {COLOR_ACCENT_CYAN};
        color: {p['BG_BASE']};
        border: none;
        border-radius: 4px;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {COLOR_ACCENT_CYAN_HOVER};
    }}
    QPushButton#PrimaryButton:pressed {{
        background-color: {COLOR_ACCENT_CYAN_PRESSED};
    }}

    QPushButton:focus, QComboBox:focus, QCheckBox:focus, QSlider:focus {{
        border: 1px solid {COLOR_ACCENT_CYAN};
    }}

    #StatusFooter {{
        background-color: {p['BG_SURFACE']};
        border-top: 1px solid {p['BORDER']};
    }}

    #StatusIndicator {{
        background-color: {p['BG_ELEVATED']};
        border: 1px solid {p['BORDER']};
        border-radius: 12px;
    }}

    #FPSLabel {{
        color: {COLOR_ACCENT_CYAN};
        font-weight: bold;
    }}

    QCheckBox {{
        spacing: 8px;
        color: {p['TEXT_PRIMARY']};
    }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border: 1.5px solid {p['BORDER']};
        border-radius: 2px;
        background-color: {p['BG_ELEVATED']};
    }}
    QCheckBox::indicator:hover {{
        border-color: {COLOR_ACCENT_CYAN};
    }}
    QCheckBox::indicator:checked {{
        background-color: {COLOR_ACCENT_CYAN};
        border: 3px solid {p['BG_SURFACE']};
    }}

    QScrollBar:vertical {{
        border: none;
        background: transparent;
        width: 6px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {p['BORDER']};
        min-height: 20px;
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {p['TEXT_SECONDARY']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        border: none;
        background: transparent;
        height: 6px;
        margin: 0px;
    }}
    QScrollBar::handle:horizontal {{
        background: {p['BORDER']};
        min-width: 20px;
        border-radius: 3px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {p['TEXT_SECONDARY']};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    #AnalyticsCard {{
        background-color: {p['BG_SURFACE']};
        border: 1px solid {p['BORDER']};
        border-radius: 6px;
    }}

    /* Left Navigation Sidebar */
    #NavSidebar {{
        background-color: #070B0E;
        border-right: 1px solid rgba(0, 199, 255, 0.15);
    }}
    
    #NavSidebar QLabel#NavLogo {{
        color: {p['TEXT_PRIMARY']};
        margin: 4px 0px;
        font-weight: 700;
        font-size: 11px;
        letter-spacing: 0.5px;
        text-align: center;
    }}
    
    #NavSidebar QLabel#NavStatus {{
        color: {COLOR_STATUS_OK};
        margin: 0px 0px 10px 0px;
        font-weight: 600;
        font-size: 8px;
        text-align: center;
    }}
    
    QFrame#NavButton {{
        background: transparent;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 4px;
        margin: 0px 2px;
    }}
    
    QFrame#NavButton:hover {{
        background-color: rgba(255, 255, 255, 0.03);
    }}
    
    QFrame#NavButton[active="true"] {{
        border-left: 3px solid #00D2FF;
        background-color: rgba(0, 210, 255, 0.08);
    }}
    
    QLabel#NavButtonText {{
        color: {p['TEXT_SECONDARY']};
        font-size: 9px;
        font-weight: 500;
    }}
    
    QFrame#NavButton[active="true"] QLabel#NavButtonText {{
        color: {p['TEXT_PRIMARY']};
        font-weight: bold;
    }}
    
    #NavSidebar QFrame#ToolbarDivider {{
        background-color: transparent;
        border-top: 1px solid rgba(0, 199, 255, 0.1);
        margin: 4px 6px;
        max-height: 1px;
    }}

    /* Sidebar Group Card */
    #SidebarGroupCard {{
        background-color: {p['BG_SURFACE']};
        border: 1px solid {p['BORDER']};
        border-radius: 6px;
    }}
    
    #ArmedStatus {{
        background-color: rgba(0, 255, 157, 0.1);
        border: 1px solid #00FF9D;
        border-radius: 6px;
        padding: 2px 10px;
        color: #00FF9D;
    }}
    
    /* QSplitter Handle */
    QSplitter::handle {{
        background-color: {p['BORDER']};
    }}
    
    /* Analytics Dashboard & Elements */
    #AnalyticsHeader {{
        letter-spacing: 0.5px;
        color: {p['TEXT_SECONDARY']};
    }}
    
    #AdvisoryStatusVal {{
        color: {p['TEXT_SECONDARY']};
    }}
    
    #AdvisoryStatusDesc {{
        color: {p['TEXT_SECONDARY']};
    }}
    
    #AnalyticsScroll {{
        border: none;
        background: transparent;
    }}
    
    #AnalyticsSectionHeader {{
        border-top: 1px solid {p['BORDER']};
        padding-top: 10px;
        margin-top: 5px;
        color: {p['TEXT_SECONDARY']};
    }}

    #CommitReworkButton {{
        background-color: {COLOR_STATUS_OK};
        color: #09090B;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        padding: 10px 16px;
    }}
    #CommitReworkButton:hover {{
        background-color: #33FFAE;
    }}
    
    #ExportTicketButton, #NextBoardButton {{
        padding: 10px 12px;
        border-radius: 4px;
    }}

    /* Segmented Control Options */
    #SegmentedControl {{
        background-color: {p['BG_ELEVATED']};
        border-radius: 6px;
        padding: 2px;
        border: 1px solid {p['BORDER']};
    }}

    .SegmentedOption {{
        color: {p['TEXT_SECONDARY']};
        font-weight: bold;
        font-size: 11px;
        letter-spacing: 0.5px;
        text-align: center;
        border: none;
        background: transparent;
        padding: 6px 12px;
        border-radius: 4px;
    }}
    
    .SegmentedOption:hover {{
        color: {p['TEXT_PRIMARY']};
    }}

    .SegmentedOption[active="true"] {{
        color: {p['BG_BASE']};
        font-weight: 800;
    }}

    #SegmentedIndicator {{
        background-color: {COLOR_ACCENT_CYAN};
        border-radius: 4px;
    }}

    /* Compute Backend Indicator HUD style */
    #BackendIndicator {{
        background-color: {p['BG_ELEVATED']};
        border: 1px solid {p['BORDER']};
        border-radius: 4px;
        padding: 4px 8px;
        color: {p['TEXT_SECONDARY']};
    }}

    /* User Onboarding Tour Tooltips */
    #OnboardingPopover {{
        background-color: #0B0F19;
        border: 1px solid rgba(0, 210, 255, 0.4);
        border-radius: 6px;
    }}

    #OnboardingTitle {{
        color: {p['TEXT_PRIMARY']};
        font-size: 11px;
        font-weight: 700;
    }}

    #OnboardingDesc {{
        color: {p['TEXT_SECONDARY']};
        font-size: 9px;
    }}

    #OnboardingProgress {{
        color: {COLOR_ACCENT_CYAN};
        font-size: 8px;
        font-weight: 600;
    }}

    #OnboardingSkipBtn {{
        color: {p['TEXT_SECONDARY']};
        background: transparent;
        border: none;
        font-size: 9px;
        font-weight: 600;
        padding: 4px 6px;
    }}
    #OnboardingSkipBtn:hover {{
        color: {p['TEXT_PRIMARY']};
        text-decoration: underline;
    }}

    #OnboardingBackBtn {{
        background-color: transparent;
        border: 1px solid {p['BORDER']};
        border-radius: 3px;
        color: {p['TEXT_PRIMARY']};
        padding: 4px 8px;
        font-size: 9px;
        font-weight: 600;
    }}
    #OnboardingBackBtn:hover {{
        background-color: rgba(255, 255, 255, 0.05);
        border-color: {p['TEXT_PRIMARY']};
    }}

    #OnboardingNextBtn {{
        background-color: {COLOR_ACCENT_CYAN};
        border: 1px solid {COLOR_ACCENT_CYAN};
        border-radius: 3px;
        color: #09090B;
        padding: 4px 10px;
        font-size: 9px;
        font-weight: 700;
    }}
    #OnboardingNextBtn:hover {{
        background-color: #00E1FF;
        border-color: #00E1FF;
    }}
    """