"""
ui/control_panel.py
-------------------
Compatibility shim — ControlPanel has been superseded by SidePanel in ui/sidebar.py.

This file re-exports the new types under the old names so that tests written
against ControlPanel continue to import cleanly during the migration period.
"""
# Re-export the new SidePanel as ControlPanel for backward compatibility
from ui.sidebar import SidePanel as ControlPanel, PreprocessingSlider  # noqa: F401

__all__ = ["ControlPanel", "PreprocessingSlider"]