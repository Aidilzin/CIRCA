"""
ui/control_panel.py
-------------------
Migration shim — ControlPanel has been superseded by SidePanel in ui/sidebar.py.

This file re-exports the new types under the old names so that any tests or
external code written against ControlPanel continue to import cleanly.

TODO: Once all call-sites have been updated to import from ui.sidebar directly,
      delete this file entirely. It is not part of the production runtime path.
"""
# Re-export the new SidePanel as ControlPanel for backward compatibility
from ui.sidebar import SidePanel as ControlPanel, PreprocessingSlider  # noqa: F401

__all__ = ["ControlPanel", "PreprocessingSlider"]