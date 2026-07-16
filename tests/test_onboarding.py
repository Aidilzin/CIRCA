import pytest
from unittest.mock import patch
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

@pytest.fixture
def main_window(qapp):
    # Reset QSettings before test
    settings = QSettings("CIRCA", "VisionStudio")
    settings.setValue("hasRunOnboarding", False)
    
    with patch("ui.main_window.MainWindow._start_async_scan"):
        win = MainWindow(model_path="models/yolov12_int8.xml")
        yield win
        # Clean up threads
        win.camera_thread.quit()
        win.camera_thread.wait()
        win.inference_thread.quit()
        win.inference_thread.wait()

def test_onboarding_persistence_and_reset(main_window):
    settings = QSettings("CIRCA", "VisionStudio")
    settings.setValue("hasRunOnboarding", False)
    
    # Relaunch tour
    main_window.start_onboarding_tour()
    controller = main_window.onboarding_controller
    assert controller is not None
    assert not controller.overlay.isHidden()
    
    # Finish/skip tour and verify key is true
    controller.finish_tour()
    assert controller.overlay.isHidden()
    assert settings.value("hasRunOnboarding", type=bool) is True

    # Trigger reset and verify key is false and overlay is open
    main_window.side_panel.resetOnboardingTour()
    assert settings.value("hasRunOnboarding", type=bool) is False
    assert not controller.overlay.isHidden()

def test_onboarding_step_navigation_and_drawer_auto_toggle(main_window):
    main_window.start_onboarding_tour()
    controller = main_window.onboarding_controller
    controller.init_steps()
    
    # Step 1: Nav sidebar
    assert controller.current_step == 0
    controller.show_step()
    assert not main_window.side_panel._expanded
    
    # Step 2: Next -> inspect_widget
    controller.next_step()
    assert controller.current_step == 1
    assert not main_window.side_panel._expanded

    # Step 4: Advance to side_panel (index 3)
    controller.next_step() # step 3 (index 2)
    controller.next_step() # step 4 (index 3 - side_panel)
    assert controller.current_step == 3
    # Verify SidePanel auto-expanded
    assert main_window.side_panel._expanded
    
    # Step 3: Back -> index 2
    controller.back_step()
    assert controller.current_step == 2
    # Verify SidePanel auto-collapsed when leaving step
    assert not main_window.side_panel._expanded

    # Finish tour -> collapses panel
    controller.finish_tour()
    assert not main_window.side_panel._expanded
