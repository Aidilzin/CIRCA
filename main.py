import faulthandler
import logging
import os
import sys

# C-Level crash reporting: enables traceback for segfaults
with open('crash_report.log', 'w') as f:
    faulthandler.enable(file=f)

from core.debug import trace_execution


def _get_model_path() -> str:
    """
    Resolve the OpenVINO model path.
    """
    model_relative = os.path.join("models", "yolov12_int8.xml")
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, model_relative)  # type: ignore[attr-defined]
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, model_relative)


def _configure_logging() -> None:
    """
    Set up module-level logging for the application.
    """
    level = logging.DEBUG if os.environ.get("CIRCA_DEBUG") else logging.INFO

    # Configure logging to both console and file
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("execution_trace.log", mode="w")
        ]
    )
    logging.getLogger("circa").setLevel(level)


def exception_hook(exctype, value, traceback):
    """
    Global exception handler to ensure unhandled Python exceptions
    are logged to execution_trace.log before the app dies.
    """
    logger = logging.getLogger(__name__)
    logger.critical("Unhandled Exception:", exc_info=(exctype, value, traceback))
    sys.__excepthook__(exctype, value, traceback)


sys.excepthook = exception_hook


@trace_execution
def main() -> int:
    """
    Application entry point.
    """
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.debug("[TRIPWIRE] Entered main()")

    try:
        logger.info("CIRCA starting up…")

        # Must create QApplication before any QWidget
        from PyQt6.QtWidgets import QApplication
        logger.debug("[TRIPWIRE] About to instantiate QApplication")
        app = QApplication(sys.argv)
        logger.debug("[TRIPWIRE] QApplication instantiated")

        app.setApplicationName("CIRCA")
        app.setApplicationDisplayName("CIRCA — PCB Defect Detection")
        app.setOrganizationName("FYP")

        # Apply global dark theme QSS (all tokens from ui/theme.py)
        from ui.theme import build_qss
        logger.debug("[TRIPWIRE] About to apply QSS theme")
        app.setStyleSheet(build_qss())
        logger.debug("Global QSS theme applied.")

        # Resolve model path
        logger.debug("[TRIPWIRE] About to resolve model path")
        model_path = _get_model_path()
        logger.info("Model path resolved: %s", model_path)
        if not os.path.isfile(model_path):
            logger.warning("Model file not found at '%s'.", model_path)

        # Build and show the main window
        from ui.main_window import MainWindow
        logger.debug("[TRIPWIRE] About to instantiate MainWindow")
        window = MainWindow(model_path=model_path)
        logger.debug("[TRIPWIRE] MainWindow instantiated")

        logger.debug("[TRIPWIRE] About to call window.show()")
        window.show()
        logger.info("MainWindow shown — entering Qt event loop.")

        logger.debug("[TRIPWIRE] About to call app.exec()")
        exit_code = app.exec()
        logger.info("CIRCA exiting — code %d.", exit_code)
        return exit_code

    except Exception as e:
        logger.exception("CRITICAL CRASH in main(): %s", e)
        return 1
    finally:
        logger.debug("[TRIPWIRE] Exiting main()")


if __name__ == "__main__":
    sys.exit(main())

