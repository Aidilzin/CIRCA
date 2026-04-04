"""
tests/conftest.py
-----------------
Shared pytest fixtures for the entire CIRCA test suite.

QApplication strategy:
  All PyQt6 tests require a QApplication (or QCoreApplication) instance.
  Since VideoWidget tests require QApplication (a QWidget subclass), and
  worker tests used QCoreApplication, we create a single QApplication at
  session scope.

  QApplication IS-A QCoreApplication — using it for all tests is safe and
  avoids the Windows STATUS_STACK_BUFFER_OVERRUN crash (exit -1073740791)
  that occurs when a QCoreApplication is created before a QApplication in
  the same Python process (Qt does not allow downgrading or re-creating
  the application instance).

  All per-file `qapp` fixtures now delegate to this session-level fixture.
"""

import sys
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qapp_session():
    """
    Create a single QApplication for the entire test session.

    This replaces the per-file QCoreApplication fixtures: QApplication
    is a strict superset, so all signal/slot, threading, and widget tests
    work correctly with it.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    yield app
    # Do NOT call app.quit() or exec() here — pytest manages teardown.
