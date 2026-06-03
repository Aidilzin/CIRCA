"""
core/utils.py
-------------
Shared utility functions for the CIRCA camera pipeline.

Architectural note — LOCALIZED PyQt6 EXCEPTION:
  This is the ONLY file in the core/ directory permitted to import from PyQt6.
  The import is restricted to PyQt6.QtGui (QImage and QImage.Format) and
  PyQt6.QtMultimedia (QMediaDevices for device enumeration).
  PyQt6.QtWidgets remains strictly forbidden here and in all other core/ files.
  Rationale: bgr_frame_to_qimage() is the canonical BGR→QImage conversion
  function (architecture decision); placing it in core/ avoids duplication
  across workers/ and prevents inconsistency.

Functions:
  bgr_frame_to_qimage()  — converts OpenCV BGR frame to a Qt-displayable QImage.
                           Used exclusively by workers/camera_worker.py when
                           emitting preview_frame_ready signal.
  enumerate_cameras()    — discovers available UVC devices via Windows DirectShow
                           and retrieves human-readable names via QtMultimedia.
                           Used by ui/sidebar.py to populate the camera QComboBox.

Functional requirements covered:
  FR2  — Camera device enumeration  → enumerate_cameras()
  FR12 — Live feed display           → bgr_frame_to_qimage() (canonical conversion)
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Localized PyQt6 exception: QtGui only — NO QtWidgets import allowed.
from PyQt6.QtGui import QImage

logger = logging.getLogger(__name__)

# Maximum camera index to probe during enumeration.
# Indices 0–5 covers the vast majority of real repair bench setups
# (1 built-in webcam + up to 5 USB cameras).
_MAX_CAMERA_INDEX: int = 5


# ---------------------------------------------------------------------------
# FR12 — BGR → QImage conversion (canonical, single source of truth)
# ---------------------------------------------------------------------------


def bgr_frame_to_qimage(frame: np.ndarray) -> QImage:
    """
    Convert an OpenCV BGR frame to a Qt-displayable QImage (RGB888 format).

    This is the ONLY place in the codebase where this conversion is performed.
    All display code in workers/ and ui/ must call this function — never
    inline the conversion (architecture enforcement rule).

    Memory safety notes:
      QImage constructed from a raw buffer pointer does NOT copy the data;
      it holds a reference to the underlying numpy array memory. To prevent
      a dangling-pointer crash when the numpy array is garbage collected,
      this function calls .copy() on the QImage, which forces Qt to allocate
      its own internal buffer and copy the pixel data into it.

    bytesPerLine precision:
      Using rgb.strides[0] (actual bytes per row in memory) instead of
      width * 3 prevents image skewing/slanting in two edge cases:
        1. Non-contiguous arrays (e.g. numpy slices with non-unit stride).
        2. Row-aligned arrays where OpenCV pads rows to a multiple of N bytes.
      For a standard contiguous (H, W, 3) uint8 array, strides[0] == width * 3,
      so there is zero performance cost in the common case.

    Args:
        frame: OpenCV BGR frame as a (H, W, 3) uint8 numpy array.
               Must not be None or empty.

    Returns:
        A QImage in Format_RGB888 that owns its pixel data (safe to use after
        the source numpy array goes out of scope).

    Raises:
        ValueError: If frame is None, empty, or not a 3-channel image.
    """
    if frame is None or frame.size == 0:
        raise ValueError("bgr_frame_to_qimage received an empty frame.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"bgr_frame_to_qimage expects a (H, W, 3) BGR frame; "
            f"got shape {frame.shape}."
        )

    try:
        # Convert BGR → RGB (YOLO/Qt expects RGB; OpenCV uses BGR by default).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure C-contiguous memory layout before passing to QImage.
        # Non-contiguous arrays (e.g. from numpy slicing) would produce
        # incorrect strides and must be made contiguous first.
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)

        h, w, _ = rgb.shape
        bytes_per_line = rgb.strides[0]  # exact bytes per row (includes any padding)

        # Construct QImage pointing at the numpy buffer, then immediately .copy()
        # so Qt owns the data independently of the numpy array's lifetime.
        qimage = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        return qimage.copy()
    except Exception as e:
        logger.error("bgr_frame_to_qimage: color conversion failed — %s", e)
        # Re-raise to let worker decide how to handle (e.g. drop frame)
        raise ValueError(f"Color conversion failed: {e}") from e


# ---------------------------------------------------------------------------
# FR2 — Camera enumeration (Windows DirectShow / UVC)
# ---------------------------------------------------------------------------


def enumerate_cameras(
    max_index: int = _MAX_CAMERA_INDEX,
    active_device_index: int = -1,
    qt_names: Optional[List[str]] = None,
) -> List[Tuple[int, str]]:
    """
    Probe UVC camera indices 0 through max_index and return all that respond.

    Uses cv2.VideoCapture(index, cv2.CAP_DSHOW) — the CAP_DSHOW backend flag
    is mandatory on Windows. Human-readable names should be retrieved via
    PyQt6.QtMultimedia on the Main GUI Thread and passed in via qt_names.

    Active Frame Validation (Robustified):
      Every detected device is briefly opened and a frame is captured via
      cap.read(). Only devices that yield a valid, non-empty, and non-flat
      frame are returned. This filters out virtual ghost cameras, driver
      placeholders (grey/black), and Windows Hello IR sources.

    Name-Based Trust System:
      If a device has a real human-readable name from QtMultimedia, we are
      more lenient (trusting the OS registration). If it has a generic
      fallback label, we apply strict noise checks to reject digital ghost
      buffers common on Lenovo and other laptop hardware.

    Probe Collision Fix:
      If index == active_device_index, the validation step is skipped because
      the CameraWorker is already using this device. Attempting to open it
      again would cause a DirectShow collision/crash.

    Args:
        max_index: Highest camera index to probe.
        active_device_index: Index of the camera currently being used by CameraWorker.
        qt_names: List of human-readable device names from QMediaDevices.videoInputs().
                  MUST be retrieved on the GUI thread and passed in.

    Returns:
        A list of (index, label) tuples.
    """
    names = qt_names if qt_names is not None else []
    available: List[Tuple[int, str]] = []

    # String denylist: items to ignore if present in the device name.
    denylist = ["Camera 1", "Virtual", "OBS"]

    for index in range(max_index + 1):
        # Resolve the human-readable label (even if validation is skipped).
        has_real_name = index < len(names) and names[index]
        if has_real_name:
            label = names[index]
        else:
            label = f"Camera {index} (USB)"

        # ── String denylist check (real names only) ──
        # Only apply the denylist to real OS-registered device names from QtMultimedia.
        # Never apply it to our own auto-generated fallback labels like "Camera N (USB)"
        # which would inadvertently block index 1 via the "Camera 1" denylist entry.
        if has_real_name and any(bad_str in label for bad_str in denylist):
            logger.debug("enumerate_cameras: skipping denylisted device: '%s'", label)
            continue

        # ── Collision avoidance ──
        if index == active_device_index:
            logger.debug("enumerate_cameras: skipping validation for active camera index %d.", index)
            available.append((index, label))
            continue

        logger.debug("enumerate_cameras: probing index %d ('%s')...", index, label)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                # Verify the camera actually produces a functional data stream.
                ret, frame = cap.read()

                # Check 1: Did the read succeed?
                if not ret or frame is None or frame.size == 0:
                    logger.debug("enumerate_cameras: index %d read() failed or empty.", index)
                    continue

                # Check 2: Statistical validation (reject placeholders/noise)
                # CALIBRATION (Lenovo):
                # Ghost noise (switch off): std ~6.0, mean ~0.2 (Generic Fallback Name)
                # Real dark sensor: std ~2.4, mean ~0.03 (Proper Name: "Integrated Camera")
                frame_std = np.std(frame)
                frame_mean = np.mean(frame)

                logger.info("enumerate_cameras: index %d ('%s') diagnostics: std=%.4f, mean=%.4f", index, label, frame_std, frame_mean)

                # Use name trust: generic fallback labels must pass stricter tests.
                if has_real_name:
                    # Named hardware: only reject if completely dead/flat.
                    if frame_std < 0.1:
                        logger.warning("enumerate_cameras: named device %d rejected as dead buffer (std < 0.1).", index)
                        continue
                    # Also reject digital garbage if mean is low
                    if frame_std > 6.0 and frame_mean < 0.5:
                        logger.warning("enumerate_cameras: named device %d rejected as digital garbage (std > 6.0, mean < 0.5).", index)
                        continue
                else:
                    # Generic fallback: reject perfectly flat OR digital garbage.
                    if frame_std < 1.0:
                        logger.warning("enumerate_cameras: generic device %d rejected as dead buffer (std < 1.0).", index)
                        continue
                    if frame_std > 3.0 and frame_mean < 0.5:
                        logger.warning("enumerate_cameras: generic device %d rejected as digital garbage (std > 3.0, mean < 0.5).", index)
                        continue

                available.append((index, label))
                logger.debug("enumerate_cameras: SUCCESS - found valid camera at index %d: '%s'", index, label)
            else:
                logger.debug("enumerate_cameras: index %d is not opened.", index)
        except Exception as e:
            logger.error("enumerate_cameras: exception probing index %d: %s", index, e)
        finally:
            cap.release()

    if not available:
        logger.warning("enumerate_cameras: no valid UVC cameras found.")

    return available