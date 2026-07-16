"""
core/pcb_guard.py
-----------------
Lightweight PCB subject-matter heuristic guard for CIRCA.

Runs BEFORE inference to reject images that clearly do not contain a PCB
(e.g., face photos, plain walls, arbitrary objects loaded by mistake).

Architecture constraints:
  - NO PyQt6 / PySide6 imports.
  - Pure OpenCV + numpy — no model inference, no heavy computation.
  - Target latency: < 1 ms on any resolution (operates on 256px thumbnail).

Heuristic pipeline (all three gates must pass):
  1. Skin-tone rejection  — rejects face/hand photos
  2. Edge density gate    — PCBs have dense right-angle edges; plain surfaces don't
  3. Solder-mask color    — detects green/blue/red/black PCB laminate

Limitations:
  - Black-mask PCBs may occasionally fail gate 3 (very dark, low saturation).
    This is an acceptable false-reject for the repair-shop use case; technicians
    can disable the guard via the UI toggle.
  - Very close-up macro shots of a single component may fail edge density if the
    field of view is too narrow. Same escape: disable guard.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Tunable thresholds ────────────────────────────────────────────────────────
# How many pixels (fraction of image) must be skin-tone to reject as "face/hand"
SKIN_REJECT_RATIO: float = 0.38

# Minimum fraction of pixels that must be detected as edges for a PCB
EDGE_DENSITY_MIN: float = 0.020

# Minimum fraction of pixels matching any PCB solder-mask color
PCB_COLOR_MIN: float = 0.06

# Working thumbnail size for all checks (keeps latency under 1 ms)
_THUMB_SIZE: int = 256
# ──────────────────────────────────────────────────────────────────────────────


def is_likely_pcb(frame: np.ndarray) -> tuple[bool, str]:
    """
    Heuristically determine whether a BGR image plausibly contains a PCB.

    Returns:
        (True, "")           — image looks like a PCB, proceed with inference
        (False, reason_str)  — image rejected; reason_str explains why
    """
    if frame is None or frame.size == 0:
        return False, "Empty frame"

    # Downsample to a fixed thumbnail for speed
    h, w = frame.shape[:2]
    scale = _THUMB_SIZE / max(h, w)
    if scale < 1.0:
        thumb = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        thumb = frame.copy()

    total = thumb.shape[0] * thumb.shape[1]
    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)

    # ── Gate 1: Edge density (Computed first to evaluate image complexity) ──
    grey = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey, 40, 120)
    edge_density = np.count_nonzero(edges) / total
    if edge_density < EDGE_DENSITY_MIN:
        logger.debug("PCBGuard: rejected — edge density %.2f%% < %.1f%% threshold",
                     edge_density * 100, EDGE_DENSITY_MIN * 100)
        return False, f"No PCB detected — insufficient edge density ({edge_density*100:.1f}%); plain surface or object"

    # ── Gate 2: Skin-tone rejection (Bypassed if edge density is high, representing complex traces) ──
    skin_mask = cv2.inRange(hsv,
                            np.array([0, 20, 70], dtype=np.uint8),
                            np.array([25, 170, 235], dtype=np.uint8))
    skin_ratio = np.count_nonzero(skin_mask) / total
    
    # Human hands/faces have smooth skin tones with very low edge density (typically < 3%).
    # Complex substrates (like yellow/beige prototype boards) have high edge density (> 5%).
    if skin_ratio > SKIN_REJECT_RATIO and edge_density < 0.05:
        logger.debug("PCBGuard: rejected — skin-tone %.1f%% > %.0f%% threshold and low edge density",
                     skin_ratio * 100, SKIN_REJECT_RATIO * 100)
        return False, f"No PCB detected — image appears to be a face or hand ({skin_ratio*100:.0f}% skin tones)"

    # ── Gate 3: Solder-mask color presence ────────────────────────────────
    # Green (most common PCB color): H 35-85, S > 45
    green = cv2.inRange(hsv,
                        np.array([35, 45, 30], dtype=np.uint8),
                        np.array([85, 255, 255], dtype=np.uint8))
    # Blue PCB: H 100-130
    blue = cv2.inRange(hsv,
                       np.array([100, 50, 30], dtype=np.uint8),
                       np.array([130, 255, 255], dtype=np.uint8))
    # Red PCB: H 0-12 or H 168-180
    red1 = cv2.inRange(hsv,
                       np.array([0, 50, 30], dtype=np.uint8),
                       np.array([12, 255, 255], dtype=np.uint8))
    red2 = cv2.inRange(hsv,
                       np.array([168, 50, 30], dtype=np.uint8),
                       np.array([180, 255, 255], dtype=np.uint8))
    # Yellow/Beige/Brown PCB: H 12-35, S 15-255, V 25-255
    yellow_beige = cv2.inRange(hsv,
                               np.array([12, 15, 25], dtype=np.uint8),
                               np.array([35, 255, 255], dtype=np.uint8))
    # Black/dark PCB: very low saturation and low value, but must have some edge density
    dark_mask = cv2.inRange(hsv,
                            np.array([0, 0, 0], dtype=np.uint8),
                            np.array([180, 60, 80], dtype=np.uint8))

    combined = cv2.bitwise_or(green, blue)
    combined = cv2.bitwise_or(combined, red1)
    combined = cv2.bitwise_or(combined, red2)
    combined = cv2.bitwise_or(combined, yellow_beige)
    combined = cv2.bitwise_or(combined, dark_mask)

    pcb_ratio = np.count_nonzero(combined) / total
    if pcb_ratio < PCB_COLOR_MIN:
        logger.debug("PCBGuard: rejected — PCB color coverage %.1f%% < %.0f%% threshold",
                     pcb_ratio * 100, PCB_COLOR_MIN * 100)
        return False, f"No PCB detected — no PCB solder mask color found ({pcb_ratio*100:.0f}% coverage)"

    logger.debug(
        "PCBGuard: PASS — skin=%.1f%% edge=%.1f%% pcb_color=%.1f%%",
        skin_ratio * 100, edge_density * 100, pcb_ratio * 100,
    )
    return True, ""
