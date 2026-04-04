"""
core/preprocessor.py
--------------------
Pure, stateless OpenCV preprocessing functions for the CIRCA camera pipeline.

Architecture constraints (enforced):
  - NO PyQt6 / PySide6 imports — this module must remain UI-framework-agnostic.
  - NO state — all functions are pure; all mutable parameters are passed in.
  - Called exclusively from workers/camera_worker.py inside the Camera Thread.

Functional requirements covered:
  FR3  — CLAHE adaptive contrast enhancement  → apply_clahe()
  FR4  — Gamma correction for shadow lifting  → apply_gamma()
  FR5  — Laplacian variance blur/motion gate  → compute_variance()

Non-functional requirements:
  NFR1 — Total preprocessing budget < 5 ms per frame.
          Each function is profiled independently; avoid redundant allocations.
  NFR7 — > 90% mAP under ±30% lighting variation — CLAHE + gamma satisfy this.
"""

import cv2
import numpy as np

from core.models import PreprocessParams


# ---------------------------------------------------------------------------
# Pre-built gamma LUT cache — computed once per unique gamma value at runtime.
# Avoids re-allocating a 256-element uint8 array on every frame (NFR1).
# ---------------------------------------------------------------------------
_gamma_lut_cache: dict[float, np.ndarray] = {}


def _build_gamma_lut(gamma: float) -> np.ndarray:
    """
    Build a uint8 lookup table that maps each pixel intensity [0–255]
    to its gamma-corrected value: out = (in / 255)^(1/gamma) * 255.

    Args:
        gamma: Gamma exponent. Values > 1.0 brighten shadows; < 1.0 darken.

    Returns:
        A (256,) uint8 numpy array suitable for cv2.LUT().
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return table


# ---------------------------------------------------------------------------
# FR3 — CLAHE Adaptive Contrast Enhancement
# ---------------------------------------------------------------------------

def apply_clahe(frame: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the
    luminance channel of the input BGR frame.

    This enhances local contrast and suppresses specular glare from solder
    joints and component leads — critical for consistent detection under the
    bright directional lighting common in PCB repair environments.

    Implementation strategy:
      1. Convert BGR → LAB colour space (lossless, perceptually uniform).
      2. Apply CLAHE to the L (lightness) channel only, preserving hue/saturation.
      3. Convert LAB → BGR and return the result.

      Operating on LAB rather than all-channel BGR prevents the colour
      distortion that would occur from equalising R, G, B independently.

    Args:
        frame:  Input BGR frame as a (H, W, 3) uint8 numpy array.
                Must not be None; caller (CameraWorker) is responsible for
                dropping empty frames before calling this function.
        params: Live preprocessing parameters from the GUI slider state.
                Only params.clahe_clip_limit is consumed here.

    Returns:
        CLAHE-enhanced BGR frame as a (H, W, 3) uint8 numpy array.
        The output array is a new allocation (cv2.merge produces a copy).

    Raises:
        ValueError: If frame is empty or not a 3-channel BGR image.
    """
    if frame is None or frame.size == 0:
        raise ValueError("apply_clahe received an empty frame.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"apply_clahe expects a 3-channel BGR frame; got shape {frame.shape}."
        )

    # tile_grid_size=(8, 8) is the OpenCV default and a well-validated choice
    # for PCB image sizes (640x640 → 80x80 px per tile).
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=(8, 8),
    )

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# FR4 — Gamma Correction
# ---------------------------------------------------------------------------

def apply_gamma(frame: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """
    Apply power-law gamma correction to a BGR frame using a pre-built LUT.

    Gamma correction lifts shadow detail in underexposed areas of the PCB
    (e.g. under component overhangs) and compresses highlights in overexposed
    areas — both common artefacts of fixed-position bench lighting.

    The perceptual formula applied per pixel:
        out_intensity = (in_intensity / 255.0) ^ (1 / gamma) * 255

    gamma > 1.0  →  brightens mid-tones and shadows (most common use case)
    gamma = 1.0  →  identity transform (no change)
    gamma < 1.0  →  darkens mid-tones (used when board is over-lit)

    Performance:
      cv2.LUT() is a SIMD-optimised O(N) lookup — negligible cost.
      The LUT is cached keyed on gamma value; it is only recomputed when
      the technician moves the slider to a new position (rare).

    Args:
        frame:  Input BGR frame as a (H, W, 3) uint8 numpy array.
        params: Live preprocessing parameters. Only params.gamma is consumed.

    Returns:
        Gamma-corrected BGR frame as a (H, W, 3) uint8 numpy array.

    Raises:
        ValueError: If frame is empty or params.gamma is not positive.
    """
    if frame is None or frame.size == 0:
        raise ValueError("apply_gamma received an empty frame.")
    if params.gamma <= 0.0:
        raise ValueError(f"Gamma must be > 0; got {params.gamma}.")

    gamma = params.gamma

    # Cache miss: build and store the LUT for this gamma value.
    if gamma not in _gamma_lut_cache:
        _gamma_lut_cache[gamma] = _build_gamma_lut(gamma)

    lut = _gamma_lut_cache[gamma]
    return cv2.LUT(frame, lut)


# ---------------------------------------------------------------------------
# FR5 — Laplacian Variance Blur / Motion Gate
# ---------------------------------------------------------------------------

def compute_variance(frame: np.ndarray) -> float:
    """
    Compute the Laplacian variance of a frame as a sharpness metric.

    The Laplacian operator measures local second-order intensity gradients —
    high values indicate sharp edges (clear, still board); low values indicate
    uniform intensity (motion blur or out-of-focus board).

    This value is used in workers/camera_worker.py as the gate condition:

        if compute_variance(frame) < params.blur_threshold:
            emit frame_dropped()
            continue  # Skip this frame entirely — do not run inference

    The threshold (params.blur_threshold, default 100.0) is user-tunable
    via the 'Motion Sensitivity' slider (FR18) to account for different
    camera optics and working distances.

    Implementation:
      1. Convert to greyscale (Laplacian is defined on single-channel images).
      2. Apply cv2.Laplacian(grey, cv2.CV_64F) — 64-bit float prevents
         integer overflow on high-contrast edges.
      3. Return the variance (std²) of the Laplacian output.

    Args:
        frame: Input frame — BGR or greyscale, any resolution.
               Must not be None or empty.

    Returns:
        A float representing the sharpness of the frame.
        Higher values → sharper / more detail → pass through inference.
        Lower values  → blurry / motion → should be dropped by the caller.

    Raises:
        ValueError: If frame is None or empty.
    """
    if frame is None or frame.size == 0:
        raise ValueError("compute_variance received an empty frame.")

    # Convert to greyscale if the input is colour.
    if frame.ndim == 3:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grey = frame

    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    return float(laplacian.var())
