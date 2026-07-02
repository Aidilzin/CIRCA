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
# Pre-built gamma LUT cache and CLAHE object cache.
# Avoids re-allocating heavy objects on every frame (NFR1).
# ---------------------------------------------------------------------------
_gamma_lut_cache: dict[float, np.ndarray] = {}
_clahe_cache: dict[float, cv2.CLAHE] = {}


def clear_gamma_lut_cache() -> None:
    """Clear the pre-built gamma LUT cache (primarily for test isolation)."""
    _gamma_lut_cache.clear()


def clear_clahe_cache() -> None:
    """Clear the CLAHE object cache (primarily for test isolation)."""
    _clahe_cache.clear()


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

    ... (rest of docstring)

    Implementation strategy:
      1. Convert BGR → LAB colour space.
      2. Apply CLAHE to the L (luminance) channel only.
      3. Convert LAB → BGR and return the result.

    Colour space choice: LAB is used here (not YUV) to ensure train/inference
    consistency.  The offline preprocessing pipeline (train_engine.py §CIRCA
    Thesis Preprocessing) also uses LAB.  The ~0.3 ms extra cost vs YUV is
    well within the 5 ms NFR1 budget at 1080p.
    ...
    """
    if frame is None or frame.size == 0:
        raise ValueError("apply_clahe received an empty frame.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"apply_clahe expects a 3-channel BGR frame; got shape {frame.shape}."
        )

    # NFR1 Optimization: Reuse CLAHE object if clipLimit hasn't changed.
    clip_limit = params.clahe_clip_limit
    if clip_limit not in _clahe_cache:
        _clahe_cache[clip_limit] = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(8, 8),
        )
    clahe = _clahe_cache[clip_limit]

    # Use LAB to match the offline training preprocessing (train_engine.py).
    # Applying CLAHE only to the L channel leaves chrominance (a, b) untouched,
    # preserving the diagnostic colour of solder and copper under variable lighting.
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

    ... (rest of docstring)

    Implementation:
      1. Convert to greyscale (Laplacian is defined on single-channel images).
      2. Apply cv2.Laplacian(grey, cv2.CV_32F) — 32-bit float is significantly
         faster than CV_64F while providing sufficient range for variance.
      3. Use cv2.meanStdDev for high-performance variance calculation.

    Args:
        frame: Input frame — BGR or greyscale, any resolution.
               Must not be None or empty.

    Returns:
        A float representing the sharpness of the frame.
    """
    if frame is None or frame.size == 0:
        raise ValueError("compute_variance received an empty frame.")

    # Convert to greyscale if the input is colour.
    if frame.ndim == 3:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grey = frame

    # NFR1 Optimization: Downsample to half-res for variance calculation.
    # 1080p -> 540p reduces pixel count 4x. INTER_NEAREST is fastest.
    small = cv2.resize(grey, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

    # NFR1 Optimization: Use CV_32F instead of CV_64F for speed.
    laplacian = cv2.Laplacian(small, cv2.CV_32F)
    
    # NFR1 Optimization: cv2.meanStdDev is faster than laplacian.var().
    _, stddev = cv2.meanStdDev(laplacian)
    return float(stddev[0, 0] ** 2)


# ---------------------------------------------------------------------------
# Noise reduction filter & Auto-optimisation logic
# ---------------------------------------------------------------------------

def apply_denoise(frame: np.ndarray) -> np.ndarray:
    """
    Apply edge-preserving bilateral filtering to suppress camera sensor noise
    (graininess) while retaining crisp defect edges critical for detection.

    Bilateral filter is chosen over Gaussian because it preserves edge sharpness:
    it weights pixels by both spatial proximity AND intensity similarity, so edges
    between solder and copper remain sharp while uniform regions are smoothed.

    Parameters chosen for real-time performance at 720p:
      d=5        — neighbourhood diameter; d>9 is very slow
      sigmaColor=35 — compresses colour range; keeps component edges intact
      sigmaSpace=5  — spatial kernel width
    """
    if frame is None or frame.size == 0:
        raise ValueError("apply_denoise received an empty frame.")
    return cv2.bilateralFilter(frame, d=5, sigmaColor=35, sigmaSpace=5)


def auto_tune_parameters(frame: np.ndarray) -> tuple[float, float]:
    """
    Analyze frame luminance histogram to calculate optimal CLAHE and Gamma values.
    
    Returns:
        tuple[clahe_clip_limit, gamma]
    """
    if frame is None or frame.size == 0:
        return 2.0, 1.0
        
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val, std_val = cv2.meanStdDev(grey)
    mean_brightness = float(mean_val[0, 0])
    std_brightness = float(std_val[0, 0])

    # 1. Calculate dynamic Gamma based on mean brightness (target range: 110-130)
    if mean_brightness < 40:
        gamma = 2.2      # Extremely dark shadows
    elif mean_brightness < 85:
        gamma = 1.6      # Dark bench light
    elif mean_brightness < 115:
        gamma = 1.3      # Slightly dark
    elif mean_brightness < 135:
        gamma = 1.0      # Target range, no correction
    elif mean_brightness < 170:
        gamma = 0.8      # Bright highlights
    else:
        gamma = 0.6      # Glare / overexposed

    # 2. Calculate dynamic CLAHE based on standard deviation (contrast indicator)
    # Capped at 2.2 max to prevent sensor noise/graininess amplification.
    if std_brightness < 20:
        clahe_clip = 2.2  # Low contrast
    elif std_brightness < 35:
        clahe_clip = 1.8  # Moderately low contrast
    elif std_brightness < 50:
        clahe_clip = 1.5  # Normal contrast
    elif std_brightness < 65:
        clahe_clip = 1.2  # High contrast
    else:
        clahe_clip = 1.0  # Very high contrast

    return clahe_clip, gamma