"""
tests/test_preprocessor.py
--------------------------
Unit tests for core/preprocessor.py.

All tests use synthetic numpy frames — no camera hardware required.
No PyQt6 import anywhere in this file (validates core/ boundary rule).

Test strategy:
  - Input validation: None, empty, wrong shape → ValueError
  - Identity transforms: gamma=1.0 → no-op; clahe on flat frame → unchanged
  - Directional contracts: gamma > 1.0 brightens; low variance = blurry
  - Output shape/dtype: all functions must preserve (H, W, 3) uint8
  - Blur gate: solid colour frame has near-zero variance (blurry)
  - Sharp frame: frame with edges has high variance (sharp)
"""

import numpy as np
import pytest

from core.models import PreprocessParams
from core.preprocessor import apply_clahe, apply_gamma, compute_variance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bgr_frame(height: int = 64, width: int = 64, value: int = 128) -> np.ndarray:
    """Produce a flat (uniform colour) BGR frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def make_checkerboard_frame(height: int = 64, width: int = 64) -> np.ndarray:
    """
    Produce a high-contrast black-and-white checkerboard BGR frame.
    This frame has many sharp edges → high Laplacian variance.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if (x // 8 + y // 8) % 2 == 0:
                frame[y, x] = [255, 255, 255]
    return frame


def make_default_params(**overrides) -> PreprocessParams:
    """Return default PreprocessParams with optional field overrides."""
    params = PreprocessParams()
    for key, val in overrides.items():
        setattr(params, key, val)
    return params


# ===========================================================================
# apply_clahe — FR3
# ===========================================================================


class TestApplyClahe:
    def test_returns_same_shape_and_dtype(self):
        frame = make_bgr_frame(64, 64, 128)
        params = make_default_params()
        result = apply_clahe(frame, params)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_raises_on_none_frame(self):
        params = make_default_params()
        with pytest.raises(ValueError, match="empty frame"):
            apply_clahe(None, params)  # type: ignore[arg-type]

    def test_raises_on_empty_frame(self):
        params = make_default_params()
        with pytest.raises(ValueError, match="empty frame"):
            apply_clahe(np.array([]), params)

    def test_raises_on_single_channel_frame(self):
        """core/preprocessor expects BGR (3-channel); greyscale must be rejected."""
        grey_frame = np.full((64, 64), 128, dtype=np.uint8)
        params = make_default_params()
        with pytest.raises(ValueError, match="3-channel BGR"):
            apply_clahe(grey_frame, params)

    def test_output_is_new_array_not_in_place(self):
        """apply_clahe must not mutate the input frame."""
        frame = make_bgr_frame(64, 64, 128)
        original_copy = frame.copy()
        params = make_default_params()
        apply_clahe(frame, params)
        np.testing.assert_array_equal(frame, original_copy)

    def test_flat_frame_remains_flat_after_clahe(self):
        """
        A perfectly uniform frame (all pixels identical) contains no local
        contrast variation. CLAHE should leave it unchanged (all pixels map
        to themselves after equalisation of a flat histogram).
        """
        frame = make_bgr_frame(64, 64, 200)
        params = make_default_params(clahe_clip_limit=2.0)
        result = apply_clahe(frame, params)
        # The BGR→LAB→BGR round-trip introduces a deterministic drift of up to 6
        # pixel units on high-intensity values (e.g. 200→206 after CLAHE
        # redistributes the flat histogram peak). This is correct CLAHE behaviour —
        # the flat frame's histogram is legally modified; the output is still a
        # near-uniform frame. Tolerance of 8 covers the measured drift + margin.
        assert np.abs(result.astype(int) - frame.astype(int)).max() <= 8

    def test_high_clip_limit_does_not_crash(self):
        """Boundary: clip_limit=8.0 (slider maximum) must not raise."""
        frame = make_checkerboard_frame()
        params = make_default_params(clahe_clip_limit=8.0)
        result = apply_clahe(frame, params)
        assert result.shape == frame.shape

    def test_minimum_clip_limit_does_not_crash(self):
        """Boundary: clip_limit=1.0 (slider minimum) must not raise."""
        frame = make_checkerboard_frame()
        params = make_default_params(clahe_clip_limit=1.0)
        result = apply_clahe(frame, params)
        assert result.shape == frame.shape


# ===========================================================================
# apply_gamma — FR4
# ===========================================================================


class TestApplyGamma:
    def test_returns_same_shape_and_dtype(self):
        frame = make_bgr_frame(64, 64, 128)
        params = make_default_params(gamma=1.0)
        result = apply_gamma(frame, params)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_raises_on_none_frame(self):
        params = make_default_params(gamma=1.0)
        with pytest.raises(ValueError, match="empty frame"):
            apply_gamma(None, params)  # type: ignore[arg-type]

    def test_raises_on_zero_gamma(self):
        frame = make_bgr_frame()
        params = make_default_params(gamma=0.0)
        with pytest.raises(ValueError, match="Gamma must be > 0"):
            apply_gamma(frame, params)

    def test_raises_on_negative_gamma(self):
        frame = make_bgr_frame()
        params = make_default_params(gamma=-1.0)
        with pytest.raises(ValueError, match="Gamma must be > 0"):
            apply_gamma(frame, params)

    def test_gamma_one_is_identity(self):
        """gamma=1.0 must produce output pixel-identical to the input."""
        frame = make_bgr_frame(64, 64, 150)
        params = make_default_params(gamma=1.0)
        result = apply_gamma(frame, params)
        np.testing.assert_array_equal(result, frame)

    def test_gamma_greater_than_one_brightens(self):
        """
        gamma > 1.0 → shadow lifting: a mid-tone pixel (e.g. 100)
        should map to a brighter value in the output.
        """
        # Single mid-tone value in a 2x2 frame for simple assertion.
        frame = np.full((2, 2, 3), 100, dtype=np.uint8)
        params = make_default_params(gamma=2.0)
        result = apply_gamma(frame, params)
        # out = (100/255)^(1/2) * 255 ≈ 159 — all output pixels must be brighter.
        assert result[0, 0, 0] > 100

    def test_gamma_less_than_one_darkens(self):
        """gamma < 1.0 → darkening: a mid-tone pixel should map to a darker value."""
        frame = np.full((2, 2, 3), 200, dtype=np.uint8)
        params = make_default_params(gamma=0.5)
        result = apply_gamma(frame, params)
        # out = (200/255)^(1/0.5) * 255 = (200/255)^2 * 255 ≈ 157 — darker.
        assert result[0, 0, 0] < 200

    def test_lut_is_cached_across_calls(self):
        """
        Calling apply_gamma twice with the same gamma must not crash
        and must produce identical results (verifies cache consistency).
        """
        from core.preprocessor import _gamma_lut_cache

        frame = make_bgr_frame(64, 64, 120)
        params = make_default_params(gamma=1.5)

        result_1 = apply_gamma(frame, params)
        assert 1.5 in _gamma_lut_cache  # LUT was cached on first call.

        result_2 = apply_gamma(frame, params)
        np.testing.assert_array_equal(result_1, result_2)

    def test_output_is_new_array_not_in_place(self):
        """apply_gamma must not mutate the input frame."""
        frame = make_bgr_frame(64, 64, 128)
        original_copy = frame.copy()
        params = make_default_params(gamma=1.8)
        apply_gamma(frame, params)
        np.testing.assert_array_equal(frame, original_copy)


# ===========================================================================
# compute_variance — FR5
# ===========================================================================


class TestComputeVariance:
    def test_returns_float(self):
        frame = make_checkerboard_frame()
        result = compute_variance(frame)
        assert isinstance(result, float)

    def test_raises_on_none_frame(self):
        with pytest.raises(ValueError, match="empty frame"):
            compute_variance(None)  # type: ignore[arg-type]

    def test_raises_on_empty_frame(self):
        with pytest.raises(ValueError, match="empty frame"):
            compute_variance(np.array([]))

    def test_flat_frame_has_near_zero_variance(self):
        """
        A solid-colour frame has zero edges → Laplacian output is all zeros
        → variance ≈ 0.0. This is the primary blurry-board detection case.
        """
        frame = make_bgr_frame(64, 64, 150)
        variance = compute_variance(frame)
        assert variance < 1e-6, f"Expected ~0 variance for flat frame; got {variance}"

    def test_checkerboard_has_high_variance(self):
        """
        A checkerboard frame has many hard edges → high Laplacian response
        → high variance. Should be well above any realistic blur_threshold.
        """
        frame = make_checkerboard_frame(64, 64)
        variance = compute_variance(frame)
        # 100.0 is the PreprocessParams default blur_threshold — sharp frames
        # must exceed this to pass through to inference.
        assert variance > 100.0, (
            f"Expected high variance for checkerboard; got {variance}"
        )

    def test_sharp_frame_has_higher_variance_than_blurred(self):
        """
        Blurring a sharp frame reduces its Laplacian variance.
        Validates the monotonic relationship the blur gate depends on.
        """
        import cv2 as _cv2

        sharp = make_checkerboard_frame(128, 128)
        blurred = _cv2.GaussianBlur(sharp, (21, 21), sigmaX=5)

        var_sharp = compute_variance(sharp)
        var_blurred = compute_variance(blurred)

        assert var_sharp > var_blurred, (
            f"Sharp ({var_sharp:.2f}) should exceed blurred ({var_blurred:.2f})"
        )

    def test_accepts_greyscale_frame(self):
        """compute_variance must also accept single-channel greyscale input."""
        grey_frame = np.full((64, 64), 128, dtype=np.uint8)
        result = compute_variance(grey_frame)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_variance_is_non_negative(self):
        """Variance is always ≥ 0 by definition."""
        for value in [0, 64, 128, 200, 255]:
            frame = make_bgr_frame(32, 32, value)
            assert compute_variance(frame) >= 0.0


# ===========================================================================
# Integration — pipeline order (CLAHE → Gamma applied on the same frame)
# ===========================================================================


class TestPipelineIntegration:
    def test_clahe_then_gamma_preserves_shape_and_dtype(self):
        """Simulates the CameraWorker pipeline order: CLAHE → Gamma."""
        frame = make_checkerboard_frame(128, 128)
        params = make_default_params(clahe_clip_limit=2.0, gamma=1.2)

        after_clahe = apply_clahe(frame, params)
        after_gamma = apply_gamma(after_clahe, params)

        assert after_gamma.shape == frame.shape
        assert after_gamma.dtype == np.uint8

    def test_sharp_frame_passes_blur_gate_after_preprocessing(self):
        """
        After CLAHE + Gamma, a sharp checkerboard frame must still register
        as sharp (variance above default threshold = 100.0).
        This validates that preprocessing does not destroy edge information.
        """
        frame = make_checkerboard_frame(128, 128)
        params = make_default_params()

        after_clahe = apply_clahe(frame, params)
        after_gamma = apply_gamma(after_clahe, params)
        variance = compute_variance(after_gamma)

        assert variance > params.blur_threshold, (
            f"Preprocessed sharp frame variance ({variance:.2f}) fell below "
            f"blur_threshold ({params.blur_threshold}). Preprocessing is "
            f"destroying edge information."
        )
