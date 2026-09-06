"""The measurement functions, against inputs whose answers are known
independently of the implementation.

A test that computes the expected value the same way the code does proves
nothing. So: a black frame is 0.0 brightness because black is, a pure red
frame is hue 0 because red is, a flat frame has zero Laplacian variance
because it has no edges. Where a real answer isn't available analytically
(motion), the test asserts a *separation* between two clips built to
differ, which is the property that actually matters.
"""
from __future__ import annotations

import numpy as np
import pytest

from vmcore.signals import (brightness, composition_centroid, contrast,
                            luma, motion, saturation_and_hue, sharpness)

H, W = 32, 48


def solid(b: int, g: int, r: int) -> np.ndarray:
    f = np.zeros((H, W, 3), dtype=np.uint8)
    f[..., 0], f[..., 1], f[..., 2] = b, g, r
    return f


BLACK = solid(0, 0, 0)
WHITE = solid(255, 255, 255)
GREY = solid(128, 128, 128)
RED = solid(0, 0, 255)      # BGR
GREEN = solid(0, 255, 0)
BLUE = solid(255, 0, 0)


# --------------------------------------------------------------------------
# Brightness / contrast
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frame,expected", [
    (BLACK, 0.0), (WHITE, 1.0), (GREY, 128 / 255),
])
def test_brightness_of_known_greys(frame, expected):
    assert brightness([frame]) == pytest.approx(expected, abs=1e-4)


def test_luma_weights_sum_to_one():
    """Any neutral grey must come back as itself, not scaled."""
    assert luma(GREY).mean() == pytest.approx(128 / 255, abs=1e-6)


def test_a_flat_frame_has_no_contrast():
    assert contrast([GREY]) == pytest.approx(0.0, abs=1e-6)


def test_half_black_half_white_has_maximum_contrast():
    """Two equal populations at 0 and 1: standard deviation is exactly 0.5."""
    f = np.zeros((H, W, 3), dtype=np.uint8)
    f[:, W // 2:] = 255
    assert contrast([f]) == pytest.approx(0.5, abs=1e-3)


def test_brightness_averages_across_frames():
    assert brightness([BLACK, WHITE]) == pytest.approx(0.5, abs=1e-4)


# --------------------------------------------------------------------------
# Saturation / hue
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frame,hue", [
    (RED, 0.0), (GREEN, 120.0), (BLUE, 240.0),
])
def test_primary_colours_land_on_their_hue(frame, hue):
    sat, got = saturation_and_hue([frame])
    assert sat == pytest.approx(1.0, abs=1e-4)
    assert got == pytest.approx(hue, abs=0.5)


@pytest.mark.parametrize("frame", [BLACK, WHITE, GREY])
def test_greys_have_no_saturation(frame):
    sat, _ = saturation_and_hue([frame])
    assert sat == pytest.approx(0.0, abs=1e-6)


def test_hue_wraps_instead_of_averaging_through_the_middle():
    """350 and 10 degrees must average to 0, not 180. This is the whole
    reason the mean is circular."""
    lo = solid(30, 0, 255)   # ~350 deg
    hi = solid(0, 30, 255)   # ~10 deg
    _, hue = saturation_and_hue([lo, hi])
    assert min(hue, 360 - hue) < 5.0


def test_near_black_pixels_report_unstable_saturation():
    """A known property, pinned so it doesn't surprise calibration later.

    HSV saturation is delta/max, so a frame at luma ~0.06 with two codes of
    chroma noise reads as ~0.12 saturated while looking black. Standard HSV
    (OpenCV agrees), left as-is rather than quietly reweighted — but a
    sweep over saturation across dark footage is reading noise there.
    """
    dark_noise = solid(16, 16, 16)
    dark_noise[..., 0] = 18          # two codes of blue, invisible
    sat, _ = saturation_and_hue([dark_noise])
    assert sat > 0.1, "the trap is real; if this drops, the measure changed"

    # The same absolute chroma difference on a bright frame is negligible.
    bright_noise = solid(253, 255, 255)
    bright_sat, _ = saturation_and_hue([bright_noise])
    assert bright_sat < 0.02


def test_a_mostly_grey_frame_is_not_shouted_down_by_a_few_pixels():
    """Grey pixels carry arbitrary hue; weighting by saturation is what
    stops them voting. A grey frame with a red corner should read red."""
    f = solid(128, 128, 128)
    f[:4, :4] = (0, 0, 255)
    _, hue = saturation_and_hue([f])
    assert min(hue, 360 - hue) < 5.0


# --------------------------------------------------------------------------
# Sharpness
# --------------------------------------------------------------------------

def test_a_flat_frame_has_zero_sharpness():
    assert sharpness(GREY) == pytest.approx(0.0, abs=1e-9)


def test_noise_is_sharper_than_a_gradient():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    gradient = np.tile(
        np.linspace(0, 255, W, dtype=np.uint8)[None, :, None], (H, 1, 3))
    assert sharpness(noise) > sharpness(gradient) * 10


def test_blurring_reduces_sharpness():
    """A box blur over the same content must score strictly lower."""
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    f = sharp.astype(np.float32)
    blurred = ((f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:]
                + f[1:-1, 1:-1]) / 5.0).astype(np.uint8)
    assert sharpness(blurred) < sharpness(sharp)


def test_tiny_frames_do_not_crash():
    assert sharpness(np.zeros((2, 2, 3), dtype=np.uint8)) == 0.0


# --------------------------------------------------------------------------
# Motion
# --------------------------------------------------------------------------

def test_identical_frames_have_no_motion():
    assert motion([GREY, GREY, GREY]) == (0.0, 0.0)


def test_a_single_frame_has_no_motion():
    assert motion([GREY]) == (0.0, 0.0)
    assert motion([]) == (0.0, 0.0)


def test_black_to_white_is_full_motion():
    mean, peak = motion([BLACK, WHITE])
    assert mean == pytest.approx(1.0, abs=1e-4)
    assert peak == pytest.approx(1.0, abs=1e-4)


def test_peak_separates_a_spike_from_steady_change():
    """The case this exists for: one whip-pan and a steady drift can share
    a mean while being nothing alike to cut with."""
    spike = motion([GREY, GREY, WHITE, WHITE, WHITE])
    steady = motion([solid(v, v, v) for v in (100, 120, 140, 160, 180)])
    assert spike[1] > steady[1]


# --------------------------------------------------------------------------
# Composition centroid
# --------------------------------------------------------------------------

def _textured(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Noise, not a real subject — but it has the one property that
    matters here: strong edge energy, the same way a person or a tool
    against a flat background does."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_a_flat_frame_centres_by_convention():
    """No detail anywhere to weight toward - reporting dead centre is the
    honest way to say "nothing to report", not an arbitrary corner."""
    assert composition_centroid([GREY]) == pytest.approx((0.5, 0.5), abs=1e-9)


def test_empty_frames_centre_by_convention():
    assert composition_centroid([]) == (0.5, 0.5)


def test_a_patch_on_the_right_pulls_the_centroid_right():
    f = solid(128, 128, 128).copy()
    f[:, -W // 4:] = _textured(H, W // 4, seed=1)
    cx, cy = composition_centroid([f])
    assert cx > 0.6
    assert cy == pytest.approx(0.5, abs=0.05)


def test_a_patch_at_the_top_pulls_the_centroid_up():
    f = solid(128, 128, 128).copy()
    f[:H // 4, :] = _textured(H // 4, W, seed=2)
    cx, cy = composition_centroid([f])
    assert cy < 0.4
    assert cx == pytest.approx(0.5, abs=0.05)


def test_symmetric_content_averages_back_to_centre():
    """Equal texture on both sides has no reason to favour either."""
    f = solid(128, 128, 128).copy()
    f[:, :W // 6] = _textured(H, W // 6, seed=3)
    f[:, -W // 6:] = _textured(H, W // 6, seed=4)
    cx, _ = composition_centroid([f])
    assert cx == pytest.approx(0.5, abs=0.05)


def test_centroid_averages_across_frames():
    """A patch that's on the right in one sampled frame and the left in
    the next should land near the middle on average, the same way a
    per-clip mean smooths any other signal here."""
    right = solid(128, 128, 128).copy()
    right[:, -W // 4:] = _textured(H, W // 4, seed=5)
    left = solid(128, 128, 128).copy()
    left[:, :W // 4] = _textured(H, W // 4, seed=6)
    cx, _ = composition_centroid([right, left])
    assert cx == pytest.approx(0.5, abs=0.1)
