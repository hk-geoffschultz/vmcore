"""Cheap per-frame measurements: how bright, how colourful, how sharp,
how much movement.

Everything here is a *measurement*. Nothing returns a label, a bucket or a
verdict, and nothing compares against a threshold — turning "motion 0.031"
into "high energy" needs a cut-off, and no cut-off exists until it has been
measured against real footage (see CLAUDE.md).

Ported in spirit from the birthday pipeline's `describe.py` and
`frame_sharpness`, but on numpy alone. The originals use OpenCV for
resizing, BGR→HSV and the Laplacian; we get resizing free from
`vmcore.frames` (ffmpeg has already scaled to `long_edge` by the time a
frame arrives), and the other two are a few lines each. That trades a
~60MB wheel for about thirty lines, and keeps the dependency list short
enough to stay honest.

Frames arrive as HxWx3 uint8 in **BGR** order, the way ffmpeg's rawvideo
pipe emits them.
"""
from __future__ import annotations

import numpy as np

# Rec.601 luma weights, in BGR order to match the frame layout.
_LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def luma(frame: np.ndarray) -> np.ndarray:
    """Perceptual grey channel, float32 in 0..1."""
    return (frame.astype(np.float32) @ _LUMA_BGR) / 255.0


def brightness(frames: list[np.ndarray]) -> float:
    """Mean luma across every frame, 0..1."""
    if not frames:
        return 0.0
    return float(np.mean([luma(f).mean() for f in frames]))


def contrast(frames: list[np.ndarray]) -> float:
    """Standard deviation of luma, 0..~0.5.

    Within-frame spread, averaged over frames — a flat grey wall scores
    near zero whether it is dark or bright, a backlit doorway scores high.
    """
    if not frames:
        return 0.0
    return float(np.mean([luma(f).std() for f in frames]))


def saturation_and_hue(frames: list[np.ndarray]) -> tuple[float, float]:
    """Mean HSV saturation (0..1) and dominant hue (degrees, 0..360).

    Hue is a **saturation-weighted circular mean**, for two reasons the
    naive version gets wrong: hue wraps, so averaging 350° and 10° must
    give 0° rather than 180°; and grey pixels have an arbitrary hue, so
    weighting by saturation stops a mostly-grey frame from voting loudly
    for whatever noise its few coloured pixels happen to carry.

    Returns hue 0.0 for frames with no colour at all, where the value is
    meaningless rather than zero-meaning-red.

    **Known trap for calibration: saturation is unstable on dark pixels.**
    HSV saturation is `delta / max`, so when `max` is small a couple of
    codes of chroma noise reads as a large fraction — a near-black frame
    measured here comes back around 0.13 saturated despite looking
    black. That is standard HSV, not a bug, and OpenCV does the same, so
    the measure is left alone rather than quietly reweighted. But a
    threshold swept over `saturation` across a corpus with dark footage in
    it will be reading noise in those clips; pair it with `brightness`
    when the time comes. Pinned by a test so it stays a known property.
    """
    if not frames:
        return 0.0, 0.0

    sat_means: list[float] = []
    x = y = 0.0
    for f in frames:
        arr = f.astype(np.float32) / 255.0
        b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
        mx = np.maximum(np.maximum(b, g), r)
        mn = np.minimum(np.minimum(b, g), r)
        delta = mx - mn

        # HSV saturation: 0 where the pixel is pure grey (mx == 0 or no spread)
        sat = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0.0)
        sat_means.append(float(sat.mean()))

        # Standard HSV hue in degrees, guarding the achromatic case.
        safe = np.maximum(delta, 1e-9)
        hue = np.where(
            delta <= 1e-9, 0.0,
            np.where(
                mx == r, ((g - b) / safe) % 6.0,
                np.where(mx == g, ((b - r) / safe) + 2.0,
                         ((r - g) / safe) + 4.0),
            ),
        ) * 60.0

        rad = np.deg2rad(hue)
        x += float((np.cos(rad) * sat).sum())
        y += float((np.sin(rad) * sat).sum())

    hue_deg = float(np.degrees(np.arctan2(y, x)) % 360.0)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        hue_deg = 0.0
    return float(np.mean(sat_means)), hue_deg


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap focus measure.

    Motion-blurred and out-of-focus frames carry little high-frequency
    energy and score low. The absolute number is scene-dependent (a busy
    build scores higher than a blank wall at equal focus), so it is only
    ever meaningful for ranking clips against each other, never against a
    fixed threshold.

    The 5-point stencil, written out rather than convolved: interior
    pixels only, so no edge-padding convention has to be invented.
    """
    g = luma(frame)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (4.0 * g[1:-1, 1:-1]
           - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def _edge_weight(frame: np.ndarray) -> np.ndarray:
    """The same 5-point Laplacian `sharpness` uses, kept as a map instead
    of collapsed to one number — high-frequency detail (edges, texture,
    a person, a tool, product packaging) reads high; a flat wall or open
    sky reads near zero."""
    g = luma(frame)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return np.zeros((1, 1), dtype=np.float32)
    lap = (4.0 * g[1:-1, 1:-1]
           - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return np.abs(lap)


def composition_centroid(frames: list[np.ndarray]) -> tuple[float, float]:
    """Mean fractional position (x, y), each 0..1, of the frame's edge
    energy — a proxy for "where does the eye land", not a subject
    detector. No face or object model: it's a purely geometric read of
    where local contrast concentrates, on the same reasoning `deliver.py`'s
    crop already needs (a booth build's subject is the work, not a face,
    so the crop centre wants motion or saliency — this is that signal).

    A frame with no discernible detail (flat colour, blown-out white, pure
    black) returns the frame centre (0.5, 0.5) by convention rather than
    an arbitrary corner — there's nothing to weight toward, and reporting
    "centred" is honest about that instead of implying a real reading.

    This says nothing about composition on its own — "is the weight near a
    rule-of-thirds line" is a question for calibration against hand labels,
    the same as every other signal here. It only reports where the mass is.
    """
    if not frames:
        return 0.5, 0.5

    xs_list, ys_list = [], []
    for f in frames:
        weight = _edge_weight(f)
        total = float(weight.sum())
        if total <= 1e-9:
            xs_list.append(0.5)
            ys_list.append(0.5)
            continue
        h, w = weight.shape
        ys, xs = np.mgrid[0:h, 0:w]
        xs_list.append(float((xs * weight).sum() / total) / max(w - 1, 1))
        ys_list.append(float((ys * weight).sum() / total) / max(h - 1, 1))
    return float(np.mean(xs_list)), float(np.mean(ys_list))


def motion(frames: list[np.ndarray]) -> tuple[float, float]:
    """Mean and peak absolute luma difference between consecutive samples.

    Returns `(mean, peak)`, both 0..1. Peak is kept separately because a
    single whip-pan and a steadily drifting handheld shot can average to
    the same number while being nothing alike to cut with.

    Note what this actually measures: change between *sampled* frames, not
    true interframe motion. At 2 fps those samples are half a second apart,
    so a fast pan can alias. That is why the sampling parameters are stored
    alongside the value — the number is only comparable across clips
    measured the same way.
    """
    if len(frames) < 2:
        return 0.0, 0.0
    grays = [luma(f) for f in frames]
    diffs = [float(np.abs(b - a).mean()) for a, b in zip(grays, grays[1:])]
    return float(np.mean(diffs)), float(np.max(diffs))
