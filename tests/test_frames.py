"""Frame sampling.

The load-bearing property here is byte alignment: every frame read off the
raw pipe is a fixed-size block, and if the predicted size disagrees with
what ffmpeg emits by even one byte, every subsequent frame is skewed and
the failure looks like garbled content rather than an error. The real-media
tests below exist to catch exactly that, which is why they shell out to
ffmpeg rather than mocking the decode.
"""
from __future__ import annotations

import pytest

from vmcore.testing import make_mp4
from vmcore.frames import analysis_vf, sample_frames, thinned_fps
from vmcore.probe import display_dims, scaled_dims

LONG_EDGE = 64  # keep the decode tiny; the arithmetic is what's under test


# --------------------------------------------------------------------------
# Rate thinning
# --------------------------------------------------------------------------

def test_short_clip_keeps_its_sample_rate():
    assert thinned_fps(duration=10.0, fps=2.0, max_frames=60) == 2.0


def test_long_clip_is_thinned_to_the_frame_budget():
    # 20 minutes at 2fps would be 2400 frames; the budget is 60.
    fps = thinned_fps(duration=1200.0, fps=2.0, max_frames=60)
    assert fps * 1200.0 == pytest.approx(60)


def test_thinning_is_exactly_at_the_boundary():
    """duration * fps == max_frames must not be thinned."""
    assert thinned_fps(duration=30.0, fps=2.0, max_frames=60) == 2.0


def test_zero_duration_does_not_divide_by_zero():
    assert thinned_fps(duration=0.0, fps=2.0, max_frames=60) == 2.0


def test_filter_chain_pins_the_long_edge_on_either_orientation():
    vf = analysis_vf(2.0, 960)
    assert "fps=2.000000" in vf
    # Landscape pins width, portrait pins height; -2 keeps dims even.
    assert "if(gt(iw,ih),960,-2)" in vf
    assert "if(gt(iw,ih),-2,960)" in vf


# --------------------------------------------------------------------------
# Real decode
# --------------------------------------------------------------------------

def test_frames_match_the_predicted_block_size(tiny_clip):
    """The contract that keeps the raw pipe aligned."""
    src = display_dims(tiny_clip)
    assert src is not None
    w, h = scaled_dims(src[0], src[1], LONG_EDGE)

    got = list(sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE))

    assert got, "expected at least one frame from a 1s clip at 2fps"
    for _, frame in got:
        assert frame.shape == (h, w, 3)
        assert frame.dtype.name == "uint8"


def test_timestamps_step_by_the_sample_period(tiny_clip):
    got = list(sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE))
    times = [t for t, _ in got]
    assert times == sorted(times)
    for i, t in enumerate(times):
        assert t == pytest.approx(i / 2.0)


def test_frame_budget_is_not_exceeded(tmp_path):
    clip = tmp_path / "longer.mp4"
    make_mp4(clip, duration=2.0)
    got = list(sample_frames(clip, 2.0, fps=8.0, max_frames=3,
                             long_edge=LONG_EDGE))
    assert len(got) <= 3


def test_zero_duration_yields_nothing(tiny_clip):
    assert list(sample_frames(tiny_clip, 0.0, long_edge=LONG_EDGE)) == []


def test_unprobeable_file_yields_nothing_rather_than_raising(tmp_path):
    """A corrupt file in a dump of hundreds must not stop the walk."""
    junk = tmp_path / "not-really.mp4"
    junk.write_bytes(b"not a video")
    assert list(sample_frames(junk, 5.0, long_edge=LONG_EDGE)) == []
