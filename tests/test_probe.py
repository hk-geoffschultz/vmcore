from __future__ import annotations

from pathlib import Path

import pytest

import vmcore.probe as probe
from vmcore.probe import _creation_time, _parse_fps, _rotation_from_stream


@pytest.mark.parametrize("rate,expect", [
    ("30000/1001", 30000 / 1001),
    ("24/1", 24.0),
    ("25", 25.0),
    ("0/0", 0.0),
    ("", 0.0),
    ("1/0", 0.0),
    ("abc", 0.0),
    ("x/2", 0.0),
])
def test_parse_fps(rate, expect):
    assert _parse_fps(rate) == pytest.approx(expect)


def test_rotation_from_side_data_list():
    assert _rotation_from_stream({"side_data_list": [{"rotation": -90}]}) == 270
    assert _rotation_from_stream({"side_data_list": [{"rotation": 90.0}]}) == 90
    assert _rotation_from_stream({"side_data_list": [
        {"displaymatrix": "..."}, {"rotation": "180"}]}) == 180
    assert _rotation_from_stream({"side_data_list": [{"rotation": "x"}]}) == 0


def test_rotation_from_tags_rotate():
    assert _rotation_from_stream({"tags": {"rotate": "90"}}) == 90
    assert _rotation_from_stream({"tags": {"rotate": "270"}}) == 270
    assert _rotation_from_stream({"tags": {"rotate": "-90"}}) == 270
    assert _rotation_from_stream({"tags": {"rotate": "bad"}}) == 0


def test_rotation_side_data_wins_and_default_zero():
    s = {"side_data_list": [{"rotation": -90}], "tags": {"rotate": "180"}}
    assert _rotation_from_stream(s) == 270
    assert _rotation_from_stream({}) == 0
    assert _rotation_from_stream({"side_data_list": None, "tags": None}) == 0


def test_creation_time_prefers_format_then_stream():
    fmt = {"tags": {"creation_time": "2026-01-02T03:04:05.000000Z"}}
    st = {"tags": {"com.apple.quicktime.creationdate": "2026-05-06T07:08:09-0700"}}
    assert _creation_time(fmt, st) == "2026-01-02T03:04:05.000000Z"
    assert _creation_time({}, st) == "2026-05-06T07:08:09-0700"
    assert _creation_time({"tags": {"date": "2025"}}, {}) == "2025"
    assert _creation_time({"tags": None}, {"tags": {}}) == ""


# The -2 side is av_rescale of HALF the dimension, then doubled: every
# expectation below was read off ffmpeg's own raw output (see the
# birthday-pipeline fork this was ported from).
@pytest.mark.parametrize("w,h,long_edge,expect", [
    (1920, 1080, 960, (960, 540)),
    (3840, 2160, 960, (960, 540)),
    (1080, 1920, 960, (540, 960)),     # portrait: long edge is the height
    (1920, 1078, 960, (960, 540)),     # 539 -> half 269.5 -> 270 -> 540
    (2560, 1080, 960, (960, 406)),     # 405 -> 406 (old code said 404)
    (1080, 2560, 960, (406, 960)),
    (1920, 810, 960, (960, 406)),
    (640, 350, 960, (960, 526)),
    (1920, 1082, 960, (960, 542)),
    (1000, 222, 960, (960, 214)),
    (4096, 2160, 960, (960, 506)),     # 506.25 -> 506
    (1080, 1920, 640, (360, 640)),
    (640, 2, 960, (960, 4)),           # 3 -> half 1.5 -> 2 -> 4
    (10000, 10, 960, (960, 2)),        # floor at 2
    (100, 100, 64, (64, 64)),
])
def test_scaled_dims_even_rounding(w, h, long_edge, expect):
    assert probe.scaled_dims(w, h, long_edge) == expect


@pytest.mark.parametrize("w,h", [(2560, 1080), (1000, 222), (1920, 1078)])
def test_scaled_dims_matches_ffmpeg_byte_count(w, h):
    # the contract any raw-pipe reader relies on: one frame is exactly
    # scaled_dims(...) rgb24 bytes, for awkward sizes too
    import subprocess

    nw, nh = probe.scaled_dims(w, h, 960)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"color=c=red:size={w}x{h}:rate=1:d=1", "-frames:v", "1",
         "-vf", "scale='if(gt(iw,ih),960,-2)':'if(gt(iw,ih),-2,960)'",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    assert len(proc.stdout) == nw * nh * 3


def test_display_dims_exists(tiny_clip):
    assert callable(probe.display_dims)
    assert probe.display_dims(tiny_clip) == (64, 36)
