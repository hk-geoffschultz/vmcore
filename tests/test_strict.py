"""`sample_frames(strict=True)`: refuse to guess.

Two kinds of input cannot be sized, and `sample_frames` yields nothing for
both by default. A file `probe.display_dims` cannot size has no frame size.
A duration of zero or less, NaN or infinity gives no sampling rate. A test
in test_frames.py pins both for a consumer that walks hundreds of files.
`strict=True` turns each into `UnsizableInput`, a `DecodeError`, before
anything else happens. Its `reason`, "dimensions" or "duration", tells them
apart.

Tests named `test_default_*` pin what the code does WITHOUT the argument;
they must pass on the commit before this one as well, and are run there.
"""
from __future__ import annotations

import json
import math
import pickle
import subprocess
from pathlib import Path

import pytest

from vmcore import frames as F
from vmcore.frames import DecodeError, sample_frames, thinned_fps
from vmcore.probe import display_dims, probe_clip
from vmcore.testing import _encode, make_mp4

from optins_helpers import (LONG_EDGE, SpyPopen, can_encode_proxy, decoders,
                            make_audio_only, make_headless_mts, spy)


@pytest.fixture
def unsizable(tmp_path) -> dict[str, Path]:
    """Five inputs `display_dims` returns None for, for five reasons."""
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    whole = tmp_path / "whole.mp4"
    make_mp4(whole, duration=2.0)
    truncated = tmp_path / "truncated.mp4"       # the moov atom is at the end
    truncated.write_bytes(whole.read_bytes()[: whole.stat().st_size // 2])
    audio = tmp_path / "audio.m4a"
    make_audio_only(audio)
    mts = tmp_path / "headless.mts"
    make_headless_mts(mts)
    missing = tmp_path / "gone.mp4"
    return {"junk": junk, "truncated": truncated, "audio-only": audio,
            "width-0 mts": mts, "missing": missing}


def test_the_fixtures_really_are_unsizable(unsizable):
    for name, path in unsizable.items():
        assert display_dims(path) is None, name


def test_the_width_zero_mts_probes_as_zero_by_zero(unsizable):
    """The input the app's ERROR-not-EMPTY rule was written against, built
    here rather than carried: ffprobe reports the stream, with no size."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height",
         "-print_format", "json", str(unsizable["width-0 mts"])],
        capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    s = json.loads(out)["streams"][0]
    assert (s["codec_name"], s["width"], s["height"]) == ("h264", 0, 0)


def test_default_unsizable_inputs_yield_nothing_and_start_no_decode(
        unsizable, monkeypatch):
    started = spy(monkeypatch)
    for name, path in unsizable.items():
        assert list(sample_frames(path, 5.0, long_edge=LONG_EDGE)) == [], name
    assert decoders(started) == []


@pytest.mark.parametrize("kw", [{}, {"full_rate_vf": "null"}],
                         ids=["plain", "fragment"])
def test_strict_raises_for_each_unsizable_input(unsizable, monkeypatch, kw):
    started = spy(monkeypatch)
    for name, path in unsizable.items():
        with pytest.raises(F.UnsizableInput) as e:
            list(sample_frames(path, 5.0, long_edge=LONG_EDGE, strict=True,
                               **kw))
        err = e.value
        assert isinstance(err, DecodeError) and isinstance(err, RuntimeError)
        assert err.reason == "dimensions", name
        assert err.returncode is None and err.yielded == 0, name
        assert err.stderr == "" and err.cmd == [], name
        assert str(path) in str(err), name
    assert decoders(started) == []


def test_strict_with_a_fragment_raises_before_the_fragment_is_used(
        unsizable, tmp_path):
    sink = tmp_path / "sink.txt"
    with pytest.raises(F.UnsizableInput):
        list(sample_frames(unsizable["junk"], 5.0, long_edge=LONG_EDGE,
                           strict=True,
                           full_rate_vf=f"metadata=mode=print:file={sink}"))
    assert not sink.exists()


@pytest.mark.skipif(not can_encode_proxy(),
                    reason="no libx264 in the proxy ffmpeg build")
@pytest.mark.parametrize("which", ["dimensions", "duration"])
def test_strict_with_proxy_out_raises_and_makes_no_directory(
        unsizable, tiny_clip, tmp_path, which):
    """Both raises come before the proxy directory is made. The default
    path makes it first, as the next test pins. Without strict, the same
    call leaves an empty directory behind."""
    proxy = tmp_path / "px" / "x.mp4"
    path, duration = ((unsizable["junk"], 5.0) if which == "dimensions"
                      else (tiny_clip, float("nan")))
    with pytest.raises(F.UnsizableInput) as e:
        list(sample_frames(path, duration, long_edge=LONG_EDGE, strict=True,
                           proxy_out=proxy))
    assert e.value.reason == which
    assert not proxy.exists() and not proxy.parent.exists()


def test_default_an_unsizable_file_with_proxy_out_still_makes_the_directory(
        unsizable, tmp_path):
    """Pinned because 757f484 does it: the directory is made before the
    file is probed, so a file that cannot be sized leaves it empty."""
    proxy = tmp_path / "px" / "x.mp4"
    assert list(sample_frames(unsizable["junk"], 5.0, long_edge=LONG_EDGE,
                              proxy_out=proxy)) == []
    assert proxy.parent.is_dir() and list(proxy.parent.iterdir()) == []


def test_default_a_zero_duration_with_proxy_out_makes_nothing(
        unsizable, tmp_path):
    proxy = tmp_path / "px" / "x.mp4"
    assert list(sample_frames(unsizable["junk"], 0.0, long_edge=LONG_EDGE,
                              proxy_out=proxy)) == []
    assert not proxy.parent.exists()


def test_strict_is_invisible_on_a_clip_that_sizes(tmp_path):
    clip = tmp_path / "ok.mp4"
    make_mp4(clip, duration=2.0)
    a = list(sample_frames(clip, 2.0, fps=2.0, long_edge=LONG_EDGE))
    b = list(sample_frames(clip, 2.0, fps=2.0, long_edge=LONG_EDGE,
                           strict=True))
    assert len(a) == len(b) == 4
    assert [t for t, _ in a] == [t for t, _ in b]
    for (_, x), (_, y) in zip(a, b):
        assert x.tobytes() == y.tobytes()


@pytest.mark.parametrize("d", [float("nan"), float("inf"), -float("inf"),
                               0.0, -1.0],
                         ids=["nan", "inf", "-inf", "zero", "negative"])
def test_strict_raises_for_a_duration_it_cannot_sample_from(
        tiny_clip, monkeypatch, d):
    started = spy(monkeypatch)
    with pytest.raises(F.UnsizableInput) as e:
        list(sample_frames(tiny_clip, d, long_edge=LONG_EDGE, strict=True))
    err = e.value
    assert err.reason == "duration" and err.cmd == [] and repr(d) in str(err)
    assert decoders(started) == [], "no ffmpeg is started"


def test_strict_samples_from_a_tiny_positive_duration(tiny_clip):
    """A tiny positive duration is a duration: it samples."""
    assert len(list(sample_frames(tiny_clip, 1e-9, long_edge=LONG_EDGE,
                                  strict=True))) > 0


def test_a_file_that_cannot_be_probed_is_dimensions_even_with_duration_zero(
        unsizable):
    """`probe_clip` reports 0.0 for a file it could not probe. A caller that
    passes that duration would be told 'duration' about a corrupt file. So
    the file is checked first."""
    with pytest.raises(F.UnsizableInput) as e:
        list(sample_frames(unsizable["junk"], 0.0, long_edge=LONG_EDGE,
                           strict=True))
    assert e.value.reason == "dimensions"


@pytest.mark.parametrize("d", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_default_a_non_finite_duration_yields_nothing(tiny_clip, d):
    """Today: `fps=nan` / `fps=0.000000`, which ffmpeg rejects (exit 234,
    nothing written), so the walk gets an empty clip with no explanation."""
    assert list(sample_frames(tiny_clip, d, long_edge=LONG_EDGE)) == []


def test_default_a_non_finite_duration_with_a_fragment_raises_as_before(
        tiny_clip):
    with pytest.raises(DecodeError) as e:
        list(sample_frames(tiny_clip, float("nan"), long_edge=LONG_EDGE,
                           full_rate_vf="null"))
    assert type(e.value) is DecodeError
    assert e.value.returncode == 234


@pytest.mark.parametrize("d", [0.0, -1.0], ids=["zero", "negative"])
def test_default_a_duration_of_zero_or_less_yields_nothing_and_starts_nothing(
        tiny_clip, monkeypatch, d):
    """Zero is the documented way to ask for no frames, and
    test_frames.py::test_zero_duration_yields_nothing pins it. By default
    nothing is started, not even the sizing probe."""
    started = spy(monkeypatch)
    assert list(sample_frames(tiny_clip, d, long_edge=LONG_EDGE)) == []
    assert started == []


def test_default_thinned_fps_is_unchanged_for_non_finite_durations():
    """Pinned so that a later 'fix' to the fallback is a decision. NaN is
    returned as NaN and +inf as 0.0; neither is a usable rate."""
    assert math.isnan(thinned_fps(float("nan"), 2.0, 60))
    assert thinned_fps(float("inf"), 2.0, 60) == 0.0
    assert thinned_fps(-float("inf"), 2.0, 60) == 2.0


# --- containers that decode while ffprobe gives no duration ----------------

def _raw_h264(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=24:duration=4",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-f", "h264", str(path)], check=True, capture_output=True,
        stdin=subprocess.DEVNULL)


def _live_mkv(path: Path) -> None:
    _encode(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=24:duration=4",
             "-pix_fmt", "yuv420p", "-f", "matroska", "-live", "1"], path)


def _piped_mkv(path: Path) -> None:
    """Muxed to a pipe, so the muxer cannot seek back to write a duration."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=24:duration=4",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-f", "matroska", "-"], check=True, capture_output=True,
        stdin=subprocess.DEVNULL)
    path.write_bytes(r.stdout)


NO_DURATION = {"raw-h264": (_raw_h264, ".h264"), "live-mkv": (_live_mkv, ".mkv"),
               "piped-mkv": (_piped_mkv, ".mkv")}


@pytest.fixture(params=sorted(NO_DURATION))
def no_duration_clip(request, tmp_path) -> Path:
    if not can_encode_proxy():
        pytest.skip("no libx264 in the ffmpeg build")
    build, ext = NO_DURATION[request.param]
    path = tmp_path / ("clip" + ext)
    build(path)
    return path


def test_some_containers_decode_fine_while_ffprobe_gives_no_duration(
        no_duration_clip):
    """Measured 2026-10-22, ffmpeg 6.1.1, on 21 synthetic containers. Of the
    18 whose video ffprobe can size, 14 give a duration and 4 do not. The
    four are a raw H.264 stream, Matroska with `-live 1`, Matroska muxed to
    a pipe, and the last two together. These all give one: MP4 in five
    forms, MOV, MPEG-TS, a TS cut at 20000 bytes, FLV, AVI, NUT, ordinary
    Matroska and an MPEG program stream. The program stream reports less
    than the true 4 s, which still samples. How much less depends on the
    build, and this test does not pin it. `probe_clip` reports 0.0 for the
    four, and all four decode every frame with exit 0. This test builds
    three of them."""
    info = probe_clip(no_duration_clip, with_audio_peak=False)
    assert info.duration == 0.0 and info.error == ""
    assert (info.width, info.height) == (64, 36)
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i",
         str(no_duration_clip), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt",
         "gray", "-"], capture_output=True, stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and len(r.stdout) // (64 * 36) == 96


def test_default_a_decodable_file_with_no_duration_yields_nothing(
        no_duration_clip):
    """The silent skip: given the duration `probe_clip` returned, the walk
    gets an empty clip. Given the true duration, the same file samples."""
    info = probe_clip(no_duration_clip, with_audio_peak=False)
    assert list(sample_frames(no_duration_clip, info.duration,
                              long_edge=LONG_EDGE)) == []
    assert len(list(sample_frames(no_duration_clip, 4.0,
                                  long_edge=LONG_EDGE))) == 8


def test_strict_raises_duration_for_a_decodable_file_with_no_duration(
        no_duration_clip):
    info = probe_clip(no_duration_clip, with_audio_peak=False)
    with pytest.raises(F.UnsizableInput) as e:
        list(sample_frames(no_duration_clip, info.duration,
                           long_edge=LONG_EDGE, strict=True))
    assert e.value.reason == "duration"
    assert len(list(sample_frames(no_duration_clip, 4.0, long_edge=LONG_EDGE,
                                  strict=True))) == 8


# --- the exception ---------------------------------------------------------

def test_unsizable_input_message_and_fields():
    err = F.UnsizableInput(["ffmpeg", "-i", "x"], "dimensions", "because")
    assert str(err) == "cannot size the decode by dimensions: because"
    assert err.reason == "dimensions" and err.detail == "because"
    assert err.cmd == ["ffmpeg", "-i", "x"]
    assert err.returncode is None and err.stderr == "" and err.yielded == 0
    assert str(F.UnsizableInput([], "duration")) == (
        "cannot size the decode by duration")


def test_unsizable_input_knows_exactly_two_reasons():
    assert F.UnsizableInput.REASONS == ("dimensions", "duration")
    with pytest.raises(ValueError, match="reason"):
        F.UnsizableInput([], "no size")


def test_default_decode_error_message_is_the_one_it_always_was():
    err = DecodeError(["ffmpeg"], 8, "line one\nNo such filter: 'nosuch'\n")
    assert str(err) == ("ffmpeg exited 8 having yielded no frames: "
                        "line one | No such filter: 'nosuch'")


def test_decode_error_family_survives_a_pickle_round_trip():
    """The exceptions take several arguments, so the default pickle (which
    rebuilds from the formatted message) fails to load them in the parent
    of a worker process, hiding the error behind a TypeError."""
    for err in (DecodeError(["ffmpeg"], 1, "boom\n", yielded=3),
                F.UnsizableInput(["ffmpeg"], "dimensions", "no size")):
        back = pickle.loads(pickle.dumps(err))
        assert type(back) is type(err)
        assert str(back) == str(err)
        assert back.__dict__ == err.__dict__
