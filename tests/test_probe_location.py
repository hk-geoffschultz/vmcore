"""Location and the local creation date, through the public functions only.

A fake `ffprobe` on PATH prints canned JSON for each file, so these tests
pin what `probe_clip` and `probe_location` make of the tags a phone writes
without needing a geotagged clip. The fake also logs its argv, which pins
that both functions make the one ffprobe call `probe_clip` always made.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vmcore.probe import ClipInfo, probe_clip, probe_location, scaled_dims

PROBE_ARGV = ["-v", "error", "-print_format", "json",
              "-show_format", "-show_streams"]

FAKE = """#!{python}
import sys
from pathlib import Path
with open({log!r}, "a") as fh:
    fh.write("\\x1f".join(sys.argv[1:]) + "\\n")
canned = Path(sys.argv[-1] + ".json")
if canned.exists():
    sys.stdout.write(canned.read_text())
"""


def _stream(tags: dict | None = None) -> dict:
    s = {"codec_type": "video", "width": 1920, "height": 1080,
         "avg_frame_rate": "30/1", "duration": "2.0"}
    if tags is not None:
        s["tags"] = tags
    return s


@pytest.fixture
def fake_ffprobe(tmp_path, monkeypatch):
    """Returns `make(name, fmt_tags, stream_tags) -> Path` and the argv log."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "argv.log"
    exe = bindir / "ffprobe"
    exe.write_text(FAKE.format(python=sys.executable, log=str(log)))
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    def make(name, fmt_tags=None, stream_tags=None, raw=None):
        clip = tmp_path / name
        clip.write_bytes(b"")
        if raw is None:
            fmt = {"duration": "2.0", "size": "1234"}
            if fmt_tags is not None:
                fmt["tags"] = fmt_tags
            raw = json.dumps({"format": fmt, "streams": [_stream(stream_tags)]})
        Path(str(clip) + ".json").write_text(raw)
        return clip

    def calls():
        if not log.exists():
            return []
        return [line.split("\x1f") for line in log.read_text().splitlines()]

    make.calls = calls
    return make


IPHONE_UTC = "2026-07-14T16:12:31.000000Z"
IPHONE_LOCAL = "2026-07-14T09:12:31-0700"

# (name, format tags, stream tags, lat, lon, loc_source, created_local)
CASES = [
    ("mp4_location.mp4", {"location": "+48.8584+002.2945/"}, None,
     48.8584, 2.2945, "iso6709", ""),
    ("apple_altitude.mov",
     {"com.apple.quicktime.location.ISO6709": "+37.7749-122.4194+010.000/"},
     None, 37.7749, -122.4194, "iso6709", ""),
    ("stream_only.mp4", None, {"location-eng": "+51.5007-000.1246/"},
     51.5007, -0.1246, "iso6709", ""),
    ("malformed.mp4", {"location": "37.7749,-122.4194"}, None,
     None, None, "", ""),
    ("absent.mp4", {"encoder": "Lavf"}, {"handler_name": "VideoHandler"},
     None, None, "", ""),
    ("no_tags.mp4", None, None, None, None, "", ""),
    ("negative.mov",
     {"com.apple.quicktime.location.ISO6709": "-33.8568-151.2153+005.000/"},
     None, -33.8568, -151.2153, "iso6709", ""),
    ("both_dates.mov",
     {"creation_time": IPHONE_UTC,
      "com.apple.quicktime.creationdate": IPHONE_LOCAL,
      "com.apple.quicktime.location.ISO6709": "+37.7749-122.4194+010.000/"},
     None, 37.7749, -122.4194, "iso6709", IPHONE_LOCAL),
    ("local_in_stream.mov", {"creation_time": IPHONE_UTC},
     {"com.apple.quicktime.creationdate": IPHONE_LOCAL},
     None, None, "", IPHONE_LOCAL),
    # Format tags win over stream tags, and the first key in order wins.
    ("format_wins.mp4", {"location": "+10.0+020.0/"},
     {"com.apple.quicktime.location.ISO6709": "+30.0+040.0/"},
     10.0, 20.0, "iso6709", ""),
    ("key_order.mov",
     {"location": "+10.0+020.0/",
      "com.apple.quicktime.location.ISO6709": "+30.0+040.0/"},
     None, 30.0, 40.0, "iso6709", ""),
    # Degrees-minutes matches the regex but is not a coordinate.
    ("out_of_range.mov", {"location": "+3746.494-12225.164/"}, None,
     None, None, "", ""),
    ("no_solidus.mp4", {"location": "+1-2"}, None, 1.0, -2.0, "iso6709", ""),
]


@pytest.mark.parametrize("name,fmt,st,lat,lon,src,local", CASES,
                         ids=[c[0] for c in CASES])
def test_probe_clip_location(fake_ffprobe, name, fmt, st, lat, lon, src, local):
    clip = fake_ffprobe(name, fmt, st)
    info = probe_clip(clip)
    assert info.error == ""
    assert (info.lat, info.lon, info.loc_source, info.created_local) == (
        lat, lon, src, local)
    assert fake_ffprobe.calls() == [PROBE_ARGV + [str(clip)]]


@pytest.mark.parametrize("name,fmt,st,lat,lon,src,local", CASES,
                         ids=[c[0] for c in CASES])
def test_probe_location_agrees(fake_ffprobe, name, fmt, st, lat, lon, src, local):
    clip = fake_ffprobe(name, fmt, st)
    assert probe_location(clip) == (lat, lon, src, local)
    assert fake_ffprobe.calls() == [PROBE_ARGV + [str(clip)]]


def test_created_stays_utc_on_an_iphone_clip(fake_ffprobe):
    clip = fake_ffprobe("iphone.mov", {
        "creation_time": IPHONE_UTC,
        "com.apple.quicktime.creationdate": IPHONE_LOCAL})
    info = probe_clip(clip)
    assert info.created == IPHONE_UTC
    assert info.created_local == IPHONE_LOCAL


def test_created_falls_back_to_the_apple_key_as_before(fake_ffprobe):
    clip = fake_ffprobe("apple_only.mov",
                        {"com.apple.quicktime.creationdate": IPHONE_LOCAL})
    info = probe_clip(clip)
    assert info.created == IPHONE_LOCAL
    assert info.created_local == IPHONE_LOCAL


def test_existing_fields_are_unchanged_and_new_keys_trail(fake_ffprobe):
    clip = fake_ffprobe("rowshape.mov", {
        "creation_time": IPHONE_UTC,
        "com.apple.quicktime.creationdate": IPHONE_LOCAL,
        "location": "+37.7749-122.4194/"})
    row = probe_clip(clip).as_row()
    assert list(row) == [
        "path", "name", "duration", "width", "height", "fps", "rotation",
        "created", "audio_peak_db", "size_bytes", "error",
        "lat", "lon", "loc_source", "created_local"]
    assert {k: row[k] for k in list(row)[:11]} == {
        "path": str(clip), "name": "rowshape.mov", "duration": 2.0,
        "width": 1920, "height": 1080, "fps": 30.0, "rotation": 0,
        "created": IPHONE_UTC, "audio_peak_db": -99.0, "size_bytes": 1234,
        "error": ""}


def test_positional_construction_still_works():
    info = ClipInfo("p", "n", 0, 0, 0, 0, 0, "", -99.0, 0, error="x")
    assert (info.lat, info.lon, info.loc_source, info.created_local) == (
        None, None, "", "")


@pytest.mark.parametrize("raw", ["", "not json", "[]", "null"])
def test_probe_location_on_an_unreadable_probe(fake_ffprobe, raw):
    clip = fake_ffprobe("broken.mp4", raw=raw)
    assert probe_location(clip) == (None, None, "", "")


def test_probe_location_reads_format_tags_without_a_video_stream(fake_ffprobe):
    clip = fake_ffprobe("audio.m4a", raw=json.dumps({
        "format": {"tags": {"location": "+1.5-2.5/",
                            "com.apple.quicktime.creationdate": IPHONE_LOCAL}},
        "streams": [{"codec_type": "audio"}]}))
    assert probe_location(clip) == (1.5, -2.5, "iso6709", IPHONE_LOCAL)
    # probe_clip still makes this an error row, with the new fields defaulted
    info = probe_clip(clip, with_audio_peak=False)
    assert info.error == "no video stream"
    assert (info.lat, info.lon, info.loc_source, info.created_local) == (
        None, None, "", "")


def test_probe_location_without_ffprobe(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        probe_location(tmp_path / "x.mp4")


def test_real_mp4_location_round_trip(tmp_path):
    """The real ffprobe reports a `location` tag the real ffmpeg wrote."""
    clip = tmp_path / "geo.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "color=c=red:size=64x36:rate=1:d=1", "-metadata",
         "location=+37.7749-122.4194/", str(clip)], check=True)
    assert probe_location(clip)[:3] == (37.7749, -122.4194, "iso6709")
    info = probe_clip(clip)
    assert (info.lat, info.lon, info.loc_source) == (37.7749, -122.4194,
                                                     "iso6709")


# scaled_dims: a square source takes the -2 WIDTH, as gt(iw,ih) does. Every
# non-square row is the value it was before the change.
SCALED = [
    (100, 100, 961, (962, 961)),     # square, odd L: was (961, 962)
    (100, 100, 960, (960, 960)),     # square, even L
    (1080, 1080, 961, (962, 961)),
    (1080, 1080, 64, (64, 64)),
    (2560, 1080, 960, (960, 406)),
    (1920, 1078, 960, (960, 540)),
    (1080, 1920, 960, (540, 960)),
    (65, 37, 960, (960, 546)),
    (2560, 1080, 961, (961, 406)),
    (1080, 1920, 961, (540, 961)),
]


@pytest.mark.parametrize("w,h,L,expect", SCALED)
def test_scaled_dims_table(w, h, L, expect):
    assert scaled_dims(w, h, L) == expect


@pytest.mark.parametrize("w,h,L,expect", SCALED)
def test_scaled_dims_matches_ffmpeg_dimensions(tmp_path, w, h, L, expect):
    """The exact width and height ffmpeg's scale filter emits, not just the
    byte count: a transposed answer has the same byte count."""
    out = tmp_path / "f.nut"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         # format=rgb24 in the source graph: a yuv420p colour source
         # rounds 65x37 down to 64x36 before the scale ever sees it
         f"color=c=red:size={w}x{h}:rate=1:d=1,format=rgb24",
         "-frames:v", "1",
         "-vf", f"scale='if(gt(iw,ih),{L},-2)':'if(gt(iw,ih),-2,{L})'",
         "-c:v", "rawvideo", "-pix_fmt", "rgb24", str(out)],
        check=True)
    s = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-print_format", "json", str(out)],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    assert (s["width"], s["height"]) == expect == scaled_dims(w, h, L)
