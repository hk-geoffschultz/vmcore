"""Walk a folder of clips and pull technical metadata via ffprobe."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from .media import VIDEO_EXTS


def source_stamp(path: Path) -> tuple[int, int]:
    """`(size, mtime_ns)` — cheap identity for a file too big to hash.

    A source overwritten with different bytes (a re-export, a re-transcode)
    gets a new stamp, so anything keyed on it treats the file as changed
    rather than serving stale results under the old path. The segment cache
    keys renders on this; ingest uses it to decide whether a clip still
    matches the record it already has.
    """
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


@dataclass
class ClipInfo:
    path: str
    name: str
    duration: float           # seconds
    width: int
    height: int
    fps: float
    rotation: int             # degrees, from display matrix
    created: str              # best-effort creation timestamp, ISO or ""
    # Max audio level in dBFS, from ffmpeg's volumedetect. Digital silence
    # reads -91.0 (measured, ffmpeg 6.1.1): a real measurement, not a clamp.
    # -99.0 is a sentinel for "nothing measured" and means one of: no audio
    # stream, `with_audio_peak=False`, the peak timed out, volumedetect
    # printed no level, or the probe itself failed (the error rows).
    audio_peak_db: float
    size_bytes: int
    error: str = ""
    # Added in 0.5.0, after every existing field and all defaulted, so a
    # positional construction keeps working and `as_row()` only grows four
    # keys at its end. Where the clip was shot, from an ISO 6709 tag
    # (`_location`): degrees, None when the file carries no parseable one.
    lat: float | None = None
    lon: float | None = None
    loc_source: str = ""      # "iso6709", or "" when lat/lon are None
    # The Apple local creation date, verbatim with its offset
    # (`2026-07-14T09:12:31-0700`), or "". `created` stays the UTC
    # `creation_time` for an iPhone clip; this is the one that knows the day.
    created_local: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: int = 120) -> str:
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    return proc.stdout


def _parse_fps(rate: str) -> float:
    """ffprobe gives fps as a rational string like '30000/1001'."""
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def _rotation_from_stream(stream: dict) -> int:
    """Phone footage carries rotation in a display matrix side-data entry."""
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    tags = stream.get("tags", {}) or {}
    if "rotate" in tags:
        try:
            return int(float(tags["rotate"])) % 360
        except (TypeError, ValueError):
            pass
    return 0


def _sizing_rotation(stream: dict) -> int:
    """Rotation ffmpeg's decoder will actually APPLY (autorotate).

    fftools autorotates from the display-matrix side data only; a bare
    `tags.rotate` with no matrix is metadata ffmpeg does not act on, so
    swapping w/h for it would size the raw pipe for frames ffmpeg never
    rotates - the reshape then runs on transposed dims (sheared frames,
    exit 0). `_rotation_from_stream` keeps honouring the tag for the
    metadata column; sizing must follow what the pipe emits.
    """
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    return 0


def display_dims(path: Path) -> tuple[int, int] | None:
    """Post-rotation (display) width and height of the first video stream.

    One ffprobe call, and the ONE place rotation is parsed for sizing:
    the display-matrix side data, which is what ffmpeg's autorotate
    reads (see `_sizing_rotation` for why the legacy tag is ignored
    here). None when the file cannot be probed (or the probe hangs past
    `_run`'s timeout).
    """
    try:
        raw = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height:stream_side_data=rotation:stream_tags=rotate",
            "-print_format", "json", str(path),
        ])
        s = json.loads(raw)["streams"][0]
        w, h = int(s["width"]), int(s["height"])
    except (subprocess.TimeoutExpired, OSError, ValueError, KeyError,
            IndexError, TypeError):
        return None
    if w <= 0 or h <= 0:
        return None
    if _sizing_rotation(s) in (90, 270):
        w, h = h, w
    return (w, h)


def scaled_dims(w: int, h: int, long_edge: int) -> tuple[int, int]:
    """Output size of ffmpeg's scale='if(gt(iw,ih),L,-2)':'if(gt(iw,ih),-2,L)'.

    Pure arithmetic, shared by every raw-pipe reader so the byte count per
    frame always matches what ffmpeg emits. ffmpeg computes a `-2` side
    as av_rescale(L, other, this * 2) * 2: round-to-nearest of HALF the
    scaled dimension, then doubled. "Round, then floor to even" (the old
    arithmetic) disagreed whenever the true value was odd or sat next to
    an odd integer - 2560x1080 -> 406 not 404, 1920x1078 -> 540 not 538 -
    and the pipe was then read misaligned. Verified against ffmpeg on
    lavfi sources of those sizes, and for a square source at an odd and an
    even long edge. Floor at 2.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"non-positive dimensions {w}x{h}")
    # Strictly greater, as gt() is: a square source takes the -2 width, so
    # at an odd L ffmpeg emits (L+1)xL. `>=` said Lx(L+1), the same byte
    # count, which a raw-pipe reader would read as sheared frames, exit 0.
    if w > h:
        nw = long_edge
        nh = 2 * ((h * long_edge + w) // (2 * w))
    else:
        nh = long_edge
        nw = 2 * ((w * long_edge + h) // (2 * h))
    return (max(nw, 2), max(nh, 2))


def _creation_time(fmt: dict, stream: dict) -> str:
    for tags in (fmt.get("tags", {}) or {}, stream.get("tags", {}) or {}):
        for key in ("creation_time", "com.apple.quicktime.creationdate", "date"):
            if tags.get(key):
                return str(tags[key])
    return ""


def _creation_local(fmt: dict, stream: dict) -> str:
    """`com.apple.quicktime.creationdate` verbatim, offset included: format
    tags, then stream tags, "" when neither has it. One key on purpose:
    `creation_time` is UTC and `date` may be a bare year, so neither can
    give the local day."""
    for tags in (fmt.get("tags", {}) or {}, stream.get("tags", {}) or {}):
        if tags.get("com.apple.quicktime.creationdate"):
            return str(tags["com.apple.quicktime.creationdate"])
    return ""


_LOCATION_KEYS = ("com.apple.quicktime.location.ISO6709", "location",
                  "location-eng")
# ISO 6709 decimal degrees: latitude, longitude, an optional altitude
# (ignored), an optional trailing solidus. `+37.7749-122.4194+010.000/`.
_ISO6709 = re.compile(
    r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)(?:[+-]\d+(?:\.\d+)?)?/?$")


def _location(fmt: dict, stream: dict) -> tuple[float | None, float | None, str]:
    """`(lat, lon, "iso6709")` from the first non-empty location tag, format
    tags then stream tags, keys in `_LOCATION_KEYS` order. A value that does
    not parse, or names a latitude beyond 90 or a longitude beyond 180 (the
    degrees-minutes form `+3746.49-12225.16` matches the regex too), gives
    `(None, None, "")`. A bad first value does not fall through to a later
    key, so a damaged tag reads as no location rather than being masked by
    another. Never raises."""
    value = ""
    for tags in (fmt.get("tags", {}) or {}, stream.get("tags", {}) or {}):
        for key in _LOCATION_KEYS:
            if tags.get(key):
                value = str(tags[key])
                break
        if value:
            break
    m = _ISO6709.match(value)
    if not m:
        return None, None, ""
    lat, lon = float(m.group(1)), float(m.group(2))
    if abs(lat) > 90 or abs(lon) > 180:
        return None, None, ""
    return lat, lon, "iso6709"


def _probe_cmd(path: Path) -> list[str]:
    return ["ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path)]


def probe_location(path: Path) -> tuple[float | None, float | None, str, str]:
    """`(lat, lon, loc_source, created_local)` for one file: the same
    ffprobe call `probe_clip` makes, without the audio peak and without
    building a `ClipInfo`, for locating records already analysed.

    Reads the format tags and the first video stream's tags, as
    `probe_clip` does; a file with no video stream is read from its format
    tags alone. A timeout, an empty or unreadable probe gives
    `(None, None, "", "")`. Raises `FileNotFoundError` with no ffprobe.
    """
    try:
        data = json.loads(_run(_probe_cmd(path)))
    except (subprocess.TimeoutExpired, ValueError):
        return None, None, "", ""
    if not isinstance(data, dict):
        return None, None, "", ""
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    lat, lon, source = _location(fmt, v)
    return lat, lon, source, _creation_local(fmt, v)


def audio_peak(path: Path, timeout: int = 180) -> float:
    """Max audio level in dBFS via ffmpeg's volumedetect.

    Useful as a proxy for energy: squealing, laughing and crowds peak high,
    which correlates with clips that work as montage openers.
    Returns -99.0 when there is no audio or detection fails.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-i", str(path),
         "-af", "volumedetect", "-vn", "-sn", "-dn", "-f", "null", "-"],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    for line in proc.stderr.splitlines():
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].strip().split()[0])
            except (IndexError, ValueError):
                return -99.0
    return -99.0


def probe_clip(path: Path, with_audio_peak: bool = True) -> ClipInfo:
    try:
        raw = _run(_probe_cmd(path))
    except subprocess.TimeoutExpired:
        # A dead volume or a corrupt container can wedge ffprobe; that is
        # an ERROR row for this clip, not a crashed ingest.
        return ClipInfo(str(path), path.name, 0, 0, 0, 0, 0, "", -99.0,
                        path.stat().st_size if path.exists() else 0,
                        error="ffprobe timed out")
    if not raw.strip():
        return ClipInfo(str(path), path.name, 0, 0, 0, 0, 0, "", -99.0,
                        path.stat().st_size if path.exists() else 0,
                        error="ffprobe returned nothing")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ClipInfo(str(path), path.name, 0, 0, 0, 0, 0, "", -99.0,
                        path.stat().st_size if path.exists() else 0,
                        error=f"bad ffprobe json: {exc}")

    streams = data.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not vstreams:
        return ClipInfo(str(path), path.name, 0, 0, 0, 0, 0, "", -99.0,
                        int(data.get("format", {}).get("size", 0) or 0),
                        error="no video stream")

    v = vstreams[0]
    fmt = data.get("format", {})
    width = int(v.get("width", 0) or 0)
    height = int(v.get("height", 0) or 0)
    rotation = _rotation_from_stream(v)
    # Portrait phone clips report landscape dims plus a 90/270 rotation.
    if rotation in (90, 270):
        width, height = height, width

    duration = 0.0
    for src in (fmt.get("duration"), v.get("duration")):
        try:
            duration = float(src)
            break
        except (TypeError, ValueError):
            continue

    peak = -99.0
    if with_audio_peak and has_audio:
        try:
            peak = audio_peak(path)
        except subprocess.TimeoutExpired:
            # The peak is a ranking hint, not a classification input: a
            # slow volume must not turn a good clip into bucket=ERROR.
            # Leading newline: ingest's progress line ends in \r, and the
            # bridge filters whole lines on the WARNING: prefix.
            print(f"\nWARNING: audio peak timed out for {path.name}",
                  file=sys.stderr, flush=True)

    lat, lon, loc_source = _location(fmt, v)
    return ClipInfo(
        path=str(path),
        name=path.name,
        duration=round(duration, 3),
        width=width,
        height=height,
        fps=round(_parse_fps(v.get("avg_frame_rate", "0/0")), 4),
        rotation=rotation,
        created=_creation_time(fmt, v),
        audio_peak_db=peak,
        size_bytes=int(fmt.get("size", 0) or 0),
        lat=lat,
        lon=lon,
        loc_source=loc_source,
        created_local=_creation_local(fmt, v),
    )


def find_clips(folder: Path, recursive: bool = True) -> list[Path]:
    """All video files under folder, sorted, skipping junk."""
    it = folder.rglob("*") if recursive else folder.glob("*")
    out = []
    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith("._") or p.name.startswith("."):
            continue  # macOS resource forks and dotfiles
        if p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return sorted(out)
