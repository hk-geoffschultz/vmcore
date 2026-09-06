"""Walk a folder of clips and pull technical metadata via ffprobe."""
from __future__ import annotations

import json
import os
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
    audio_peak_db: float      # max audio level, -inf floor clamped to -99
    size_bytes: int
    error: str = ""

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
    lavfi sources of those sizes. Floor at 2.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"non-positive dimensions {w}x{h}")
    if w >= h:
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


def audio_peak(path: Path, timeout: int = 180) -> float:
    """Max audio level in dBFS via ffmpeg's volumedetect.

    Useful as a proxy for energy: squealing, laughing and crowds peak high,
    which correlates with clips that work as montage openers.
    Returns -99.0 when there is no audio or detection fails.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
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
        raw = _run([
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ])
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
