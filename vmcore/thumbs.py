"""Evenly spaced JPEG thumbnails from a clip, for human review.

One ffmpeg call per clip, using the same `fps` + `scale` chain that
`vmcore.frames` pipes raw — so a thumbnail shows what the analysis saw,
at a size a person can actually look at.

Deliberately not the origin's `contact.py` approach. That one opens each
clip with `cv2.VideoCapture`, seeks by frame index, and composites a sheet
with `cv2.putText`. It needs OpenCV for all of it, and it bakes a
classifier's verdict into every cell. Writing files and letting HTML do the
layout costs nothing, renders text without a font dependency, and works
for labelling clips that have no verdict yet.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# Wide enough to judge framing on a laptop, small enough that forty clips
# of them load instantly from disk.
THUMB_LONG_EDGE = 320


def thumb_paths(out_dir: Path, key: str, count: int) -> list[Path]:
    """Where `extract` will put this clip's thumbnails, in order."""
    return [out_dir / f"{key}.{i + 1:02d}.jpg" for i in range(count)]


def extract(path: Path, out_dir: Path, key: str, duration: float,
            count: int = 3, long_edge: int = THUMB_LONG_EDGE,
            timeout: int = 120) -> list[Path]:
    """Write `count` thumbnails spread across the clip. Returns what exists.

    Sampled at `count / duration` fps rather than by seeking, because
    seeking on a long-GOP phone clip is both slow and inexact — the same
    reason the analysis path pipes frames instead of seeking. The first
    thumbnail lands slightly after t=0, which is usually an improvement:
    the literal first frame of a handheld clip is often the camera still
    being raised.
    """
    if duration <= 0 or count < 1:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = count / duration
    vf = (f"fps={fps:.6f},"
          f"scale='if(gt(iw,ih),{long_edge},-2)':"
          f"'if(gt(iw,ih),-2,{long_edge})'")
    pattern = str(out_dir / f"{key}.%02d.jpg")

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-vf", vf, "-frames:v", str(count),
         "-q:v", "4", "-y", pattern],
        capture_output=True, timeout=timeout, check=False,
    )
    written = [p for p in thumb_paths(out_dir, key, count) if p.exists()]
    if proc.returncode != 0 and not written:
        return []
    return written
