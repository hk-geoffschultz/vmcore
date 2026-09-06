"""Pull sampled frames out of a clip as raw BGR, without touching disk.

Ported from the birthday pipeline's `bbp/faces.py`, where this generator
only ever lived because it happened to be the thing feeding InsightFace.
Nothing in it is about faces: it decodes, thins to a frame rate, scales,
and yields arrays. That is why it belongs here and not in an app.

Two deliberate changes from the original:

1. **No Config object.** The original took the pipeline's `Config` and read
   three fields off it, which quietly made a decode primitive depend on one
   app's settings schema. Sampling parameters are now plain arguments.

2. **Sizing goes through `probe.display_dims` + `probe.scaled_dims`.** The
   original carried its own `_scaled_dims`, which read rotation from the
   display matrix only and rounded scaled dimensions by "round, then floor
   to even". Both were later fixed in `probe.py` — rotation needs the
   display-matrix/legacy-tag split (a bare `tags.rotate` must NOT drive
   sizing, or frames arrive sheared with exit code 0), and the even-rounding
   had to match ffmpeg's `av_rescale` or the pipe is read misaligned. Using
   probe's versions means one implementation of that contract, already
   covered by 18 tests, instead of a second copy drifting behind it.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from .probe import display_dims, scaled_dims


def analysis_vf(fps: float, long_edge: int) -> str:
    """The filter chain the raw pipe reads through.

    Kept as its own function because `scaled_dims` has to predict this
    chain's output size exactly; when one changes the other must.
    """
    return (
        f"fps={fps:.6f},"
        f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
    )


def thinned_fps(duration: float, fps: float, max_frames: int) -> float:
    """Drop the sample rate so a long clip still yields at most max_frames.

    Protects against a 20-minute accidental recording costing 20 minutes of
    decode. Returns `fps` unchanged when the clip is short enough.
    """
    if duration <= 0 or fps <= 0:
        return fps
    return fps if duration * fps <= max_frames else max_frames / duration


def sample_frames(
    path: Path,
    duration: float,
    *,
    fps: float = 2.0,
    max_frames: int = 60,
    long_edge: int = 960,
    timeout: int = 900,
    proxy_out: Path | None = None,
):
    """Yield `(timestamp_seconds, BGR ndarray)` sampled across the clip.

    ffmpeg applies the display-matrix rotation itself, so portrait phone
    clips arrive upright without us doing anything.

    With `proxy_out`, the SAME decode also writes a colour-normalised review
    proxy as a second output branch, split before any sampling so the
    analysis branch is untouched. One decode of the library instead of two —
    the proxy costs an encode, not a second read of 20GB. The fused form
    needs the full ffmpeg build (tonemap for HLG sources); analysis-only
    keeps using the system binary, whose decode behaviour is load-bearing.
    """
    if duration <= 0:
        return

    fps = thinned_fps(duration, fps, max_frames)
    vf = analysis_vf(fps, long_edge)

    if proxy_out is None:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-vf", vf,
            "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-",
        ]
    else:
        from . import proxy as _proxy

        proxy_out.parent.mkdir(parents=True, exist_ok=True)
        pvf = _proxy.proxy_vf(_proxy.color_transfer(path))
        cmd = [
            _proxy.full_ffmpeg(), "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-filter_complex",
            f"[0:v:0]split=2[an][px];[an]{vf}[aout];[px]{pvf}[pout]",
            "-map", "[aout]", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            "-map", "[pout]",
        ] + _proxy.proxy_encode_args(proxy_out)

    # Every frame is a fixed-size block, so the byte count has to match what
    # ffmpeg emits exactly — one byte out and every later frame is skewed.
    src = display_dims(path)
    if src is None:
        return
    w, h = scaled_dims(src[0], src[1], long_edge)
    frame_bytes = w * h * 3

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=frame_bytes * 2,
    )
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            if idx < max_frames:
                yield (idx / fps,
                       np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3))
            idx += 1
            if idx >= max_frames and proxy_out is None:
                break
    finally:
        _shutdown(proc, frame_bytes, timeout, proxy_out)


def _shutdown(proc, frame_bytes: int, timeout: int, proxy_out: Path | None):
    """Stop the decode without truncating a proxy that's still being written.

    Analysis-only: enough frames means we're done, so kill the decode.
    Fused: the same process is still writing the proxy, and breaking the
    pipe here truncates it. Drain the rawvideo tail — fps-thinned, so a
    handful of frames at most — and let ffmpeg finish.
    """
    if proxy_out is None:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return

    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not proc.stdout.read(frame_bytes * 16):
                break
        proc.stdout.close()
        proc.wait(timeout=max(1, deadline - time.monotonic()))
    except Exception:
        proc.kill()
    # A non-zero exit means the proxy is partial; a partial proxy that looks
    # complete is worse than none, because nothing downstream re-renders it.
    if proc.returncode != 0 and proxy_out.exists():
        proxy_out.unlink()
