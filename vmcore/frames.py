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

One addition since (0.3.0): `full_rate_vf`, a filter fragment a caller
can ride on this decode ahead of the `fps=` thinning, where it sees every
native frame. It exists because the decode is the most expensive thing a
consumer does with a clip, and a second one for a per-frame measurement
doubles it, while a filter on the existing chain costs nothing that
separates from run-to-run noise. vmcore splices the fragment in and lets
ffmpeg finish, so a file the filter writes is complete when the decode
ends on its own; it does not know what the fragment measures. What a
number means stays with the caller. What vmcore does own is the FORMAT
of what the `metadata` filter prints - `metadata_print_blocks` reads it
into one dict per frame, the same way the pipe's block size is a fact
about ffmpeg and not about any footage - and ffmpeg's own complaint: a
fragment it rejects raises `DecodeError` instead of yielding nothing.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .probe import display_dims, scaled_dims


class DecodeError(RuntimeError):
    """ffmpeg exited non-zero having yielded nothing, with a fragment set.

    Raised by `sample_frames` on that path only. Without a fragment the
    same outcome yields nothing, as it always has, because a corrupt file
    in a dump of hundreds must not stop the walk; with one, nothing
    yielded is ambiguous - the fragment is the caller's code, and a
    filter ffmpeg rejects looks exactly like a file it cannot decode - so
    what ffmpeg said is handed back instead of thrown away. `stderr` is
    the tail of its output, `returncode` its exit, `cmd` the argv.

    What this does NOT tell apart on its own: a rejected fragment from a
    file that probes but does not decode. Both exit non-zero with nothing
    yielded (measured 2026-09-27, ffmpeg 6.1.1: an unknown filter name
    exits 8 "No such filter"; a sink path with an unescaped ':' exits 234
    "Invalid argument"; a valid container whose payload is zeroed exits 69
    "Decode error rate 1 exceeds maximum"). The message says which. A
    walker that must not stop catches this per clip and records it.
    """

    def __init__(self, cmd: list[str], returncode: int | None, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        said = stderr.strip().splitlines()
        super().__init__(
            f"ffmpeg exited {returncode} having yielded no frames"
            + (": " + " | ".join(said[-4:]) if said else ""))


def analysis_vf(fps: float, long_edge: int,
                full_rate_vf: str | None = None) -> str:
    r"""The filter chain the raw pipe reads through.

    Kept as its own function because `scaled_dims` has to predict this
    chain's output size exactly; when one changes the other must.

    `full_rate_vf` is a filter fragment spliced in BEFORE the `fps=`
    thinning, so it sees every native frame rather than the sampled ones -
    a scene-change detector (`scdet`) or a `metadata=mode=print:file=PATH`
    sink is the intended kind of thing. With it unset (or empty) the chain
    is byte-identical to what it was before the argument existed.

    The contract the fragment has to keep: it observes and passes through.
    It must not change the frame's aspect, drop or duplicate frames, move
    timestamps, or write to stdout. `scaled_dims` predicts the pipe's
    block size from the `fps=`/`scale=` tail alone and `sample_frames`
    computes each timestamp from the sample index, so (measured
    2026-09-27, ffmpeg 6.1.1, a 64x36 clip at long_edge 64): a fragment
    that changes the aspect - `crop=32:36`, `pad=96:36` - misaligns the
    pipe with exit code 0 (6 and 2 "frames" read where 4 were emitted);
    one that scales without changing it (`scale=32:18`) keeps the block
    size, because the tail scales it back, but the analysis then sees
    resampled pixels; one that drops frames puts the wrong instant under
    every timestamp; and a `metadata=...:file=-` sink writes its text
    into the frame pipe (9 blocks read where 8 were emitted). ffmpeg may
    auto-insert a pixel-format conversion ahead of a fragment that does
    not accept the decoder's format; that leaves dimensions alone, so the
    byte contract holds, but the analysis then sees the converted pixels.

    The fragment goes in verbatim, and vmcore stays dumb about ffmpeg's
    syntax, so escaping a `file=` path is the caller's job - and it has
    to survive TWO parsers, not one. The filtergraph parser ends a filter's
    arguments at ',' ';' '[' ']' and strips one level of quotes or
    backslashes; the option parser then splits what is left on ':' and
    strips another. One level is therefore not enough: on this ffmpeg a
    bare path, `'…'` quoted once, or `\:` escaped once are all rejected
    with "Invalid argument" (and see `sample_frames` for what that raises).
    Two forms are pinned by a test on a path with a space and a colon:
    quote the value and backslash the colon inside the quotes,
    `file='/a b\:c/sink.txt'`, or double-backslash it bare,
    `file=/a\\ b\\:c/sink.txt`. The simplest thing is to choose a sink
    path with none of those characters (a temporary file in a plain
    directory) and escape nothing.
    """
    tail = (
        f"fps={fps:.6f},"
        f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
    )
    if full_rate_vf:
        return f"{full_rate_vf},{tail}"
    return tail


def metadata_print_blocks(text: str) -> list[dict]:
    """What ffmpeg's `metadata=mode=print` filter wrote, one dict per frame.

    The format (ffmpeg 6.1.1, pinned by a test on a real cut): a header
    line `frame:N pts:P pts_time:T` per frame, then `key=value` lines
    until the next header. Each dict carries `frame` (int) and `pts_time`
    (float) from the header and every `key=value` as a STRING - the
    filter prints `0` for the first frame's `pts_time`, `1` for a
    `lavfi.scd.time` of one second and `85.547` for a score, so a caller
    that wants numbers parses floats and never matches a decimal shape.
    Lines before the first header are ignored; the frame numbers are
    ffmpeg's own and run from 0 in order, so a caller counting whether a
    sink is whole compares `len` against the file's native frame count.

    This is a reader of a format, not of a measurement: it carries no
    opinion about what any key means. It lives here rather than in every
    consumer because a filter's print format is a fact about ffmpeg, like
    the pipe's block size, and two copies of a parser of the same text
    would drift the way two copies of `_scaled_dims` once did.
    """
    blocks: list[dict] = []
    for line in text.splitlines():
        if line.startswith("frame:"):
            fields = dict(tok.split(":", 1) for tok in line.split())
            blocks.append({"frame": int(fields["frame"]),
                           "pts_time": float(fields["pts_time"])})
        elif "=" in line and blocks:
            k, v = line.split("=", 1)
            blocks[-1][k] = v
    return blocks


def thinned_fps(duration: float, fps: float, max_frames: int) -> float:
    """Drop the sample rate so a long clip still yields at most max_frames.

    What this bounds is the number of frames read off the pipe and handed
    to numpy, not the decode: ffmpeg decodes every native frame either way
    and `fps=` drops after decoding, so a 20-minute accidental recording is
    still a 20-minute decode - it yields 60 frames instead of 2400. The
    budget break in `sample_frames` does not shorten it either: with
    thinning, the last sample falls in the clip's final interval and the
    fps filter emits it only at EOF, so ffmpeg has finished by the time
    Python stops reading (measured 2026-09-27, an 8s clip, every run).
    Returns `fps` unchanged when the clip is short enough.
    """
    if duration <= 0 or fps <= 0:
        return fps
    return fps if duration * fps <= max_frames else max_frames / duration


def decode_cmd(path: Path, vf: str, proxy_out: Path | None = None) -> list[str]:
    """The exact argv `sample_frames` runs, given its finished analysis chain.

    Its own function so a test can pin the command's shape - which binary,
    where the chain sits, that the proxy branch is untouched by whatever
    was spliced into the analysis one - without decoding anything. (The
    fused form still probes the source's colour transfer, because the proxy
    branch's chain depends on it; that is ffprobe, not a decode.)
    """
    if proxy_out is None:
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-vf", vf,
            "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-",
        ]

    from . import proxy as _proxy

    pvf = _proxy.proxy_vf(_proxy.color_transfer(path))
    return [
        _proxy.full_ffmpeg(), "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-filter_complex",
        f"[0:v:0]split=2[an][px];[an]{vf}[aout];[px]{pvf}[pout]",
        "-map", "[aout]", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        "-map", "[pout]",
    ] + _proxy.proxy_encode_args(proxy_out)


def sample_frames(
    path: Path,
    duration: float,
    *,
    fps: float = 2.0,
    max_frames: int = 60,
    long_edge: int = 960,
    timeout: int = 900,
    proxy_out: Path | None = None,
    full_rate_vf: str | None = None,
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

    With `full_rate_vf`, a filter fragment rides the same decode ahead of
    the `fps=` thinning, on the analysis branch, where it sees every native
    frame (see `analysis_vf` for what the fragment may and may not do, and
    for the escaping the caller owes). The frames yielded are unchanged by
    it. What changes is how the decode ends: with the fragment set the
    process is never terminated at the frame budget - the pipe is drained
    to EOF and ffmpeg exits on its own, the way the fused proxy form
    already ends, so a filter writing a file (`metadata=mode=print:file=`)
    gets to close and flush it. A killed ffmpeg still runs its cleanup, so
    the file is not empty - it is cut short at wherever the decode was,
    and reads as complete to anything that does not count. How short is
    load-dependent and moves run to run: on an 8s clip of 192 native
    frames the old code's kill left roughly a quarter to a half of them
    behind, never zero (2026-09-27, 64x36 and 320x180, 5 runs each,
    three sets of runs on one box gave three different ranges). The cost
    is that a clip past the budget is decoded to its end instead of being
    cut off; that is the point of asking to see every frame.

    `timeout` is not a bound on the decode, on any path. The read loop
    has no deadline; `timeout` covers only `_shutdown`'s drain after the
    consumer stops early, checked between reads of sixteen sampled frames,
    and the wait for ffmpeg's exit after EOF. An ffmpeg that stalls
    mid-stream is not what it guards against (measured 2026-09-27: a 4s
    clip through a `realtime` fragment with `timeout=1` took 4.1s whether
    the consumer took everything, two frames, or one and closed).

    With the fragment set, ffmpeg exiting non-zero having yielded nothing
    raises `DecodeError` with what it said, rather than yielding nothing:
    a rejected fragment is the caller's code and is otherwise
    indistinguishable from a file that does not decode - and the raise
    covers both, see the exception. A file `probe` cannot read still
    yields nothing, on either path, before ffmpeg is run at all. Without a
    fragment, stderr is discarded and nothing yielded is nothing yielded,
    as before.

    vmcore does not read the sink or know its path: a caller that needs to
    know the file is complete counts its frames against the source.
    """
    if duration <= 0:
        return

    fps = thinned_fps(duration, fps, max_frames)
    vf = analysis_vf(fps, long_edge, full_rate_vf)
    # Either way the process has to be allowed to finish on its own: the
    # proxy is still being written, or a full-rate filter's sink is.
    drain = proxy_out is not None or bool(full_rate_vf)

    if proxy_out is not None:
        proxy_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = decode_cmd(path, vf, proxy_out)

    # Every frame is a fixed-size block, so the byte count has to match what
    # ffmpeg emits exactly — one byte out and every later frame is skewed.
    src = display_dims(path)
    if src is None:
        return
    w, h = scaled_dims(src[0], src[1], long_edge)
    frame_bytes = w * h * 3

    # stderr goes to a file, not a pipe, on the fragment path: a pipe that
    # nobody reads while the frame loop blocks on stdout would fill and
    # stall ffmpeg on a file with many decode errors. Without a fragment it
    # is discarded, as it always was.
    with (tempfile.TemporaryFile() if full_rate_vf else nullcontext()) as err:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if err is None else err,
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
                if idx >= max_frames and not drain:
                    break
        finally:
            _shutdown(proc, frame_bytes, timeout, drain=drain,
                      proxy_out=proxy_out)
        if err is not None and idx == 0 and proc.returncode != 0:
            err.seek(0)
            raise DecodeError(cmd, proc.returncode,
                              err.read()[-4096:].decode("utf-8", "replace"))


def _shutdown(proc, frame_bytes: int, timeout: int, *, drain: bool,
              proxy_out: Path | None):
    """Stop the decode without truncating a file that's still being written.

    Not draining: enough frames means we're done, so kill the decode.
    Draining: the same process is still writing something - the proxy, or
    a full-rate filter's sink - and breaking the pipe here truncates it.
    Read the rawvideo tail - fps-thinned, so a handful of frames at most -
    to EOF and let ffmpeg finish. `timeout` is checked between those reads
    and bounds the wait after EOF; a read that never returns is not
    bounded by it (see `sample_frames`).

    Every kill is followed by a wait, so `returncode` is a number by the
    time anyone reads it: before, a killed process was left for Popen's
    finalizer to reap and `returncode` stayed None - which the proxy
    unlink below treated as "not 0", correct by accident.
    """
    if not drain:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
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
        proc.wait()
    # A non-zero exit means the proxy is partial; a partial proxy that looks
    # complete is worse than none, because nothing downstream re-renders it.
    # A full-rate filter's sink is the caller's file, so nothing is done to
    # it here; the caller checks it against the source.
    if proxy_out is not None and proc.returncode != 0 and proxy_out.exists():
        proxy_out.unlink()
