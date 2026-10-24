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

import math
import subprocess
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .probe import display_dims, scaled_dims


class DecodeError(RuntimeError):
    """A decode that did not end well, raised where `sample_frames` was
    asked to say so.

    By default one path raises it: ffmpeg exiting non-zero having yielded
    nothing, with a `full_rate_vf` fragment set. Without a fragment the
    same outcome yields nothing, as it always has, because a corrupt file
    in a dump of hundreds must not stop the walk; with one,
    nothing yielded is ambiguous - the fragment is the caller's code, and a
    filter ffmpeg rejects looks exactly like a file it cannot decode - so
    what ffmpeg said is handed back instead of thrown away. The keyword-only
    opt-ins `strict`, `check_exit` and `stall_timeout` raise it on more
    paths, each through a subclass named below; a walker that must not stop
    catches this base class per clip and records it.

    `stderr` is the tail of ffmpeg's output, `returncode` its exit (negative
    for a signal, None when ffmpeg was never started), `cmd` the argv and
    `yielded` how many frames the caller had already been handed. Frames
    handed over before a raise are a PREFIX of the clip: discard them.

    What this does NOT tell apart on its own: a rejected fragment from a
    file that probes but does not decode. Both exit non-zero with nothing
    yielded (measured 2026-09-27, ffmpeg 6.1.1: an unknown filter name
    exits 8 "No such filter"; a sink path with an unescaped ':' exits 234
    "Invalid argument"; a valid container whose payload is zeroed exits 69
    "Decode error rate 1 exceeds maximum"). The message says which.
    """

    def __init__(self, cmd: list[str], returncode: int | None, stderr: str,
                 *, yielded: int = 0):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        self.yielded = yielded
        said = stderr.strip().splitlines()
        super().__init__(
            self._summary()
            + (": " + " | ".join(said[-4:]) if said else ""))

    def _summary(self) -> str:
        if self.yielded:
            return (f"ffmpeg exited {self.returncode} after yielding "
                    f"{self.yielded} frames")
        return f"ffmpeg exited {self.returncode} having yielded no frames"

    def __reduce__(self):
        # The default rebuilds from `args`, which is the formatted message,
        # and this signature does not take one: an instance sent across a
        # process boundary (concurrent.futures, multiprocessing) would
        # fail to unpickle and hide the error behind a TypeError.
        return (_rebuild_decode_error, (type(self), self.args, self.__dict__))


def _rebuild_decode_error(cls, args, state):
    exc = cls.__new__(cls)
    exc.args = args
    exc.__dict__.update(state)
    return exc


class UnsizableInput(DecodeError):
    """`sample_frames(strict=True)` was given an input it cannot size.

    ffmpeg was not started and no command was built, so `cmd` is `[]`.
    `reason` says which of the two facts the decode needs was missing.

    - `"dimensions"`: how big a frame is. `probe.display_dims` found no
      video stream, a zero width or height, or no answer in time.
    - `"duration"`: how often to sample. The duration is zero, negative,
      NaN or infinite.

    `detail` says it in full. Without `strict` the same inputs yield
    nothing. The second reason is the one `probe_clip` produces by itself.
    It reports 0.0 for any file whose container gives ffprobe no duration.
    A raw H.264 stream and a live Matroska file are two, and both decode
    fine.
    """

    REASONS = ("dimensions", "duration")

    def __init__(self, cmd: list[str], reason: str, detail: str = ""):
        if reason not in self.REASONS:
            raise ValueError(f"reason must be one of {self.REASONS}, "
                             f"not {reason!r}")
        self.reason = reason
        self.detail = detail
        super().__init__(cmd, None, "")

    def _summary(self) -> str:
        said = f": {self.detail}" if self.detail else ""
        return f"cannot size the decode by {self.reason}{said}"


class DecodeTimeout(DecodeError):
    """vmcore killed an ffmpeg for taking too long.

    `phase` says which limit. "stall": `stall_timeout` - ffmpeg stopped
    producing frames, `limit` is the stall that was allowed. "finish":
    `check_exit` was set and ffmpeg had not exited `timeout` seconds after
    its last frame (it was still finalising a proxy or a sink), `limit` is
    that `timeout`. `yielded` is how many frames the caller was handed
    before it. ffmpeg is dead and reaped, and a `proxy_out` it was writing
    is gone. A file a `full_rate_vf` fragment was writing is the caller's
    and is left cut short: delete it. `returncode` is -9, the SIGKILL
    vmcore sent.
    """

    def __init__(self, cmd: list[str], returncode: int | None, stderr: str,
                 *, yielded: int, limit: float, phase: str = "stall"):
        self.limit = limit
        self.phase = phase
        super().__init__(cmd, returncode, stderr, yielded=yielded)

    def _summary(self) -> str:
        if self.phase == "finish":
            after = (f" after its last of {self.yielded} frames"
                     if self.yielded else "")
            return (f"ffmpeg had not exited {self.limit:g}s{after} "
                    f"and was killed")
        if self.yielded:
            return (f"ffmpeg produced no frame for {self.limit:g}s after "
                    f"{self.yielded} frames and was killed")
        return (f"ffmpeg produced no frame within {self.limit:g}s "
                f"and was killed")


class _StallWatch:
    """Kills ffmpeg when the reader has waited too long for its next bytes.

    The clock runs only while the reader is blocked in a read of the pipe
    (`waiting` .. `idle`). Time the consumer spends between frames, with the
    generator suspended at its `yield`, never counts: ffmpeg is alive but
    blocked on a full pipe then, which is not a stall. The birthday
    pipeline's version of this timed wall clock from `Popen` and killed a
    healthy decode feeding a slow consumer (measured 2026-10-22: a 4 s clip,
    `timeout=2`, 0.6 s of consumer work per frame, killed after 4 of 8
    frames at 2.6 s).

    Only a decoder that is STILL running has stalled: a process that has
    already exited is left alone, so a decode that ended while the reader
    was about to wake is never "timed out" after the fact. The kill is
    SIGKILL, because a wedged ffmpeg ignores the polite signals (measured
    2026-10-22, ffmpeg 6.1.1: one blocked opening its input was still
    running 5 s after SIGTERM and after SIGINT, and ended 0.01 s after
    SIGKILL). The reader's pending `read` then returns EOF; the caller
    waits on the process afterwards (`_shutdown`), so it is reaped.
    """

    def __init__(self, proc, limit: float):
        self.proc = proc
        self.limit = limit
        self.fired = False
        self._since: float | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._tick = min(0.5, limit / 4)
        self._thread = threading.Thread(
            target=self._run, name="vmcore-decode-watchdog", daemon=True)
        self._thread.start()

    def waiting(self) -> None:
        with self._lock:
            self._since = time.monotonic()

    def idle(self) -> None:
        with self._lock:
            self._since = None

    def stop(self) -> None:
        self._done.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._done.wait(self._tick):
            with self._lock:
                if (self._since is None
                        or time.monotonic() - self._since < self.limit):
                    continue
                if self.proc.poll() is not None:
                    return
                self.fired = True
                self.proc.kill()
                return


def _read(stream, n: int, watch: "_StallWatch | None") -> bytes:
    """`stream.read(n)`, with the stall clock running around it if there is one."""
    if watch is None:
        return stream.read(n)
    watch.waiting()
    try:
        return stream.read(n)
    finally:
        watch.idle()


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

    A non-finite duration is not handled here, on purpose: NaN returns NaN
    and +inf returns 0.0 (every comparison with NaN is False, and
    `max_frames / inf` is 0.0). Both become an `fps=` ffmpeg rejects, exit
    234 with nothing written (measured 2026-10-22, ffmpeg 6.1.1), and
    `sample_frames` then yields nothing. Falling back to `fps` instead
    would not be safe: the thinning is what spreads the samples across the
    clip, and without it a long clip is sampled from its first
    `max_frames / fps` seconds only, with exit 0. `sample_frames(strict=
    True)` refuses a non-finite duration before this is reached.
    """
    if duration <= 0 or fps <= 0:
        return fps
    return fps if duration * fps <= max_frames else max_frames / duration


def decode_cmd(path: Path, vf: str, proxy_out: Path | None = None, *,
               proxy_ffmpeg: str | Path | None = None) -> list[str]:
    """The exact argv `sample_frames` runs, given its finished analysis chain.

    Both branches carry `-nostdin` straight after `-hide_banner`. Without
    it ffmpeg reads the caller's stdin for keystrokes. A job in a background
    process group whose stdin is still the terminal is stopped by SIGTTOU.
    A stray `q` ends the decode early without an error, and bytes meant for
    the caller are eaten. The frame pipe is stdout and nothing here reads
    stdin as data, so the flag costs nothing. The test file
    tests/test_nostdin.py pins it for every ffmpeg vmcore runs. It also says
    which of these effects were measured.

    Its own function so a test can pin the command's shape - which binary,
    where the chain sits, that the proxy branch is untouched by whatever
    was spliced into the analysis one - without decoding anything. (The
    fused form still probes the source's colour transfer, because the proxy
    branch's chain depends on it; that is ffprobe, not a decode.)

    Two binaries, on purpose. The analysis-only form runs `ffmpeg` from
    PATH. The fused form, which also writes a proxy, runs
    `proxy.full_ffmpeg(proxy_ffmpeg=...)`. `proxy_ffmpeg` therefore changes
    argv[0] of the fused form and nothing else; the analysis-only form
    ignores it, so a caller can pass it on every call without moving its
    analysis decode onto another build. In the fused form one process
    decodes for both outputs, so the named build also decodes the frames
    handed to the analysis; that was already true of `full_ffmpeg()`.
    """
    if proxy_out is None:
        return [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-i", str(path),
            "-vf", vf,
            "-map", "0:v:0",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-",
        ]

    from . import proxy as _proxy

    # Inline, and only public names of the module. A consumer may replace the
    # whole `vmcore.proxy` module with an object that has `full_ffmpeg`,
    # `proxy_vf`, `color_transfer` and `proxy_encode_args`, as it could at
    # 757f484. A private helper here would not exist on that object.
    #
    # The order is 757f484's: the colour probe and the filter, then the
    # binary. A named binary is checked first, so a bad name probes nothing.
    if proxy_ffmpeg is not None:
        binary = _proxy.full_ffmpeg(proxy_ffmpeg=proxy_ffmpeg)
    pvf = _proxy.proxy_vf(_proxy.color_transfer(path))
    if proxy_ffmpeg is None:
        binary = _proxy.full_ffmpeg()
    return [
        binary, "-hide_banner", "-nostdin", "-loglevel", "error",
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
    stall_timeout: float | None = None,
    strict: bool = False,
    check_exit: bool = False,
    proxy_out: Path | None = None,
    full_rate_vf: str | None = None,
    proxy_ffmpeg: str | Path | None = None,
):
    """Yield `(timestamp_seconds, BGR ndarray)` sampled across the clip.

    ffmpeg applies the display-matrix rotation itself, so portrait phone
    clips arrive upright without us doing anything.

    Argument errors and `UnsizableInput` are raised when the generator is
    first advanced, not when `sample_frames` is called.

    With `proxy_out`, the SAME decode also writes a colour-normalised review
    proxy as a second output branch, split before any sampling so the
    analysis branch is untouched. One decode of the library instead of two —
    the proxy costs an encode, not a second read of 20GB. The fused form
    needs the full ffmpeg build (tonemap for HLG sources); analysis-only
    keeps using the system binary, whose decode behaviour is load-bearing.
    `proxy_ffmpeg` names the full build for the fused form (see
    `proxy.full_ffmpeg` for the order it is looked for in otherwise). It
    never changes the analysis-only decode, so it is safe to pass on every
    call; without `proxy_out` it is not used and not checked. With
    `proxy_out` the one process decodes for both outputs, so the named
    build also decodes the analysis frames.

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
    the consumer took everything, two frames, or one and closed; and
    2026-10-22: an ffmpeg blocked opening its input, `timeout=1`, was still
    blocked when the harness killed the process 12s later).

    `stall_timeout` is the bound the read loop lacks. It is in seconds and
    defaults to None, which is off. It is a stall limit and not a total
    deadline. The clock runs only while vmcore waits for ffmpeg to produce
    the next block of the frame pipe, or to end it. The time the consumer
    spends between frames does not count. A decode that delivers a frame
    every 0.4 s is never killed by a limit of 1 s, however long it runs.
    Choose a limit larger than ffmpeg needs for one sampling interval, plus
    its start-up. One interval is `duration / max_frames` seconds of source
    when the clip is thinned.

    On a stall ffmpeg is sent SIGKILL and waited for. A `proxy_out` it was
    writing is removed. `DecodeTimeout`, a `DecodeError`, is raised whether
    or not frames were already yielded, so a stall is never a short clip. It
    is not raised into a consumer that has already closed the generator.
    The wait for ffmpeg's exit after EOF is bounded by `timeout`, and not by
    this limit.

    Two things it cannot end. A process stuck in uninterruptible sleep on a
    dead network volume ignores SIGKILL until the kernel lets go. Nothing in
    user space can end that wait. And the watchdog is a thread. If the
    thread cannot be started, the call raises RuntimeError. On the
    analysis-only path ffmpeg is killed first. With `proxy_out` or
    `full_rate_vf` the decode is drained to EOF before the error comes out,
    so a wedged ffmpeg leaves that wait unbounded.

    With the fragment set, ffmpeg exiting non-zero having yielded nothing
    raises `DecodeError` with what it said, rather than yielding nothing:
    a rejected fragment is the caller's code and is otherwise
    indistinguishable from a file that does not decode - and the raise
    covers both, see the exception. A file `probe` cannot read still
    yields nothing, on either path, before ffmpeg is run at all. Without a
    fragment, stderr is discarded and nothing yielded is nothing yielded,
    as before.

    `check_exit` (default False) makes an ffmpeg that ends on its own with
    a non-zero exit raise `DecodeError`, with its stderr, whether or not
    frames were yielded. Today a decode that dies after the Nth frame -
    the proxy branch of a fused decode losing its disk, a stream that
    breaks, an OOM kill - yields a PREFIX and returns as if the clip were
    that short, and with `proxy_out` or no fragment a decode that dies at
    start yields nothing, also without a word. The raise comes after the
    last frame the caller was handed, because that is the first moment it
    is known: frames received before it are a prefix of the clip, and a
    consumer that records a result per clip must discard them. A
    `proxy_out` is removed first, as it always was on a non-zero exit. "Ends
    on its own" excludes the cases where vmcore stopped it: the frame budget
    on the analysis-only path (it is terminated, its exit says nothing),
    and a consumer that closes the generator (nobody is left to tell). If
    vmcore had to kill it for running past `timeout` after its last frame,
    the error is `DecodeTimeout` with `phase="finish"`.

    What it cannot see is damage ffmpeg tolerates. The recipe is ffmpeg
    6.1.1 and an 8 s clip of 16 expected frames. The payload is zeroed from
    a chosen point to the end of the data. Zeroed from the start, ffmpeg
    exits non-zero and nothing is yielded. Zeroed from 25, 50, 75 or 90
    percent in, it exits 0 and the clip comes back short. The later the
    damage starts, the more frames arrive. The counts depend on the clip
    and the encoder, and no test pins them. A short clip with exit 0 stays
    a short clip here; a consumer that needs the count compares it with what
    it expected. Nor does it cover a trailing partial frame in a stream that
    exits 0.

    `strict` (default False) turns the silent exits that mean "this input
    cannot be sized" into a raise. By default each yields nothing, before
    ffmpeg is run. A corrupt file in a dump of hundreds must not stop the
    walk. With `strict=True` each raises `UnsizableInput`, a `DecodeError`,
    before anything else happens. No proxy directory is made and no ffmpeg
    is started. Its `reason` tells the two cases apart.

    - `"dimensions"`: `probe.display_dims` cannot size the file. There is
      no video stream, or the width or height is zero, or the probe timed
      out. A transport stream cut before its SPS reads 0x0.
    - `"duration"`: `duration` is zero or less, or NaN or infinite. By
      default zero or less is the documented way to ask for no frames. A
      NaN or infinite duration becomes `fps=nan` or `fps=0`, which ffmpeg
      rejects.

    The file is checked first, so a file that cannot be probed is reported
    as "dimensions" even when the duration is also bad. "duration" then
    means the file is fine and only the duration is missing. A caller that
    passes `probe_clip(...).duration` meets this with files that decode
    fine. ffprobe gives some containers no duration and `probe_clip`
    reports 0.0. Four forms gave none, with ffmpeg 6.1.1 on a set of
    synthetic containers. They are a raw H.264 stream, Matroska with
    `-live 1`, Matroska muxed to a pipe, and the last two together. All
    four decode every frame with exit 0. `tests/test_strict.py` builds three of them.

    The default stays the walk rule above. A walker that wants the raise
    catches `DecodeError`. A caller that wants every failure that ffmpeg
    reports to be loud passes `strict=True`, `check_exit=True` and a
    `stall_timeout`. Damage that ffmpeg tolerates stays silent even then, as
    the `check_exit` paragraph above says.

    vmcore does not read the sink or know its path: a caller that needs to
    know the file is complete counts its frames against the source.
    """
    limit = _stall_limit(stall_timeout)
    src = None
    if strict:
        # Both raises come here, before anything else happens: no proxy
        # directory is made, no command is built and no ffmpeg is started.
        # The file comes first. A caller that passes `probe_clip(...)
        # .duration` gets 0.0 for a file that could not be probed at all, and
        # "dimensions" names that cause. "duration" then means the file is
        # fine and only the duration is missing.
        src = display_dims(path)
        if src is None:
            raise UnsizableInput(
                [], "dimensions",
                f"ffprobe found no usable video stream in {path}. That "
                f"means no stream, a zero width or height, or no answer")
        if not (math.isfinite(duration) and duration > 0):
            raise UnsizableInput(
                [], "duration",
                f"duration {duration!r} is not a positive finite number, "
                f"so no sampling rate can be chosen")
    elif duration <= 0:
        return

    fps = thinned_fps(duration, fps, max_frames)
    vf = analysis_vf(fps, long_edge, full_rate_vf)
    # Either way the process has to be allowed to finish on its own: the
    # proxy is still being written, or a full-rate filter's sink is.
    drain = proxy_out is not None or bool(full_rate_vf)

    if proxy_out is not None:
        proxy_out.parent.mkdir(parents=True, exist_ok=True)
    # The keyword is passed only when it is set. A consumer that replaced
    # `decode_cmd` with the three-argument form it has always had keeps working.
    if proxy_ffmpeg is None:
        cmd = decode_cmd(path, vf, proxy_out)
    else:
        cmd = decode_cmd(path, vf, proxy_out, proxy_ffmpeg=proxy_ffmpeg)

    # Every frame is a fixed-size block, so the byte count has to match what
    # ffmpeg emits exactly — one byte out and every later frame is skewed.
    if src is None:
        src = display_dims(path)
        if src is None:
            return
    w, h = scaled_dims(src[0], src[1], long_edge)
    frame_bytes = w * h * 3

    # stderr goes to a file, not a pipe, on the fragment path and when a
    # stall limit or check_exit wants to report it: a pipe that nobody
    # reads while the frame loop blocks on stdout would fill and stall
    # ffmpeg on a file with many decode errors. Otherwise it is discarded,
    # as it always was.
    keep_stderr = bool(full_rate_vf) or limit is not None or check_exit
    with (tempfile.TemporaryFile() if keep_stderr else nullcontext()) as err:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if err is None else err,
            bufsize=frame_bytes * 2,
        )
        idx = 0
        watch = None
        eof = False     # the pipe ended on its own, not because we stopped
        late = False    # we had to kill ffmpeg for outliving `timeout`
        try:
            if limit is not None:
                watch = _StallWatch(proc, limit)
            while True:
                buf = _read(proc.stdout, frame_bytes, watch)
                if not buf or len(buf) < frame_bytes:
                    eof = True
                    break
                if idx < max_frames:
                    yield (idx / fps,
                           np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3))
                idx += 1
                if idx >= max_frames and not drain:
                    break
        finally:
            try:
                late = _shutdown(proc, frame_bytes, timeout, drain=drain,
                                 proxy_out=proxy_out, watch=watch,
                                 settle=check_exit and eof)
            finally:
                if watch is not None:
                    watch.stop()
        yielded = min(idx, max_frames)
        if watch is not None and watch.fired:
            raise DecodeTimeout(cmd, proc.returncode, _stderr_tail(err),
                                yielded=yielded, limit=limit)
        if (full_rate_vf or check_exit) and proc.returncode != 0:
            if check_exit and eof:
                if late:
                    raise DecodeTimeout(cmd, proc.returncode,
                                        _stderr_tail(err), yielded=yielded,
                                        limit=timeout, phase="finish")
                raise DecodeError(cmd, proc.returncode, _stderr_tail(err),
                                  yielded=yielded)
            if full_rate_vf and idx == 0:
                raise DecodeError(cmd, proc.returncode, _stderr_tail(err))


def _stderr_tail(err) -> str:
    """What ffmpeg last said, from the temp file its stderr was sent to."""
    if err is None:
        return ""
    err.seek(0)
    return err.read()[-4096:].decode("utf-8", "replace")


def _stall_limit(stall_timeout: float | None) -> float | None:
    """`stall_timeout` as a limit in seconds, or None when there is none.

    A number that is not positive, or is NaN, is a mistake worth a loud
    error. It would be a watchdog that fires at once or never. A bool is a
    mistake too: `True` would pass as a limit of one second. That holds for
    numpy's bool as well. Anything that is not a number is a TypeError that
    names the argument. Infinity means no limit, and so does an integer too
    large for a float.
    """
    if stall_timeout is None:
        return None
    if isinstance(stall_timeout, (bool, np.bool_)):
        raise TypeError(
            f"stall_timeout must be a number of seconds or None, "
            f"not the bool {stall_timeout!r}")
    try:
        positive = bool(stall_timeout > 0)
        try:
            limit = float(stall_timeout)
        except OverflowError:           # an int too large for a float
            limit = math.inf
    except (TypeError, ValueError):
        raise TypeError(
            f"stall_timeout must be a number of seconds or None, "
            f"not {type(stall_timeout).__name__} {stall_timeout!r}") from None
    if not positive:
        raise ValueError(
            f"stall_timeout must be a positive number of seconds or None, "
            f"not {stall_timeout!r}")
    return None if math.isinf(limit) else limit


def _shutdown(proc, frame_bytes: int, timeout: int, *, drain: bool,
              proxy_out: Path | None, watch: "_StallWatch | None" = None,
              settle: bool = False) -> bool:
    """Stop the decode without truncating a file that's still being written.

    Not draining: enough frames means we're done, so kill the decode.
    Draining: the same process is still writing something - the proxy, or
    a full-rate filter's sink - and breaking the pipe here truncates it.
    Read the rawvideo tail - fps-thinned, so a handful of frames at most -
    to EOF and let ffmpeg finish. `timeout` is checked between those reads
    and bounds the wait after EOF; a read that never returns is not
    bounded by it (see `sample_frames`).

    `settle` is for the analysis-only path once EOF is reached. ffmpeg
    closed the pipe, so it is exiting. Wait for it to. Do not send SIGTERM
    first, so `returncode` is its own exit and not the signal sent to a
    process already on its way out. The wait is bounded by `timeout`,
    as the drain is. A process that outlives it is killed, and the return
    value is True, so both paths report the same `DecodeTimeout`.

    Every kill is followed by a wait, so `returncode` is a number by the
    time anyone reads it: before, a killed process was left for Popen's
    finalizer to reap and `returncode` stayed None - which the proxy
    unlink below treated as "not 0", correct by accident.

    Returns True when ffmpeg had to be killed for outliving `timeout`, on
    the drain or in the settle wait.
    """
    late = False
    if not drain:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if settle:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                late = True
                proc.kill()
                proc.wait()
                return late
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return late

    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _read(proc.stdout, frame_bytes * 16, watch):
                break
        proc.stdout.close()
        proc.wait(timeout=max(1, deadline - time.monotonic()))
    except Exception:
        late = True
        proc.kill()
        proc.wait()
    # A non-zero exit means the proxy is partial; a partial proxy that looks
    # complete is worse than none, because nothing downstream re-renders it.
    # A full-rate filter's sink is the caller's file, so nothing is done to
    # it here; the caller checks it against the source.
    if proxy_out is not None and proc.returncode != 0 and proxy_out.exists():
        proxy_out.unlink()
    return late
