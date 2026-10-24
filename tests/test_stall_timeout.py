"""`sample_frames(stall_timeout=...)`: a stalled decode is killed, not waited on.

Without it, `timeout` bounds only the drain after the consumer leaves, so
an ffmpeg that stops producing (a dead volume, a stream that spins) blocks
the read loop forever. The watchdog is a stall limit on the pipe read: it
does not count the consumer's own time, it is not a total, and it kills and
reaps the process whatever the path.

Real ffmpeg is used for the wedge (its input swapped for a FIFO nobody
writes, so it blocks in open(), which is what a read on a dead volume looks
like from the pipe) and for the healthy-but-slow cases (`-re`). The scripted
decoder from optins_helpers is used where the test needs "three frames, then
nothing" on every run.

Tests named test_default_* pin behaviour without the argument.
"""
from __future__ import annotations

import inspect
import os
import pickle
import threading
import time

import numpy as np
import pytest

from vmcore import frames as F
from vmcore.frames import DecodeError, sample_frames
from vmcore.testing import make_mp4

from optins_helpers import (LONG_EDGE, can_encode_proxy, decoders,
                            install_fake, is_reaped, no_hang, realtime, spy)

needs_x264 = pytest.mark.skipif(not can_encode_proxy(),
                                reason="no libx264 in the proxy ffmpeg build")


def watchdogs() -> list[threading.Thread]:
    return [t for t in threading.enumerate()
            if t.name == "vmcore-decode-watchdog"]


@pytest.fixture
def clip8(tmp_path):
    p = tmp_path / "eight.mp4"
    make_mp4(p, duration=8.0)
    return p


def wedge_input(monkeypatch, tmp_path):
    """Swap the real argv's input for a FIFO nobody writes."""
    fifo = tmp_path / "nobody.fifo"
    os.mkfifo(fifo)
    real = F.decode_cmd

    def wedged(path, vf, proxy_out=None, **kw):
        cmd = real(path, vf, proxy_out, **kw)
        cmd[cmd.index("-i") + 1] = str(fifo)
        return cmd

    monkeypatch.setattr(F, "decode_cmd", wedged)


# --- the argument ----------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 0.0, float("nan")],
                         ids=["0", "-1", "0.0", "nan"])
def test_stall_timeout_must_be_a_positive_number(tiny_clip, bad):
    with pytest.raises(ValueError, match="positive number of seconds"):
        list(sample_frames(tiny_clip, 1.0, long_edge=LONG_EDGE,
                           stall_timeout=bad))


@pytest.mark.parametrize("bad", ["5", b"5", [1], (1,), {}, 1j, object()],
                         ids=["str", "bytes", "list", "tuple", "dict",
                              "complex", "object"])
def test_stall_timeout_must_be_a_number(tiny_clip, bad):
    """The TypeError names the argument. The bare comparison error that
    Python would raise does not."""
    with pytest.raises(TypeError, match="stall_timeout must be a number"):
        list(sample_frames(tiny_clip, 1.0, long_edge=LONG_EDGE,
                           stall_timeout=bad))


@pytest.mark.parametrize("bad", [True, False, np.True_, np.False_],
                         ids=["True", "False", "np.True_", "np.False_"])
def test_stall_timeout_rejects_a_bool(tiny_clip, bad):
    """`True` would otherwise pass as a limit of one second. numpy's bool
    compares like a number, so it needs its own check."""
    with pytest.raises(TypeError, match="bool"):
        list(sample_frames(tiny_clip, 1.0, long_edge=LONG_EDGE,
                           stall_timeout=bad))


@pytest.mark.parametrize("ok", [np.float64(5), np.int64(5), np.float32(0.5)],
                         ids=["float64", "int64", "float32"])
def test_stall_timeout_accepts_numpy_numbers(ok):
    assert F._stall_limit(ok) == float(ok)


def test_an_integer_too_large_for_a_float_means_no_limit():
    assert F._stall_limit(10**400) is None
    with pytest.raises(ValueError, match="positive"):
        F._stall_limit(-10**400)


def test_stall_timeout_errors_come_with_the_first_next_not_the_call(tiny_clip):
    """`sample_frames` is a generator. Argument errors are raised when it is
    first advanced, and a call that is never iterated raises nothing."""
    gen = sample_frames(tiny_clip, 1.0, long_edge=LONG_EDGE, stall_timeout=-1)
    with pytest.raises(ValueError, match="positive number"):
        next(gen)
    sample_frames(tiny_clip, 1.0, long_edge=LONG_EDGE, stall_timeout="5")


def test_the_decode_timeout_name_is_gone(tiny_clip):
    """The argument is `stall_timeout`, not `decode_timeout`. No release used
    the second name. The limit is a stall limit, so the name says so."""
    assert "decode_timeout" not in inspect.signature(sample_frames).parameters
    with pytest.raises(TypeError, match="decode_timeout"):
        sample_frames(tiny_clip, 1.0, decode_timeout=5)
    p = inspect.signature(sample_frames).parameters["stall_timeout"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None


@pytest.mark.parametrize("ok", [None, float("inf"), 30, 5])
def test_stall_timeout_accepts_none_inf_and_positive(tiny_clip, ok):
    assert len(list(sample_frames(tiny_clip, 1.0, long_edge=LONG_EDGE,
                                  stall_timeout=ok))) == 2


@pytest.mark.parametrize("small", [0.25, 1e-9, 0.5])
def test_stall_timeout_accepts_a_small_positive_number(small):
    """The validator alone. A 0.25 s limit used to run a real decode. That
    needs longer than 0.25 s to start, and it failed under CPU load with 'no
    frame within 0.25s'. What a small limit does to a running decode is
    covered by the stand-in decoder tests, which are deterministic."""
    assert F._stall_limit(small) == small


def test_inf_means_no_limit_and_starts_no_thread(tiny_clip):
    seen = []
    for t, _ in sample_frames(tiny_clip, 1.0, long_edge=LONG_EDGE,
                              stall_timeout=float("inf")):
        seen.append(len(watchdogs()))
    assert seen and max(seen) == 0


# --- identical when nothing is wrong --------------------------------------

def test_a_healthy_decode_is_identical_with_and_without_a_limit(clip8):
    a = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE))
    b = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, stall_timeout=30))
    assert len(a) == len(b) == 16
    assert [t for t, _ in a] == [t for t, _ in b]
    assert all(x.tobytes() == y.tobytes() for (_, x), (_, y) in zip(a, b))
    assert watchdogs() == []


def test_the_frame_budget_still_ends_the_decode_early_with_a_limit(clip8,
                                                                    monkeypatch):
    started = spy(monkeypatch)
    got = list(sample_frames(clip8, 8.0, fps=8.0, max_frames=3,
                             long_edge=LONG_EDGE, stall_timeout=30))
    assert len(got) == 3
    (proc,) = decoders(started)
    assert is_reaped(proc) and watchdogs() == []


# --- the wedge --------------------------------------------------------------

@pytest.mark.slow
def test_a_decode_that_never_starts_is_killed_reaped_and_raised(
        clip8, tmp_path, monkeypatch):
    """Real ffmpeg blocked in open(). By default nothing bounds this
    (measured 2026-10-22: `timeout=1` and still blocked when the harness
    killed the process 12 s later); with the limit it ends in about the
    limit."""
    wedge_input(monkeypatch, tmp_path)
    started = spy(monkeypatch)
    t0 = time.monotonic()
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, timeout=1,
                           stall_timeout=1.0))
    took = time.monotonic() - t0
    assert 1.0 <= took < 4.0, took
    err = e.value
    assert isinstance(err, DecodeError) and isinstance(err, RuntimeError)
    assert err.limit == 1.0 and err.yielded == 0 and err.returncode == -9
    assert "no frame within 1s" in str(err)
    (proc,) = decoders(started)
    assert is_reaped(proc)
    assert watchdogs() == []


@pytest.mark.slow
def test_a_decode_that_stalls_after_three_frames_yields_them_then_raises(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "f3,h")
    started = spy(monkeypatch)
    got = []
    with pytest.raises(F.DecodeTimeout) as e:
        for t, frame in sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                                      stall_timeout=1.0):
            got.append(t)
    assert got == [0.0, 0.5, 1.0]
    assert e.value.yielded == 3 and "after 3 frames" in str(e.value)
    assert all(is_reaped(p) for p in decoders(started))
    assert watchdogs() == []


@pytest.mark.slow
def test_a_decode_that_stalls_mid_frame_is_not_a_short_clip(
        clip8, tmp_path, monkeypatch):
    """Two frames and half of a third, then silence: the reader's read()
    holds a partial block. Killing the process turns that into EOF, which
    the loop treats as the end of the clip; it must raise instead."""
    install_fake(monkeypatch, tmp_path, clip8, "f2,p,h")
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, stall_timeout=1.0))
    assert e.value.yielded == 2


@pytest.mark.slow
def test_stderr_up_to_the_stall_comes_back_with_the_error(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "eopening the volume,h")
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, stall_timeout=1.0))
    assert "opening the volume" in e.value.stderr
    assert "opening the volume" in str(e.value)


@pytest.mark.slow
def test_a_decoder_that_ignores_sigterm_is_still_killed(
        clip8, tmp_path, monkeypatch):
    """The kill is SIGKILL. A wedged ffmpeg ignores the polite signals
    (measured: still running 5 s after SIGTERM), so a watchdog that
    terminated would leave the reader blocked."""
    install_fake(monkeypatch, tmp_path, clip8, "i,f2,h")
    started = spy(monkeypatch)
    t0 = time.monotonic()
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                           stall_timeout=1.0))
    assert time.monotonic() - t0 < 4.0
    assert e.value.returncode == -9 and e.value.yielded == 2
    assert all(is_reaped(p) for p in decoders(started))


def test_a_watchdog_thread_that_cannot_start_still_cleans_up(
        clip8, tmp_path, monkeypatch):
    """Thread creation can fail under pressure. The process is already
    running by then, so it has to be killed and reaped on the way out."""
    install_fake(monkeypatch, tmp_path, clip8, "f2,h")
    started = spy(monkeypatch)

    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, stall_timeout=5))
    assert all(is_reaped(p) for p in decoders(started))


@pytest.mark.slow
def test_a_refused_watchdog_leaves_the_drain_path_unbounded(
        clip8, tmp_path, monkeypatch):
    """The limit this paragraph of `sample_frames` documents. With a
    fragment set the decode is drained to EOF before the RuntimeError comes
    out, and nothing in the drain is watched. The decoder here ends by
    itself after 2 s, so the test can see the call wait that long under a
    0.5 s limit. A decoder that never ended would hold the call for ever,
    which is why the test does not use one."""
    install_fake(monkeypatch, tmp_path, clip8, "f2,s2,c")

    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="can't start new thread"):
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                           full_rate_vf="null", stall_timeout=0.5))
    assert time.monotonic() - t0 >= 1.9


@pytest.mark.slow
def test_a_stall_on_the_fragment_path_raises_the_same_error(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "f4,h")
    started = spy(monkeypatch)
    sink = tmp_path / "sink.txt"
    sink.write_text("the caller's file, cut short")
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                           full_rate_vf="null", stall_timeout=1.0))
    assert e.value.yielded == 4
    assert sink.exists(), "vmcore does not own or touch the fragment's sink"
    assert all(is_reaped(p) for p in decoders(started))


@pytest.mark.slow
def test_a_stall_after_the_last_frame_while_draining_still_raises(
        clip8, tmp_path, monkeypatch):
    """max_frames reached on the fragment path: the loop keeps reading to
    EOF (the decode is drained, not cut off), and that read is covered."""
    install_fake(monkeypatch, tmp_path, clip8, "f3,h")
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, max_frames=3, long_edge=LONG_EDGE,
                           full_rate_vf="null", stall_timeout=1.0))
    assert e.value.yielded == 3


# --- the guard against killing a process that already ended -----------------

class _Proc:
    """Just enough of a Popen for the watch: poll() and kill()."""

    def __init__(self, exit_status):
        self.exit_status = exit_status
        self.killed = False

    def poll(self):
        return self.exit_status

    def kill(self):
        self.killed = True


@pytest.mark.slow
def test_the_watch_leaves_a_process_that_has_already_exited_alone():
    """The race the guard exists for: the reader is still marked as waiting
    but the decode has ended, so there is nothing to kill and nothing that
    stalled. There is no way to make this happen reliably with a real
    ffmpeg, so the watch is driven directly."""
    gone = _Proc(exit_status=0)
    watch = F._StallWatch(gone, 0.2)
    watch.waiting()
    time.sleep(0.8)
    watch.stop()
    assert not gone.killed and not watch.fired


@pytest.mark.slow
def test_the_watch_kills_a_process_that_is_still_running():
    alive = _Proc(exit_status=None)
    watch = F._StallWatch(alive, 0.2)
    watch.waiting()
    time.sleep(0.8)
    watch.stop()
    assert alive.killed and watch.fired


@pytest.mark.slow
def test_the_watch_does_not_run_while_the_reader_is_not_waiting():
    alive = _Proc(exit_status=None)
    watch = F._StallWatch(alive, 0.2)
    time.sleep(0.8)
    watch.stop()
    assert not alive.killed and not watch.fired


# --- proxy_out partials -----------------------------------------------------

@needs_x264
@pytest.mark.slow
def test_a_stalled_fused_decode_removes_its_partial_proxy(
        clip8, tmp_path, monkeypatch):
    proxy = tmp_path / "px" / "eight.mp4"
    proxy.parent.mkdir()
    install_fake(monkeypatch, tmp_path, clip8, "w,f2,h", partial=proxy)
    started = spy(monkeypatch)
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, proxy_out=proxy,
                           stall_timeout=1.0))
    assert e.value.yielded == 2
    assert not proxy.exists(), "a partial proxy must not survive a kill"
    assert all(is_reaped(p) for p in decoders(started))


@needs_x264
def test_a_proxy_without_a_stall_is_untouched_by_the_limit(clip8, tmp_path):
    proxy = tmp_path / "px" / "eight.mp4"
    got = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, proxy_out=proxy,
                             stall_timeout=60))
    assert len(got) == 16 and proxy.stat().st_size > 0


# --- what must not be killed ------------------------------------------------

@pytest.mark.slow
def test_a_slow_but_progressing_decode_is_never_killed(clip8, tmp_path,
                                                       monkeypatch):
    """A frame every 0.5 s for 2 s against a 1.5 s limit: the total is
    longer than the limit, no single wait is. A wall-clock timer would
    have fired."""
    install_fake(monkeypatch, tmp_path, clip8,
                 ",".join(["s0.5,f1"] * 4))
    t0 = time.monotonic()
    got = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                             stall_timeout=1.5))
    assert len(got) == 4 and time.monotonic() - t0 > 1.9


@pytest.mark.slow
def test_the_consumers_own_time_does_not_count(clip8, monkeypatch):
    """Real ffmpeg, healthy, feeding a consumer that works 1.2 s per frame
    against a 1.0 s limit. The frames are upscaled to 960 px so each one
    fills the pipe: ffmpeg is alive and blocked writing for the whole
    sleep, which the test checks, because a process that has already
    exited is never killed and would make this pass for the wrong reason.
    The birthday pipeline's wall-clock timer killed exactly this
    (measured 2026-10-22: 'decode timed out after 2s' on a healthy 4 s clip
    after 4 of 8 frames, with 0.6 s of consumer work per frame)."""
    started = spy(monkeypatch)
    alive = []
    for t, _ in sample_frames(clip8, 8.0, fps=1.0, max_frames=2,
                              long_edge=960, stall_timeout=1.0):
        time.sleep(1.2)
        alive.append(decoders(started)[-1].poll() is None)
    assert len(alive) == 2
    assert alive[0], "ffmpeg should be blocked on the pipe, not finished"


@pytest.mark.slow
def test_a_clip_decoded_in_real_time_is_not_killed_by_a_short_limit(
        tmp_path, monkeypatch):
    clip3 = tmp_path / "three.mp4"
    make_mp4(clip3, duration=3.0)
    realtime(monkeypatch)
    t0 = time.monotonic()
    got = list(sample_frames(clip3, 3.0, long_edge=LONG_EDGE,
                             stall_timeout=1.5))
    assert len(got) == 6 and time.monotonic() - t0 > 2.5


# --- leaving early ----------------------------------------------------------

@pytest.mark.slow
def test_closing_the_generator_early_never_raises_a_timeout(
        clip8, tmp_path, monkeypatch):
    """The consumer has gone; there is nobody to tell. The stalled process
    is still killed and reaped, and the proxy still removed."""
    proxy = tmp_path / "px.mp4"
    install_fake(monkeypatch, tmp_path, clip8, "w,f2,h", partial=proxy)
    started = spy(monkeypatch)
    gen = sample_frames(clip8, 8.0, long_edge=LONG_EDGE, proxy_out=proxy,
                        stall_timeout=1.0)
    next(gen)
    t0 = time.monotonic()
    gen.close()
    assert time.monotonic() - t0 < 3.0
    assert not proxy.exists()
    assert all(is_reaped(p) for p in decoders(started))
    assert watchdogs() == []


def test_default_closing_a_plain_generator_early_still_terminates_ffmpeg(
        clip8, monkeypatch):
    realtime(monkeypatch)
    started = spy(monkeypatch)
    gen = sample_frames(clip8, 8.0, long_edge=LONG_EDGE)
    next(gen)
    gen.close()
    assert all(is_reaped(p) for p in decoders(started))


# --- the exception ---------------------------------------------------------

def test_the_decode_timeout_exception_pickles():
    err = F.DecodeTimeout(["ffmpeg"], -9, "x\n", yielded=3, limit=1.5)
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is F.DecodeTimeout and back.limit == 1.5
    assert str(back) == str(err) and back.yielded == 3


def test_the_decode_timeout_exception_messages():
    a = F.DecodeTimeout(["ffmpeg"], -9, "", yielded=0, limit=30)
    b = F.DecodeTimeout(["ffmpeg"], -9, "", yielded=7, limit=30)
    assert str(a) == "ffmpeg produced no frame within 30s and was killed"
    assert str(b) == ("ffmpeg produced no frame for 30s after 7 frames "
                      "and was killed")
