"""`sample_frames(check_exit=True)`: an ffmpeg that dies is not a short clip.

Today, an ffmpeg that exits non-zero after yielding some frames, or before
yielding any, ends the generator quietly: the consumer classifies a clip on
a fraction of its frames. On the fused path (`proxy_out`) that means a
proxy branch that loses its disk mid-stream; on the fragment path only
"nothing yielded" raised. `check_exit=True` raises `DecodeError` carrying
stderr and the frame count for any non-zero exit ffmpeg reaches on its own.

Real ffmpeg is killed mid-stream here (SIGKILL after the third frame, at
`-re` so it is still decoding) on all three paths, and a real proxy branch
is killed by a file-size limit in a fresh interpreter. A scripted decoder
is used where the test needs a specific exit status, stderr text or timing.

Tests named test_default_* pin behaviour without the argument.
"""
from __future__ import annotations

import os
import pickle
import resource
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from vmcore import frames as F
from vmcore.frames import DecodeError, sample_frames
from vmcore.testing import make_mp4

from optins_helpers import (LONG_EDGE, can_encode_proxy, decoders,
                            install_fake, is_reaped, no_hang, realtime, spy,
                            zero_payload)

needs_x264 = pytest.mark.skipif(not can_encode_proxy(),
                                reason="no libx264 in the proxy ffmpeg build")

PATHS = ["plain", "fragment", pytest.param("fused", marks=needs_x264)]


@pytest.fixture
def clip8(tmp_path):
    p = tmp_path / "eight.mp4"
    make_mp4(p, duration=8.0)
    return p


def path_kwargs(kind: str, tmp_path: Path) -> dict:
    if kind == "fragment":
        return {"full_rate_vf": f"metadata=mode=print:file={tmp_path/'s.txt'}"}
    if kind == "fused":
        return {"proxy_out": tmp_path / "px" / "eight.mp4"}
    return {}


def kill_third_frame(clip, kind, tmp_path, monkeypatch, **extra):
    """Run real ffmpeg at -re, SIGKILL it after the third frame. Returns
    (frames seen, exception or None, decoder process, kwargs)."""
    realtime(monkeypatch)
    started = spy(monkeypatch)
    kw = path_kwargs(kind, tmp_path)
    seen, err = [], None
    try:
        for t, frame in sample_frames(clip, 8.0, long_edge=LONG_EDGE, **kw,
                                      **extra):
            seen.append(t)
            if len(seen) == 3:
                os.kill(decoders(started)[-1].pid, signal.SIGKILL)
    except DecodeError as e:
        err = e
    return seen, err, decoders(started)[-1], kw


# --- today: a prefix, no word -----------------------------------------------

@pytest.mark.parametrize("kind", PATHS)
def test_default_ffmpeg_killed_mid_stream_yields_a_prefix_quietly(
        clip8, tmp_path, monkeypatch, kind):
    seen, err, proc, kw = kill_third_frame(clip8, kind, tmp_path, monkeypatch)
    assert err is None and seen == [0.0, 0.5, 1.0]
    assert proc.returncode == -9 and is_reaped(proc)
    if kind == "fused":
        assert not kw["proxy_out"].exists(), "the partial proxy is removed"


def test_default_a_decoder_that_exits_1_after_three_frames_yields_three(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "f3,eboom,x1")
    assert len(list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE))) == 3


@needs_x264
def test_default_a_fused_decode_that_fails_at_start_yields_nothing_quietly(
        clip8, tmp_path, monkeypatch):
    proxy = tmp_path / "px" / "eight.mp4"
    install_fake(monkeypatch, tmp_path, clip8,
                 "eboom: libx264 not found,x3")
    assert list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                              proxy_out=proxy)) == []
    assert not proxy.exists()


PROXY_DEATH = textwrap.dedent('''
    import resource, signal, sys
    from pathlib import Path
    from vmcore import frames
    src, out, check = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
    full = len(list(frames.sample_frames(src, 40.0, long_edge=64)))
    # A file-size limit on this process, inherited by ffmpeg, stands in for
    # a full disk: the proxy branch dies when its file reaches it.
    resource.setrlimit(resource.RLIMIT_FSIZE, (400_000, 400_000))
    n = 0
    try:
        for _ in frames.sample_frames(src, 40.0, long_edge=64, proxy_out=out,
                                      **({"check_exit": True} if check else {})):
            n += 1
        print("quiet", n, full, out.exists())
    except frames.DecodeError as e:
        print("raised", n, full, e.returncode, e.yielded, out.exists())
''')


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory):
    p = tmp_path_factory.mktemp("long") / "long.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=40",
         "-c:v", "libx264", "-preset", "ultrafast", str(p)],
        check=True, stdin=subprocess.DEVNULL)
    return p


def run_proxy_death(clip: Path, out: Path, check: bool) -> list[str]:
    r = subprocess.run(
        [sys.executable, "-c", PROXY_DEATH, str(clip), str(out),
         "1" if check else "0"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120,
        env={**os.environ, "PYTHONPATH": str(Path(F.__file__).parent.parent)})
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


@needs_x264
def test_default_a_proxy_branch_that_dies_mid_stream_yields_a_prefix_quietly(
        long_clip, tmp_path):
    verdict, n, full, exists = run_proxy_death(long_clip, tmp_path / "p.mp4",
                                               check=False)
    assert verdict == "quiet" and 0 < int(n) < int(full)
    assert exists == "False"


# --- the opt-in ---------------------------------------------------------------

@pytest.mark.parametrize("kind", PATHS)
@pytest.mark.slow
def test_ffmpeg_killed_mid_stream_raises_after_the_frames_it_did_send(
        clip8, tmp_path, monkeypatch, kind):
    seen, err, proc, kw = kill_third_frame(clip8, kind, tmp_path, monkeypatch,
                                           check_exit=True)
    assert seen == [0.0, 0.5, 1.0], "the prefix is delivered first"
    assert isinstance(err, DecodeError) and type(err) is DecodeError
    assert err.returncode == -9 and err.yielded == 3
    assert "after yielding 3 frames" in str(err)
    assert proc.returncode == -9 and is_reaped(proc)
    if kind == "fused":
        assert not kw["proxy_out"].exists()


def test_the_raise_is_where_the_generator_would_have_ended(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "f3,eboom,x1")
    gen = sample_frames(clip8, 8.0, long_edge=LONG_EDGE, check_exit=True)
    assert [next(gen)[0] for _ in range(3)] == [0.0, 0.5, 1.0]
    with pytest.raises(DecodeError) as e:
        next(gen)
    assert e.value.returncode == 1 and e.value.yielded == 3
    assert "boom" in e.value.stderr and "boom" in str(e.value)


@pytest.mark.parametrize("kind", ["plain", "fragment"])
def test_an_error_exit_with_nothing_yielded_raises_with_stderr(
        clip8, tmp_path, monkeypatch, kind):
    install_fake(monkeypatch, tmp_path, clip8, "eno such thing,x8")
    with pytest.raises(DecodeError) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, check_exit=True,
                           **path_kwargs(kind, tmp_path)))
    assert e.value.returncode == 8 and e.value.yielded == 0
    assert str(e.value) == ("ffmpeg exited 8 having yielded no frames: "
                            "no such thing")


@needs_x264
def test_a_fused_decode_that_fails_at_start_raises_with_what_it_said(
        clip8, tmp_path, monkeypatch):
    proxy = tmp_path / "px" / "eight.mp4"
    install_fake(monkeypatch, tmp_path, clip8,
                 "eboom: libx264 not found,x3")
    with pytest.raises(DecodeError) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, proxy_out=proxy,
                           check_exit=True))
    assert e.value.returncode == 3 and "libx264 not found" in str(e.value)
    assert not proxy.exists()


@needs_x264
@pytest.mark.slow
def test_a_proxy_branch_that_dies_mid_stream_raises(long_clip, tmp_path):
    """Real ffmpeg, real proxy branch: the process is killed by a file-size
    limit when the proxy reaches it. The first five or so frames have been
    handed over by then."""
    verdict, n, full, rc, yielded, exists = run_proxy_death(
        long_clip, tmp_path / "p.mp4", check=True)
    assert verdict == "raised" and 0 < int(n) < int(full)
    assert int(rc) < 0 and int(yielded) == int(n)
    assert exists == "False"


def test_a_file_whose_payload_is_gone_raises_without_a_fragment(
        clip8, tmp_path):
    """The existing suite pins that this raises with a fragment and yields
    nothing without one. `check_exit` is what turns the second into a raise:
    an empty clip that is really a failed decode."""
    bad = tmp_path / "zeroed.mp4"
    zero_payload(clip8, bad)
    assert list(sample_frames(bad, 8.0, long_edge=LONG_EDGE)) == []
    with pytest.raises(DecodeError) as e:
        list(sample_frames(bad, 8.0, long_edge=LONG_EDGE, check_exit=True))
    assert e.value.returncode == 69 and e.value.stderr.strip()


def test_damage_ffmpeg_tolerates_is_not_seen(clip8, tmp_path):
    """The limit, pinned so it is not forgotten: a payload zeroed from the
    halfway point exits 0 with a short clip, and `check_exit` does not
    raise. A consumer that needs the count compares it with what it
    expected. Skipped on an ffmpeg that exits non-zero here, where the
    premise no longer holds and the docstring needs updating."""
    bad = tmp_path / "half.mp4"
    zero_payload(clip8, bad, from_fraction=0.5)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel",
                        "error", "-i", str(bad), "-map", "0:v:0", "-f",
                        "null", "-"], capture_output=True,
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        pytest.skip(f"this ffmpeg exits {r.returncode} on a half-zeroed file")
    got = list(sample_frames(bad, 8.0, long_edge=LONG_EDGE, check_exit=True))
    assert 0 < len(got) < 16


# --- nothing is raised for what vmcore did itself --------------------------

@pytest.mark.parametrize("kind", PATHS)
def test_a_healthy_decode_is_identical_with_check_exit(clip8, tmp_path, kind):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    kw = path_kwargs(kind, tmp_path / "a")
    kw2 = path_kwargs(kind, tmp_path / "b")
    a = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, **kw))
    b = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, check_exit=True,
                           **kw2))
    assert len(a) == len(b) == 16
    assert [t for t, _ in a] == [t for t, _ in b]
    assert all(x.tobytes() == y.tobytes() for (_, x), (_, y) in zip(a, b))


@pytest.mark.slow
def test_the_frame_budget_ending_the_decode_early_is_not_a_failure(
        clip8, monkeypatch):
    """ffmpeg is still decoding when the budget is reached, so vmcore
    terminates it and its exit is non-zero (255: ffmpeg catches SIGTERM and
    exits that way, measured 2026-10-22 on 6.1.1). That is not a decode
    error, and the reason the check applies only to an ffmpeg that ended
    on its own.

    The caller declares 3 s for an 8 s file, so the thinned rate puts the
    third sample at 2 s, mid-file; with the true duration the last sample
    would fall in the final interval and ffmpeg would already be done
    (the `thinned_fps` docstring), which would test nothing."""
    realtime(monkeypatch)
    started = spy(monkeypatch)
    got = list(sample_frames(clip8, 3.0, fps=2.0, max_frames=3,
                             long_edge=LONG_EDGE, check_exit=True))
    assert len(got) == 3
    (proc,) = decoders(started)
    assert proc.returncode not in (0, None) and is_reaped(proc)


def test_closing_a_plain_generator_early_raises_nothing_and_does_not_wait(
        clip8, monkeypatch):
    realtime(monkeypatch)
    started = spy(monkeypatch)
    gen = sample_frames(clip8, 8.0, long_edge=LONG_EDGE, check_exit=True)
    next(gen)
    t0 = time.monotonic()
    gen.close()
    assert time.monotonic() - t0 < 5.0
    assert all(is_reaped(p) for p in decoders(started))


def test_closing_a_draining_generator_early_raises_nothing(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "f4,eboom,x1")
    gen = sample_frames(clip8, 8.0, long_edge=LONG_EDGE, full_rate_vf="null",
                        check_exit=True)
    next(gen)
    gen.close()


# --- the exit status is ffmpeg's own ---------------------------------------

def test_a_failing_exit_after_eof_is_reported_not_overwritten_by_sigterm(
        clip8, tmp_path, monkeypatch):
    """The decoder closes its pipe, lingers 0.4 s, then exits 1. Terminating
    it as soon as the pipe closes would turn that into -15 and hide the
    status; the analysis-only path waits for a process that is exiting."""
    install_fake(monkeypatch, tmp_path, clip8, "f2,c,s0.4,eearly goodbye,x1")
    with pytest.raises(DecodeError) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, check_exit=True))
    assert e.value.returncode == 1 and e.value.yielded == 2
    assert "early goodbye" in e.value.stderr


@pytest.mark.slow
def test_a_decoder_that_outlives_timeout_after_its_last_frame_is_killed(
        clip8, tmp_path, monkeypatch):
    """Draining, pipe closed, process hung (an encoder finishing a proxy
    that never finishes). `timeout` kills it; by default that is silent, as
    it always was, and with `check_exit` it is `DecodeTimeout`."""
    install_fake(monkeypatch, tmp_path, clip8, "f2,c,h")
    t0 = time.monotonic()
    quiet = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                               full_rate_vf="null", timeout=1))
    assert len(quiet) == 2 and time.monotonic() - t0 < 6.0
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                           full_rate_vf="null", timeout=1, check_exit=True))
    err = e.value
    assert err.phase == "finish" and err.limit == 1 and err.yielded == 2
    assert err.returncode == -9
    assert "had not exited 1s after its last of 2 frames" in str(err)


def _lingering_decoder(kind, clip, tmp_path, monkeypatch):
    """Three frames, then the pipe closes and the process stays for 30 s."""
    install_fake(monkeypatch, tmp_path, clip, "f3,c,s30")
    return {"full_rate_vf": "null"} if kind == "fragment" else {}


@pytest.mark.parametrize("kind", ["plain", "fragment"])
@pytest.mark.slow
def test_a_decoder_that_lingers_after_eof_is_a_decode_timeout_on_both_paths(
        clip8, tmp_path, monkeypatch, kind):
    """The analysis-only path used to wait a fixed 10 s, whatever `timeout`
    was, and then SIGTERM the process. The error was a plain DecodeError
    with returncode -15, vmcore's own signal reported as ffmpeg's exit. It
    now waits `timeout` and kills, as the drain always did. A caller that
    catches DecodeTimeout to find hangs sees the same error on both."""
    kw = _lingering_decoder(kind, clip8, tmp_path, monkeypatch)
    started = spy(monkeypatch)
    t0 = time.monotonic()
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, timeout=2,
                           check_exit=True, **kw))
    took = time.monotonic() - t0
    err = e.value
    assert 1.9 <= took < 6.0, took
    assert (err.phase, err.limit, err.returncode, err.yielded) == (
        "finish", 2, -9, 3)
    assert "had not exited 2s after its last of 3 frames" in str(err)
    assert all(is_reaped(p) for p in decoders(started))


@pytest.mark.slow
def test_the_two_paths_report_the_same_error_for_a_lingering_decoder(
        clip8, tmp_path, monkeypatch):
    seen = {}
    for kind in ("plain", "fragment"):
        kw = _lingering_decoder(kind, clip8, tmp_path, monkeypatch)
        with pytest.raises(F.DecodeTimeout) as e:
            list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, timeout=1,
                               check_exit=True, **kw))
        err = e.value
        seen[kind] = (type(err), err.phase, err.limit, err.returncode,
                      err.yielded, str(err))
    assert seen["plain"] == seen["fragment"]


def test_default_a_lingering_decoder_on_the_plain_path_is_still_terminated_at_once(
        clip8, tmp_path, monkeypatch):
    """Without `check_exit` nothing waits at EOF: the pipe is closed and the
    process is terminated, as it always was."""
    _lingering_decoder("plain", clip8, tmp_path, monkeypatch)
    started = spy(monkeypatch)
    t0 = time.monotonic()
    got = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, timeout=30))
    assert len(got) == 3 and time.monotonic() - t0 < 5.0
    assert all(is_reaped(p) for p in decoders(started))


def test_default_the_timeout_after_the_last_frame_is_silent(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "f2,c,h")
    started = spy(monkeypatch)
    got = list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                             full_rate_vf="null", timeout=1))
    assert len(got) == 2
    assert all(is_reaped(p) for p in decoders(started))


@pytest.mark.slow
def test_a_stall_is_reported_as_a_stall_even_with_check_exit(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "f2,h")
    with pytest.raises(F.DecodeTimeout) as e:
        list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE, check_exit=True,
                           stall_timeout=1.0))
    assert e.value.phase == "stall" and e.value.yielded == 2


def test_the_fragment_rule_for_nothing_yielded_is_unchanged_by_check_exit(
        clip8, tmp_path, monkeypatch):
    install_fake(monkeypatch, tmp_path, clip8, "eNo such filter,x8")
    for check in (False, True):
        with pytest.raises(DecodeError) as e:
            list(sample_frames(clip8, 8.0, long_edge=LONG_EDGE,
                               full_rate_vf="nosuch", check_exit=check))
        assert str(e.value) == ("ffmpeg exited 8 having yielded no frames: "
                                "No such filter")


def test_the_finish_phase_of_the_decode_timeout_pickles():
    err = F.DecodeTimeout(["ffmpeg"], -9, "", yielded=2, limit=1,
                          phase="finish")
    back = pickle.loads(pickle.dumps(err))
    assert back.phase == "finish" and str(back) == str(err)
