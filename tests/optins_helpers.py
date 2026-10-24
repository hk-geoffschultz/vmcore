"""Shared pieces for the opt-in tests: a scripted stand-in for ffmpeg, a spy
that keeps every process `subprocess.Popen` starts, and the awkward inputs
(a transport stream that sizes to 0x0, a file whose second half is zeroed).

The stand-in exists because "kill ffmpeg mid-stream" and "ffmpeg hangs after
three frames" have to happen at a known frame on every run, on every
machine. Real ffmpeg is used wherever the claim is about ffmpeg itself (the
tests say so); the script is used where the claim is about what
`sample_frames` does with a process that behaves in a given way.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vmcore import frames
from vmcore.probe import display_dims, scaled_dims

LONG_EDGE = 64

# A scripted decoder. argv: frame_bytes, spec, [partial_path]. spec is a
# comma-separated list run left to right:
#   fN    write N frames (each filled with its own index)
#   sX    sleep X seconds
#   eTXT  write TXT and a newline to stderr
#   xN    exit N
#   k     SIGKILL itself
#   p     write half a frame (the reader's read() is left holding a partial)
#   c     close stdout (the reader sees EOF; the process lives on)
#   i     ignore SIGTERM from here on (a wedged ffmpeg does)
#   h     hang for an hour
#   w     write 4096 bytes to partial_path (a proxy that never got a moov)
FAKE_SRC = textwrap.dedent('''
    import os, signal, sys, time
    fb = int(sys.argv[1])
    path = sys.argv[3] if len(sys.argv) > 3 else None
    n = 0
    for tok in sys.argv[2].split(","):
        if not tok:
            continue
        op, arg = tok[0], tok[1:]
        if op == "f":
            for _ in range(int(arg)):
                sys.stdout.buffer.write(bytes([n % 251]) * fb)
                sys.stdout.buffer.flush()
                n += 1
        elif op == "s":
            time.sleep(float(arg))
        elif op == "e":
            sys.stderr.write(arg + "\\n"); sys.stderr.flush()
        elif op == "x":
            os._exit(int(arg))
        elif op == "k":
            os.kill(os.getpid(), signal.SIGKILL)
        elif op == "p":
            sys.stdout.buffer.write(b"\\0" * (fb // 2))
            sys.stdout.buffer.flush()
        elif op == "c":
            sys.stdout.flush()
            os.close(1)
        elif op == "i":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        elif op == "h":
            time.sleep(3600)
        elif op == "w":
            with open(path, "wb") as fh:
                fh.write(b"\\0" * 4096)
''')


def frame_bytes_for(clip: Path, long_edge: int = LONG_EDGE) -> int:
    w, h = scaled_dims(*display_dims(clip), long_edge)
    return w * h * 3


def install_fake(monkeypatch, tmp_path: Path, clip: Path, spec: str,
                 *, partial: Path | None = None) -> list[str]:
    """Make `sample_frames` run the scripted decoder instead of ffmpeg.

    Returns the argv it will run. `decode_cmd` is replaced, so the sizing
    probe, the read loop, the shutdown and the exit handling are all the
    real code; only the process at the end of the pipe is scripted.
    """
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(FAKE_SRC)
    argv = [sys.executable, str(script), str(frame_bytes_for(clip)), spec]
    if partial is not None:
        argv.append(str(partial))

    def fake_decode_cmd(path, vf, proxy_out=None, **kw):
        return list(argv)

    monkeypatch.setattr(frames, "decode_cmd", fake_decode_cmd)
    return argv


class SpyPopen(subprocess.Popen):
    """Keeps every process started through `subprocess.Popen`."""

    started: list = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        SpyPopen.started.append(self)


def spy(monkeypatch) -> list:
    SpyPopen.started = []
    monkeypatch.setattr(subprocess, "Popen", SpyPopen)
    return SpyPopen.started


def decoders(started) -> list:
    """The processes that were a decode (not an ffprobe)."""
    return [p for p in started if "ffprobe" not in Path(str(p.args[0])).name]


def is_reaped(proc) -> bool:
    """Exit status collected, and the pid no longer exists as a zombie."""
    if proc.returncode is None:
        return False
    try:
        os.kill(proc.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def realtime(monkeypatch):
    """Run the real `decode_cmd` argv at its native frame rate (`-re`), so a
    test has time to act while ffmpeg is still decoding."""
    real = frames.decode_cmd

    def slow(path, vf, proxy_out=None, **kw):
        cmd = real(path, vf, proxy_out, **kw)
        cmd.insert(cmd.index("-i"), "-re")
        return cmd

    monkeypatch.setattr(frames, "decode_cmd", slow)


def can_encode_proxy() -> bool:
    from vmcore.proxy import full_ffmpeg

    r = subprocess.run([full_ffmpeg(), "-hide_banner", "-encoders"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.returncode == 0 and " libx264 " in r.stdout


def make_headless_mts(path: Path) -> None:
    """A transport stream whose video ffprobe sizes as 0x0.

    Encoded with one IDR (so SPS/PPS appear once, at the start) and the
    first half of the packets removed, which is what a camera file that
    lost its head looks like. Measured 2026-10-22, ffmpeg 6.1.1: ffprobe
    reports `width=0 height=0` for it.
    """
    full = path.with_name(path.stem + "_full.mts")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=6",
         "-c:v", "libx264", "-preset", "ultrafast", "-g", "9999",
         "-keyint_min", "9999", "-pix_fmt", "yuv420p", "-f", "mpegts",
         str(full)], check=True, capture_output=True,
        stdin=subprocess.DEVNULL)
    data = full.read_bytes()
    cut = (len(data) // 2) // 188 * 188
    path.write_bytes(data[cut:])
    full.unlink()


def make_audio_only(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "1",
         "-c:a", "aac", str(path)], check=True, capture_output=True,
        stdin=subprocess.DEVNULL)


def zero_payload(src: Path, dst: Path, *, from_fraction: float = 0.0) -> None:
    """Copy `src` with the bytes of its `mdat` zeroed from `from_fraction`
    of the way through it. 0.0 is the existing suite's 'probes but does not
    decode' file."""
    data = bytearray(src.read_bytes())
    i = data.find(b"mdat")
    size = int.from_bytes(data[i - 4:i], "big")
    start = i + 4 + int((size - 8) * from_fraction)
    data[start:i - 4 + size] = b"\0" * (i - 4 + size - start)
    dst.write_bytes(data)


@pytest.fixture(autouse=True)
def no_hang():
    """A watchdog that fails must fail the test, not hang the suite: the
    thing under test is a read that can block forever."""
    def boom(*_):
        raise TimeoutError("test did not finish in 90 s: a read is blocked")

    old = signal.signal(signal.SIGALRM, boom)
    signal.alarm(90)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
