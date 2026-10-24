"""Run a command as a background job of a terminal that is also its stdin.

A helper for tests/test_nostdin.py, not a test. A pty is made the
controlling terminal. The command runs in its OWN process group, so it is a
background group of that terminal. Its fd 0 is still the tty, and its
stdout and stderr go to a log file. A process in that state which asks the
terminal to change its settings is stopped with SIGTTOU. So is every
process in its group.

This is the job a bare `tool &` leaves. GNU nohup would also redirect a
terminal stdin from an unreadable file, and the stop would not happen.
Measured 2026-10-22 with coreutils 9.4 on 757f484: the bare job stopped,
and the same job under nohup exited 0. A nohup that leaves stdin alone, as
rule 17 of the birthday pipeline reports for macOS, was not measured here.

    python tests/ttyrun.py LOG TIMEOUT_SECONDS CMD [ARG ...]

prints one JSON line: {"verdict": "EXITED(0)" | "STOPPED(SIGTTOU)" |
"TIMEOUT"}. Stdlib only; needs fork and a pty (Linux, macOS).
"""
from __future__ import annotations

import json
import os
import pty
import signal
import sys
import time


def run_in_background_on_tty(argv: list[str], log: str, timeout: float) -> str:
    rpipe, wpipe = os.pipe()
    pid, master = pty.fork()
    if pid == 0:                                   # session leader on the pty
        os.close(rpipe)
        worker = os.fork()
        if worker == 0:                            # the background job
            os.setpgid(0, 0)                       # a background process group
            fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.dup2(fd, 1)
            os.dup2(fd, 2)                         # stdin stays the tty
            try:
                os.execvp(argv[0], argv)
            finally:
                os._exit(127)
        verdict = "TIMEOUT"
        deadline = time.time() + timeout
        while time.time() < deadline:
            done, status = os.waitpid(worker, os.WUNTRACED | os.WNOHANG)
            if done == worker:
                if os.WIFSTOPPED(status):
                    sig = signal.Signals(os.WSTOPSIG(status)).name
                    verdict = f"STOPPED({sig})"
                    os.killpg(worker, signal.SIGKILL)
                    os.waitpid(worker, 0)
                else:
                    verdict = f"EXITED({os.waitstatus_to_exitcode(status)})"
                break
            time.sleep(0.05)
        else:
            os.killpg(worker, signal.SIGKILL)
            os.waitpid(worker, 0)
        os.write(wpipe, verdict.encode())
        os._exit(0)
    os.close(wpipe)
    got = b""
    while True:                       # drain the pty, collect the verdict
        try:
            chunk = os.read(rpipe, 256)
        except OSError:
            break
        if not chunk:
            break
        got += chunk
    os.close(rpipe)
    try:
        os.close(master)
    except OSError:
        pass
    os.waitpid(pid, 0)
    return got.decode() or "NO-VERDICT"


if __name__ == "__main__":
    print(json.dumps({"verdict": run_in_background_on_tty(
        sys.argv[3:], sys.argv[1], float(sys.argv[2]))}))
