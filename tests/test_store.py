from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import vmcore.store as store_mod
from vmcore.store import Store


def test_store_json_round_trip(tmp_path):
    st = Store(tmp_path / "work")
    assert st.work.is_dir()
    p = st.work / "x.json"
    obj = {"a": [1, 2, {"b": None}], "c": "café", "d": 1.5}
    st.write_json(p, obj)
    assert st.read_json(p) == obj
    assert st.read_json(st.work / "missing.json", default="dflt") == "dflt"
    p.write_text("{not json")
    assert st.read_json(p, default={"fallback": 1}) == {"fallback": 1}


# post-merge: pins A7 (atomic_write_text in bbp/store.py)
def test_atomic_write_text_leaves_no_tmp_and_backs_up(tmp_path):
    p = tmp_path / "state.json"
    store_mod.atomic_write_text(p, "one")
    assert p.read_text() == "one"
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "state.json.bak").exists()

    store_mod.atomic_write_text(p, "two", backup=True)
    assert p.read_text() == "two"
    assert (tmp_path / "state.json.bak").read_text() == "one"
    assert not list(tmp_path.glob("*.tmp"))


# post-merge: pins A7 (atomic_write_json in bbp/store.py)
def test_atomic_write_json_round_trip(tmp_path):
    p = tmp_path / "nested" / "obj.json"
    obj = {"k": [1, 2.5, "s"], "n": None, "u": "é"}
    store_mod.atomic_write_json(p, obj)
    assert json.loads(p.read_text()) == obj
    assert not list(p.parent.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Concurrent writers of one file. The temp name used to be `<name>.tmp`,
# shared by every writer of that file.
# ---------------------------------------------------------------------------

def _spy_replace(monkeypatch):
    calls = []
    real = os.replace

    def spy(src, dst):
        calls.append((os.fspath(src), os.fspath(dst)))
        return real(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", spy)
    return calls


def test_the_temp_name_is_this_writers_own_and_still_ends_in_tmp(
        tmp_path, monkeypatch):
    calls = _spy_replace(monkeypatch)
    p = tmp_path / "state.json"
    store_mod.atomic_write_text(p, "one")
    [(src, dst)] = calls
    assert dst == str(p)
    assert Path(src).name == (
        f"state.json.{os.getpid()}.{threading.get_ident()}.tmp")
    assert Path(src).suffix == ".tmp"        # a *.tmp glob still finds it
    assert Path(src).parent == p.parent


def test_two_threads_never_share_a_temp_name(tmp_path, monkeypatch):
    """Both threads exist before either writes, so their identifiers differ
    (the OS reuses an identifier only after its thread has exited)."""
    calls = _spy_replace(monkeypatch)
    p = tmp_path / "state.json"
    both_alive = threading.Barrier(2)

    def write():
        both_alive.wait()
        store_mod.atomic_write_text(p, "x")

    ts = [threading.Thread(target=write) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len({src for src, _ in calls}) == 2


def test_concurrent_writers_never_tear_or_fail(tmp_path):
    """The case the fixed name lost. At 757f484, 8 threads x 150 writes of
    one file raised FileNotFoundError in about half of the writes and let a
    reader see unparseable JSON in about a quarter of its reads. Sized here
    to stay under a second and still fail reliably against that code."""
    p = tmp_path / "project.json"
    store_mod.atomic_write_json(p, {"writer": -1, "pad": ""})
    n_threads, n_writes, pad = 8, 40, 20000
    errors, torn, reads = [], [], []
    stop = threading.Event()
    start = threading.Barrier(n_threads + 1)

    def writer(i):
        body = {"writer": i, "pad": chr(97 + i) * (pad * (i + 1) // n_threads)}
        start.wait()
        for _ in range(n_writes):
            try:
                store_mod.atomic_write_json(p, body, indent=1)
            except Exception as exc:                  # noqa: BLE001
                errors.append(exc)

    def reader():
        start.wait()
        while not stop.is_set():
            try:
                text = p.read_text()
            except FileNotFoundError:
                continue
            reads.append(1)
            try:
                json.loads(text)
            except ValueError:
                torn.append(text[:40])

    ts = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    rd = threading.Thread(target=reader)
    for t in ts + [rd]:
        t.start()
    for t in ts:
        t.join()
    stop.set()
    rd.join()

    assert errors == []
    assert torn == []
    assert reads, "the reader never ran, so the check above proved nothing"
    final = json.loads(p.read_text())
    assert len(final["pad"]) == pad * (final["writer"] + 1) // n_threads
    assert [x.name for x in tmp_path.iterdir()] == ["project.json"]


def test_a_backup_is_never_visible_half_written(tmp_path, monkeypatch):
    """The .bak is the recovery source, so it lands the way the main file
    does: copied to a temp name, then renamed."""
    p = tmp_path / "state.json"
    store_mod.atomic_write_text(p, "old " * 1000)
    os.utime(p, ns=(1_000_000_000_000, 1_000_000_000_000))
    os.chmod(p, 0o640)
    copies = []
    real = store_mod.shutil.copy2

    def spy(src, dst, **kw):
        copies.append(Path(dst).name)
        return real(src, dst, **kw)

    monkeypatch.setattr(store_mod.shutil, "copy2", spy)
    store_mod.atomic_write_text(p, "new", backup=True)

    [dst] = copies
    assert dst != "state.json.bak" and dst.endswith(".tmp")
    bak = tmp_path / "state.json.bak"
    assert bak.read_text() == "old " * 1000
    assert stat.S_IMODE(bak.stat().st_mode) == 0o640      # copy2 semantics
    assert bak.stat().st_mtime_ns == 1_000_000_000_000
    assert sorted(x.name for x in tmp_path.iterdir()) == [
        "state.json", "state.json.bak"]


@pytest.mark.parametrize("umask, mode", [(0o022, 0o644), (0o077, 0o600)])
def test_the_file_mode_is_the_umask_applied_to_0666(tmp_path, umask, mode):
    """Not tempfile.mkstemp, which creates 0o600 whatever the umask is and
    which os.replace then carries onto the target."""
    old = os.umask(umask)
    try:
        p = tmp_path / "state.json"
        store_mod.atomic_write_text(p, "x")
        assert stat.S_IMODE(p.stat().st_mode) == mode
    finally:
        os.umask(old)


def test_an_existing_files_mode_is_not_kept_it_is_a_new_file(tmp_path):
    """Unchanged from the fixed-name writer; pinned so a change is chosen."""
    old = os.umask(0o022)
    try:
        p = tmp_path / "state.json"
        p.write_text("x")
        os.chmod(p, 0o640)
        store_mod.atomic_write_text(p, "y")
        assert stat.S_IMODE(p.stat().st_mode) == 0o644
    finally:
        os.umask(old)


def test_a_failed_write_leaves_the_old_file_and_no_temp(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    store_mod.atomic_write_text(p, "good")

    def boom(fd):
        raise OSError("disk gone")

    monkeypatch.setattr(store_mod.os, "fsync", boom)
    with pytest.raises(OSError, match="disk gone"):
        store_mod.atomic_write_text(p, "bad", backup=True)
    assert p.read_text() == "good"
    assert sorted(x.name for x in tmp_path.iterdir()) == [
        "state.json", "state.json.bak"]          # the backup preceded the failure


def test_a_failed_backup_copy_leaves_no_temp_and_does_not_write(
        tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    store_mod.atomic_write_text(p, "good")

    def boom(src, dst, **kw):
        Path(dst).write_text("par")              # a partial copy
        raise OSError("disk full")

    monkeypatch.setattr(store_mod.shutil, "copy2", boom)
    with pytest.raises(OSError, match="disk full"):
        store_mod.atomic_write_text(p, "new", backup=True)
    assert p.read_text() == "good"
    assert [x.name for x in tmp_path.iterdir()] == ["state.json"]


def test_a_kill_between_write_and_rename_leaves_the_target_whole(tmp_path):
    """The cost of a per-writer name, pinned so it stays a known property:
    SIGKILL at the rename leaves the OLD file intact and exactly one
    `<name>.<pid>.<thread>.tmp` behind. The fixed name was overwritten by
    the next write; this one is not, because the next process has another
    pid. It is a few kilobytes and a `*.tmp` glob finds it."""
    p = tmp_path / "state.json"
    store_mod.atomic_write_text(p, "old")
    code = (
        "import os, signal, sys\n"
        "import vmcore.store as s\n"
        "s.os.replace = lambda a, b: os.kill(os.getpid(), signal.SIGKILL)\n"
        f"s.atomic_write_text({str(p)!r}, 'new')\n")
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-c", code], cwd=root)
    assert r.returncode == -9
    assert p.read_text() == "old"
    left = [x.name for x in tmp_path.iterdir() if x.name != "state.json"]
    assert len(left) == 1
    assert re.fullmatch(r"state\.json\.\d+\.\d+\.tmp", left[0])
    assert (tmp_path / left[0]).read_text() == "new"
