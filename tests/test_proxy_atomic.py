"""`proxy.build_proxy`: never a half-written file at the final path.

The encode used to be written straight to `out`. A timeout left a partial
file there with no moov atom, which a caller that skips clips whose proxy
exists treated as done forever. Now the encode goes to a hidden partial
beside `out` and is renamed onto it only on success.

A scripted ffmpeg is used where the test needs a given exit, sleep, or
output; real ffmpeg is used for the claims about its bytes and about
concurrent builds. Tests named test_default_* pin behaviour that is the
same as before this change and pass on the commit before it.
"""
from __future__ import annotations

import errno
import hashlib
import inspect
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

try:
    import fcntl
except ImportError:           # Windows
    fcntl = None

from vmcore import proxy
from vmcore.testing import make_mp4

from optins_helpers import can_encode_proxy, no_hang

needs_flock = pytest.mark.skipif(fcntl is None, reason="no flock here")
needs_x264 = pytest.mark.skipif(not can_encode_proxy(),
                                reason="no libx264 in the proxy ffmpeg build")

FAKE = textwrap.dedent('''\
    #!{python}
    import os, sys, time
    out = sys.argv[-1]
    spec = os.environ.get("FAKE_SPEC", "")
    log = os.environ.get("FAKE_LOG")
    for tok in spec.split(","):
        if not tok:
            continue
        op, arg = tok[0], tok[1:]
        if op == "l" and log:
            d = os.path.dirname(out) or "."
            with open(log, "w") as fh:
                fh.write(out + "\\n" + "\\n".join(sorted(os.listdir(d))))
        elif op == "w":
            with open(out, "wb") as fh:
                fh.write(b"\\x01" * int(arg))
        elif op == "s":
            time.sleep(float(arg))
        elif op == "x":
            os._exit(int(arg))
''')


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "src.mp4"
    make_mp4(p, duration=2.0)
    return p


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Make build_proxy run a scripted ffmpeg; FAKE_SPEC says what it does."""
    script = tmp_path / "fake_ff"
    script.write_text(FAKE.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setattr(proxy, "full_ffmpeg", lambda *a, **k: str(script))

    def spec(s: str, log: Path | None = None):
        monkeypatch.setenv("FAKE_SPEC", s)
        if log is not None:
            monkeypatch.setenv("FAKE_LOG", str(log))

    return spec


def listing(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


def _interrupt_the_encode(monkeypatch):
    """Make the encode, and only the encode, raise KeyboardInterrupt after
    it has written half a file. The ffprobe in `color_transfer` must reach
    the real `subprocess.run`. If every call is stubbed, the ffprobe is
    interrupted instead, before the partial exists, and the test proves
    nothing."""
    real = subprocess.run

    def run(cmd, **kw):
        if "ffprobe" in Path(str(cmd[0])).name:
            return real(cmd, **kw)
        Path(cmd[-1]).write_bytes(b"half")
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", run)


def _unlink_fails_for_partials(monkeypatch):
    real = Path.unlink

    def unlink(self, missing_ok=False):
        if self.name.endswith(".part.mp4") and self.exists():
            raise PermissionError(errno.EACCES, "refused", str(self))
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)


# --- what is the same as before ---------------------------------------------

def test_signature_adds_only_the_proxy_ffmpeg_keyword():
    """No switch for the timeout. A caller that wants one more failed clip
    instead of an exception catches `subprocess.TimeoutExpired`."""
    sig = inspect.signature(proxy.build_proxy)
    assert list(sig.parameters) == ["path", "out", "timeout", "proxy_ffmpeg"]
    p = sig.parameters["proxy_ffmpeg"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
    assert sig.parameters["timeout"].default == 1800


def test_default_a_successful_build_returns_true_and_leaves_only_the_proxy(
        clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert out.read_bytes() == b"\x01" * 500
    assert listing(out.parent) == ["src.mp4"]


def test_default_a_failed_build_returns_false_and_removes_a_previous_proxy(
        clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    out.write_bytes(b"the previous proxy")
    fake("w500,x1")
    assert proxy.build_proxy(clip, out) is False
    assert listing(out.parent) == []


def test_default_an_empty_output_is_a_failure(clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    fake("w0")
    assert proxy.build_proxy(clip, out) is False
    assert listing(out.parent) == []


def test_default_a_missing_ffmpeg_raises_and_leaves_a_previous_proxy(
        clip, tmp_path, monkeypatch):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    out.write_bytes(b"the previous proxy")
    monkeypatch.setattr(proxy, "full_ffmpeg",
                        lambda *a, **k: str(tmp_path / "no-such-ffmpeg"))
    with pytest.raises(FileNotFoundError):
        proxy.build_proxy(clip, out)
    assert out.read_bytes() == b"the previous proxy"
    assert listing(out.parent) == ["src.mp4"]


@needs_x264
def test_default_a_junk_input_returns_false(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    out = tmp_path / "px" / "junk.mp4"
    assert proxy.build_proxy(junk, out) is False
    assert listing(out.parent) == []


# --- the bytes -----------------------------------------------------------------

@needs_x264
def test_a_proxy_is_byte_identical_to_a_direct_write(clip, tmp_path):
    """Only the name ffmpeg writes to differs. The same argv, written
    straight to the final path, gives the same bytes."""
    direct = tmp_path / "direct" / "src.mp4"
    direct.parent.mkdir()
    vf = proxy.proxy_vf(proxy.color_transfer(clip))
    subprocess.run(
        [proxy.full_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i",
         str(clip), "-map", "0:v:0", "-vf", vf] + proxy.proxy_encode_args(direct),
        check=True, stdin=subprocess.DEVNULL)
    again = tmp_path / "again.mp4"
    subprocess.run(
        [proxy.full_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i",
         str(clip), "-map", "0:v:0", "-vf", vf] + proxy.proxy_encode_args(again),
        check=True, stdin=subprocess.DEVNULL)
    if direct.read_bytes() != again.read_bytes():
        pytest.skip("this ffmpeg is not deterministic run to run")

    built = tmp_path / "built" / "src.mp4"
    assert proxy.build_proxy(clip, built) is True
    assert (hashlib.sha256(built.read_bytes()).hexdigest()
            == hashlib.sha256(direct.read_bytes()).hexdigest())


def test_the_encode_is_written_to_a_hidden_partial_beside_out(
        clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    log = tmp_path / "log.txt"
    fake("l,w500", log)
    assert proxy.build_proxy(clip, out) is True
    target, *seen = log.read_text().splitlines()
    target = Path(target)
    assert target.parent == out.parent and target != out
    assert target.name.startswith(".") and target.name.endswith(".part.mp4")
    assert seen == [target.name], (
        "when ffmpeg ran, only the empty, locked partial existed; "
        "out did not")
    assert listing(out.parent) == ["src.mp4"]


def test_out_does_not_exist_while_the_encode_runs(clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    log = tmp_path / "log.txt"
    fake("w100,l,w500", log)
    assert proxy.build_proxy(clip, out) is True
    _, *seen = log.read_text().splitlines()
    assert len(seen) == 1 and seen[0].startswith(".") and "src.mp4" not in seen


def test_partial_names_are_distinct_per_thread_and_keep_the_suffix(tmp_path):
    out = tmp_path / "a.b.mov"
    here = proxy._partial_path(out)
    seen = []
    t = threading.Thread(target=lambda: seen.append(proxy._partial_path(out)))
    t.start(); t.join()
    assert here != seen[0]
    for p in (here, seen[0]):
        assert p.parent == out.parent and p.suffix == ".mov"
        assert p.name.startswith(".") and p.name.endswith(".part.mov")
        assert p.name != out.name
    assert proxy._partial_path(tmp_path / "noext").name.endswith(".part")


def test_the_partial_name_does_not_depend_on_the_stem(tmp_path):
    """The point of the hash token: the same length for any stem, so the
    partial is writable whenever `out` is."""
    lengths = {len(proxy._partial_path(tmp_path / (stem + ".mp4")).name)
               for stem in ("a", "a" * 100, "a" * 251, "\u5199" * 83)}
    assert len(lengths) == 1
    assert lengths.pop() < 64


@pytest.mark.parametrize("name, ending", [
    ("x.mp4", ".part.mp4"),
    ("clip-0001.part.mp4", ".part.mp4"),     # a consumer's own staging name
    (".mp4", ".part.mp4"),                   # Path.suffix is '' for this one
    (".mov", ".part.mov"),
    ("a.b.c.mp4", ".part.mp4"),
    ("UPPER.MP4", ".part.MP4"),
    ("noext", ".part"),
], ids=["plain", "staging-name", "only-an-extension", "only-mov", "dots",
        "upper", "no-dot"])
def test_the_partial_ends_in_part_plus_the_text_after_the_last_dot(
        tmp_path, name, ending):
    """The pin a consumer lists its folder by. For an `out` ending `.mp4`
    the hidden partial ends `.part.mp4`, whatever the name before it.
    The muxer comes from the text after the last dot. So the partial must
    end with the same text `out` does. The consumer's own reading of its
    code is that it drops names ending `.part.mp4` from a `*.mp4` listing.
    This repository cannot check that."""
    part = proxy._partial_path(tmp_path / name)
    assert part.name.endswith(ending)
    assert part.name.startswith(".") and part.parent == tmp_path
    assert part.name != name


def test_a_name_that_is_only_an_extension_is_built_as_757f484_built_it(
        clip, tmp_path, fake):
    """`.mp4` has an empty `Path.suffix`. 757f484 passed it to ffmpeg, which
    reads the text after the last dot, so it built. A partial named from
    `Path.suffix` has no extension and ffmpeg finds no muxer."""
    out = tmp_path / "px" / ".mp4"
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert out.read_bytes() == b"\x01" * 500
    assert listing(out.parent) == [".mp4"]


@needs_x264
@pytest.mark.parametrize("name", [".mp4", ".mov"])
def test_a_real_encode_into_a_name_that_is_only_an_extension(
        clip, tmp_path, name):
    out = tmp_path / "px" / name
    assert proxy.build_proxy(clip, out) is True
    assert listing(out.parent) == [name]
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and float(r.stdout) > 1.0


def test_two_outs_in_one_directory_get_two_partials(tmp_path):
    a = proxy._partial_path(tmp_path / "one.mp4")
    b = proxy._partial_path(tmp_path / "two.mp4")
    assert a != b and a.parent == b.parent


def test_a_name_with_undecodable_bytes_gets_a_partial(tmp_path):
    odd = tmp_path / os.fsdecode(b"clip-\xff\xfe.mp4")
    assert proxy._partial_path(odd).name.endswith(".part.mp4")


# --- long names: the partial must be writable whenever `out` is -----------

NAME_MAX_STEM = 251        # + ".mp4" = 255 bytes, the ext4 limit


def _stem(n_bytes: int) -> str:
    return "s" * n_bytes


def test_every_stem_from_200_to_251_bytes_builds_as_757f484_did(
        clip, tmp_path, fake, monkeypatch):
    """These stems were first tried against real files. 757f484 returned True
    for all of them. A partial named from the stem failed from 224 up, with
    an OSError out of the cleanup. The scripted ffmpeg opens whatever name it
    is given, so the kernel's NAME_MAX is what is being tested."""
    monkeypatch.setattr(proxy, "color_transfer", lambda p: "")
    fake("w500")
    failed = {}
    for n in range(200, NAME_MAX_STEM + 1):
        out = tmp_path / f"px{n}" / f"{_stem(n)}.mp4"
        try:
            ok = proxy.build_proxy(clip, out)
        except OSError as exc:
            failed[n] = exc
            continue
        if ok is not True or out.read_bytes() != b"\x01" * 500 \
                or listing(out.parent) != [out.name]:
            failed[n] = (ok, listing(out.parent))
    assert failed == {}


@needs_x264
@pytest.mark.parametrize("stem", [_stem(223), _stem(224), _stem(225),
                                  _stem(NAME_MAX_STEM), "\u5199" * 83],
                         ids=["223", "224", "225", "251", "cjk-249-bytes"])
def test_a_real_encode_at_the_name_length_limit_builds(clip, tmp_path, stem):
    out = tmp_path / "px" / f"{stem}.mp4"
    assert len(out.name.encode()) <= 255
    assert proxy.build_proxy(clip, out) is True
    assert listing(out.parent) == [out.name]
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and float(r.stdout) > 1.0


# --- long paths: the partial must be writable whenever `out` is --------------
#
# The hash token fixed the NAME_MAX case. The full path has a limit too, and
# the hidden name is about 40 bytes longer than a short `out`. A path within
# that margin of PATH_MAX built at 757f484. A hidden name that is not checked
# against the limit cannot be opened there.

def _path_max(where: Path) -> int:
    return os.pathconf(where, "PC_PATH_MAX")


def _deep_dir(base: Path, total: int) -> Path:
    """A directory under `base` whose path is exactly `total` bytes. It is
    made one component at a time, so no system call sees a path over the
    limit."""
    base = base.resolve()
    left = total - len(str(base))
    parts = []
    while left > 0:
        n = min(200, left - 1)
        if 0 < left - (n + 1) < 2:
            n -= 1
        assert n >= 1, f"cannot make a path of {total} bytes"
        parts.append("d" * n)
        left -= n + 1
    deep = base.joinpath(*parts)
    os.makedirs(deep)
    assert len(str(deep)) == total
    return deep


def _std_len() -> int:
    """Bytes in the full hidden name for an `out` ending `.mp4`, this
    process and this thread. Spelled out here, not read from the code."""
    return (1 + 12 + 1 + len(str(os.getpid())) + 1
            + len(str(threading.get_ident())) + len(".part.mp4"))


def _short_name() -> str:
    """The shorter hidden name, this process and this thread."""
    return f".{os.getpid()}.{threading.get_ident()}.part.mp4"


def _short_len() -> int:
    return len(_short_name())


def _logged_target(log: Path) -> Path:
    return Path(log.read_text().splitlines()[0])


def _pathconf_says(monkeypatch, path_max=None, name_max=None):
    real = os.pathconf

    def pathconf(where, name):
        if name == "PC_PATH_MAX" and path_max is not None:
            return path_max
        if name == "PC_NAME_MAX" and name_max is not None:
            return name_max
        return real(where, name)

    monkeypatch.setattr(os, "pathconf", pathconf)


@pytest.mark.parametrize("path_max", ["tight", 1024, 4096])
def test_the_partial_name_by_room_left_under_the_path_limit(
        tmp_path, monkeypatch, path_max):
    """Every room from none to well over the margin, against a limit the
    test chooses. The rule: the full name when its path is under the limit,
    else the short name when its path is, else None."""
    base = tmp_path.resolve()
    if path_max == "tight":
        path_max = len(str(base)) + 100
    _pathconf_says(monkeypatch, path_max=path_max)
    checked = 0
    for out_len in range(path_max - 70, path_max + 3):
        pad = out_len - len(str(base)) - 2 - len("p.mp4")
        if pad < 1:
            continue
        out = base / ("d" * pad) / "p.mp4"        # not created: only measured
        assert len(str(out)) == out_len
        got = proxy._partial_path(out)
        full = out_len - len("p.mp4") + _std_len()
        short = out_len - len("p.mp4") + _short_len()
        if full < path_max:
            assert got is not None and len(got.name) == _std_len()
        elif short < path_max:
            assert got == out.with_name(_short_name())
        else:
            assert got is None
        if got is not None:
            assert got.name.endswith(".part.mp4") and got.parent == out.parent
        checked += 1
    assert checked >= 60


def test_an_unknown_limit_falls_back_to_4096_and_255(tmp_path, monkeypatch):
    base = tmp_path.resolve()

    def raising(where, name):
        raise OSError(errno.EINVAL, "no such limit")

    monkeypatch.setattr(os, "pathconf", raising)
    assert proxy._limits(base) == (255, 4096)
    monkeypatch.setattr(os, "pathconf", lambda where, name: -1)
    assert proxy._limits(base) == (255, 4096)


def test_a_name_over_the_name_limit_gets_no_hidden_partial(
        tmp_path, monkeypatch):
    """A 252-byte name whose extension is 250 bytes long. Both hidden names
    would pass 255 bytes, so the build is in place, as at 757f484."""
    out = tmp_path.resolve() / ("a." + "x" * 250)
    assert len(out.name) == 252
    assert proxy._partial_path(out) is None
    ok = tmp_path.resolve() / ("a." + "x" * 150)
    assert proxy._partial_path(ok) is not None


def _build_at(clip, root, fake, log, d_len, name="p.mp4"):
    """Run `build_proxy` into a directory `d_len` bytes deep and return the
    path of the file ffmpeg was told to write."""
    deep = _deep_dir(root, d_len)
    out = deep / name
    fake("l,w500", log)
    assert proxy.build_proxy(clip, out) is True
    assert out.read_bytes() == b"\x01" * 500
    assert listing(deep) == [name]
    return out, _logged_target(log)


def test_default_a_path_at_the_limit_builds_in_place_as_757f484_did(
        clip, tmp_path, fake):
    """`out` is the longest path the kernel opens. No hidden name can fit
    beside it, so ffmpeg writes to `out` itself, which is all 757f484 did."""
    root = tmp_path / "r"
    root.mkdir()
    limit = _path_max(root)
    out, target = _build_at(clip, root, fake, tmp_path / "log.txt",
                            limit - 1 - 1 - len("p.mp4"))
    assert len(str(out)) == limit - 1
    assert target == out


def test_default_a_path_over_the_limit_fails_as_757f484_did(
        clip, tmp_path, fake):
    """`out` is one byte too long to open. 757f484 ran ffmpeg, which could
    not open it, then raised ENAMETOOLONG from `out.unlink`. The same
    happens now, with nothing left behind."""
    root = tmp_path / "r"
    root.mkdir()
    deep = _deep_dir(root, _path_max(root) - 1 - len("p.mp4"))
    out = deep / "p.mp4"
    assert len(str(out)) == _path_max(root)
    fake("w500")
    with pytest.raises(OSError) as e:
        proxy.build_proxy(clip, out)
    assert e.value.errno == errno.ENAMETOOLONG
    assert listing(deep) == []


@pytest.mark.parametrize("where, expect", [
    ("full-fits-exactly", "full"),
    ("full-one-too-long", "short"),
    ("short-fits-exactly", "short"),
    ("short-one-too-long", "in-place"),
])
def test_a_deep_path_uses_the_longest_name_that_fits(
        clip, tmp_path, fake, where, expect):
    """The four edges of the two margins, on the real kernel and not on a
    pretend limit. The scripted ffmpeg opens whatever name it is given, so
    an ENAMETOOLONG from it would fail the build."""
    root = tmp_path / "r"
    root.mkdir()
    limit = _path_max(root)
    room = {"full-fits-exactly": limit - 1 - _std_len(),
            "full-one-too-long": limit - _std_len(),
            "short-fits-exactly": limit - 1 - _short_len(),
            "short-one-too-long": limit - _short_len()}[where]
    out, target = _build_at(clip, root, fake, tmp_path / "log.txt", room - 1)
    if expect == "full":
        assert len(target.name) == _std_len() and target.parent == out.parent
    elif expect == "short":
        assert target == out.with_name(_short_name())
    else:
        assert target == out
    assert len(str(target)) < limit


def test_the_short_name_is_never_shared_by_two_threads(tmp_path, monkeypatch):
    """The short name has no hash, so it cannot say which `out` it is for.
    It carries the pid and the thread instead, so two builds that run at
    once cannot meet in it, whether they build one `out` or two."""
    base = tmp_path.resolve()
    path_max = len(str(base)) + 1 + 2 + _short_len() + 1
    _pathconf_says(monkeypatch, path_max=path_max)
    out_a, out_b = base / "a.mp4", base / "b.mp4"
    seen = {}
    both = threading.Barrier(2)     # alive together, so no thread id is reused

    def ask(key, out):
        seen[key] = proxy._partial_path(out)
        both.wait(10)

    ts = [threading.Thread(target=ask, args=(k, o))
          for k, o in (("a", out_a), ("b", out_b))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen["a"] is not None and seen["b"] is not None
    assert seen["a"] != seen["b"]
    assert all(len(v.name) <= _short_len() + 4 for v in seen.values())
    assert all(v.name.count(".") == 4 for v in seen.values())   # short form


def test_two_threads_build_two_outs_in_the_short_band(clip, tmp_path, fake):
    """The case 757f484 built correctly: two different outs, one directory,
    both builds running at once, within reach of PATH_MAX. The first
    version of the short name was shared by both, and the second build
    took the first one's file."""
    root = tmp_path / "r"
    root.mkdir()
    limit = _path_max(root)
    # The short name fits with room to spare. The full one is 14 bytes longer.
    deep = _deep_dir(root, limit - 1 - _short_len() - 4)
    results = {}

    def build(name):
        results[name] = proxy.build_proxy(clip, deep / name)

    fake("w500,s1")
    ts = [threading.Thread(target=build, args=(n,)) for n in ("a.mp4", "b.mp4")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert results == {"a.mp4": True, "b.mp4": True}
    assert listing(deep) == ["a.mp4", "b.mp4"]


def test_a_relative_out_in_a_deep_directory_is_measured_as_absolute(
        clip, tmp_path, fake, monkeypatch):
    """The string handed to ffmpeg is short, but the directory is deep. The
    full name would make the absolute path too long, so the short name is
    chosen. The build still works, whichever name is picked."""
    root = tmp_path / "r"
    root.mkdir()
    deep = _deep_dir(root, _path_max(root) - _std_len())    # full: 1 too long
    monkeypatch.chdir(deep)
    log = tmp_path / "log.txt"
    fake("l,w500", log)
    assert proxy.build_proxy(clip, Path("p.mp4")) is True
    assert _logged_target(log) == Path(_short_name())
    assert listing(deep) == ["p.mp4"]


@needs_x264
@pytest.mark.parametrize("band", ["short", "in-place"])
def test_a_real_encode_near_the_path_limit_builds(clip, tmp_path, band):
    """Real ffmpeg, so a limit inside ffmpeg itself would show. Both
    fallbacks write a valid proxy."""
    root = tmp_path / "r"
    root.mkdir()
    limit = _path_max(root)
    room = (limit - 1 - _short_len() if band == "short"
            else limit - 1 - len("p.mp4") - 1)
    deep = _deep_dir(root, room - 1 if band == "short" else room)
    out = deep / "p.mp4"
    assert proxy.build_proxy(clip, out) is True
    assert listing(deep) == ["p.mp4"]
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and float(r.stdout) > 1.0


@pytest.mark.parametrize("band", ["full", "short", "in-place"])
def test_every_band_ends_the_partial_in_dot_part_dot_mp4(
        clip, tmp_path, fake, band):
    """The pin from the consumer's side, now including the fallbacks. For an
    `out` ending `.mp4`, a hidden name ends `.part.mp4`. The in-place band
    writes `out` itself and has no hidden name at all."""
    root = tmp_path / "r"
    root.mkdir()
    limit = _path_max(root)
    room = {"full": len(str(root.resolve())) + 6,
            "short": limit - 1 - _short_len(),
            "in-place": limit - 1 - len("p.mp4") - 1}[band]
    out, target = _build_at(clip, root, fake, tmp_path / "log.txt", room - 1
                            if band == "short" else room)
    if band == "in-place":
        assert target == out
    else:
        assert target.name.startswith(".")
        assert target.name.endswith(".part.mp4")


# --- a hard-killed caller: the sweep ------------------------------------------
#
# SIGKILL, an OOM kill or a container stop leaves the hidden partial behind.
# A builder holds an flock on its partial. The kernel drops it when the process
# ends, so a later build of the same `out` removes a partial it can lock, and
# touches nothing else. The pid in the name is never read.

GONE = 123          # any number: the sweep does not look at it


def _token_of(name: str) -> str:
    """The hash token, spelled out here and not read from the code."""
    return hashlib.sha1(name.encode()).hexdigest()[:12]


def _plant(directory: Path, name: str, pid=GONE, tid: int = 1,
           ext: str = ".mp4", token: str | None = None,
           data: bytes = b"left behind") -> Path:
    """Create a file named like a partial of `name`. Nobody locks it."""
    token = _token_of(name) if token is None else token
    f = directory / f".{token}.{pid}.{tid}.part{ext}"
    f.write_bytes(data)
    return f


@pytest.fixture
def held():
    """Lock files the way a live builder does, from another open file
    description. `flock` conflicts between two descriptions even inside one
    process, so no second process is needed for the unit tests."""
    fds = []

    def hold(path: Path):
        fd = os.open(path, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fds.append(fd)

    yield hold
    for fd in fds:
        os.close(fd)


def _os_unlink_fails_for_partials(monkeypatch):
    real = os.unlink

    def unlink(path, *a, **k):
        if str(path).endswith(".part.mp4"):
            raise PermissionError(errno.EACCES, "refused", str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(os, "unlink", unlink)


@needs_flock
def test_a_partial_nobody_holds_is_swept_before_the_next_build(
        clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4")
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert not old.exists()
    assert listing(out.parent) == ["src.mp4"]


@needs_flock
def test_a_partial_with_this_process_pid_is_swept_when_nobody_holds_it(
        clip, tmp_path, fake):
    """The container-restart case. The restarted process often has the pid
    of the one that was killed. A sweep that read the pid would call the
    leftover alive for ever."""
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4", pid=os.getpid(), tid=2)
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert not old.exists()


@needs_flock
@pytest.mark.parametrize("pid", ["0", "99999999999999999999"],
                         ids=["pid-zero", "huge"])
def test_the_pid_in_the_name_is_not_read(clip, tmp_path, fake, pid):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4", pid)
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert not old.exists()


@needs_flock
def test_a_partial_a_live_builder_holds_is_kept(
        clip, tmp_path, fake, held):
    """The one rule that matters: a build that is running is never touched,
    whatever pid its name carries."""
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    theirs = _plant(out.parent, "src.mp4", pid=1)
    mine = _plant(out.parent, "src.mp4", pid=os.getpid(), tid=2)
    gone = _plant(out.parent, "src.mp4", tid=3)
    held(theirs)
    held(mine)
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert theirs.exists() and mine.exists() and not gone.exists()
    assert listing(out.parent) == sorted([out.name, theirs.name, mine.name])


@needs_flock
def test_a_live_builder_in_another_process_is_not_swept(
        clip, tmp_path, monkeypatch):
    """A real second interpreter running a real `build_proxy` stands for a
    builder in another container: its pid means nothing here, and the lock
    does. Killing it releases the lock, and the next build sweeps."""
    import signal
    script = tmp_path / "fake_ff"
    script.write_text(FAKE.format(python=sys.executable))
    script.chmod(0o755)
    out = tmp_path / "px" / "src.mp4"
    child = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(proxy.__file__).parent.parent)!r})\n"
        "from pathlib import Path\n"
        "from vmcore import proxy\n"
        f"proxy.full_ffmpeg = lambda *a, **k: {str(script)!r}\n"
        f"proxy.build_proxy(Path({str(clip)!r}), Path({str(out)!r}))\n")
    p = subprocess.Popen([sys.executable, "-c", child],
                         env=dict(os.environ, FAKE_SPEC="w4096,s60"),
                         start_new_session=True, stdin=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if out.parent.exists() and any(
                    f.stat().st_size == 4096 for f in out.parent.iterdir()):
                break
            time.sleep(0.05)
        else:
            pytest.fail("the child never started its encode")
        theirs = [f for f in out.parent.iterdir()]
        assert len(theirs) == 1
        monkeypatch.setenv("FAKE_SPEC", "w500")
        monkeypatch.setattr(proxy, "full_ffmpeg", lambda *a, **k: str(script))
        assert proxy.build_proxy(clip, out) is True
        assert theirs[0].exists(), "the live builder's partial was swept"
        assert listing(out.parent) == sorted(["src.mp4", theirs[0].name])
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
    assert proxy.build_proxy(clip, out) is True
    assert listing(out.parent) == ["src.mp4"]


@needs_flock
def test_the_partial_is_locked_while_its_builder_runs_and_not_after(
        clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    fake("w10,s2")
    result = []
    t = threading.Thread(target=lambda: result.append(
        proxy.build_proxy(clip, out)))
    t.start()
    try:
        deadline = time.monotonic() + 20
        part = None
        while time.monotonic() < deadline and part is None:
            hidden = [f for f in out.parent.glob(".*.part.mp4")
                      if f.stat().st_size == 10] if out.parent.exists() else []
            part = hidden[0] if hidden else None
            time.sleep(0.02)
        assert part is not None, "the encode never started"
        fd = os.open(part, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    finally:
        t.join()
    assert result == [True]
    assert listing(out.parent) == ["src.mp4"]


@needs_flock
def test_a_hold_whose_name_was_swept_a_moment_ago_starts_again(
        tmp_path, monkeypatch):
    """A sweep can lock the file between the builder's create and its lock,
    and unlink it, and another builder can make the name again. The first
    builder then holds a lock on a file that is not at the name. It must
    notice and start again, or its encode would run unprotected."""
    part = tmp_path / ".x.1.2.part.mp4"
    real = fcntl.flock
    calls = []

    def flock(fd, op):
        real(fd, op)
        calls.append(fd)
        if len(calls) == 1:
            part.unlink()              # what the sweep did
            part.write_bytes(b"")      # and a new file under the same name

    monkeypatch.setattr(proxy.fcntl, "flock", flock)
    fd = proxy._hold(part)
    try:
        assert fd is not None and len(calls) == 2
        here, there = os.fstat(fd), os.stat(part)
        assert (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino)
    finally:
        os.close(fd)


@needs_flock
def test_the_lock_does_not_leak_a_file_descriptor(clip, tmp_path, fake):
    fds = lambda: len(os.listdir("/dev/fd"))
    out = tmp_path / "px" / "src.mp4"
    fake("w500")
    assert proxy.build_proxy(clip, out) is True      # warm anything lazy
    before = fds()
    for _ in range(3):
        assert proxy.build_proxy(clip, out) is True
    fake("w500,x1")
    assert proxy.build_proxy(clip, out) is False
    assert fds() == before


@needs_flock
def test_the_sweep_touches_no_other_out_and_no_other_file(
        clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    d = out.parent
    d.mkdir()
    token = _token_of("src.mp4")
    keep = [
        _plant(d, "other.mp4"),                          # another out, same dir
        _plant(d, "src.mp4", ext=".mov"),                # another extension
        _plant(d, "src.mp4", ext=".mp4.bak"),            # a longer ending
        _plant(d, "src.mp4", "abc"),                     # pid is not digits
        _plant(d, "src.mp4", token="0" * 12),            # a wrong token
        d / f".{GONE}.{GONE}.part.mp4",                  # the short fallback
        d / ".DS_Store",
        d / "src.mp4.old",
        d / f".{token}.{GONE}.1.part.mp4.keep",
    ]
    for f in keep:
        if not f.exists():
            f.write_bytes(b"x")
    (d / "sub").mkdir()
    nested = _plant(d / "sub", "src.mp4")                # another directory
    a_dir = d / f".{token}.{GONE}.3.part.mp4"
    a_dir.mkdir()                                        # a directory
    link = d / f".{token}.{GONE}.4.part.mp4"
    link.symlink_to(d / ".DS_Store")                     # a symlink
    before = {p.name for p in d.iterdir()}
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    after = {p.name for p in d.iterdir()}
    assert after == before | {"src.mp4"}
    assert nested.exists() and a_dir.is_dir() and link.is_symlink()
    assert (d / ".DS_Store").exists()


@needs_flock
def test_a_partial_in_another_directory_is_kept(clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    old = _plant(elsewhere, "src.mp4")
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert old.exists()


@needs_flock
def test_a_failed_build_still_sweeps_and_still_returns_false(
        clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4")
    fake("w500,x1")
    assert proxy.build_proxy(clip, out) is False
    assert listing(out.parent) == []
    assert not old.exists()


@needs_flock
def test_the_sweep_runs_after_a_named_binary_is_checked(clip, tmp_path):
    """A bad name writes nothing, and a sweep removes a file, which is a
    write. So the sweep must not run before the name is checked."""
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4")
    with pytest.raises(FileNotFoundError):
        proxy.build_proxy(clip, out, proxy_ffmpeg=tmp_path / "nope")
    assert old.exists()


@needs_flock
def test_the_sweep_runs_after_the_default_binary_is_resolved(
        clip, tmp_path, monkeypatch):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4")

    def boom():
        raise RuntimeError("no ffmpeg")

    monkeypatch.setattr(proxy, "full_ffmpeg", boom)
    with pytest.raises(RuntimeError):
        proxy.build_proxy(clip, out)
    assert old.exists()


def test_without_flock_there_is_no_sweep_and_no_lock(
        clip, tmp_path, fake, monkeypatch):
    """Windows has no `fcntl`. The build is the same, nothing is swept, and
    no lock is taken."""
    monkeypatch.setattr(proxy, "fcntl", None)
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4")
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert old.exists()
    assert proxy._hold(tmp_path / "x.part.mp4") is None
    assert not (tmp_path / "x.part.mp4").exists()


@needs_flock
def test_a_filesystem_that_refuses_flock_keeps_every_leftover(
        clip, tmp_path, fake, monkeypatch):
    """If the lock cannot be taken, the file might be live. It is kept, and
    the build runs unprotected rather than failing."""
    def refuse(fd, op):
        raise OSError(errno.ENOLCK, "no locks")

    monkeypatch.setattr(proxy.fcntl, "flock", refuse)
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4")
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert old.exists()
    assert listing(out.parent) == sorted(["src.mp4", old.name])


@needs_flock
@pytest.mark.parametrize("how", ["scandir", "unlink", "sweep-bug"])
def test_a_failing_sweep_never_changes_the_build_result(
        clip, tmp_path, fake, monkeypatch, how):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    old = _plant(out.parent, "src.mp4")
    if how == "scandir":
        def scandir(path):
            raise PermissionError(errno.EACCES, "no listing")
        monkeypatch.setattr(os, "scandir", scandir)
    elif how == "unlink":
        _os_unlink_fails_for_partials(monkeypatch)
    else:
        def broken(entry):
            raise RuntimeError("a bug in the sweep")
        monkeypatch.setattr(proxy, "_sweep_one", broken)
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert out.read_bytes() == b"\x01" * 500
    assert old.exists()


@needs_flock
@pytest.mark.parametrize("how", ["scandir", "unlink"])
def test_a_failing_sweep_does_not_hide_a_false_return_or_an_interrupt(
        clip, tmp_path, fake, monkeypatch, how):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    _plant(out.parent, "src.mp4")
    if how == "scandir":
        monkeypatch.setattr(os, "scandir", lambda p: (_ for _ in ()).throw(
            PermissionError(errno.EACCES, "no listing")))
    else:
        _os_unlink_fails_for_partials(monkeypatch)
    fake("w500,x1")
    assert proxy.build_proxy(clip, out) is False
    _interrupt_the_encode(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        proxy.build_proxy(clip, out)


def test_an_out_that_is_a_directory_raises_before_any_work(
        clip, tmp_path, monkeypatch):
    """757f484 ran the whole encode and raised IsADirectoryError from the
    unlink afterwards. Now it raises first, with the same class on Linux."""
    out = tmp_path / "px" / "src.mp4"
    out.mkdir(parents=True)
    ran = []
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: (
        ran.append(cmd), real(cmd, **k))[1])
    monkeypatch.setattr(proxy, "full_ffmpeg", lambda *a, **k: "ffmpeg")
    with pytest.raises(IsADirectoryError) as e:
        proxy.build_proxy(clip, out)
    assert e.value.errno == errno.EISDIR and str(out) in str(e.value)
    assert not any("ffmpeg" == Path(str(c[0])).name for c in ran)
    assert listing(out.parent) == ["src.mp4"] and list(out.iterdir()) == []


def test_a_build_after_a_real_kill_of_the_caller_sweeps_what_it_left(
        clip, tmp_path, monkeypatch):
    """A caller killed with SIGKILL while ffmpeg runs, as an OOM kill does.
    The hidden partial stays, and `out` does not exist. The next build, by a
    process with another pid, removes the partial and writes `out`.
    This runs a real second interpreter and a scripted ffmpeg."""
    import signal
    import time
    script = tmp_path / "fake_ff"
    script.write_text(FAKE.format(python=sys.executable))
    script.chmod(0o755)
    out = tmp_path / "px" / "src.mp4"
    child = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(proxy.__file__).parent.parent)!r})\n"
        "from pathlib import Path\n"
        "from vmcore import proxy\n"
        f"proxy.full_ffmpeg = lambda *a, **k: {str(script)!r}\n"
        f"proxy.build_proxy(Path({str(clip)!r}), Path({str(out)!r}))\n")
    env = dict(os.environ, FAKE_SPEC="w4096,s60")
    p = subprocess.Popen([sys.executable, "-c", child], env=env,
                         start_new_session=True, stdin=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if out.parent.exists() and any(
                    f.stat().st_size == 4096 for f in out.parent.iterdir()):
                break
            time.sleep(0.05)
        else:
            pytest.fail("the child never started its encode")
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
    left = listing(out.parent)
    assert len(left) == 1 and left[0].endswith(".part.mp4")
    assert not out.exists()

    monkeypatch.setenv("FAKE_SPEC", "w500")
    monkeypatch.setattr(proxy, "full_ffmpeg", lambda *a, **k: str(script))
    assert proxy.build_proxy(clip, out) is True
    assert listing(out.parent) == ["src.mp4"]


def test_a_name_757f484_could_not_write_fails_the_same_way(
        clip, tmp_path, fake, monkeypatch):
    """A 252-byte stem makes a 256-byte file name. 757f484 raised OSError
    ENAMETOOLONG from `out.unlink` after ffmpeg failed to open it. Now the
    encode succeeds into the short partial and `os.replace` raises the same
    error, and the partial does not stay behind."""
    monkeypatch.setattr(proxy, "color_transfer", lambda p: "")
    fake("w500")
    out = tmp_path / "px" / f"{_stem(NAME_MAX_STEM + 1)}.mp4"
    with pytest.raises(OSError) as e:
        proxy.build_proxy(clip, out)
    assert e.value.errno == errno.ENAMETOOLONG
    assert listing(out.parent) == []


def _run_double(monkeypatch, write):
    """Stand in for `subprocess.run` on the encode only. `write(argv)` is what
    the double does with the argv. ffprobe still runs for real."""
    real = subprocess.run

    def run(cmd, **kw):
        if "ffprobe" in Path(str(cmd[0])).name:
            return real(cmd, **kw)
        write(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(proxy, "full_ffmpeg", lambda *a, **k: "ffmpeg")


def test_a_double_that_writes_the_last_argument_is_a_success(
        clip, tmp_path, monkeypatch):
    """The advice in the docstring: the last element of the argv is the file
    to create."""
    out = tmp_path / "px" / "src.mp4"
    _run_double(monkeypatch, lambda cmd: Path(cmd[-1]).write_bytes(b"x" * 320))
    assert proxy.build_proxy(clip, out) is True
    assert out.read_bytes() == b"x" * 320
    assert listing(out.parent) == ["src.mp4"]


@pytest.mark.parametrize("how", ["nothing", "straight-to-out"])
def test_a_double_that_exits_zero_without_writing_the_last_argument_fails(
        clip, tmp_path, monkeypatch, how):
    """757f484 returned True here and kept `out`. The check now looks at the
    partial, so this is False and `out` is removed. It is documented, and
    it is pinned so the documentation stays true."""
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    out.write_bytes(b"the previous proxy")
    _run_double(monkeypatch,
                (lambda cmd: None) if how == "nothing"
                else (lambda cmd: out.write_bytes(b"x" * 320)))
    assert proxy.build_proxy(clip, out) is False
    assert listing(out.parent) == []


def test_the_partial_is_flushed_before_it_is_renamed(
        clip, tmp_path, fake, monkeypatch):
    """The durability the rename is for. The order is flush, then rename."""
    out = tmp_path / "px" / "src.mp4"
    events = []
    real_fsync, real_replace = proxy._fsync_file, os.replace

    def fsync_spy(path):
        events.append(("fsync", Path(path).name))
        real_fsync(path)

    def replace_spy(src, dst):
        events.append(("replace", Path(src).name, Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr(proxy, "_fsync_file", fsync_spy)
    monkeypatch.setattr(os, "replace", replace_spy)
    fake("w500")
    assert proxy.build_proxy(clip, out) is True
    assert [e[0] for e in events] == ["fsync", "replace"]
    assert events[0][1] == events[1][1] and events[1][2] == "src.mp4"


def test_a_failing_flush_raises_and_leaves_a_previous_proxy_as_it_was(
        clip, tmp_path, fake, monkeypatch):
    """The failure path the flush adds. 757f484 had no flush. The encode
    finished, the flush failed, and the call raises the OSError. `out` is
    untouched and the partial is gone."""
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    out.write_bytes(b"the previous proxy")

    def failing(path):
        raise OSError(errno.EIO, "the disk said no")

    monkeypatch.setattr(proxy, "_fsync_file", failing)
    fake("w500")
    with pytest.raises(OSError) as e:
        proxy.build_proxy(clip, out)
    assert e.value.errno == errno.EIO
    assert out.read_bytes() == b"the previous proxy"
    assert listing(out.parent) == ["src.mp4"]


def test_a_failing_cleanup_does_not_hide_a_false_return(
        clip, tmp_path, fake, monkeypatch):
    out = tmp_path / "px" / "src.mp4"
    fake("w500,x1")
    _unlink_fails_for_partials(monkeypatch)
    assert proxy.build_proxy(clip, out) is False


@pytest.mark.slow
def test_a_failing_cleanup_does_not_hide_a_timeout(
        clip, tmp_path, fake, monkeypatch):
    out = tmp_path / "px" / "src.mp4"
    fake("w4096,s30")
    _unlink_fails_for_partials(monkeypatch)
    with pytest.raises(subprocess.TimeoutExpired):
        proxy.build_proxy(clip, out, timeout=1)


def test_a_failing_cleanup_does_not_hide_an_interrupt(
        clip, tmp_path, monkeypatch):
    out = tmp_path / "px" / "src.mp4"
    _interrupt_the_encode(monkeypatch)
    _unlink_fails_for_partials(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        proxy.build_proxy(clip, out)


# --- the timeout --------------------------------------------------------------

@pytest.mark.slow
def test_a_timeout_raises_and_leaves_nothing_behind(clip, tmp_path, fake):
    """Measured on 757f484: the same call raised TimeoutExpired and left a
    file at `out` that ffprobe rejects ('moov atom not found'), which
    `if not out.exists()` then skipped for ever."""
    out = tmp_path / "px" / "src.mp4"
    fake("w4096,s30")
    with pytest.raises(subprocess.TimeoutExpired):
        proxy.build_proxy(clip, out, timeout=1)
    assert listing(out.parent) == []
    assert not out.exists()


@pytest.mark.slow
def test_a_timeout_leaves_a_previous_proxy_exactly_as_it_was(
        clip, tmp_path, fake):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    out.write_bytes(b"the previous proxy")
    fake("w4096,s30")
    with pytest.raises(subprocess.TimeoutExpired):
        proxy.build_proxy(clip, out, timeout=1)
    assert out.read_bytes() == b"the previous proxy"
    assert listing(out.parent) == ["src.mp4"]


@pytest.mark.slow
def test_a_skip_if_exists_loop_retries_after_a_timeout(clip, tmp_path, fake):
    """The consumer's pattern, `todo = [... if not d.exists()]`, over two
    runs. The first times out and the caller counts it as a failed clip.
    The second must still see work to do."""
    out = tmp_path / "px" / "src.mp4"
    fake("w4096,s30")
    try:
        proxy.build_proxy(clip, out, timeout=1)
        failed = False
    except subprocess.TimeoutExpired:
        failed = True
    assert failed
    todo = [(clip, out)] if not out.exists() else []
    assert todo, "a timed-out proxy must not look done"
    fake("w500")
    assert proxy.build_proxy(*todo[0]) is True
    assert out.read_bytes() == b"\x01" * 500


@pytest.mark.slow
def test_the_timeout_is_still_subprocess_timeout_expired(
        clip, tmp_path, fake):
    """The exception 757f484 raised, not a subclass or a wrapper. Its cmd
    now names the partial, which is the hidden argv change."""
    out = tmp_path / "px" / "src.mp4"
    fake("w4096,s30")
    with pytest.raises(subprocess.TimeoutExpired) as e:
        proxy.build_proxy(clip, out, timeout=1)
    assert type(e.value) is subprocess.TimeoutExpired
    assert e.value.timeout == 1
    assert Path(e.value.cmd[-1]).parent == out.parent


def test_there_is_no_on_timeout_switch(clip, tmp_path):
    out = tmp_path / "never" / "src.mp4"
    with pytest.raises(TypeError, match="on_timeout"):
        proxy.build_proxy(clip, out, on_timeout="false")
    assert not out.parent.exists()


# --- other ways out -----------------------------------------------------------

def test_an_interrupt_removes_the_partial_and_leaves_a_previous_proxy(
        clip, tmp_path, monkeypatch):
    out = tmp_path / "px" / "src.mp4"
    out.parent.mkdir()
    out.write_bytes(b"the previous proxy")
    _interrupt_the_encode(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        proxy.build_proxy(clip, out)
    assert out.read_bytes() == b"the previous proxy"
    assert listing(out.parent) == ["src.mp4"]


@needs_x264
def test_concurrent_builds_of_one_proxy_each_get_a_whole_file(clip, tmp_path):
    """The partial name carries the thread, so two builds of one `out` do
    not write into each other's file. Whichever renames last wins, and
    both files were complete."""
    out = tmp_path / "px" / "src.mp4"
    results, errors = [], []

    def go():
        try:
            results.append(proxy.build_proxy(clip, out))
        except Exception as exc:       # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == [] and results == [True] * 4
    assert listing(out.parent) == ["src.mp4"]
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and float(r.stdout) > 1.0
