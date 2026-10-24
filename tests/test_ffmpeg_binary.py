"""Which ffmpeg the proxy path runs, and who gets to say.

Two binaries are in play on purpose. The analysis-only decode, `audio_peak`
and `thumbs.extract` run `ffmpeg` from PATH, because their decode behaviour
is load-bearing for the consumers that pin them. The proxy path (`build_proxy`
and the fused decode, the ones that tonemap HLG) runs `full_ffmpeg()`, which
needs zscale. These tests pin three things:

1. The order `full_ffmpeg` looks in: argument, then a tools/ build above
   the package, then PATH. No environment variable is read.
2. A binary the caller NAMES is checked and never silently replaced.
3. Naming one moves argv[0] of the proxy path and nothing else, and leaves
   the PATH-ffmpeg sites exactly where they were.
4. A consumer that replaced `full_ffmpeg` with a zero-argument function
   keeps working, because the keyword is passed only when it is set.

Anything that must run a real binary uses a small shell wrapper around the
real ffmpeg that logs its arguments, so "the argument reached the process"
is observed rather than assumed.
"""
from __future__ import annotations

import ast
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vmcore import frames, proxy
from vmcore.frames import analysis_vf, decode_cmd, sample_frames
from vmcore.proxy import build_proxy, full_ffmpeg
from vmcore.testing import make_mp4

OLD_ENV = "VMCORE_PROXY_FFMPEG"      # read by no code; see the pin below
LONG_EDGE = 64
PACKAGE = Path(proxy.__file__).resolve().parent


@pytest.fixture
def root_without_tools(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "root_without_tools"
    root.mkdir()
    monkeypatch.setattr(proxy, "_ROOT", root)
    return root


def _wrapper(path: Path, log: Path | None = None) -> Path:
    """An executable that runs the real ffmpeg, noting its arguments."""
    real = shutil.which("ffmpeg")
    lines = ["#!/bin/sh"]
    if log is not None:
        lines.append(f'printf "%s\\n" "$@" > "{log}"')
    lines.append(f'exec "{real}" "$@"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)
    return path


def _root_with_tools(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "root_with_tools"
    binary = _wrapper(root / "tools" / "ffmpeg" / "ffmpeg")
    monkeypatch.setattr(proxy, "_ROOT", root)
    return binary


def _can(*needles: str) -> bool:
    """True when the PATH ffmpeg has every named encoder or filter."""
    out = ""
    for flag in ("-encoders", "-filters"):
        r = subprocess.run(["ffmpeg", "-hide_banner", flag],
                           capture_output=True, text=True)
        out += r.stdout
    return all(f" {n} " in out for n in needles)


# --------------------------------------------------------------------------
# The order of the search
# --------------------------------------------------------------------------

def test_default_is_the_path_ffmpeg_when_nothing_is_named(root_without_tools):
    assert full_ffmpeg() == "ffmpeg"


def test_a_tools_build_above_the_package_is_still_found(tmp_path, monkeypatch):
    """Step 2 is unchanged: exactly the string it returned before."""
    binary = _root_with_tools(tmp_path, monkeypatch)
    assert full_ffmpeg() == str(binary)


def test_argument_beats_the_tools_build(tmp_path, monkeypatch):
    _root_with_tools(tmp_path, monkeypatch)
    arg = _wrapper(tmp_path / "arg" / "ffmpeg")
    assert full_ffmpeg(proxy_ffmpeg=arg) == str(arg)


def test_no_environment_variable_is_read(tmp_path, monkeypatch):
    """VMCORE_PROXY_FFMPEG is not read. The argument is the one way in. A hidden input makes a run depend on the
    shell that started it. Exported with a bad path, it must change nothing,
    neither the result nor an error."""
    binary = _root_with_tools(tmp_path, monkeypatch)
    monkeypatch.setenv(OLD_ENV, str(tmp_path / "nowhere" / "ffmpeg"))
    assert full_ffmpeg() == str(binary)
    assert not hasattr(proxy, "ENV_PROXY_FFMPEG")
    cmd = decode_cmd(tmp_path / "x.mp4", analysis_vf(2.0, LONG_EDGE),
                     tmp_path / "px" / "c.mp4")
    assert cmd[0] == str(binary)


def test_a_path_object_is_accepted_and_a_str_comes_back(tmp_path,
                                                        root_without_tools):
    named = _wrapper(tmp_path / "bin" / "ffmpeg")
    got = full_ffmpeg(proxy_ffmpeg=named)
    assert got == str(named) and type(got) is str


def test_a_bare_command_name_is_kept_verbatim_and_beats_tools(
        tmp_path, monkeypatch):
    """`proxy_ffmpeg="ffmpeg"` is how a caller says 'the PATH one, even
    though a tools/ build exists'."""
    _root_with_tools(tmp_path, monkeypatch)
    assert full_ffmpeg(proxy_ffmpeg="ffmpeg") == "ffmpeg"


def test_the_new_argument_is_keyword_only_with_a_none_default():
    for fn in (full_ffmpeg, build_proxy, decode_cmd, sample_frames):
        p = inspect.signature(fn).parameters["proxy_ffmpeg"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        assert p.default is None, fn.__name__


def test_the_root_anchor_is_the_directory_above_the_package():
    """Step 3 reads `_ROOT / tools / ffmpeg / ffmpeg`. A refactor that
    moved the anchor would silently drop it for editable installs."""
    assert proxy._ROOT == PACKAGE.parent


# --------------------------------------------------------------------------
# A named binary is checked, and never replaced
# --------------------------------------------------------------------------

def test_missing_argument_raises_and_names_the_argument(tmp_path,
                                                        root_without_tools):
    gone = tmp_path / "nope" / "ffmpeg"
    with pytest.raises(FileNotFoundError, match="proxy_ffmpeg") as e:
        full_ffmpeg(proxy_ffmpeg=gone)
    assert str(gone) in str(e.value)


def test_a_directory_is_refused(tmp_path, root_without_tools):
    with pytest.raises(IsADirectoryError, match="proxy_ffmpeg"):
        full_ffmpeg(proxy_ffmpeg=tmp_path)


def test_a_file_without_execute_permission_is_refused(tmp_path,
                                                      root_without_tools):
    f = tmp_path / "ffmpeg"
    f.write_text("not a program\n")
    f.chmod(0o644)
    if os.access(f, os.X_OK):
        pytest.skip("this filesystem reports every file as executable")
    with pytest.raises(PermissionError, match="proxy_ffmpeg"):
        full_ffmpeg(proxy_ffmpeg=f)


def test_a_bare_name_that_is_not_on_path_is_refused(root_without_tools):
    with pytest.raises(FileNotFoundError, match="not a command on PATH"):
        full_ffmpeg(proxy_ffmpeg="no-such-ffmpeg-binary-anywhere")


def test_an_empty_string_argument_is_refused(root_without_tools):
    with pytest.raises(ValueError, match="empty string"):
        full_ffmpeg(proxy_ffmpeg="")


@pytest.mark.parametrize("bad", [3, b"/usr/bin/ffmpeg", ["ffmpeg"]])
def test_a_wrong_type_is_refused(bad, root_without_tools):
    with pytest.raises(TypeError, match="proxy_ffmpeg"):
        full_ffmpeg(proxy_ffmpeg=bad)


def test_a_named_binary_needs_ffprobe_on_path(tmp_path, monkeypatch,
                                              root_without_tools):
    """Naming an absolute ffmpeg is the one way to run with no ffprobe on
    PATH, and `color_transfer` turns that into '' - the SDR chain for an
    HLG clip, exit 0. So a named binary is refused when ffprobe is gone."""
    named = _wrapper(tmp_path / "bin" / "ffmpeg")
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(FileNotFoundError, match="ffprobe is not on PATH"):
        full_ffmpeg(proxy_ffmpeg=named)


def test_naming_nothing_does_not_look_for_ffprobe(tmp_path, monkeypatch,
                                                  root_without_tools):
    """The default path never checked, and still does not: on a PATH with no
    ffprobe it returns what it always returned."""
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert full_ffmpeg() == "ffmpeg"


def test_the_tools_build_keeps_its_old_weak_check(tmp_path, monkeypatch):
    """What the default path did before this argument existed, pinned: a
    tools/ffmpeg/ffmpeg that merely EXISTS is returned, executable or not.
    Tightening it would change a default call, so only what a caller names
    is validated."""
    root = tmp_path / "root"
    cand = root / "tools" / "ffmpeg" / "ffmpeg"
    cand.parent.mkdir(parents=True)
    cand.write_text("not a program\n")
    cand.chmod(0o644)
    monkeypatch.setattr(proxy, "_ROOT", root)
    assert full_ffmpeg() == str(cand)


# --------------------------------------------------------------------------
# The argument moves argv[0] of the proxy path, and nothing else
# --------------------------------------------------------------------------

def test_fused_decode_cmd_takes_the_named_binary_in_argv0_only(
        tiny_clip, tmp_path, root_without_tools):
    vf = analysis_vf(2.0, LONG_EDGE)
    out = tmp_path / "px" / "c.mp4"
    named = _wrapper(tmp_path / "bin" / "ffmpeg")
    default = decode_cmd(tiny_clip, vf, out)
    given = decode_cmd(tiny_clip, vf, out, proxy_ffmpeg=named)
    assert default[0] == "ffmpeg"
    assert given[0] == str(named)
    assert given[1:] == default[1:]


def test_analysis_only_decode_cmd_ignores_the_named_binary(
        tiny_clip, tmp_path, monkeypatch):
    """The split this change must not blur: naming a full build never moves
    the analysis decode off the PATH binary, so a caller can pass it on
    every call."""
    vf = analysis_vf(2.0, LONG_EDGE)
    named = _wrapper(tmp_path / "bin" / "ffmpeg")
    plain = decode_cmd(tiny_clip, vf)
    assert plain[0] == "ffmpeg"
    assert decode_cmd(tiny_clip, vf, proxy_ffmpeg=named) == plain


def test_fused_decode_cmd_with_a_missing_binary_raises(tiny_clip, tmp_path):
    with pytest.raises(FileNotFoundError, match="proxy_ffmpeg"):
        decode_cmd(tiny_clip, analysis_vf(2.0, LONG_EDGE),
                   tmp_path / "px" / "c.mp4",
                   proxy_ffmpeg=tmp_path / "nope" / "ffmpeg")


def test_build_proxy_runs_the_named_binary_with_otherwise_unchanged_args(
        tiny_clip, tmp_path, monkeypatch):
    """Two ways to name the same wrapper, the argument and a tools/ build,
    must produce the same arguments after argv[0]. The binary is the only
    thing that moved."""
    if not _can("libx264"):
        pytest.skip("no libx264 in the PATH ffmpeg")
    log_arg, log_tools = (tmp_path / n for n in ("arg.log", "tools.log"))

    assert build_proxy(tiny_clip, tmp_path / "a" / "p.mp4",
                       proxy_ffmpeg=_wrapper(tmp_path / "w1", log_arg))

    root = tmp_path / "root"
    _wrapper(root / "tools" / "ffmpeg" / "ffmpeg", log_tools)
    monkeypatch.setattr(proxy, "_ROOT", root)
    assert build_proxy(tiny_clip, tmp_path / "t" / "p.mp4")

    def args_with_outputs_masked(log: Path, out: Path) -> list[str]:
        # The encode writes a hidden partial beside `out`, so mask the
        # directory, which both names share, and not the file.
        return [a.replace(str(out.parent), "OUT")
                for a in log.read_text().splitlines()]

    a = args_with_outputs_masked(log_arg, tmp_path / "a" / "p.mp4")
    t = args_with_outputs_masked(log_tools, tmp_path / "t" / "p.mp4")
    assert a == t
    assert "-hide_banner" in a and "-vf" in a


def test_build_proxy_with_a_missing_binary_raises_before_writing(tmp_path):
    clip = tmp_path / "in.mp4"
    make_mp4(clip, 0.5)
    out = tmp_path / "made" / "p.mp4"
    with pytest.raises(FileNotFoundError, match="proxy_ffmpeg"):
        build_proxy(clip, out, proxy_ffmpeg=tmp_path / "nope" / "ffmpeg")
    assert not out.parent.exists()


def test_fused_sample_frames_runs_the_named_binary(tiny_clip, tmp_path):
    if not _can("libx264"):
        pytest.skip("no libx264 in the PATH ffmpeg")
    log = tmp_path / "fused.log"
    out = tmp_path / "px" / "p.mp4"
    got = list(sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE,
                             proxy_out=out,
                             proxy_ffmpeg=_wrapper(tmp_path / "w", log)))
    base_out = tmp_path / "base" / "p.mp4"
    base = list(sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE,
                              proxy_out=base_out))
    assert log.exists(), "the named binary was never run"
    assert "-filter_complex" in log.read_text().splitlines()
    assert out.exists() and out.stat().st_size > 0
    assert len(got) == len(base) > 0
    for (ta, a), (tb, b) in zip(got, base):
        assert ta == tb and np.array_equal(a, b)


def test_plain_sample_frames_never_runs_the_named_binary(tiny_clip, tmp_path):
    log = tmp_path / "plain.log"
    got = list(sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE,
                             proxy_ffmpeg=_wrapper(tmp_path / "w", log)))
    assert len(got) > 0
    assert not log.exists()


def test_fused_sample_frames_with_a_missing_binary_raises_on_first_frame(
        tiny_clip, tmp_path):
    it = sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE,
                       proxy_out=tmp_path / "px" / "p.mp4",
                       proxy_ffmpeg=tmp_path / "nope" / "ffmpeg")
    with pytest.raises(FileNotFoundError, match="proxy_ffmpeg"):
        next(it)


# --------------------------------------------------------------------------
# The reason this exists
# --------------------------------------------------------------------------

def _slim_path_ffmpeg(tmp_path: Path) -> Path:
    """A PATH ffmpeg that cannot do zscale, as the iMac's brew build can't."""
    real = shutil.which("ffmpeg")
    slim = tmp_path / "slimbin"
    slim.mkdir()
    (slim / "ffmpeg").write_text(
        '#!/bin/sh\nfor a in "$@"; do set -- "$@" '
        '"$(printf %s "$a" | sed s/zscale/zscale_absent/g)"; shift; done\n'
        f'exec "{real}" "$@"\n')
    (slim / "ffmpeg").chmod(0o755)
    return slim


def _hlg_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "hlg.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
         "-f", "lavfi", "-i", "smptebars=size=64x36:rate=24:duration=1",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
         "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
         "-colorspace", "bt2020nc", str(clip)],
        check=True, capture_output=True)
    return clip


def test_hlg_proxy_fails_on_a_slim_path_ffmpeg_and_works_when_one_is_named(
        tmp_path, monkeypatch, root_without_tools):
    """The measured failure: with no tools/ beside the package, the proxy
    path lands on the slim PATH binary and an HLG clip gets no proxy. Naming
    the full build fixes it, for the standalone render and the fused one."""
    if not _can("libx264", "zscale", "tonemap"):
        pytest.skip("needs libx264, zscale and tonemap in the PATH ffmpeg")
    clip = _hlg_clip(tmp_path)
    full = shutil.which("ffmpeg")
    monkeypatch.setenv("PATH", f"{_slim_path_ffmpeg(tmp_path)}:"
                               f"{os.environ['PATH']}")

    assert build_proxy(clip, tmp_path / "default.mp4") is False
    assert build_proxy(clip, tmp_path / "named.mp4",
                       proxy_ffmpeg=full) is True

    fused = tmp_path / "fused.mp4"
    assert list(sample_frames(clip, 1.0, fps=2.0, long_edge=LONG_EDGE,
                              proxy_out=tmp_path / "slim_fused.mp4")) == []
    got = list(sample_frames(clip, 1.0, fps=2.0, long_edge=LONG_EDGE,
                             proxy_out=fused, proxy_ffmpeg=full))
    assert len(got) > 0 and fused.stat().st_size > 0


def test_an_installed_copy_cannot_find_tools_but_can_be_told(tmp_path):
    """The package copied into a site-packages directory, the way a pip
    install lays it out, next to an app whose tools/ it cannot see."""
    site = tmp_path / "site-packages"
    shutil.copytree(PACKAGE, site / "vmcore",
                    ignore=shutil.ignore_patterns("__pycache__"))
    app_ffmpeg = _wrapper(tmp_path / "app" / "tools" / "ffmpeg" / "ffmpeg")
    child = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(site)!r})\n"
        "import vmcore.proxy as p\n"
        "print(json.dumps([str(p._ROOT), p.full_ffmpeg(), "
        f"p.full_ffmpeg(proxy_ffmpeg={str(app_ffmpeg)!r})]))\n")
    r = subprocess.run([sys.executable, "-I", "-c", child], cwd=tmp_path,
                       capture_output=True, text=True, check=True)
    import json
    root, default, named = json.loads(r.stdout)
    assert root == str(site)
    assert default == "ffmpeg"
    assert named == str(app_ffmpeg)


# --------------------------------------------------------------------------
# A consumer that replaced full_ffmpeg keeps working
# --------------------------------------------------------------------------

def test_a_zero_argument_full_ffmpeg_stub_still_serves_every_call_site(
        tiny_clip, tmp_path, monkeypatch):
    """The usual way to pin or redirect the proxy binary is to replace
    `vmcore.proxy.full_ffmpeg`. A stub that takes no arguments worked before
    `proxy_ffmpeg` existed. It must work on every call that names no binary:
    the fused argv, the standalone render and the fused decode. Only a call
    that names one passes the keyword."""
    if not _can("libx264"):
        pytest.skip("no libx264 in the PATH ffmpeg")
    real = shutil.which("ffmpeg")
    calls = []

    def stub():
        calls.append(1)
        return real

    monkeypatch.setattr(proxy, "full_ffmpeg", stub)
    cmd = decode_cmd(tiny_clip, analysis_vf(2.0, LONG_EDGE),
                     tmp_path / "px" / "c.mp4")
    assert cmd[0] == real and len(calls) == 1

    assert build_proxy(tiny_clip, tmp_path / "bp" / "p.mp4") is True
    assert len(calls) == 2

    fused = tmp_path / "fused" / "p.mp4"
    got = list(sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE,
                             proxy_out=fused))
    assert len(got) == 2 and fused.stat().st_size > 0
    assert len(calls) == 3


def test_default_a_raising_full_ffmpeg_leaves_the_directory_as_757f484_did(
        tiny_clip, tmp_path, monkeypatch):
    """A resolver that raises, with no binary named. 757f484 made the output
    directory first, so the directory was already there when the error came
    out. Resolving first would leave nothing behind, which a caller that
    lists the tree after a failed run could see."""
    def boom():
        raise RuntimeError("no ffmpeg for you")

    monkeypatch.setattr(proxy, "full_ffmpeg", boom)
    out = tmp_path / "x" / "y" / "p.mp4"
    with pytest.raises(RuntimeError, match="no ffmpeg for you"):
        build_proxy(tiny_clip, out)
    assert out.parent.is_dir() and list(out.parent.iterdir()) == []


def test_default_the_binary_is_resolved_after_the_directory_and_the_probe(
        tiny_clip, tmp_path, monkeypatch):
    """The call order 757f484 had: make the directory, ffprobe the colour
    transfer, then resolve the binary. A stub that counts or orders those
    calls sees the same sequence."""
    out = tmp_path / "x" / "p.mp4"
    seen = []

    def probe_stub(path):
        seen.append(("color_transfer", out.parent.is_dir()))
        return ""

    def resolve_stub():
        seen.append(("full_ffmpeg", out.parent.is_dir()))
        raise RuntimeError("stop here")

    monkeypatch.setattr(proxy, "color_transfer", probe_stub)
    monkeypatch.setattr(proxy, "full_ffmpeg", resolve_stub)
    with pytest.raises(RuntimeError, match="stop here"):
        build_proxy(tiny_clip, out)
    assert seen == [("color_transfer", True), ("full_ffmpeg", True)]


def test_a_named_binary_is_resolved_before_the_directory_and_the_probe(
        tiny_clip, tmp_path, monkeypatch):
    """The one exception to the order above. A bad name raises with nothing
    written and nothing probed, which the Change 2 contract promises."""
    out = tmp_path / "x" / "p.mp4"
    seen = []

    def resolve_spy(*, proxy_ffmpeg):
        seen.append(("full_ffmpeg", out.parent.exists()))
        raise ValueError("bad name")

    def probe_spy(path):
        seen.append(("color_transfer", None))
        return ""

    monkeypatch.setattr(proxy, "full_ffmpeg", resolve_spy)
    monkeypatch.setattr(proxy, "color_transfer", probe_spy)
    with pytest.raises(ValueError, match="bad name"):
        build_proxy(tiny_clip, out, proxy_ffmpeg="elsewhere")
    assert seen == [("full_ffmpeg", False)]
    assert not out.parent.exists()


def _module_stub(**over):
    """An object standing in for the whole `vmcore.proxy` module, with the
    four public names `decode_cmd` uses and nothing else."""
    import types
    ns = dict(full_ffmpeg=lambda: "myffmpeg",
              proxy_vf=lambda transfer: "scale=1280:-2",
              color_transfer=lambda path: "",
              proxy_encode_args=lambda out: ["-y", str(out)])
    ns.update(over)
    return types.SimpleNamespace(**ns)


def test_default_decode_cmd_works_when_the_whole_proxy_module_is_replaced(
        tmp_path, monkeypatch):
    """757f484 read four public names off `vmcore.proxy`. A consumer that
    swapped the module for a namespace holding those four still got its
    binary in argv[0]. A private helper on the module breaks that."""
    import vmcore
    stub = _module_stub()
    monkeypatch.setattr(vmcore, "proxy", stub)
    monkeypatch.setitem(sys.modules, "vmcore.proxy", stub)
    cmd = decode_cmd(tmp_path / "in.mp4", "null", tmp_path / "px" / "c.mp4")
    assert cmd[0] == "myffmpeg"
    assert cmd[-2:] == ["-y", str(tmp_path / "px" / "c.mp4")]


def test_default_decode_cmd_calls_the_probe_and_the_filter_before_the_binary(
        tmp_path, monkeypatch):
    """757f484 ran `proxy_vf(color_transfer(path))` and then `full_ffmpeg()`.
    A recording stub sees the order, and so does a `full_ffmpeg` that raises:
    the probe has already run when it does."""
    import vmcore
    calls = []

    def rec(name, value):
        def f(*a, **k):
            calls.append(name)
            return value
        return f

    monkeypatch.setattr(vmcore, "proxy", _module_stub(
        full_ffmpeg=rec("full_ffmpeg", "myffmpeg"),
        proxy_vf=rec("proxy_vf", "scale=1280:-2"),
        color_transfer=rec("color_transfer", ""),
        proxy_encode_args=rec("proxy_encode_args", ["-y", "p.mp4"])))
    decode_cmd(tmp_path / "in.mp4", "null", tmp_path / "p.mp4")
    assert calls == ["color_transfer", "proxy_vf", "full_ffmpeg",
                     "proxy_encode_args"]


def test_a_named_binary_is_still_checked_before_the_probe(
        tmp_path, monkeypatch):
    import vmcore
    calls = []

    def named(**kw):
        calls.append("full_ffmpeg")
        return "x"

    monkeypatch.setattr(vmcore, "proxy", _module_stub(
        full_ffmpeg=named,
        color_transfer=lambda p: calls.append("color_transfer") or ""))
    decode_cmd(tmp_path / "in.mp4", "null", tmp_path / "p.mp4",
               proxy_ffmpeg="elsewhere")
    assert calls == ["full_ffmpeg", "color_transfer"]


def test_default_decode_cmd_with_a_magicmock_module_gets_a_str_binary(
        tmp_path, monkeypatch):
    """A MagicMock module returns a MagicMock for any attribute. A call to
    an unknown helper would put a mock into argv[0]. Popen would then
    receive the mock as the binary."""
    from unittest import mock
    import vmcore
    m = mock.MagicMock()
    m.full_ffmpeg.return_value = "myffmpeg"
    m.proxy_vf.return_value = "scale=1280:-2"
    m.color_transfer.return_value = ""
    m.proxy_encode_args.return_value = ["-y", "p.mp4"]
    monkeypatch.setattr(vmcore, "proxy", m)
    cmd = decode_cmd(tmp_path / "in.mp4", "null", tmp_path / "p.mp4")
    assert cmd[0] == "myffmpeg"
    assert all(isinstance(a, str) for a in cmd)
    m.full_ffmpeg.assert_called_once_with()


def test_a_named_binary_reaches_a_replaced_module_as_the_keyword(
        tmp_path, monkeypatch):
    import vmcore
    seen = []

    def spy(**kw):
        seen.append(kw)
        return "named-by-stub"

    monkeypatch.setattr(vmcore, "proxy", _module_stub(full_ffmpeg=spy))
    cmd = decode_cmd(tmp_path / "in.mp4", "null", tmp_path / "c.mp4",
                     proxy_ffmpeg="elsewhere")
    assert cmd[0] == "named-by-stub"
    assert seen == [{"proxy_ffmpeg": "elsewhere"}]


def test_decode_cmd_uses_no_private_name_of_the_proxy_module():
    """The rule behind the tests above, stated for the source: inside
    `decode_cmd`, every attribute read off the proxy module is public."""
    import ast
    tree = ast.parse(Path(frames.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "decode_cmd")
    private = [a.attr for a in ast.walk(fn)
               if isinstance(a, ast.Attribute)
               and isinstance(a.value, ast.Name) and a.value.id == "_proxy"
               and a.attr.startswith("_")]
    assert private == []


def test_a_zero_argument_stub_is_not_called_by_the_analysis_only_decode(
        tiny_clip, monkeypatch):
    def stub():
        raise AssertionError("the analysis-only decode must not resolve it")

    monkeypatch.setattr(proxy, "full_ffmpeg", stub)
    assert len(list(sample_frames(tiny_clip, 1.0, fps=2.0,
                                  long_edge=LONG_EDGE))) == 2


def test_naming_a_binary_calls_full_ffmpeg_with_the_keyword(
        tiny_clip, tmp_path, monkeypatch):
    seen = []

    def spy(**kw):
        seen.append(kw)
        return "ffmpeg"

    monkeypatch.setattr(proxy, "full_ffmpeg", spy)
    decode_cmd(tiny_clip, analysis_vf(2.0, LONG_EDGE), tmp_path / "c.mp4",
               proxy_ffmpeg="elsewhere")
    assert seen == [{"proxy_ffmpeg": "elsewhere"}]


def test_a_three_argument_decode_cmd_replacement_still_serves_sample_frames(
        tiny_clip, monkeypatch):
    """The same rule one level up: `sample_frames` passes the keyword to
    `decode_cmd` only when the caller set it."""
    real = frames.decode_cmd
    seen = []

    def old_shape(path, vf, proxy_out=None):
        seen.append(proxy_out)
        return real(path, vf, proxy_out)

    monkeypatch.setattr(frames, "decode_cmd", old_shape)
    assert len(list(sample_frames(tiny_clip, 1.0, fps=2.0,
                                  long_edge=LONG_EDGE))) == 2
    assert seen == [None]


# --------------------------------------------------------------------------
# No new call site can pass the keyword unconditionally
# --------------------------------------------------------------------------

def _full_ffmpeg_calls():
    """Every call of `full_ffmpeg` in the package, with the `if` statements
    that enclose it."""
    for src in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(src.read_text())
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else "")
                if name != "full_ffmpeg":
                    continue
                guards, up = [], node
                while up in parents:
                    up = parents[up]
                    if isinstance(up, ast.If):
                        guards.append(ast.unparse(up.test))
                yield src.name, node, guards


def test_full_ffmpeg_is_called_bare_unless_a_binary_is_named():
    """A call that passes `proxy_ffmpeg=` unconditionally would break a
    zero-argument replacement of `full_ffmpeg` for every caller. The
    keyword may appear only under an `if` that tests `proxy_ffmpeg`. A
    bare call is always fine."""
    calls = list(_full_ffmpeg_calls())
    assert {f for f, _, _ in calls} >= {"proxy.py"}, (
        "the scan found no call site in proxy.py")
    keyword_calls = [(f, n.lineno, guards) for f, n, guards in calls
                     if n.keywords or n.args]
    assert keyword_calls, "no call passes the keyword: the argument is dead"
    unguarded = [(f, line) for f, line, guards in keyword_calls
                 if not any("proxy_ffmpeg" in g for g in guards)]
    assert not unguarded, (
        f"full_ffmpeg called with an argument and no guard at {unguarded}")
