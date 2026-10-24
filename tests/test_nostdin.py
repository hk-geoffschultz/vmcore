"""Every ffmpeg vmcore runs carries `-nostdin`.

ffmpeg treats the caller's stdin as a keyboard unless told not to. Measured
on ffmpeg 6.1.1, running vmcore's own call sites:

  * A job in a background process group whose stdin is still the terminal
    is stopped by SIGTTOU. It happens as soon as ffmpeg asks the terminal
    to change its settings. The Python process around it stops too, and it
    sits there until someone notices. GNU nohup redirects a terminal stdin
    from an unreadable file, so `nohup tool &` avoids the stop there. That
    was measured with coreutils 9.4 on 757f484. The bare job stopped at
    audio_peak, sample_frames and build_proxy. The same job under nohup
    exited 0. The macOS case rests on rule 17 of the birthday pipeline,
    where `nohup ... &` suspended a job. It was not measured here.
  * A `q` on stdin ends the run early. At the first poll that leaves
    nothing: `audio_peak` returns -99.0, the value for "no audio", and
    `sample_frames` yields no frames. Later it leaves a valid file with
    exit code 0 that is a fraction of the source, which `build_proxy`
    reports as success.
  * Each poll takes a byte of the caller's stdin, and a line that starts
    with `c` is taken whole as a filter command. A loop that reads clip
    paths from stdin and calls vmcore per clip loses the start of the
    next path.

`-nostdin` is a global option that changes none of the decode, only
whether ffmpeg reads fd 0. ffprobe never reads stdin and rejects the flag
("Option not found"), and ffmpeg's print-and-exit options (`-encoders`,
`-filters`, `-version`) never stop and need none. No vmcore site feeds ffmpeg from stdin, but the flag is also
harmless to one that does: `-i -` and `-i pipe:0` produce identical frames
with and without it.

The rule has four layers because a comment never kept one:

  1. the argv each site builds is pinned (test_frames.py for decode_cmd),
  2. every ffmpeg process any site spawns is observed and checked,
  3. the source of vmcore/ is scanned for ffmpeg argv lists without the
     flag and for subprocess calls the scan cannot see into,
  4. the symptoms themselves are reproduced on a pty and a stdin pipe.

The scan covers vmcore/ and not tests/. Under pytest's default capture fd 0
is /dev/null for the whole run, so an ffmpeg started by a test cannot meet
the terminal (measured: tests/test_probe.py as a background job on a pty
ran to the end by default and was stopped only under `pytest -s`). The
suite's own ffmpeg calls carry the flag anyway, so it is also safe under
`-s`; that is not enforced, because a test about some other ffmpeg
behaviour should not have to carry a flag it is not about.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vmcore import frames, probe, proxy, testing, thumbs

ROOT = Path(__file__).resolve().parent.parent
FLAG = "-nostdin"
LIBRARY = sorted((ROOT / "vmcore").glob("*.py"))


# --------------------------------------------------------------------------
# 1 and 2: the argv every site builds, and every ffmpeg process it spawns
# --------------------------------------------------------------------------

def _record_spawns(monkeypatch) -> list[list[str]]:
    """Every argv handed to subprocess.Popen (so also run), as strings."""
    seen: list[list[str]] = []
    real = subprocess.Popen

    class Spy(real):
        def __init__(self, args, *a, **kw):
            seen.append([str(x) for x in args])
            super().__init__(args, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", Spy)
    return seen


def _ffmpegs(argvs: list[list[str]]) -> list[list[str]]:
    return [a for a in argvs if Path(a[0]).name.startswith("ffmpeg")]


def test_decode_cmd_carries_the_flag_on_both_branches(tiny_clip, tmp_path):
    vf = frames.analysis_vf(2.0, 64)
    for cmd in (frames.decode_cmd(tiny_clip, vf),
                frames.decode_cmd(tiny_clip, vf, tmp_path / "p.mp4")):
        assert cmd[1:3] == ["-hide_banner", FLAG]
        assert cmd.count(FLAG) == 1


def _exercise(site: str, clip: Path, tmp: Path) -> None:
    if site == "decode":
        list(frames.sample_frames(clip, 1.0, fps=2.0, long_edge=64))
    elif site == "decode_fragment":
        list(frames.sample_frames(clip, 1.0, fps=2.0, long_edge=64,
                                  full_rate_vf="scdet=t=10"))
    elif site == "decode_fused":
        list(frames.sample_frames(clip, 1.0, fps=2.0, long_edge=64,
                                  proxy_out=tmp / "fused.mp4"))
    elif site == "build_proxy":
        proxy.build_proxy(clip, tmp / "proxy.mp4")
    elif site == "audio_peak":
        probe.audio_peak(clip)
    elif site == "thumbs":
        thumbs.extract(clip, tmp / "thumbs", "k", 1.0)
    elif site == "make_mp4":
        testing.make_mp4(tmp / "m.mp4", duration=0.5)
    elif site == "make_cut_mp4":
        testing.make_cut_mp4(tmp / "c.mp4", before=0.5, after=0.5)
    elif site == "make_wav":
        testing.make_wav(tmp / "w.wav", duration=0.5)
    else:
        raise AssertionError(site)


SITES = ["decode", "decode_fragment", "decode_fused", "build_proxy",
         "audio_peak", "thumbs", "make_mp4", "make_cut_mp4", "make_wav"]


@pytest.mark.parametrize("site", SITES)
def test_every_ffmpeg_a_site_spawns_carries_the_flag(
        site, tiny_clip, tmp_path, monkeypatch):
    seen = _record_spawns(monkeypatch)
    _exercise(site, tiny_clip, tmp_path)
    ffmpegs = _ffmpegs(seen)
    assert ffmpegs, f"{site} ran no ffmpeg - the spy saw {seen}"
    for argv in ffmpegs:
        assert argv[1:3] == ["-hide_banner", FLAG], argv
        # a global option: it has to sit before the first input
        assert argv.index(FLAG) < argv.index("-i"), argv


def test_ffprobe_is_left_alone(tiny_clip, monkeypatch):
    """ffprobe never reads stdin, and it rejects the flag: on 6.1.1
    `ffprobe -nostdin ...` exits 1 with "Option not found". Adding the flag
    to every argv in sight would break probing."""
    seen = _record_spawns(monkeypatch)
    probe.display_dims(tiny_clip)
    proxy.color_transfer(tiny_clip)
    probes = [a for a in seen if Path(a[0]).name.startswith("ffprobe")]
    assert len(probes) == 2
    assert all(FLAG not in a for a in probes)


# --------------------------------------------------------------------------
# 3: the source scan
# --------------------------------------------------------------------------

# Print-and-exit options. Measured on a background process group on a tty:
# ffmpeg returns before it touches the terminal, so these need no flag.
PRINT_AND_EXIT = {"-encoders", "-filters", "-version"}

SUBPROCESS_FUNCS = {"run", "Popen", "call", "check_call", "check_output",
                    "getoutput", "getstatusoutput"}
OS_FUNCS = {"system", "popen"}
ASYNCIO_FUNCS = {"create_subprocess_exec", "create_subprocess_shell"}


def _consts(lst: ast.AST) -> set[str]:
    return {e.value for e in lst.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)}


def _is_option(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value.startswith("-"))


def _head(lst: ast.AST) -> str:
    return ast.unparse(lst.elts[0]).lower() if lst.elts else ""


# What can stand for a program at the head of an argv. A tuple, a list or a
# comprehension cannot, which keeps a table of test cases out of the scan.
_PROGRAM = (ast.Constant, ast.Name, ast.Attribute, ast.Call, ast.BoolOp,
            ast.IfExp, ast.Subscript)


def _is_full_argv(lst: ast.AST) -> bool:
    """A list that starts with a program, not with an option: a fragment
    such as `["-c:v", "libx264", ...]` is the tail of something else."""
    return (bool(lst.elts) and isinstance(lst.elts[0], _PROGRAM)
            and not _is_option(lst.elts[0]))


def _is_ffmpeg_argv(lst: ast.AST) -> bool:
    """By name where the program is spelled out (`"ffmpeg"`,
    `full_ffmpeg()`, a variable called `ffmpeg`), by shape where it is not
    (a list with `-hide_banner` or `-i`): a program held in a variable
    called `binary` is still caught."""
    if not _is_full_argv(lst) or "ffprobe" in _head(lst):
        return False
    return ("ffmpeg" in _head(lst)
            or bool({"-hide_banner", "-i"} & _consts(lst)))


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)}


def _chain_lists(node: ast.AST, parents: dict) -> list[ast.AST]:
    """The list displays of `[...] + x + [...]`, the whole sum, from any
    one of them: the flag may sit in any piece."""
    top = node
    while (isinstance(parents.get(top), ast.BinOp)
           and isinstance(parents[top].op, ast.Add)):
        top = parents[top]
    found: list[ast.AST] = []

    def leaves(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            leaves(n.left)
            leaves(n.right)
        elif isinstance(n, ast.List):
            found.append(n)

    leaves(top)
    return found


def _enclosing(node: ast.AST, parents: dict) -> str:
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return "<module>"


def _spawn_calls(tree: ast.AST) -> list[ast.Call]:
    from_subprocess = {
        a.asname or a.name for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "subprocess"
        for a in n.names if a.name in SUBPROCESS_FUNCS}
    calls = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            mod, attr = f.value.id, f.attr
            if ((mod == "subprocess" and attr in SUBPROCESS_FUNCS)
                    or (mod == "os" and attr in OS_FUNCS)
                    or (mod == "asyncio" and attr in ASYNCIO_FUNCS)):
                calls.append(n)
        elif isinstance(f, ast.Name) and f.id in from_subprocess:
            calls.append(n)
    return calls


def scan(source: str, filename: str = "<src>"):
    """`(missing, heads, opaque)` for one module's source.

    missing: `(line, function)` of each ffmpeg argv that lacks the flag
    heads:   function names of every ffmpeg argv the scan found
    opaque:  function names of spawn calls whose argv has no list display
             beginning with a program, so the scan cannot see what runs
    """
    tree = ast.parse(source, filename=filename)
    parents = _parents(tree)
    missing, heads, opaque = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and _is_ffmpeg_argv(node):
            where = _enclosing(node, parents)
            heads.append(where)
            pieces = _chain_lists(node, parents)
            union = set().union(*(_consts(p) for p in pieces))
            if FLAG not in union and not (union & PRINT_AND_EXIT):
                missing.append((node.lineno, where))
    for call in _spawn_calls(tree):
        arg = call.args[0] if call.args else next(
            (k.value for k in call.keywords if k.arg == "args"), None)
        chain = (_chain_lists(arg, {})
                 if isinstance(arg, (ast.List, ast.BinOp)) else [])
        if not any(_is_full_argv(p) for p in chain):
            opaque.append(_enclosing(call, parents))
    return missing, heads, opaque


# Spawn calls the scan cannot see into, and why each is covered anyway.
OPAQUE_OK = {
    ("probe", "_run"): "runs the ffprobe lists its callers pass",
    ("frames", "sample_frames"): "runs the argv decode_cmd builds",
    ("testing", "_encode"): "extends the lists make_mp4 and make_cut_mp4 build",
}

# What the scan must still find in each module: a count it falls below is a
# scan gone blind, not a module that improved. Extra sites are fine; they
# are held to the rule like any other.
KNOWN_HEADS = {"frames": 2, "probe": 1, "proxy": 1, "thumbs": 1, "testing": 3}


@pytest.mark.parametrize("path", LIBRARY, ids=lambda p: p.name)
def test_no_ffmpeg_argv_in_the_source_lacks_the_flag(path):
    missing, _, _ = scan(path.read_text(), str(path))
    assert not missing, (
        f"{path.name}: ffmpeg argv without {FLAG} at {missing}. ffmpeg reads "
        f"the caller's stdin for keystrokes unless told not to. A job in a "
        f"background process group whose stdin is the terminal is stopped "
        f"by SIGTTOU, and a stray q ends the run early. Put {FLAG} straight "
        f"after -hide_banner.")


@pytest.mark.parametrize("path", LIBRARY, ids=lambda p: p.name)
def test_every_subprocess_call_in_vmcore_is_one_the_scan_understands(path):
    _, _, opaque = scan(path.read_text(), str(path))
    unexplained = [fn for fn in opaque if (path.stem, fn) not in OPAQUE_OK]
    assert not unexplained, (
        f"{path.name}: {unexplained} spawn a process from an argv the scan "
        f"cannot read. Build the argv as a list in the call so it can be "
        f"checked, or add the function to OPAQUE_OK with the reason.")


def test_the_scan_still_finds_the_known_sites():
    found: dict[str, int] = {}
    for path in LIBRARY:
        _, heads, _ = scan(path.read_text(), str(path))
        if heads:
            found[path.stem] = len(heads)
    for module, n in KNOWN_HEADS.items():
        assert found.get(module, 0) >= n, (
            f"the scan found {found.get(module, 0)} ffmpeg argv lists in "
            f"{module}.py, expected at least {n}: a site moved or the scan "
            f"went blind. Found: {found}")


@pytest.mark.parametrize("src,ok", [
    ('subprocess.run(["ffmpeg", "-hide_banner", "-i", p])', False),
    ('subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-i", p])', True),
    ('subprocess.run(["ffmpeg", "-nostdin", "-i", p])', True),
    ('x = [full_ffmpeg(), "-hide_banner", "-i", p] + tail', False),
    ('x = [_p.full_ffmpeg(), "-hide_banner", "-nostdin", "-i", p] + tail', True),
    ('x = [binary, "-i", p]', False),
    ('x = [binary, "-nostdin", "-i", p]', True),
    ('x = [ffmpeg_bin or "ffmpeg", "-i", p, "-f", "null", "-"]', False),
    ('x = ["ffmpeg"] + ["-hide_banner", "-nostdin"] + ["-i", p]', True),
    ('x = ["ffmpeg"] + opts + ["-i", p]', False),
    ('x = ["ffmpeg", "-hide_banner", "-encoders"]', True),
    ('x = ["ffmpeg", "-hide_banner", "-codecs"]', False),
    # not ffmpeg argvs at all
    ('x = ["ffprobe", "-v", "error", "-i", p]', True),
    ('x = ["-c:v", "libx264", "-i", p]', True),
    ('x = ["ls", "-l"]', True),
])
def test_scan_classifies_argv_lists(src, ok):
    missing, _, _ = scan(src)
    assert (not missing) is ok, (src, missing)


def test_scan_sees_spawn_calls_it_cannot_read():
    src = textwrap.dedent('''
        import subprocess
        from subprocess import check_output as co
        def a(cmd):
            subprocess.run(cmd)
        def b(cmd):
            co(cmd + ["-x"])
        def c():
            subprocess.Popen(["ffmpeg", "-nostdin", "-i", "x"])
        def d(base):
            os.system("ffmpeg -i x")
    ''')
    _, heads, opaque = scan(src)
    assert heads == ["c"]
    assert sorted(opaque) == ["a", "b", "d"]


def test_the_flag_does_not_disturb_an_input_fed_through_stdin(tiny_clip):
    """The reason the flag is safe to put on every argv, including one that
    reads its input from a pipe later: ffmpeg turns its keyboard handling
    off by itself for `-i -`, so the flag changes nothing there. The bytes
    of a real stream contain plenty of 0x71 ('q'); the frames out are the
    same with and without it."""
    nut = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i",
         str(tiny_clip), "-c", "copy", "-f", "nut", "-"],
        check=True, capture_output=True, stdin=subprocess.DEVNULL).stdout
    assert nut.count(b"q") > 0

    def decode(with_flag):
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
               "-i", "-", "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt",
               "bgr24", "-"]
        if not with_flag:
            cmd.remove("-nostdin")
        got = subprocess.run(cmd, input=nut, capture_output=True)
        assert got.returncode == 0, got.stderr
        return got.stdout

    assert len(decode(False)) > 0
    assert decode(True) == decode(False)


# --------------------------------------------------------------------------
# 4: the symptoms
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ramp_clip(tmp_path_factory) -> Path:
    """4 s of picture and a tone that climbs to full level at the very end,
    so the loudest sample is the last one and anything that stops the read
    early lowers or loses the peak."""
    p = tmp_path_factory.mktemp("ramp") / "ramp.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=24:duration=4",
         "-f", "lavfi", "-i",
         r"aevalsrc=sin(2*PI*440*t)*min(0.9\,t/4):d=4:s=8000",
         "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(p)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL)
    return p


_PRELUDE = f"""
import json, os, sys
from hashlib import sha256
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
CLIP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
"""

_SITE_CODE = {
    "audio_peak": """
from vmcore.probe import audio_peak
result = audio_peak(CLIP)
""",
    "sample_frames": """
from vmcore.frames import sample_frames
got = list(sample_frames(CLIP, 4.0, fps=2.0, long_edge=64))
blob = b"".join(f.tobytes() for _, f in got)
result = [len(got), sha256(blob).hexdigest()[:16]]
""",
    "thumbs": """
from vmcore.thumbs import extract
got = extract(CLIP, OUT, "k", 4.0)
blob = b"".join(p.read_bytes() for p in got)
result = [len(got), sha256(blob).hexdigest()[:16]]
""",
    "build_proxy": """
from vmcore.proxy import build_proxy
ok = build_proxy(CLIP, OUT / "p.mp4")
result = [ok, (OUT / "p.mp4").stat().st_size > 0 if ok else None]
""",
}
_EPILOGUE = """
if os.environ.get("STDIN_OFFSET"):
    result = [result, os.lseek(0, 0, os.SEEK_CUR)]
print(json.dumps(result))
"""


def _site_script(site: str) -> str:
    return _PRELUDE + _SITE_CODE[site] + _EPILOGUE


def _skip_if_cannot_run(site: str) -> None:
    """build_proxy needs libx264 in the proxy ffmpeg; the argv checks above
    do not, because they only look at what was started."""
    if site == "build_proxy":
        encoders = subprocess.run(
            [proxy.full_ffmpeg(), "-hide_banner", "-encoders"],
            capture_output=True, text=True).stdout
        if " libx264 " not in encoders:
            pytest.skip("no libx264 in the proxy ffmpeg build")


def _run_site(site, clip, out, env_extra=None, **popen_kwargs):
    out.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-c", _site_script(site), str(clip), str(out)],
        capture_output=True, text=True, timeout=120, env=env, **popen_kwargs)


@pytest.mark.parametrize("site", ["audio_peak", "sample_frames", "thumbs",
                                  "build_proxy"])
def test_a_q_on_stdin_changes_nothing(site, ramp_clip, tmp_path):
    """The measured symptom: with a q waiting on stdin, audio_peak read
    -99.0, sample_frames and thumbs.extract produced nothing and
    build_proxy returned False."""
    _skip_if_cannot_run(site)
    quiet = _run_site(site, ramp_clip, tmp_path / "a", stdin=subprocess.DEVNULL)
    noisy = _run_site(site, ramp_clip, tmp_path / "b", input="q\n")
    assert quiet.returncode == noisy.returncode == 0, (quiet.stderr, noisy.stderr)
    assert json.loads(noisy.stdout) == json.loads(quiet.stdout)
    # and the baseline itself is not the degenerate answer
    assert json.loads(quiet.stdout) not in (
        -99.0, [0, "e3b0c44298fc1c14"], [False, None])


@pytest.mark.parametrize("site", ["audio_peak", "sample_frames", "thumbs",
                                  "build_proxy"])
def test_no_byte_of_the_callers_stdin_is_taken(site, ramp_clip, tmp_path):
    """fd 0 is a regular file, so its shared offset after the call says how
    much ffmpeg took. A line starting with c is the worst case: ffmpeg
    reads a whole line as a filter command."""
    _skip_if_cannot_run(site)
    payload = tmp_path / "stdin.txt"
    payload.write_bytes(b"clip2.mp4 is the next path\nand the one after\n")
    with open(payload, "rb") as fh:
        got = _run_site(site, ramp_clip, tmp_path / "o",
                        {"STDIN_OFFSET": "1"}, stdin=fh)
    assert got.returncode == 0, got.stderr
    _, offset = json.loads(got.stdout)
    assert offset == 0


def _pty_available() -> bool:
    try:
        import pty
        master, slave = pty.openpty()
        os.close(master)
        os.close(slave)
        return hasattr(os, "fork")
    except (ImportError, OSError):
        return False


@pytest.mark.skipif(not _pty_available(), reason="needs fork and a pty")
@pytest.mark.parametrize("site", ["audio_peak", "sample_frames", "thumbs",
                                  "build_proxy"])
def test_a_background_job_on_a_terminal_is_not_stopped(site, ramp_clip, tmp_path):
    """Rule 17 of the birthday pipeline, reproduced without nohup. The job
    is a background process group whose stdin is still the terminal. Without
    the flag ffmpeg is stopped by SIGTTOU within a second, and the job never
    finishes. A nohup that redirects a terminal stdin, as GNU nohup does,
    would hide the stop, so this test does not use one."""
    _skip_if_cannot_run(site)
    out = tmp_path / "o"
    out.mkdir()
    log = tmp_path / "job.log"
    got = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "ttyrun.py"), str(log), "30",
         sys.executable, "-c", _site_script(site), str(ramp_clip), str(out)],
        capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    verdict = json.loads(got.stdout)["verdict"]
    assert verdict == "EXITED(0)", (verdict, log.read_text())
