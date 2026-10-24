"""Review proxies: 720p H.264, colour-normalised, one per clip.

Two ways to get them:

- Fused into ingest: `sample_frames(..., proxy_out=...)` adds a second
  output branch to the decode ffmpeg the analysis pass already runs, so
  the proxy costs one encode, not a second decode of the whole library.
- Standalone backfill for an already-ingested library: `build_proxy` per
  clip, decode + encode from scratch.

Colour: phone footage regularly mixes HLG HDR and full-range video in the
same shoot. A proxy that ignores either looks wrong with exit code 0, so
every proxy is normalised to limited-range bt709 SDR here (verified
against every HLG clip in the birthday-pipeline library this module was
forked from - 7/431, all iPhone). The HLG chain needs zscale, which the
slim system ffmpeg lacks, so the proxy path runs the binary `full_ffmpeg()`
picks. That is the caller's `proxy_ffmpeg=` argument, else a build under
tools/ above the vmcore package, else `ffmpeg` on PATH. A pip-installed
vmcore sits in site-packages and has no tools/ beside it. An installed
consumer therefore names its build with the argument. No environment
variable is read.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows has no flock; the sweep and the lock are off
    fcntl = None

PROXY_LONG_EDGE = 1280
PROXY_CRF = 23
PROXY_PRESET = "veryfast"

# The directory above this package. For a source tree or an editable install
# that is the checkout; for a pip install it is site-packages, which holds
# no tools/. Read at call time, so a test or a stopgap can reassign it.
_ROOT = Path(__file__).resolve().parent.parent

# HLG -> SDR: linearise, hable tonemap, back to bt709 limited. Validated
# against all 7 HLG clips in the library (exit 0, bt709 output).
TONEMAP = ("zscale=transfer=linear:npl=100,tonemap=hable:desat=0,"
           "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv")

HDR_TRANSFERS = ("arib-std-b67", "smpte2084")


def full_ffmpeg(*, proxy_ffmpeg: str | Path | None = None) -> str:
    """The ffmpeg that writes proxies and does colour work.

    First match wins:

    1. `proxy_ffmpeg`, the argument `build_proxy`, `decode_cmd` and
       `sample_frames` pass down.
    2. `tools/ffmpeg/ffmpeg` in the directory above this package, if it
       exists. That is the checkout for an editable install and
       site-packages for a pip install. vmcore ships no tools/, so this
       finds a consumer's build only if the package is vendored beside it.
    3. `ffmpeg` on PATH, which may be a slim build without zscale.

    No environment variable is read. A consumer that wants one reads its
    own and passes the value as `proxy_ffmpeg`.

    A binary named in step 1 is checked. It must exist and be executable,
    and `ffprobe` must be on PATH. A bad name raises. It never falls
    through to a lower step, because a silent fallback to a build without
    zscale is the failure this exists to remove. A bare command name is
    looked up on PATH. Steps 2 and 3 keep the checks they always had:
    `exists()`, and none.

    The ffprobe check is here for one reason. Naming an absolute ffmpeg is
    the one way to run it with no ffprobe on PATH. `color_transfer` turns a
    missing ffprobe into '', which selects the SDR chain. An HLG clip then
    gets a washed-out proxy and exit code 0.

    Only the proxy and colour path asks this. The analysis-only decode,
    `probe.audio_peak` and `thumbs.extract` run the PATH `ffmpeg` on
    purpose, because their decode behaviour is load-bearing. Nothing
    here changes which binary they use.
    """
    if proxy_ffmpeg is not None:
        return _named_binary(proxy_ffmpeg, "proxy_ffmpeg")
    cand = _ROOT / "tools" / "ffmpeg" / "ffmpeg"
    return str(cand) if cand.exists() else "ffmpeg"


def _named_binary(value, source: str) -> str:
    """`value` as the argv[0] it will be, or an error that names `source`."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise TypeError(f"{source} must be a str or Path, "
                        f"not {type(value).__name__}")
    if not value:
        raise ValueError(f"{source} is an empty string; "
                         f"pass None to use the default search")
    if os.sep in value or (os.altsep and os.altsep in value):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(
                f"{source}={value!r} does not exist. vmcore will not fall "
                f"back to another ffmpeg when one is named.")
        if path.is_dir():
            raise IsADirectoryError(
                f"{source}={value!r} is a directory. Name the ffmpeg "
                f"binary inside it.")
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"{source}={value!r} is not executable.")
    elif shutil.which(value) is None:
        raise FileNotFoundError(
            f"{source}={value!r} is not a command on PATH. vmcore will not "
            f"fall back to another ffmpeg when one is named.")
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError(
            f"{source}={value!r} is usable, but ffprobe is not on PATH. "
            f"Without it an HLG clip would be rendered as SDR.")
    return value


def _full_ffmpeg_for(proxy_ffmpeg: str | Path | None) -> str:
    """`full_ffmpeg()` as it has always been called, unless a binary is named.

    The keyword is passed only when the caller set it. Some consumers
    replace `vmcore.proxy.full_ffmpeg` with a function that takes no
    arguments, to pin or redirect the binary. That keeps working on every
    call that names nothing. Only `build_proxy` resolves the binary here.
    `frames.decode_cmd` makes the same two calls inline. It must work when a
    consumer has replaced the whole `vmcore.proxy` module with an object
    that has only the public names.
    """
    if proxy_ffmpeg is None:
        return full_ffmpeg()
    else:
        return full_ffmpeg(proxy_ffmpeg=proxy_ffmpeg)


def color_transfer(path: Path) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer", "-of", "json",
             str(path)],
            capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout)["streams"][0].get("color_transfer", "") or ""
    except Exception:
        return ""


def proxy_vf(transfer: str, long_edge: int = PROXY_LONG_EDGE) -> str:
    """Filter chain for the proxy branch of a given source clip."""
    scale = (f"scale='if(gt(iw,ih),{long_edge},-2)':"
             f"'if(gt(iw,ih),-2,{long_edge})':"
             "in_range=auto:out_range=tv")
    head = TONEMAP + "," if transfer in HDR_TRANSFERS else ""
    return f"{head}{scale},format=yuv420p"


def proxy_encode_args(out: Path) -> list[str]:
    return ["-c:v", "libx264", "-preset", PROXY_PRESET, "-crf", str(PROXY_CRF),
            "-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-y", str(out)]


def _suffix(name: str) -> str:
    """The text from the last dot of `name`, dot included, or '' without one.

    ffmpeg picks the muxer from the text after the last dot of the file name.
    `Path.suffix` reads a name that is only an extension, such as `.mp4`, as
    having none. A partial built from it then has no extension and ffmpeg
    finds no muxer, where 757f484 wrote the file.
    """
    dot = name.rfind(".")
    return name[dot:] if dot >= 0 else ""


def _token(name: str) -> str:
    """12 hex digits of the SHA-1 of the file name, the same for any stem."""
    return hashlib.sha1(os.fsencode(name),
                        usedforsecurity=False).hexdigest()[:12]


def _limits(directory: Path) -> tuple[int, int]:
    """NAME_MAX and PATH_MAX for the filesystem of `directory`, in bytes.

    255 and 4096 where the system will not say. PATH_MAX counts the
    terminating NUL, so a path of PATH_MAX bytes is already too long. That
    was measured on Linux with ffmpeg 6.1.1: 4095 bytes opened and 4096 gave
    ENAMETOOLONG. macOS is documented to use 1024. That was not measured.
    """
    def ask(name: str, default: int) -> int:
        try:
            value = os.pathconf(directory, name)
        except (OSError, ValueError, AttributeError):
            return default
        return value if value > 0 else default

    return ask("PC_NAME_MAX", 255), ask("PC_PATH_MAX", 4096)


def _fits(candidate: Path, parent_length: int, name_max: int,
          path_max: int) -> bool:
    """True when `candidate` is a name the kernel will open.

    Two path lengths must stay under PATH_MAX. One is the path as it will be
    handed to ffmpeg and `os.replace`. The other is the resolved parent plus
    the name, which is what a caller that made the path absolute would
    hand them. The second can only be longer when the path is relative or
    reaches its directory through a symlink. Taking both costs nothing and
    cannot pick a name that fails.
    """
    name = os.fsencode(candidate.name)
    return (len(name) <= name_max
            and len(os.fsencode(str(candidate))) < path_max
            and parent_length + 1 + len(name) < path_max)


def _partial_path(out: Path) -> Path | None:
    """Where `build_proxy` encodes before it renames onto `out`.

    The partial sits in the same directory, so the rename cannot cross a
    filesystem and stays atomic. It ends in `.part` plus the text after the
    last dot of `out`. ffmpeg picks the muxer from that text. A consumer
    that lists the folder can tell a partial from a proxy by `.part.mp4`
    when `out` ends in `.mp4`. It starts with a dot, which
    `probe.find_clips` already treats as not-a-clip. It carries
    the process and the thread, so two builds of one `out` each write their
    own file. That is the lesson `segcache.put` records, and the watcher and
    an editor can meet there.

    The name holds 12 hex digits of the SHA-1 of `out.name`, never the name
    itself, so its length is the same for every stem. A name built from the
    stem was the stem plus 32 to 34 bytes. A stem of 224 bytes then raised
    ENAMETOOLONG where 757f484 wrote the proxy. That was measured on ext4
    with NAME_MAX 255.

    The same name is also longer than a short `out` by about 45 bytes, and
    the whole path must stay under PATH_MAX. 757f484 wrote an `out` of any
    length up to the limit. So there are two more outcomes.

    - The full name does not fit but a shorter one does:
      `.<pid>.<thread>.part<ext>`. It still ends in `.part` plus the
      extension. It has no hash, so it does not say which `out` it is for.
      It does carry the process and the thread. So two builds that run at
      once never share it, whether they build one `out` or two.
    - Neither fits: None. `build_proxy` then writes to `out` itself, exactly
      as 757f484 did. If `out` is itself too long, the failure is 757f484's.
    """
    token = _token(out.name)
    ext = _suffix(out.name)
    who = f"{os.getpid()}.{threading.get_ident()}"
    names = (f".{token}.{who}.part{ext}", f".{who}.part{ext}")
    name_max, path_max = _limits(out.parent)
    try:
        parent_length = len(os.fsencode(os.path.realpath(out.parent)))
    except Exception:
        parent_length = len(os.fsencode(str(out.parent)))
    for name in names:
        candidate = out.with_name(name)
        if _fits(candidate, parent_length, name_max, path_max):
            return candidate
    return None


def _discard(path: Path) -> None:
    """Remove `path` if it is there. Never raises.

    The partial is removed on every way out of `build_proxy`, including the
    ones that are already raising or returning something. An unlink that
    fails then must not replace that outcome with its own error.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _hold(part: Path) -> int | None:
    """Create `part` and lock it, for as long as the returned fd stays open.

    The lock is how a later build tells a live builder from a dead one. The
    kernel drops it when the process ends by any means, including SIGKILL, an
    OOM kill and a container stop. It does not depend on pids. So it holds
    across containers that share the directory, and across a restart that
    gets the same pid again. ffmpeg opens the file by name and keeps the
    inode, so the lock stays on the file it writes.

    Returns None where there is no lock to take: Windows, or a filesystem that
    refuses `flock`. The build then runs unprotected. A sweep on that
    filesystem cannot take the lock either, so it removes nothing.
    """
    if fcntl is None:
        return None
    for _ in range(5):
        fd = -1
        try:
            fd = os.open(part, os.O_RDWR | os.O_CREAT | getattr(
                os, "O_NOFOLLOW", 0), 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # A sweep that held the lock a moment ago may have removed the
            # name it found. Keep the fd only if the name is still this file.
            here, there = os.fstat(fd), os.stat(part)
            if (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino):
                return fd
        except OSError:
            pass
        if fd >= 0:
            os.close(fd)
    return None


def _sweep_dead_partials(out: Path) -> None:
    """Remove earlier partials of `out` whose builder is gone. Never raises.

    A hard kill of the caller leaves its hidden partial behind. SIGKILL, an
    OOM kill and a container stop do. The name carries a pid and a thread id
    that no later build reuses, so nothing else would ever remove it. This
    runs before a build and removes the files that match all of these:

    - the same directory as `out`, with no recursion;
    - the name `.<token>.<pid>.<thread>.part<ext>` for THIS `out`, so the
      same hash token and the same extension, and not any other file;
    - a regular file, not a symlink;
    - not locked. `_hold` locks a partial for as long as its builder lives,
      so a lock that can be taken means the builder is gone.

    The pid in the name is not read. Liveness is a property of the file.
    A live builder in another container is never swept. A dead one is swept
    even when the new process has the same pid. The shorter fallback name has
    no token, so it cannot be tied to an `out` and is never touched. A hard
    kill in that band leaves its partial where it is.

    This needs a filesystem that implements `flock` for both sides. Where it
    does not, the sweep's lock attempt raises, and the file is kept.

    The sweep lists the directory once per build. Any failure is swallowed,
    because it must never decide the result of the build.
    """
    if fcntl is None:
        return
    try:
        prefix = f".{_token(out.name)}."
        tail = re.compile(r"\d+\.\d+" + re.escape(f".part{_suffix(out.name)}"))
        with os.scandir(out.parent) as entries:
            for entry in entries:
                if (not entry.name.startswith(prefix)
                        or tail.fullmatch(entry.name, len(prefix)) is None):
                    continue
                try:
                    _sweep_one(entry)
                except OSError:
                    pass
    except Exception:
        pass


def _sweep_one(entry: os.DirEntry) -> None:
    if not entry.is_file(follow_symlinks=False):
        return
    fd = os.open(entry.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0))
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return                      # its builder is alive
        here, there = os.fstat(fd), os.lstat(entry.path)
        if (here.st_dev, here.st_ino) == (there.st_dev, there.st_ino):
            os.unlink(entry.path)       # while the lock is still held
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build_proxy(path: Path, out: Path, timeout: int = 1800, *,
                proxy_ffmpeg: str | Path | None = None) -> bool:
    """Standalone proxy render for one clip. Returns True on success.

    `proxy_ffmpeg` names the binary that renders it. `full_ffmpeg` gives the
    order it is looked for in when not given. A named binary that is
    missing raises before anything is written.

    The encode goes to a hidden partial beside `out`. It is renamed onto
    `out` only when ffmpeg exited 0 and wrote something, so `out` is never
    a half-written file. What a caller can rely on:

    - True: `out` is a complete new proxy.
    - False: ffmpeg failed or wrote nothing. There is no file at `out`.
      A previous proxy is removed, as a failed run has always done.
    - An exception, including `subprocess.TimeoutExpired` after `timeout`
      seconds and `KeyboardInterrupt`: `out` is exactly as it was before the
      call, and the partial is gone. The exception is the one 757f484
      raised.

    The state after a timeout is the one change to a call that used to fail
    in a different way. The recipe is ffmpeg 6.1.1, a 40 s 1080p source and
    `timeout=2`. On 757f484 the call raised after about two seconds and left
    a file at `out`. Its size was whatever the encode had reached, so it
    differs from run to run and with the load. In every run measured,
    ffprobe rejected it with "moov atom not found". A caller that skips
    clips whose proxy exists treated it as done for ever. Over a good
    previous proxy, 757f484 replaced it with such a file. Now no file is
    left when there was none. A previous proxy survives byte for byte.

    A caller that is killed outright leaves its partial behind and no `out`.
    SIGKILL, an OOM kill and a container stop do this. At 757f484 the encode
    went straight to `out`, and an ffmpeg orphaned by a kill of Python alone
    could finish into it. Now `out` is absent until the next build of it, and
    a consumer that skips clips whose `out` exists rebuilds that clip. Before
    it encodes, `build_proxy` removes earlier partials of the same `out`
    that no live builder holds a lock on. `_sweep_dead_partials` gives the
    exact rule.
    Measured with ffmpeg 6.1.1 on a 40 s 1080p source, killed after 3 s.
    SIGKILL to the whole process group left an unplayable file, at `out` on
    757f484 and as a partial now. SIGKILL to Python alone let the orphan
    finish a playable file, at `out` on 757f484 and as a partial now. A
    later build of the same `out` from another process removed the partial
    in both cases.

    Two cases at the edge of PATH_MAX are outside those guarantees. The
    hidden name is about 45 bytes longer than a short `out`, and the whole
    path must stay under the limit. Within about 45 bytes of it, a shorter
    name `.<pid>.<thread>.part<ext>` is used and the build stays atomic.
    Within about 30 bytes, no hidden name fits. Then ffmpeg writes to `out`
    itself, exactly as at 757f484, and a timeout leaves a partial file there.
    The limit is 4096 on Linux, and 1024 on macOS by its documentation.
    Measured on Linux with ffmpeg 6.1.1, for an `out` named `p.mp4`: every
    absolute length from 4040 to 4095 bytes built, as at 757f484. A length of
    4096 raised ENAMETOOLONG on both trees. A hard kill in the shorter-name
    band leaves its partial, because that name cannot be tied to an `out`.

    An `out` that is a directory raises `IsADirectoryError` before any work.
    757f484 ran the whole encode and raised it from the unlink after.

    The bytes of a successful proxy are identical to a direct write, and a
    test pins it. Only the name ffmpeg writes to differs while it runs.
    A caller that wants a timeout to count as one more failed clip catches
    `subprocess.TimeoutExpired` itself.

    The finished partial is flushed to disk with `fsync` before the rename.
    That is what makes the rename durable. After a power loss `out` is the
    old file or a whole new one, and never a name that points at unwritten
    data. 757f484 did not flush. The cost is the time to write back whatever
    the kernel still holds. It grows with the size of the proxy and depends
    on the device. On the one VM disk measured it was linear in size. For a
    proxy of a few megabytes it was lost in the noise of the encode itself.
    The flush also adds a failure path after a finished encode. If `fsync`
    raises `OSError`, `build_proxy` raises it, `out` is exactly as it was and
    the partial is gone. 757f484 had no such path.

    A test double for `subprocess.run` must write the LAST element of the
    argv, which is the partial. 757f484 looked at `out`, and now the check
    looks at the partial. A double that exits 0 and writes nothing gets False
    and loses `out`. So does a double that writes straight to `out`. 757f484
    returned True for both and kept `out`. Make the double create the file
    that the last argument names.
    """
    # The order is 757f484's: make the directory, probe, then resolve the
    # binary. A resolver that raises then leaves the directory behind, as it
    # did. A NAMED binary is the exception. It is checked first, so a bad
    # name writes nothing and probes nothing.
    named = proxy_ffmpeg is not None
    if named:
        binary = _full_ffmpeg_for(proxy_ffmpeg)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = proxy_vf(color_transfer(path))
    if not named:
        binary = _full_ffmpeg_for(proxy_ffmpeg)
    if out.is_dir():
        # 757f484 ran the whole encode and then failed on the unlink. Say it
        # now, with the same exception class on Linux.
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR),
                                str(out))
    part = _partial_path(out)
    target = out if part is None else part
    _sweep_dead_partials(out)
    lock = None if part is None else _hold(part)
    try:
        r = subprocess.run(
            [binary, "-hide_banner", "-nostdin", "-loglevel", "error",
             "-i", str(path), "-map", "0:v:0", "-vf", vf]
            + proxy_encode_args(target),
            capture_output=True, text=True, timeout=timeout)
        if (r.returncode != 0 or not target.exists()
                or target.stat().st_size == 0):
            out.unlink(missing_ok=True)
            return False
        if part is not None:
            _fsync_file(part)
            os.replace(part, out)
        return True
    finally:
        if part is not None:
            _discard(part)
        if lock is not None:
            os.close(lock)
