"""Content-addressed cache for rendered video segments.

Every recut re-rendered every segment even when only a few clips changed -
~150 x264 encodes for a five-clip edit. A segment's pixels are a function
of its inputs, so caching by a hash of those inputs turns an unchanged
segment into a file copy instead of a decode+encode.

TWO WAYS TO KEY ONE, AND THE HONEST COMPARISON (2026-09-08). This module
offers `segment_key`, which ENUMERATES the inputs: source, in-point, frame
rate, frame count, speed, HDR-ness, encode profile, crop centre. An
enumeration is readable and it is what a caller reaches for first. It is
also structurally behind the command it stands for, and this was measured
rather than argued: the list shipped without `fps` at all, so one cut
rendered at 24 and at 30fps hashed identically, and nothing about the
function's shape would have caught that. `fps` is now a required argument
(see `segment_key`) - but the class of problem is not fixed by fixing one
instance of it.

What an enumeration cannot reach, by construction:

- Module constants spliced into the filter graph. A tonemap chain, a
  `setparams` VUI workaround, colour tags, `-pix_fmt`: retune any of them
  and every affected segment renders different pixels under an unchanged
  key. `hdr` says WHICH branch was taken, never what that branch contains.
- Anything with no source to stat. A gap slate's pixels are entirely
  described by a colour, a size, a rate and a duration; `segment_key`
  stats the source first, so it cannot key one at all.
- Whatever the caller adds next. The enumeration only drifts in one
  direction.

The alternative is to hash the ffmpeg argv itself, plus the source's
identity, the encoder build and a per-renderer tag. That cannot drift,
because it IS what produced the pixels. It is what the `bbp` events
pipeline does (`bbp/timeline.py`, `RenderCache`), after this module's
enumeration was tried and found short; only the store half below is used
there. Prefer it for a new caller. `segment_key` stays supported, and
correct for the inputs it names, for a caller whose command is fixed.

What must NOT go through either key: anything that is a property of where
a cut sits in a particular sequence rather than of the source window - a
burned-in caption carrying a cut index or a timeline position, most of
all. Two sequences placing the same clip at different positions need
different pixels there, and a cache serving the earlier one hands back a
correct-length segment with the wrong words on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

from .probe import source_stamp


def profile_hash(profile: dict) -> str:
    """Hash of the ENCODE-AFFECTING fields of a profile, not its name.

    Mirrors `pipeline.config_stamp`'s pattern: keying on the fields that
    actually change output means a ladder retune (a crf bump) invalidates
    stale cache entries instead of silently reusing them under a name that
    now means something else.
    """
    blob = json.dumps({"w": profile.get("w"), "h": profile.get("h"),
                       "mode": profile.get("mode"),
                       "video": list(profile.get("video") or [])},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _src_stamp(src: Path) -> tuple[int, int]:
    """(size, mtime_ns) - the "don't hash 19GB" identity. A source
    overwritten with different bytes (a re-export, a re-transcode) gets a
    new stamp and a cache miss, rather than silently serving the old pixels
    under the old path. Delegates to probe.source_stamp so ingest and the
    render cache agree on what "the same file" means."""
    return source_stamp(src)


def segment_key(src: Path, in_point: float, frames: int, speed, hdr: bool,
                profile: dict, *, fps: float, cx: float | None = None,
                cy: float | None = None) -> str:
    """The cache key for one rendered segment, from its named inputs.

    Read the module docstring first: this enumerates, and an enumeration
    is behind the command it stands for. Correct for what it names.

    `fps` is REQUIRED and keyword-only, which is a deliberate break. It
    was absent, and its absence was a silent wrong-pixels hit: the rate
    reaches a segment three ways - the `fps=` resample filter deciding
    which source instants are sampled, the output `-r`, and a slate's own
    `rate=`/duration - so the same source window at 24 and at 30fps
    hashed identically and the cache served whichever ran first. A
    caller written against the old signature now gets a TypeError naming
    the argument, which is the only acceptable way to find out.

    `in_point` is formatted exactly as it is sent to ffmpeg (`f"{in:.6f}"`)
    rather than compared as a raw float: frame-level precision (~0.0417s
    at 24fps) is far coarser than float round-off between two carve
    computations that should be considered identical. `fps` is formatted
    the same way, so a rate that arrives as 23.976023976023978 from one
    reader and 24000/1001 from another does not split the key.

    `cx`/`cy` are the crop centre (frame fractions, rounded to the `.4f`
    baked into the crop filter expression). Passing them is what makes a
    hand-edited crop override correctly bust the cache; a fit-mode
    segment passes neither. They are a POINT, so both or neither: passing
    one alone used to raise a bare `TypeError` from `float(None)` deep
    inside the payload build, which named nothing.
    """
    if (cx is None) != (cy is None):
        raise ValueError(
            "segment_key takes cx and cy together or not at all - a crop "
            f"centre is a point (got cx={cx!r}, cy={cy!r})")
    size, mtime_ns = _src_stamp(src)
    payload = {
        "src": str(Path(src).resolve()), "size": size, "mtime_ns": mtime_ns,
        "in": f"{in_point:.6f}", "fps": f"{float(fps):.6f}",
        "frames": int(frames),
        "speed": float(speed) if speed else 0.0, "hdr": bool(hdr),
        "profile": profile_hash(profile),
    }
    if cx is not None:
        payload["cx"] = round(float(cx), 4)
        payload["cy"] = round(float(cy), 4)
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class SegmentCache:
    """One directory of content-addressed segment files.

    No eviction: a distinct input tuple always gets a distinct file, so
    nothing already in the cache can ever be stale. The directory can grow
    unboundedly across many recuts of a long project; pruning by age is a
    follow-up, not a correctness requirement (open risk, see CLAUDE.md).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.mp4"

    def get(self, key: str) -> Path | None:
        p = self.path_for(key)
        return p if p.exists() and p.stat().st_size > 0 else None

    def put(self, key: str, rendered_path: Path) -> Path:
        """Adopt `rendered_path` into the cache, atomically, and return the
        cache's own copy.

        The caller's temp file is left in place - some callers still need
        it at its original location for the current render (e.g. to stage
        into a concat list) and copy rather than move for that reason.

        The temp file this writes carries the writing PROCESS AND THREAD
        in its name. It used to be `{key}.mp4.tmp`, derived from the key
        alone, which held against a concurrent reader and not against a
        concurrent writer - and one key reached by two writers at once is
        not exotic, it is a montage using the same source window twice, or
        two renders of one job. Measured at 12 threads on one key: the
        shared name raised `FileNotFoundError` in 9 to 11 of 12 across
        runs (each thread kept writing into an inode another had already
        renamed, then `os.replace` found no temp of its own), the
        qualified name in 0 of 12, with the entry intact either way and
        nothing stray left behind."""
        dest = self.path_for(key)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}."
                             f"{threading.get_ident()}.tmp")
        try:
            shutil.copyfile(rendered_path, tmp)
            os.replace(tmp, dest)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise
        return dest
