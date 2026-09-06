"""Content-addressed cache for rendered preview/delivery segments.

Every recut re-rendered every segment even when only a few clips changed -
~150 x264 encodes for a five-clip edit. A segment's pixels are a pure
function of a handful of inputs (source file, in-point, frame count, speed,
HDR-ness, and the encode profile), so caching by a hash of those inputs
turns an unchanged segment into a file copy instead of a decode+encode.

What is deliberately NOT cached: `preview.py`'s burned-in label (cut index,
timeline position, source name) is a property of where a cut sits in THIS
recut, not of the source window - two recuts placing the same clip at
different timeline positions must show different labels. `preview.py`
therefore renders a small, label-free geometry segment through this cache
and applies the label as a second, always-cheap, never-cached pass.

Smart-crop segments (deliver.py's social profiles) add the face-centred
crop point to the key: `cut_center()`'s result, or a hand-edited
`.crops.json` override, changes the rendered pixels even when nothing else
about the cut did, so the cache must not serve a stale crop after an
override edit.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
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
                profile: dict, *, cx: float | None = None,
                cy: float | None = None) -> str:
    """The cache key for one rendered segment.

    `in_point` is formatted exactly as it is sent to ffmpeg (`f"{in:.6f}"`
    in both `preview._segment_cmd` and `deliver.segment_cmd`) rather than
    compared as a raw float: frame-level precision (~0.0417s at 24fps) is
    far coarser than float round-off between two carve computations that
    should be considered identical.

    `cx`/`cy` are the smart-crop centre (frame fractions, already rounded
    to the precision baked into the crop filter expression, `.4f`). Passing
    them is what makes a `.crops.json` override correctly bust the cache -
    a fit-mode segment (master/mobile) passes neither.
    """
    size, mtime_ns = _src_stamp(src)
    payload = {
        "src": str(Path(src).resolve()), "size": size, "mtime_ns": mtime_ns,
        "in": f"{in_point:.6f}", "frames": int(frames),
        "speed": float(speed) if speed else 0.0, "hdr": bool(hdr),
        "profile": profile_hash(profile),
    }
    if cx is not None or cy is not None:
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
        cache's own copy. The caller's temp file is left in place - some
        callers still need it at its original location for the current
        render (e.g. to stage into a concat list) and copy rather than
        move for that reason; `put` itself always uses a tmp+replace on
        the cache side so a concurrent reader never sees a partial file."""
        dest = self.path_for(key)
        tmp = dest.with_suffix(".mp4.tmp")
        shutil.copyfile(rendered_path, tmp)
        os.replace(tmp, dest)
        return dest
