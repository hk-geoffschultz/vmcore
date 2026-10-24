"""Where the pipeline keeps its state between stages.

Everything lives in a single work directory so a run is resumable and
inspectable. Whether the folder is disposable is the consumer's call: a
consumer that keeps hand corrections in it (the birthday pipeline keeps
work/overrides.csv there) cannot delete it and re-run.

This is trimmed down from the birthday pipeline's Store: that one grew
path properties for a specific stage graph (clips.csv, frames.json,
the face-embedding cache under analysis/) which don't apply here yet.
The atomic-write discipline is the part worth keeping unconditionally;
add path properties back here as the new stage graph gets designed,
following the same pattern (Phase 1 of the birthday pipeline — one
record per clip, an index written only after a complete walk — is a
good model to copy for whatever gets cached here, once we know what
per-clip signal actually gets computed for event footage).
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Iterator


def _temp_sibling(path: Path) -> Path:
    """`<name>.<pid>.<thread>.tmp`, next to `path`.

    The name used to be `<name>.tmp`, derived from the target alone, which
    two writers of one file share. They then write into one inode: the first
    to `os.replace` publishes whatever bytes are in it so far, and the second
    finds no temp of its own and raises FileNotFoundError. Measured at
    757f484, 8 threads x 150 writes of one file, five runs: 654 to 692 of
    1200 writes raised, and 89 to 272 of 462 to 800 concurrent reads saw
    unparseable JSON. A fixed interleave of two writers left
    `{"status": "x"}ivered", ...` in place of the file. With the name
    qualified: 0 of 1200 writes and 0 of 572 to 907 reads in all five runs.
    The process and thread in the name make it one writer's own
    (the rule `segcache.SegmentCache.put` adopted for the same reason),
    and the `.tmp` suffix is kept so a `*.tmp` glob still finds it.
    """
    return path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def atomic_write_text(path: Path, text: str, backup: bool = False,
                      encoding: str = "utf-8") -> None:
    """Write `text` to `path` so a crash can never leave a torn file.

    Write a sibling temp file, flush + fsync, then os.replace, which is
    atomic on POSIX. With `backup=True` the previous version is kept as
    `<name>.bak` first, copied to a temp name and renamed into place so a
    reader of the backup never sees it half written. Every stage file
    another process reads should go through here, not a bare
    `Path.write_text`.

    The temp name is `<name>.<pid>.<thread>.tmp` (see `_temp_sibling`), so
    concurrent writers of one file each get their own. The last
    `os.replace` wins whole; the file is never a mixture. A crash between
    the write and the rename leaves that one temp file behind. The old fixed
    name was overwritten by the next write. The temp name adds a dot,
    the pid, a dot, the thread id and `.tmp`. With a 5-digit pid and a
    15-digit thread id that is 26 bytes of file name. The longest target that
    can be written on ext4 then falls from 251 to 229, or to 225 with
    `backup=True`. Each digit more in the pid or the thread id costs one more
    byte, and each digit fewer gives one back. With a 4-digit pid and a
    15-digit thread id the measured limits were 230 and 226.

    The temp is created with `open()`, so the file's mode is the process
    umask applied to 0o666, exactly as before. `tempfile.mkstemp` would
    create it 0o600, and `os.replace` would carry that onto `path`. An
    existing `path` keeps no mode of its own: the replacement is a new file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        bak = path.with_name(path.name + ".bak")
        bak_tmp = _temp_sibling(bak)
        try:
            shutil.copy2(path, bak_tmp)
            os.replace(bak_tmp, bak)
        finally:
            bak_tmp.unlink(missing_ok=True)
    tmp = _temp_sibling(path)
    try:
        with open(tmp, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, obj, indent: int = 2,
                      backup: bool = False, **dumps_kw) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=indent, **dumps_kw),
                      backup=backup)


class Store:
    def __init__(self, work: Path):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)

    @property
    def config_path(self) -> Path:
        return self.work / "config.json"

    def write_json(self, path: Path, obj) -> None:
        atomic_write_json(path, obj, indent=2)

    def read_json(self, path: Path, default=None):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return default

    # ---- stages ----------------------------------------------------------
    # A stage is a directory of one JSON record per item, plus a CSV index
    # written only once a walk finishes. The split is what makes a stage
    # resumable: a crash costs the one clip in flight, and the index never
    # exists in a half-written state that a later stage would read as
    # complete. The birthday pipeline arrived at this the hard way; the
    # pattern is the reusable part, so the stage NAMES stay in the app.

    def stage_dir(self, stage: str) -> Path:
        d = self.work / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record_path(self, stage: str, key: str) -> Path:
        return self.stage_dir(stage) / f"{key}.json"

    def read_record(self, stage: str, key: str, default=None):
        return self.read_json(self.record_path(stage, key), default)

    def write_record(self, stage: str, key: str, obj) -> Path:
        p = self.record_path(stage, key)
        atomic_write_json(p, obj, indent=1)
        return p

    def records(self, stage: str) -> Iterator[tuple[str, dict]]:
        """Every record in a stage, by key, in sorted order.

        Unreadable records are skipped rather than raised on: one truncated
        file from a hard kill shouldn't make the whole stage unreadable.
        """
        for p in sorted(self.stage_dir(stage).glob("*.json")):
            rec = self.read_json(p)
            if isinstance(rec, dict):
                yield p.stem, rec

    def index_path(self, stage: str) -> Path:
        return self.work / f"{stage}.csv"

    def write_index(self, stage: str, rows: list[dict],
                    fieldnames: list[str] | None = None) -> Path:
        """Write a stage's CSV index atomically.

        Call this only after a complete walk — its existence is the signal
        that the stage finished. Rows are written through a StringIO first
        so the file lands via one os.replace rather than growing on disk.
        """
        path = self.index_path(stage)
        names = fieldnames or (list(rows[0].keys()) if rows else [])
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(path, buf.getvalue())
        return path

    def read_index(self, stage: str) -> list[dict]:
        path = self.index_path(stage)
        if not path.exists():
            return []
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
