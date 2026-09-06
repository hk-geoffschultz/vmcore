"""Shared FCP7/xmeml plumbing: the boilerplate common to every writer, one
rate/ntsc parser for every reader, and the interval-carving primitive that
"remove this span from these spans" logic collapses onto everywhere.

What this module deliberately does NOT do: build a clipitem. `emit.py`,
`alternates.py` and `rough_cut.py` each have real, different requirements
for how a clipitem is built (rough_cut.py's keep path copies a captured
clipitem element-for-element to preserve hand-authored filters; its music
clipitem intentionally carries `<masterclipid>` to match a real Premiere
export). Only the parts that never differ move here: the rate/
samplecharacteristics/sequence-skeleton fragments (identical bodies, even
identical comments, in all three writers), the pretty-print-plus-header
write tail (previously three non-atomic `out.write_text()` calls), the
`<rate>/<ntsc>` -> fps parse every reader needs, and the carve primitive.

See CLAUDE.md rule #9 (element order, `<samplecharacteristics>` placement)
and rule #13 (the v1 XML shape) for why this boilerplate is exact rather
than approximate.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from .store import atomic_write_text


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_rate(parent, fps) -> None:
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = str(fps)
    ET.SubElement(r, "ntsc").text = "FALSE"


def write_samplechars(parent, fps, w, h):
    sc = ET.SubElement(parent, "samplecharacteristics")
    write_rate(sc, fps)
    ET.SubElement(sc, "width").text = str(w)
    ET.SubElement(sc, "height").text = str(h)
    # Without an explicit PAR the importer guesses - and guesses D1/DV NTSC
    ET.SubElement(sc, "anamorphic").text = "FALSE"
    ET.SubElement(sc, "pixelaspectratio").text = "square"
    ET.SubElement(sc, "fielddominance").text = "none"
    return sc


def sequence_skeleton(parent, seq_id, name, fps, width, height):
    """Sequence skeleton in the child order Premiere's importer expects
    (rule #9): name, duration, rate, in, out, media. Returns
    (sequence, duration_el, media_el, video_el) so a caller can append
    tracks and set the final duration.
    """
    s = ET.SubElement(parent, "sequence", id="sequence-%s" % seq_id)
    ET.SubElement(s, "name").text = name
    dur = ET.SubElement(s, "duration")
    dur.text = "0"
    write_rate(s, fps)
    ET.SubElement(s, "in").text = "-1"
    ET.SubElement(s, "out").text = "-1"
    media = ET.SubElement(s, "media")
    video = ET.SubElement(media, "video")
    fmt = ET.SubElement(video, "format")
    write_samplechars(fmt, fps, width, height)
    return s, dur, media, video


def write_xmeml(xmeml_root, out_path: Path) -> None:
    """Pretty-print, prepend the exact two-line header the importer needs,
    and write atomically. The three writers each did this with a bare
    `out.write_text()`; a crash mid-write there left a torn XML behind for
    Premiere to reject with its usual empty-dialog "File Import Failure".
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = minidom.parseString(ET.tostring(xmeml_root)).toprettyxml(indent="  ")
    text = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'
            + pretty.split("\n", 1)[1])
    atomic_write_text(out_path, text)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def parse_rate(sequence_el, default: float = 24.0) -> float:
    """The one `<rate><timebase>/<ntsc></rate>` -> fps computation.

    NTSC drop-frame multiplies the nominal timebase by 1000/1001 (24 fps
    timebase, ~23.976 real fps). `rough_cut.py`'s `--preserve` reader used
    to skip the `<ntsc>` check entirely (unlike every other reader), which
    would silently mis-time every preserved clip against a genuinely
    drop-frame `--preserve` source.
    """
    tb = sequence_el.find(".//rate/timebase")
    ntsc = sequence_el.find(".//rate/ntsc")
    fps = float(tb.text) if tb is not None else float(default)
    if ntsc is not None and (ntsc.text or "").upper() == "TRUE":
        fps *= 1000.0 / 1001.0
    return fps


# ---------------------------------------------------------------------------
# Interval carving
# ---------------------------------------------------------------------------

def carve(pieces: list[tuple[float, float]],
         covers: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Every `cover` removed from every `piece`.

    A piece entirely inside a cover vanishes; a piece straddling a cover's
    edge is split into what remains on either side. `covers` are applied in
    the given order (order doesn't change the result, since removal is
    associative), and each input piece's remaining fragments stay adjacent
    and left-to-right in the output.

    This one loop is, byte-for-byte, what used to be four independent
    copies: `rough_cut.py`'s `uncovered()` (one piece, many covers),
    `preview.read_recut`'s upper-track carve (many pieces, many covers, one
    call per lower-track clip), `rough_cut.py`'s locked-span trim in
    `section_cuts` (many candidate cuts carved around every locked span),
    and `pipeline.extract_signals`'s two carves of `sig["locked"]` around a
    single hole (a RESHUFFLE range, or one banned clip's footprint) —
    `carve(spans, [hole])` covers both identically, which is also why
    `pipeline.cmd_refine` no longer needs its own copy of the ban-inside-
    lock carve: it calls this with the same hole.
    """
    out: list[tuple[float, float]] = []
    for x, y in pieces:
        parts = [(x, y)]
        for a, b in covers:
            nxt = []
            for px, py in parts:
                if py <= a or px >= b:
                    nxt.append((px, py))
                else:
                    if px < a:
                        nxt.append((px, a))
                    if py > b:
                        nxt.append((b, py))
            parts = nxt
        out.extend(parts)
    return out


def overlaps_any(hole: tuple[float, float],
                 spans: list[tuple[float, float]]) -> bool:
    """Whether `hole` intersects any span — the swap-detection half of the
    ban-inside-lock logic, kept separate from `carve` since callers that
    only need the carved result (RESHUFFLE) don't need this check."""
    ha, hb = hole
    return any(a < hb and ha < b for a, b in spans)
