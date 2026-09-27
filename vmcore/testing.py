"""Synthetic XML + tiny-media helpers shared by the suite.

Trimmed from the birthday pipeline's tests/fixtures.py: that file also
built a full synthetic clips.csv/frames.json/selects.csv work directory
through the face-classifier's own writers (ClipSpec, make_fixture,
frames_for) - none of which exists in this project yet (see
docs/scope-of-work-draft.md). What's generic and worth keeping day one:
tiny real media via ffmpeg lavfi (so pathurls resolve to real files) and
the hand-written xmeml builder + v1-shape validators, since whatever
assembler eventually emits a cut list will need to produce the same
Premiere-importable shape the birthday pipeline spent real debugging time
getting right (CLAUDE.md rules 9 and 13, ported verbatim into
bbp/xmeml.py).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

FPS = 24


def make_mp4(path: Path, duration: float, w: int = 64, h: int = 36) -> None:
    src = f"smptebars=size={w}x{h}:rate={FPS}:duration={duration:.3f}"
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", src, "-pix_fmt", "yuv420p"]
    proc = subprocess.run(base + ["-c:v", "libx264", "-preset", "ultrafast",
                                  str(path)], capture_output=True)
    if proc.returncode != 0:
        subprocess.run(base + ["-c:v", "mpeg4", str(path)], check=True,
                       capture_output=True)


def make_cut_mp4(path: Path, before: float = 1.0, after: float = 1.0,
                 w: int = 64, h: int = 36) -> int:
    """Black for `before` seconds, then white for `after`: one hard cut.

    Returns the index of the first white frame, counted at the native rate
    FPS from zero, so a test can say exactly where a scene-change detector
    must fire and nowhere else. Built from two lavfi colour sources joined
    with `concat`, so every frame on either side is identical to its
    neighbours and the only frame difference in the file is the cut.
    """
    black = f"color=c=black:s={w}x{h}:r={FPS}:d={before:.3f}"
    white = f"color=c=white:s={w}x{h}:r={FPS}:d={after:.3f}"
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", black, "-f", "lavfi", "-i", white,
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p"]
    proc = subprocess.run(base + ["-c:v", "libx264", "-preset", "ultrafast",
                                  str(path)], capture_output=True)
    if proc.returncode != 0:
        subprocess.run(base + ["-c:v", "mpeg4", str(path)], check=True,
                       capture_output=True)
    return round(before * FPS)


def make_wav(path: Path, duration: float) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
                    "-t", f"{duration:.3f}", "-c:a", "pcm_s16le", str(path)],
                   check=True, capture_output=True)


# Hand-written xmeml (what Premiere exports; what the readers consume)
# ---------------------------------------------------------------------------

def _rate(parent, fps, ntsc):
    r = ET.SubElement(parent, "rate")
    ET.SubElement(r, "timebase").text = str(fps)
    ET.SubElement(r, "ntsc").text = "TRUE" if ntsc else "FALSE"


def _samplechars(parent, fps, ntsc, w=1920, h=1080):
    sc = ET.SubElement(parent, "samplecharacteristics")
    _rate(sc, fps, ntsc)
    ET.SubElement(sc, "width").text = str(w)
    ET.SubElement(sc, "height").text = str(h)


def build_xmeml(video_tracks, markers=(), audio=(), fps=24, ntsc=False,
                name="TEST SEQUENCE") -> str:
    """A minimal Premiere-style export.

    video_tracks: list (V1 first) of lists of clip dicts with keys
      name, start, end, in (frames); optional label, enabled (bool),
      pathurl, speed (timeremap percent), file_stub (bool: <file id/> only).
    markers: dicts with name, in, optional out (frames).
    audio: clip dicts with name, start, end, in, pathurl.
    """
    xmeml = ET.Element("xmeml", version="4")
    s = ET.SubElement(xmeml, "sequence", id="sequence-1")
    ET.SubElement(s, "name").text = name
    ET.SubElement(s, "duration").text = "0"
    _rate(s, fps, ntsc)
    ET.SubElement(s, "in").text = "-1"
    ET.SubElement(s, "out").text = "-1"
    media = ET.SubElement(s, "media")
    video = ET.SubElement(media, "video")
    _samplechars(ET.SubElement(video, "format"), fps, ntsc)
    uid = 0
    for clips in video_tracks:
        track = ET.SubElement(video, "track")
        for c in clips:
            uid += 1
            ci = ET.SubElement(track, "clipitem", id=f"clipitem-{uid}")
            ET.SubElement(ci, "name").text = c["name"]
            if "enabled" in c:
                ET.SubElement(ci, "enabled").text = (
                    "TRUE" if c["enabled"] else "FALSE")
            ET.SubElement(ci, "duration").text = str(c["end"] - c["start"])
            _rate(ci, fps, ntsc)
            ET.SubElement(ci, "start").text = str(c["start"])
            ET.SubElement(ci, "end").text = str(c["end"])
            ET.SubElement(ci, "in").text = str(c.get("in", 0))
            ET.SubElement(ci, "out").text = str(
                c.get("in", 0) + c["end"] - c["start"])
            f = ET.SubElement(ci, "file", id=f"file-{uid}")
            if not c.get("file_stub"):
                ET.SubElement(f, "name").text = c.get("file", c["name"])
                ET.SubElement(f, "pathurl").text = c.get(
                    "pathurl", f"file:///media/{c['name']}")
                _rate(f, fps, ntsc)
                fm = ET.SubElement(f, "media")
                _samplechars(ET.SubElement(fm, "video"), fps, ntsc)
            if c.get("speed") is not None:
                flt = ET.SubElement(ci, "filter")
                eff = ET.SubElement(flt, "effect")
                ET.SubElement(eff, "name").text = "Time Remap"
                ET.SubElement(eff, "effectid").text = "timeremap"
                par = ET.SubElement(eff, "parameter")
                ET.SubElement(par, "parameterid").text = "speed"
                ET.SubElement(par, "value").text = str(c["speed"])
            if c.get("label"):
                L = ET.SubElement(ci, "labels")
                ET.SubElement(L, "label2").text = c["label"]
    if audio:
        atrack = ET.SubElement(ET.SubElement(media, "audio"), "track")
        for c in audio:
            uid += 1
            ci = ET.SubElement(atrack, "clipitem", id=f"aclip-{uid}")
            ET.SubElement(ci, "name").text = c["name"]
            ET.SubElement(ci, "duration").text = str(c["end"] - c["start"])
            _rate(ci, fps, ntsc)
            ET.SubElement(ci, "start").text = str(c["start"])
            ET.SubElement(ci, "end").text = str(c["end"])
            ET.SubElement(ci, "in").text = str(c.get("in", 0))
            ET.SubElement(ci, "out").text = str(
                c.get("in", 0) + c["end"] - c["start"])
            f = ET.SubElement(ci, "file", id=f"afile-{uid}")
            ET.SubElement(f, "name").text = c["name"]
            ET.SubElement(f, "pathurl").text = c.get(
                "pathurl", f"file:///media/{c['name']}")
    for m in markers:
        me = ET.SubElement(s, "marker")
        ET.SubElement(me, "name").text = m["name"]
        ET.SubElement(me, "in").text = str(m["in"])
        ET.SubElement(me, "out").text = str(m.get("out", -1))
    pretty = minidom.parseString(ET.tostring(xmeml)).toprettyxml(indent="  ")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'
            + pretty.split("\n", 1)[1])


def write_xmeml(path: Path, *args, **kw) -> Path:
    path = Path(path)
    path.write_text(build_xmeml(*args, **kw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The v1 XML shape (CLAUDE.md rules 9 and 13)
# ---------------------------------------------------------------------------

SEQUENCE_HEAD = ["name", "duration", "rate", "in", "out", "media"]


def video_clipitems(seq):
    """(track_index, clipitem) for every clipitem on a video track."""
    out = []
    for ti, track in enumerate(seq.findall("media/video/track"), 1):
        for ci in track.findall("clipitem"):
            out.append((ti, ci))
    return out


def _int(ci, tag):
    return int(ci.findtext(tag))


def v1_shape_problems(root) -> list[str]:
    """Every violation of the shape Premiere is known to import.

    A rule here was learned by an import that failed with an empty error
    dialog, ghost clips or a dead program monitor, never by parsing - so
    nothing is added to this list without an isolated import test first.
    """
    problems = []
    if root.tag != "xmeml" or root.get("version") != "4":
        problems.append("root must be <xmeml version=\"4\">")
    if root.find(".//bin") is not None:
        problems.append("top-level/any <bin> wrapper present")
    for seq in root.iter("sequence"):
        tags = [child.tag for child in seq]
        head = tags[:len(SEQUENCE_HEAD)]
        if head != SEQUENCE_HEAD:
            problems.append(f"sequence child order {head} != {SEQUENCE_HEAD}")
        if any(t != "marker" for t in tags[len(SEQUENCE_HEAD):]):
            problems.append("sequence has non-marker children after <media>")
        if seq.find("media/video/format/samplecharacteristics") is None:
            problems.append("sequence lacks media/video/format/"
                            "samplecharacteristics")
        for ti, ci in video_clipitems(seq):
            cid = ci.get("id", "?")
            for tag in ("name", "duration", "rate", "start", "end", "in",
                        "out"):
                if ci.find(tag) is None:
                    problems.append(f"{cid}: missing <{tag}>")
            if ci.find("masterclipid") is not None:
                problems.append(f"{cid}: has <masterclipid>")
            f = ci.find("file")
            if f is None or not len(f):
                problems.append(f"{cid}: no full <file> element")
                continue
            pu = f.findtext("pathurl") or ""
            if not pu.startswith("file:///"):
                problems.append(f"{cid}: pathurl {pu!r} is not Path.as_uri()")
            if pu.startswith("file://localhost"):
                problems.append(f"{cid}: localhost-form pathurl")
            if f.find("media/video/samplecharacteristics") is None:
                problems.append(f"{cid}: <file> lacks media/video/"
                                "samplecharacteristics")
            try:
                if _int(ci, "end") - _int(ci, "start") != \
                        _int(ci, "out") - _int(ci, "in"):
                    problems.append(f"{cid}: end-start != out-in")
            except (TypeError, ValueError):
                problems.append(f"{cid}: non-integer timing")
        for track in seq.findall("media/video/track"):
            starts = [int(ci.findtext("start"))
                      for ci in track.findall("clipitem")]
            if starts != sorted(starts):
                problems.append("clipitems not chronological on a track")
    return problems


def assert_v1_shape(root) -> None:
    problems = v1_shape_problems(root)
    assert not problems, "\n".join(problems)


def audio_clipitems(seq):
    """(track_index, clipitem) for every clipitem on an audio track."""
    out = []
    for ti, track in enumerate(seq.findall("media/audio/track"), 1):
        for ci in track.findall("clipitem"):
            out.append((ti, ci))
    return out


def audio_shape_problems(root) -> list[str]:
    """Audio clipitems are held to a DIFFERENT, narrower contract than
    video ones, by design: rough_cut.py's music clipitem carries
    <masterclipid>/premiereChannelType/<sourcetrack> on purpose, copied
    from a real Premiere export because without them Premiere treats the
    clip as too thin to run a real crossfade on (see CLAUDE.md). Rule #13
    forbids <masterclipid> on a VIDEO clipitem; it is explicitly PERMITTED
    here. What must still hold, on either audio shape (emit.py's bare
    beat-grid clip or rough_cut.py's richer one): a full <file> (never a
    stub), an as_uri (never localhost) pathurl, and internally consistent
    start/end/in/out arithmetic. Before this existed, `assert_v1_shape`
    only walked video tracks and so never checked audio clipitems at all -
    rough_cut.py's masterclipid divergence was invisible to every
    automated check.
    """
    problems = []
    for seq in root.iter("sequence"):
        for ti, ci in audio_clipitems(seq):
            cid = ci.get("id", "?")
            for tag in ("name", "duration", "rate", "start", "end", "in",
                        "out"):
                if ci.find(tag) is None:
                    problems.append(f"{cid}: missing <{tag}>")
            f = ci.find("file")
            if f is None or not len(f):
                problems.append(f"{cid}: no full <file> element")
                continue
            pu = f.findtext("pathurl") or ""
            if not pu.startswith("file:///"):
                problems.append(f"{cid}: pathurl {pu!r} is not Path.as_uri()")
            if pu.startswith("file://localhost"):
                problems.append(f"{cid}: localhost-form pathurl")
            try:
                if _int(ci, "end") - _int(ci, "start") != \
                        _int(ci, "out") - _int(ci, "in"):
                    problems.append(f"{cid}: end-start != out-in")
            except (TypeError, ValueError):
                problems.append(f"{cid}: non-integer timing")
    return problems


def assert_audio_shape(root) -> None:
    problems = audio_shape_problems(root)
    assert not problems, "\n".join(problems)


def parse_xml(path: Path):
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith('<?xml version="1.0"'), lines[0]
    assert lines[1] == "<!DOCTYPE xmeml>", lines[1]
    return ET.fromstring(text)
