"""The v1-shape validators in vmcore.testing, tested for the first time.

README.md says these "pin xmeml's output contract", and until this file
nothing in the suite called any of them. A weakened validator would have
passed vmcore's CI and relaxed every consumer's guard silently. The table
below pins each rule's exact problem string. It also pins three properties
of the helpers that crashed or lied at 757f484:

- a missing or non-integer video <start> raised TypeError or ValueError
  from the chronology check and threw away the problem list;
- assert_v1_shape, assert_audio_shape and parse_xml used `assert`, which
  `python -O` removes: a stubbed <file> then passed;
- parse_xml raised IndexError, not AssertionError, on a file shorter than
  two lines.

What this does NOT cover: vmcore.xmeml builds no clipitem, so the rule-13
checks pass trivially on its output. Those surfaces are written by the
consumers, which keep their own negative cases.
"""
from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from vmcore import testing as T
from vmcore.xmeml import sequence_skeleton, write_xmeml

ROOT = Path(__file__).resolve().parent.parent


def _clip(**kw):
    c = {"name": "a.mp4", "start": 0, "end": 24, "in": 0}
    c.update(kw)
    return c


def _good() -> str:
    return T.build_xmeml(
        [[_clip(), _clip(name="b.mp4", start=24, end=48)]],
        markers=[{"name": "m", "in": 3}],
        audio=[{"name": "song.wav", "start": 0, "end": 48}])


def _root(mutate=None):
    root = ET.fromstring(_good())
    if mutate:
        mutate(root)
    return root


def _vclips(root):
    return root.findall(".//media/video/track/clipitem")


def _drop(el, tag):
    el.remove(el.find(tag))


# --- the positive case ------------------------------------------------------

def test_the_reference_builder_output_passes_both_validators():
    root = _root()
    assert T.v1_shape_problems(root) == []
    assert T.audio_shape_problems(root) == []
    assert T.assert_v1_shape(root) is None
    assert T.assert_audio_shape(root) is None


def test_the_music_clipitem_may_carry_masterclipid_the_video_one_may_not():
    """Rule 13 forbids <masterclipid> on a video clipitem and the audio
    contract permits it (rough_cut.py's music item carries one on purpose)."""
    def add_audio_mc(root):
        ci = root.find(".//media/audio/track/clipitem")
        ET.SubElement(ci, "masterclipid").text = "masterclip-1"

    def add_video_mc(root):
        ET.SubElement(_vclips(root)[0], "masterclipid").text = "masterclip-1"

    assert T.audio_shape_problems(_root(add_audio_mc)) == []
    assert T.v1_shape_problems(_root(add_video_mc)) == [
        "clipitem-1: has <masterclipid>"]


# --- each rule, exact string ------------------------------------------------

def _version5(root): root.set("version", "5")
def _bin(root): ET.SubElement(root, "bin")
def _stub(root):
    f = _vclips(root)[0].find("file")
    for c in list(f):
        f.remove(c)
def _localhost(root):
    _vclips(root)[0].find("file/pathurl").text = "file://localhost/media/a.mp4"
def _relative(root):
    _vclips(root)[0].find("file/pathurl").text = "media/a.mp4"
def _no_file_chars(root):
    f = _vclips(root)[0].find("file")
    f.remove(f.find("media"))
def _arith(root): _vclips(root)[0].find("out").text = "5"
def _no_format(root):
    v = root.find(".//sequence/media/video")
    v.remove(v.find("format"))
def _head_order(root):
    s = root.find("sequence")
    d, r = s.find("duration"), s.find("rate")
    s.remove(d)
    s.insert(list(s).index(r) + 1, d)
def _late_child(root): ET.SubElement(root.find("sequence"), "name")
def _reversed(root):
    track = root.find(".//media/video/track")
    items = list(track)
    for it in items:
        track.remove(it)
    track.extend(reversed(items))
def _drop_name(root): _drop(_vclips(root)[0], "name")


@pytest.mark.parametrize("mutate, expected", [
    (_version5, ['root must be <xmeml version="4">']),
    (_bin, ["top-level/any <bin> wrapper present"]),
    (_stub, ["clipitem-1: no full <file> element"]),
    (_localhost, ["clipitem-1: localhost-form pathurl",
                  "clipitem-1: pathurl 'file://localhost/media/a.mp4' is not Path.as_uri()"]),
    (_relative, ["clipitem-1: pathurl 'media/a.mp4' is not Path.as_uri()"]),
    (_no_file_chars, ["clipitem-1: <file> lacks media/video/samplecharacteristics"]),
    (_arith, ["clipitem-1: end-start != out-in"]),
    (_no_format, ["sequence lacks media/video/format/samplecharacteristics"]),
    (_head_order, ["sequence child order ['name', 'rate', 'duration', 'in', 'out', 'media']"
                   " != ['name', 'duration', 'rate', 'in', 'out', 'media']"]),
    (_late_child, ["sequence has non-marker children after <media>"]),
    (_reversed, ["clipitems not chronological on a track"]),
    (_drop_name, ["clipitem-1: missing <name>"]),
], ids=["version5", "bin", "stub-file", "localhost", "relative-url",
        "file-without-media", "arithmetic", "no-format", "head-order",
        "late-child", "chronology", "missing-name"])
def test_each_video_rule_reports_its_exact_problem(mutate, expected):
    assert sorted(T.v1_shape_problems(_root(mutate))) == sorted(expected)


def test_audio_rules_report_their_exact_problems():
    def stub(root):
        f = root.find(".//media/audio/track/clipitem/file")
        for c in list(f):
            f.remove(c)

    def localhost(root):
        root.find(".//media/audio/track/clipitem/file/pathurl").text = \
            "file://localhost/media/song.wav"

    def arith(root):
        root.find(".//media/audio/track/clipitem/out").text = "5"

    assert T.audio_shape_problems(_root(stub)) == [
        "aclip-3: no full <file> element"]
    assert sorted(T.audio_shape_problems(_root(localhost))) == [
        "aclip-3: localhost-form pathurl",
        "aclip-3: pathurl 'file://localhost/media/song.wav' is not Path.as_uri()"]
    assert T.audio_shape_problems(_root(arith)) == [
        "aclip-3: end-start != out-in"]


# --- crashes at 757f484 ------------------------------------------------------

def test_a_missing_start_is_reported_not_raised():
    def drop_start(root): _drop(_vclips(root)[1], "start")
    assert T.v1_shape_problems(_root(drop_start)) == [
        "clipitem-2: missing <start>", "clipitem-2: non-integer timing"]


def test_a_non_integer_start_is_reported_not_raised():
    def junk(root): _vclips(root)[1].find("start").text = "abc"
    assert T.v1_shape_problems(_root(junk)) == ["clipitem-2: non-integer timing"]


def test_a_non_integer_start_on_a_stub_file_is_still_reported():
    """A stub <file> skips the arithmetic check, so nothing else would."""
    def junk(root):
        ci = _vclips(root)[1]
        ci.find("start").text = "abc"
        f = ci.find("file")
        for c in list(f):
            f.remove(c)
    assert T.v1_shape_problems(_root(junk)) == [
        "clipitem-2: no full <file> element", "clipitem-2: non-integer timing"]


def test_the_clips_around_a_bad_start_are_still_checked_for_order():
    xml = T.build_xmeml([[_clip(start=24, end=48), _clip(name="b.mp4"),
                          _clip(name="c.mp4", start=0, end=24)]])
    root = ET.fromstring(xml)
    root.findall(".//media/video/track/clipitem")[1].find("start").text = "xx"
    problems = T.v1_shape_problems(root)
    assert "clipitem-2: non-integer timing" in problems
    assert "clipitems not chronological on a track" in problems       # 24 then 0


def test_assert_v1_shape_raises_assertion_error_with_every_problem():
    def drop_start(root): _drop(_vclips(root)[1], "start")
    with pytest.raises(AssertionError) as info:
        T.assert_v1_shape(_root(drop_start))
    assert str(info.value) == "clipitem-2: missing <start>\nclipitem-2: non-integer timing"


@pytest.mark.parametrize("text, message", [
    ("", "empty file"),
    ('<?xml version="1.0" encoding="UTF-8"?>',
     "no second line: expected <!DOCTYPE xmeml>"),
    ("junk\n<!DOCTYPE xmeml>\n<xmeml/>", "junk"),
    ('<?xml version="1.0" encoding="UTF-8"?>\n<x/>\n<xmeml/>', "<x/>"),
], ids=["empty", "one-line", "bad-first", "bad-second"])
def test_parse_xml_names_the_wrong_line_as_an_assertion_error(tmp_path, text, message):
    p = tmp_path / "x.xml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(AssertionError) as info:
        T.parse_xml(p)
    assert str(info.value) == message


def test_parse_xml_reads_what_the_reference_builder_and_xmeml_write(tmp_path):
    p = T.write_xmeml(tmp_path / "ref.xml", [[_clip()]])
    assert T.parse_xml(p).tag == "xmeml"
    # and vmcore.xmeml's own writer: header, version, head order (rule 9 only)
    xmeml = ET.Element("xmeml", version="4")
    sequence_skeleton(xmeml, 1, "S", 24, 1920, 1080)
    out = tmp_path / "skeleton.xml"
    write_xmeml(xmeml, out)
    root = T.parse_xml(out)
    assert T.v1_shape_problems(root) == []


# --- python -O ---------------------------------------------------------------

_O_SCRIPT = r"""
import sys, tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from vmcore import testing as T
stub = ET.fromstring(T.build_xmeml([[{"name": "a.mp4", "start": 0, "end": 24, "file_stub": True}]]))
audio = ET.fromstring(T.build_xmeml([[]], audio=[{"name": "s.wav", "start": 0, "end": 5}]))
audio.find(".//media/audio/track/clipitem/file").clear()
bad = Path(tempfile.mkdtemp()) / "bad.xml"
bad.write_text("junk\n<!DOCTYPE xmeml>\n<xmeml/>")
out = {}
for name, fn in (("v1", lambda: T.assert_v1_shape(stub)),
                 ("audio", lambda: T.assert_audio_shape(audio)),
                 ("parse", lambda: T.parse_xml(bad))):
    try:
        fn()
        out[name] = "PASSED"
    except AssertionError:
        out[name] = "AssertionError"
    except Exception as exc:
        out[name] = type(exc).__name__
print(sys.flags.optimize, out["v1"], out["audio"], out["parse"])
"""


@pytest.mark.parametrize("flag", [[], ["-O"], ["-OO"]], ids=["plain", "-O", "-OO"])
def test_the_validators_still_fail_under_python_dash_O(flag):
    """`assert` statements are removed by -O. At 757f484 this printed
    `1 PASSED PASSED ParseError` under -O: a stubbed <file> and a clipitem
    with no file contents both passed, and a wrong header fell through to
    the XML parser."""
    r = subprocess.run([sys.executable, *flag, "-c", _O_SCRIPT], cwd=ROOT,
                       capture_output=True, text=True, check=True)
    opt, v1, audio, parse = r.stdout.split()
    assert opt == {"": "0", "-O": "1", "-OO": "2"}[flag[0] if flag else ""]
    assert (v1, audio, parse) == ("AssertionError",) * 3
