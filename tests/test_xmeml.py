# Phase 2a: bbp/xmeml.py consolidates rate/samplechars/sequence-skeleton/
# write-tail (previously copy-pasted across emit.py/alternates.py/
# rough_cut.py) and the carve primitive (previously four independent
# implementations). These pin the shared helpers directly.
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import pytest

from vmcore.xmeml import (carve, overlaps_any, parse_rate, sequence_skeleton,
                       write_rate, write_samplechars, write_xmeml)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_write_rate_shape():
    root = ET.Element("root")
    write_rate(root, 24)
    r = root.find("rate")
    assert r.findtext("timebase") == "24"
    assert r.findtext("ntsc") == "FALSE"


def test_write_samplechars_shape():
    root = ET.Element("root")
    sc = write_samplechars(root, 24, 1920, 1080)
    assert sc.findtext("width") == "1920"
    assert sc.findtext("height") == "1080"
    assert sc.findtext("anamorphic") == "FALSE"
    assert sc.findtext("pixelaspectratio") == "square"
    assert sc.findtext("fielddominance") == "none"
    assert sc.find("rate").findtext("timebase") == "24"


def test_sequence_skeleton_child_order():
    """Rule #9: name, duration, rate, in, out, media - in that order."""
    root = ET.Element("xmeml")
    s, dur, media, video = sequence_skeleton(root, "s1", "My Seq", 24, 1920,
                                             1080)
    tags = [c.tag for c in s]
    assert tags == ["name", "duration", "rate", "in", "out", "media"]
    assert s.findtext("name") == "My Seq"
    assert dur.text == "0"
    assert media is s.find("media")
    assert video is media.find("video")
    assert video.find("format/samplecharacteristics/width").text == "1920"


def test_write_xmeml_header_and_atomicity(tmp_path):
    root = ET.Element("xmeml", version="4")
    sequence_skeleton(root, "s1", "Seq", 24, 1920, 1080)
    out = tmp_path / "nested" / "out.xml"
    write_xmeml(root, out)
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith('<?xml version="1.0"')
    assert lines[1] == "<!DOCTYPE xmeml>"
    # parses back to the same document
    back = ET.fromstring(text)
    assert back.tag == "xmeml" and back.get("version") == "4"
    # no leftover temp file
    assert sorted(p.name for p in out.parent.iterdir()) == ["out.xml"]


def test_write_xmeml_matches_manual_pretty_print(tmp_path):
    """Byte-identical to the pattern it replaces in all three writers."""
    root = ET.Element("xmeml", version="4")
    sequence_skeleton(root, "s1", "Seq", 24, 1920, 1080)
    out = tmp_path / "out.xml"
    write_xmeml(root, out)
    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    expected = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'
                + pretty.split("\n", 1)[1])
    assert out.read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _seq_with_rate(timebase, ntsc=None):
    root = ET.Element("xmeml")
    seq = ET.SubElement(root, "sequence")
    r = ET.SubElement(seq, "rate")
    ET.SubElement(r, "timebase").text = str(timebase)
    if ntsc is not None:
        ET.SubElement(r, "ntsc").text = ntsc
    return seq


def test_parse_rate_plain():
    assert parse_rate(_seq_with_rate(24, "FALSE")) == 24.0


def test_parse_rate_ntsc_true_applies_drop_frame():
    fps = parse_rate(_seq_with_rate(24, "TRUE"))
    assert fps == pytest.approx(24 * 1000.0 / 1001.0)


def test_parse_rate_missing_ntsc_element_is_not_drop_frame():
    """No <ntsc> at all (some writers omit it) must not be treated as TRUE -
    this is the exact gap rough_cut.py's --preserve reader used to have."""
    assert parse_rate(_seq_with_rate(30)) == 30.0


def test_parse_rate_missing_timebase_uses_default():
    root = ET.Element("xmeml")
    seq = ET.SubElement(root, "sequence")
    assert parse_rate(seq, default=24.0) == 24.0
    assert parse_rate(seq, default=30.0) == 30.0


# ---------------------------------------------------------------------------
# carve()
# ---------------------------------------------------------------------------

def test_carve_no_overlap_is_unchanged():
    assert carve([(0, 10)], [(20, 30)]) == [(0, 10)]


def test_carve_full_cover_removes_the_piece():
    assert carve([(5, 10)], [(0, 20)]) == []


def test_carve_splits_a_straddled_piece():
    assert carve([(0, 10)], [(3, 6)]) == [(0, 3), (6, 10)]


def test_carve_trims_one_edge():
    assert carve([(0, 10)], [(7, 15)]) == [(0, 7)]
    assert carve([(0, 10)], [(-5, 3)]) == [(3, 10)]


def test_carve_multiple_covers_in_one_pass():
    # rough_cut.py's uncovered(): one piece, several covering spans
    assert carve([(0, 100)], [(10, 20), (50, 60)]) == \
        [(0, 10), (20, 50), (60, 100)]


def test_carve_many_pieces_many_covers():
    # preview.read_recut's shape: several candidate pieces against several
    # higher-track covers in one call
    pieces = [(0, 10), (10, 20), (20, 30)]
    covers = [(5, 15), (25, 28)]
    assert carve(pieces, covers) == [
        (0, 5),            # first piece trimmed
        (15, 20),          # second piece trimmed from the other side
        (20, 25), (28, 30),  # third piece split
    ]


def test_carve_reshuffle_shape_carves_many_spans_around_one_hole():
    # pipeline.extract_signals's RESHUFFLE-punches-LOCK and the (now
    # unified) ban-inside-lock carve: many spans, one hole.
    spans = [(0, 10), (20, 40)]
    hole = (5, 25)
    assert carve(spans, [hole]) == [(0, 5), (25, 40)]


def test_carve_is_order_independent_for_non_overlapping_covers():
    a = carve([(0, 100)], [(10, 20), (50, 60)])
    b = carve([(0, 100)], [(50, 60), (10, 20)])
    assert a == b


# ---------------------------------------------------------------------------
# overlaps_any()
# ---------------------------------------------------------------------------

def test_overlaps_any_true_when_intersecting():
    assert overlaps_any((5, 15), [(0, 10)])
    assert overlaps_any((5, 15), [(20, 30), (10, 20)])


def test_overlaps_any_false_when_disjoint():
    assert not overlaps_any((5, 15), [(20, 30)])
    assert not overlaps_any((5, 15), [])


def test_overlaps_any_touching_edges_do_not_count():
    # [5,15) and [15,20) share only a boundary point, not an interval
    assert not overlaps_any((5, 15), [(15, 20)])
    assert not overlaps_any((5, 15), [(0, 5)])
