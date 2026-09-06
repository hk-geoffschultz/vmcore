# Phase 2b: the content-addressed segment cache. No ffmpeg needed here -
# these test the key derivation and the cache directory contract directly.
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from vmcore.segcache import SegmentCache, profile_hash, segment_key

FIT = {"w": 1920, "h": 1080, "mode": "fit",
      "video": ["-c:v", "libx264", "-preset", "medium", "-crf", "22"]}
SMARTCROP = {"w": 1080, "h": 1920, "mode": "smartcrop",
            "video": ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]}


def _src(tmp_path, name="a.mp4", content=b"x" * 100):
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# profile_hash
# ---------------------------------------------------------------------------

def test_profile_hash_ignores_the_name_only_the_encode_fields():
    a = dict(FIT)
    b = {**FIT, "card_aspect": "16x9", "audio": ["-c:a", "aac"]}
    assert profile_hash(a) == profile_hash(b)


def test_profile_hash_changes_with_crf():
    bumped = {**FIT, "video": ["-c:v", "libx264", "-preset", "medium",
                               "-crf", "20"]}
    assert profile_hash(FIT) != profile_hash(bumped)


def test_profile_hash_changes_with_dimensions():
    assert profile_hash(FIT) != profile_hash({**FIT, "w": 1280, "h": 720})


# ---------------------------------------------------------------------------
# segment_key
# ---------------------------------------------------------------------------

def test_same_inputs_same_key(tmp_path):
    src = _src(tmp_path)
    k1 = segment_key(src, 1.5, 48, None, False, FIT)
    k2 = segment_key(src, 1.5, 48, None, False, FIT)
    assert k1 == k2


def test_key_changes_with_in_point_at_frame_precision(tmp_path):
    src = _src(tmp_path)
    k1 = segment_key(src, 1.500000, 48, None, False, FIT)
    k2 = segment_key(src, 1.500001, 48, None, False, FIT)
    assert k1 != k2


def test_key_ignores_float_noise_below_the_formatted_precision(tmp_path):
    """Two carve computations landing at 1e-9-different floats must collide
    once formatted to the same 6 decimal places ffmpeg actually receives."""
    src = _src(tmp_path)
    k1 = segment_key(src, 1.5 + 1e-10, 48, None, False, FIT)
    k2 = segment_key(src, 1.5 - 1e-10, 48, None, False, FIT)
    assert k1 == k2


def test_key_changes_with_frames_speed_hdr_profile(tmp_path):
    src = _src(tmp_path)
    base = segment_key(src, 1.0, 48, None, False, FIT)
    assert base != segment_key(src, 1.0, 49, None, False, FIT)
    assert base != segment_key(src, 1.0, 48, 200.0, False, FIT)
    assert base != segment_key(src, 1.0, 48, None, True, FIT)
    assert base != segment_key(src, 1.0, 48, None, False, SMARTCROP)


def test_speed_none_and_speed_zero_collide():
    # FCP7 uses 0/None interchangeably for "no timeremap" - both must hash
    # the same or an untouched clip's key would depend on which the XML
    # reader happened to produce.
    src_path = "does/not/need/to/exist/for/hash/purposes"
    import vmcore.segcache as sc
    orig = sc._src_stamp
    sc._src_stamp = lambda p: (100, 12345)
    try:
        assert segment_key(Path(src_path), 1.0, 48, None, False, FIT) == \
            segment_key(Path(src_path), 1.0, 48, 0.0, False, FIT)
    finally:
        sc._src_stamp = orig


def test_key_changes_when_source_bytes_change(tmp_path):
    """A source overwritten with different bytes (re-export, re-transcode)
    must miss the cache rather than serve stale pixels under the old path."""
    src = _src(tmp_path)
    k1 = segment_key(src, 1.0, 48, None, False, FIT)
    time.sleep(0.01)
    src.write_bytes(b"y" * 200)
    os.utime(src, None)   # ensure mtime actually advances on fast filesystems
    k2 = segment_key(src, 1.0, 48, None, False, FIT)
    assert k1 != k2


def test_key_takes_crop_centre_for_smartcrop(tmp_path):
    src = _src(tmp_path)
    fit_key = segment_key(src, 1.0, 48, None, False, SMARTCROP)
    cropped_key = segment_key(src, 1.0, 48, None, False, SMARTCROP,
                              cx=0.5, cy=0.5)
    assert fit_key != cropped_key


def test_key_changes_when_a_crops_override_moves_the_centre(tmp_path):
    """A hand-edited .crops.json override must correctly bust the cache -
    the whole reason cx/cy are in the key rather than derived silently."""
    src = _src(tmp_path)
    k1 = segment_key(src, 1.0, 48, None, False, SMARTCROP, cx=0.5, cy=0.5)
    k2 = segment_key(src, 1.0, 48, None, False, SMARTCROP, cx=0.3, cy=0.5)
    assert k1 != k2


def test_key_rounds_crop_centre_to_the_filter_expressions_precision(tmp_path):
    src = _src(tmp_path)
    k1 = segment_key(src, 1.0, 48, None, False, SMARTCROP,
                     cx=0.500001, cy=0.5)
    k2 = segment_key(src, 1.0, 48, None, False, SMARTCROP,
                     cx=0.500002, cy=0.5)
    assert k1 == k2   # both round to 0.5000, the crop filter's own .4f


# ---------------------------------------------------------------------------
# SegmentCache
# ---------------------------------------------------------------------------

def test_get_on_a_fresh_cache_is_none(tmp_path):
    cache = SegmentCache(tmp_path / "cache")
    assert cache.get("nonexistent") is None


def test_put_then_get_round_trips(tmp_path):
    cache = SegmentCache(tmp_path / "cache")
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"fake mp4 bytes")
    dest = cache.put("somekey", rendered)
    assert dest.exists()
    assert cache.get("somekey") == dest
    assert dest.read_bytes() == b"fake mp4 bytes"


def test_put_leaves_the_source_file_untouched(tmp_path):
    """Some callers still need their own copy at the original path (to
    stage into a concat list) after caching it."""
    cache = SegmentCache(tmp_path / "cache")
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"abc")
    cache.put("k", rendered)
    assert rendered.exists() and rendered.read_bytes() == b"abc"


def test_put_leaves_no_tmp_file_behind(tmp_path):
    cache = SegmentCache(tmp_path / "cache")
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"abc")
    cache.put("k", rendered)
    names = [p.name for p in (tmp_path / "cache").iterdir()]
    assert names == ["k.mp4"]


def test_a_zero_byte_cache_file_reads_as_a_miss(tmp_path):
    """A crash mid-copy (before put()'s os.replace) cannot be observed as
    a partial file by a reader - but guard the zero-byte case too, since a
    reader could otherwise be handed an empty, unplayable segment."""
    cache = SegmentCache(tmp_path / "cache")
    cache.path_for("k").touch()
    assert cache.get("k") is None


def test_distinct_keys_do_not_collide(tmp_path):
    cache = SegmentCache(tmp_path / "cache")
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"AAA")
    b.write_bytes(b"BBB")
    cache.put("key-a", a)
    cache.put("key-b", b)
    assert cache.get("key-a").read_bytes() == b"AAA"
    assert cache.get("key-b").read_bytes() == b"BBB"
