"""Frame sampling.

The load-bearing property here is byte alignment: every frame read off the
raw pipe is a fixed-size block, and if the predicted size disagrees with
what ffmpeg emits by even one byte, every subsequent frame is skewed and
the failure looks like garbled content rather than an error. The real-media
tests below exist to catch exactly that, which is why they shell out to
ffmpeg rather than mocking the decode.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from vmcore.testing import FPS, make_cut_mp4, make_mp4
from vmcore.frames import (DecodeError, analysis_vf, decode_cmd, sample_frames,
                           thinned_fps)
from vmcore.probe import display_dims, scaled_dims

LONG_EDGE = 64  # keep the decode tiny; the arithmetic is what's under test


# --------------------------------------------------------------------------
# Rate thinning
# --------------------------------------------------------------------------

def test_short_clip_keeps_its_sample_rate():
    assert thinned_fps(duration=10.0, fps=2.0, max_frames=60) == 2.0


def test_long_clip_is_thinned_to_the_frame_budget():
    # 20 minutes at 2fps would be 2400 frames; the budget is 60.
    fps = thinned_fps(duration=1200.0, fps=2.0, max_frames=60)
    assert fps * 1200.0 == pytest.approx(60)


def test_thinning_is_exactly_at_the_boundary():
    """duration * fps == max_frames must not be thinned."""
    assert thinned_fps(duration=30.0, fps=2.0, max_frames=60) == 2.0


def test_zero_duration_does_not_divide_by_zero():
    assert thinned_fps(duration=0.0, fps=2.0, max_frames=60) == 2.0


def test_filter_chain_pins_the_long_edge_on_either_orientation():
    vf = analysis_vf(2.0, 960)
    assert "fps=2.000000" in vf
    # Landscape pins width, portrait pins height; -2 keeps dims even.
    assert "if(gt(iw,ih),960,-2)" in vf
    assert "if(gt(iw,ih),-2,960)" in vf


# --------------------------------------------------------------------------
# Real decode
# --------------------------------------------------------------------------

def test_frames_match_the_predicted_block_size(tiny_clip):
    """The contract that keeps the raw pipe aligned."""
    src = display_dims(tiny_clip)
    assert src is not None
    w, h = scaled_dims(src[0], src[1], LONG_EDGE)

    got = list(sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE))

    assert got, "expected at least one frame from a 1s clip at 2fps"
    for _, frame in got:
        assert frame.shape == (h, w, 3)
        assert frame.dtype.name == "uint8"


def test_timestamps_step_by_the_sample_period(tiny_clip):
    got = list(sample_frames(tiny_clip, 1.0, fps=2.0, long_edge=LONG_EDGE))
    times = [t for t, _ in got]
    assert times == sorted(times)
    for i, t in enumerate(times):
        assert t == pytest.approx(i / 2.0)


def test_frame_budget_is_not_exceeded(tmp_path):
    clip = tmp_path / "longer.mp4"
    make_mp4(clip, duration=2.0)
    got = list(sample_frames(clip, 2.0, fps=8.0, max_frames=3,
                             long_edge=LONG_EDGE))
    assert len(got) <= 3


def test_zero_duration_yields_nothing(tiny_clip):
    assert list(sample_frames(tiny_clip, 0.0, long_edge=LONG_EDGE)) == []


def test_unprobeable_file_yields_nothing_rather_than_raising(tmp_path):
    """A corrupt file in a dump of hundreds must not stop the walk."""
    junk = tmp_path / "not-really.mp4"
    junk.write_bytes(b"not a video")
    assert list(sample_frames(junk, 5.0, long_edge=LONG_EDGE)) == []


# --------------------------------------------------------------------------
# A full-rate filter riding the decode
#
# `full_rate_vf` splices a fragment in ahead of the fps thinning, where it
# sees every native frame, and keeps ffmpeg alive to EOF so a file the
# fragment writes is complete. The tests pin four things: the chain and
# the argv are what they were when the fragment is absent; the frames are
# what they were when it is present; the sink is whole in the two cases
# where the old code demonstrably killed the decode (a `duration` that
# understates the file, and a consumer that stops early - the thinned
# budget case is pinned too, but the old code left that sink whole as
# well, see its docstring); and what ffmpeg rejects is raised with its
# reason rather than yielded as nothing. The scdet + metadata pair is the
# fragment these were written against, and the scdet test pins the exact
# text that filter pair writes, because a consumer parses it.
# --------------------------------------------------------------------------

# scdet's own default threshold (`ffmpeg -h filter=scdet`: t, default 10).
# A declared choice, not a measurement, and not load-bearing here: on the
# black-then-white clip the cut scores 85.5 and every other frame 0, so
# any t in (0, 85.5) gives the same result. What a score means on real
# footage is the consumer's measurement to make.
SCDET_T = 10.0

def _native_frames(path: Path) -> int:
    """How many frames the file really holds, by decoding it (ffprobe).

    The default writer, not csv: on a stream with side data (a phone clip's
    display matrix) csv prints the side-data section after the field and
    the line reads `167,`.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def _blocks(sink: Path) -> list[dict]:
    """The metadata filter's print format, one dict per frame.

    A frame is a `frame:N pts:P pts_time:T` header followed by `key=value`
    lines until the next header. Kept literal here, not shared from vmcore:
    vmcore does not parse what a fragment writes, and this is a test of
    the text a consumer will have to.
    """
    blocks: list[dict] = []
    for line in sink.read_text().splitlines():
        if line.startswith("frame:"):
            fields = dict(tok.split(":", 1) for tok in line.split())
            blocks.append({"frame": int(fields["frame"]),
                           "pts_time": float(fields["pts_time"])})
        elif "=" in line and blocks:
            k, v = line.split("=", 1)
            blocks[-1][k] = v
    return blocks


def _scdet(sink: Path | str, t: float = SCDET_T) -> str:
    return f"scdet=t={t},metadata=mode=print:file={sink}"


def test_default_chain_is_byte_identical_without_a_fragment():
    """The chain `scaled_dims` predicts, pinned as the literal it was
    before the argument existed - None and empty both mean absent."""
    expected = ("fps=2.000000,"
                "scale='if(gt(iw,ih),960,-2)':'if(gt(iw,ih),-2,960)'")
    assert analysis_vf(2.0, 960) == expected
    assert analysis_vf(2.0, 960, None) == expected
    assert analysis_vf(2.0, 960, "") == expected


def test_fragment_is_prefixed_verbatim_ahead_of_the_fps_thinning():
    frag = "scdet=t=10,metadata=mode=print:file=/tmp/x.txt"
    assert analysis_vf(2.0, 960, frag) == frag + "," + analysis_vf(2.0, 960)


def test_analysis_only_command_carries_the_chain_in_vf(tmp_path):
    vf = analysis_vf(2.0, LONG_EDGE, "scdet=t=10")
    cmd = decode_cmd(tmp_path / "c.mp4", vf)
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-vf") + 1] == vf
    assert "-filter_complex" not in cmd
    # Without the fragment the argv is exactly the old one.
    plain = decode_cmd(tmp_path / "c.mp4", analysis_vf(2.0, LONG_EDGE))
    assert plain == [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(tmp_path / "c.mp4"),
        "-vf", analysis_vf(2.0, LONG_EDGE),
        "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-",
    ]


def test_fused_command_carries_the_fragment_in_the_analysis_branch_only(
        tiny_clip, tmp_path):
    from vmcore.proxy import color_transfer, proxy_vf

    proxy = tmp_path / "proxies" / "clip.mp4"
    with_it = decode_cmd(tiny_clip, analysis_vf(2.0, LONG_EDGE, "scdet=t=10"),
                         proxy_out=proxy)
    without = decode_cmd(tiny_clip, analysis_vf(2.0, LONG_EDGE),
                         proxy_out=proxy)

    graph = with_it[with_it.index("-filter_complex") + 1]
    split, analysis, proxy_branch = graph.split(";")
    assert split == "[0:v:0]split=2[an][px]"
    assert analysis == f"[an]scdet=t=10,{analysis_vf(2.0, LONG_EDGE)}[aout]"
    assert proxy_branch == f"[px]{proxy_vf(color_transfer(tiny_clip))}[pout]"
    assert "scdet" not in proxy_branch

    # The graph is the only argument that moved: binary, maps and the
    # proxy's encode arguments are untouched by the fragment.
    i = with_it.index("-filter_complex") + 1
    assert with_it[:i] == without[:i]
    assert with_it[i + 1:] == without[i + 1:]


def _assert_whole(sink: Path, native: int) -> None:
    """Every native frame is in the sink, in order, ending on the last."""
    assert sink.exists()
    blocks = _blocks(sink)
    assert len(blocks) == native
    assert [b["frame"] for b in blocks] == list(range(native))
    last = blocks[-1]
    assert "lavfi.scd.mafd" in last and "lavfi.scd.score" in last


def test_sink_is_complete_when_the_clip_outruns_the_frame_budget(tmp_path):
    """8s at fps=2 with a budget of 6 is thinned to 0.75fps and yields 6
    frames. The frames must be what they were without the fragment, and
    the sink must hold every native frame, ending on the last one.

    Measured before this was written (2026-09-27, 64x36 to 640x360, 5 runs
    each): the OLD code also left this sink whole, every run. Thinning puts
    the last sample in the clip's final interval and the fps filter emits
    that frame only at EOF, so the old break at the budget fired after
    ffmpeg had already finished. This test pins the thinned case; the two
    below are the ones where the old code demonstrably killed the decode."""
    clip = tmp_path / "eight.mp4"
    make_mp4(clip, duration=8.0)
    native = _native_frames(clip)
    assert native > 6, "the clip has to outrun the budget for this to test"

    plain = list(sample_frames(clip, 8.0, fps=2.0, max_frames=6,
                               long_edge=LONG_EDGE))
    sink = tmp_path / "scd.txt"
    got = list(sample_frames(clip, 8.0, fps=2.0, max_frames=6,
                             long_edge=LONG_EDGE, full_rate_vf=_scdet(sink)))

    assert len(got) == len(plain) == 6
    assert [t for t, _ in got] == [t for t, _ in plain]
    for (_, a), (_, b) in zip(got, plain):
        assert a.shape == b.shape
        assert np.array_equal(a, b)
    _assert_whole(sink, native)


def test_sink_is_complete_when_the_budget_is_hit_mid_file(tmp_path):
    """Thinning trusts the caller's `duration`; when the file is longer
    than that, the budget is reached with decode still to go, and that is
    where the old code broke out of the loop and terminated ffmpeg. Here
    an 8s file is declared as 3s: six frames at 2fps, then 120 native
    frames the old code never let the sink see."""
    clip = tmp_path / "eight.mp4"
    make_mp4(clip, duration=8.0)
    native = _native_frames(clip)
    sink = tmp_path / "scd.txt"

    got = list(sample_frames(clip, 3.0, fps=2.0, max_frames=6,
                             long_edge=LONG_EDGE, full_rate_vf=_scdet(sink)))

    assert [t for t, _ in got] == [i / 2.0 for i in range(6)]
    _assert_whole(sink, native)


def test_sink_is_complete_when_the_consumer_stops_early(tmp_path):
    """Taking one frame and closing the generator is the path through
    `_shutdown` proper; with the fragment set it drains to EOF instead of
    killing the decode, so the sink is still whole. Measured 2026-09-27 on
    this clip: the old code's kill left roughly a quarter to a third of
    the 192 frames in the sink, a different range in every set of five
    runs and never zero, because how far ffmpeg got before SIGTERM is a
    matter of load. The mechanism is the finding; a range is a snapshot."""
    clip = tmp_path / "eight.mp4"
    make_mp4(clip, duration=8.0)
    native = _native_frames(clip)
    sink = tmp_path / "scd.txt"

    gen = sample_frames(clip, 8.0, fps=2.0, max_frames=6,
                        long_edge=LONG_EDGE, full_rate_vf=_scdet(sink))
    t, frame = next(gen)
    gen.close()

    assert t == 0.0 and frame.shape[2] == 3
    _assert_whole(sink, native)


def test_scdet_scores_exactly_the_cut_frame_and_tags_every_frame(tmp_path):
    """Pins the text a consumer parses: a `frame:` header per native frame,
    `lavfi.scd.mafd` and `lavfi.scd.score` on every one of them, and
    `lavfi.scd.time` only on a frame whose score clears `t`. On a
    black-then-white clip that is the first white frame and no other.
    Frame 0 prints `pts_time:0` with no decimal, which is why the parser
    above reads it as a float rather than matching a shape."""
    clip = tmp_path / "cut.mp4"
    cut = make_cut_mp4(clip, before=1.0, after=1.0)
    native = _native_frames(clip)
    assert native == cut + FPS

    sink = tmp_path / "scd.txt"
    got = list(sample_frames(clip, 2.0, fps=2.0, long_edge=LONG_EDGE,
                             full_rate_vf=_scdet(sink, t=SCDET_T)))
    assert len(got) == 4

    blocks = _blocks(sink)
    assert len(blocks) == native
    for b in blocks:
        assert "lavfi.scd.mafd" in b, b
        assert "lavfi.scd.score" in b, b
        float(b["lavfi.scd.mafd"]), float(b["lavfi.scd.score"])

    fired = [b["frame"] for b in blocks
             if float(b["lavfi.scd.score"]) >= SCDET_T]
    assert fired == [cut]
    tagged = [b["frame"] for b in blocks if "lavfi.scd.time" in b]
    assert tagged == [cut]
    assert blocks[cut]["pts_time"] == pytest.approx(cut / FPS)
    # scdet's score is min(mafd, |mafd - previous mafd|); on the frame
    # before the cut mafd is 0, so the score on the cut is its own mafd.
    assert blocks[cut]["lavfi.scd.score"] == blocks[cut]["lavfi.scd.mafd"]


def test_a_rejected_fragment_raises_with_what_ffmpeg_said(tmp_path):
    """The fragment is the caller's code. A filter ffmpeg does not have
    used to look, from the caller's side, exactly like a zero-length
    recording - nothing yielded, no exception - with ffmpeg's reason (exit
    8, "No such filter") sent to DEVNULL. Now it raises, and the message
    names the filter."""
    clip = tmp_path / "two.mp4"
    make_mp4(clip, duration=2.0)
    with pytest.raises(DecodeError) as e:
        list(sample_frames(clip, 2.0, fps=2.0, long_edge=LONG_EDGE,
                           full_rate_vf="nosuchfilter"))
    assert "nosuchfilter" in str(e.value)
    assert e.value.returncode not in (0, None)
    assert e.value.cmd[0] == "ffmpeg"


def test_an_unprobeable_file_still_yields_nothing_with_a_fragment_set(
        tmp_path):
    """The walk rule survives on the fragment path: a file `probe` cannot
    read is returned on before ffmpeg is run, fragment or not."""
    junk = tmp_path / "not-really.mp4"
    junk.write_bytes(b"not a video")
    sink = tmp_path / "scd.txt"
    assert list(sample_frames(junk, 5.0, long_edge=LONG_EDGE,
                              full_rate_vf=_scdet(sink))) == []
    assert not sink.exists()


def test_a_file_that_probes_but_does_not_decode_raises_only_with_a_fragment(
        tmp_path):
    """The boundary the raise draws, pinned honestly: a valid container
    whose payload is zeroed probes fine and exits non-zero with nothing
    yielded, exactly like a rejected fragment. Without a fragment that is
    still nothing yielded, as it always was; with one it is `DecodeError`
    carrying ffmpeg's words, and the caller that must keep walking catches
    it per clip. The message is not pinned - it is ffmpeg's, and this test
    is about which path raises, not what it says."""
    clip = tmp_path / "two.mp4"
    make_mp4(clip, duration=2.0)
    data = bytearray(clip.read_bytes())
    i = data.find(b"mdat")
    size = int.from_bytes(data[i - 4:i], "big")
    data[i + 4:i - 4 + size] = b"\0" * (size - 8)
    corrupt = tmp_path / "zeroed.mp4"
    corrupt.write_bytes(data)
    assert display_dims(corrupt) is not None, "the file has to probe"

    assert list(sample_frames(corrupt, 2.0, fps=2.0, long_edge=LONG_EDGE)) == []
    sink = tmp_path / "scd.txt"
    with pytest.raises(DecodeError) as e:
        list(sample_frames(corrupt, 2.0, fps=2.0, long_edge=LONG_EDGE,
                           full_rate_vf=_scdet(sink)))
    assert e.value.stderr.strip()


def test_sink_path_with_a_colon_and_a_space_is_escaped_for_two_parsers(
        tmp_path):
    """The escaping the caller owes, measured rather than assumed: the
    filtergraph parser strips one level, the option parser another, so a
    ':' in a `file=` path needs `\\:` inside single quotes or `\\\\:` bare.
    The one-level forms - bare, quoted once, backslashed once - are what a
    caller following the obvious rule writes, and ffmpeg rejects them; that
    is now a raise rather than a silent zero."""
    clip = tmp_path / "two.mp4"
    make_mp4(clip, duration=2.0)
    native = _native_frames(clip)
    d = tmp_path / "esc dir:x"
    d.mkdir()
    sink = d / "sink.txt"
    raw = str(sink)

    def run(form: str) -> list:
        sink.unlink(missing_ok=True)
        return list(sample_frames(clip, 2.0, fps=2.0, long_edge=LONG_EDGE,
                                  full_rate_vf=_scdet(form)))

    quoted = "'" + raw.replace(":", "\\:") + "'"
    assert len(run(quoted)) == 4
    _assert_whole(sink, native)
    bare = raw.replace(":", "\\\\:").replace(" ", "\\\\ ")
    assert len(run(bare)) == 4
    _assert_whole(sink, native)

    for wrong in (raw, f"'{raw}'", raw.replace(":", "\\:").replace(" ", "\\ ")):
        with pytest.raises(DecodeError):
            run(wrong)
        assert not sink.exists()


def _can_encode_proxy() -> bool:
    from vmcore.proxy import full_ffmpeg

    r = subprocess.run([full_ffmpeg(), "-hide_banner", "-encoders"],
                       capture_output=True, text=True)
    return r.returncode == 0 and " libx264 " in r.stdout


def test_fused_proxy_path_still_works_with_the_fragment(tmp_path):
    """Both outputs of the one decode survive the fragment: the proxy is a
    whole, probe-able file and the sink holds every native frame, while the
    frames are what the fused path yields without it."""
    if not _can_encode_proxy():
        pytest.skip("no libx264 in the proxy ffmpeg build")
    clip = tmp_path / "two.mp4"
    make_mp4(clip, duration=2.0)
    native = _native_frames(clip)

    plain_proxy = tmp_path / "plain" / "two.mp4"
    plain = list(sample_frames(clip, 2.0, fps=2.0, long_edge=LONG_EDGE,
                               proxy_out=plain_proxy))
    sink = tmp_path / "scd.txt"
    proxy = tmp_path / "with" / "two.mp4"
    got = list(sample_frames(clip, 2.0, fps=2.0, long_edge=LONG_EDGE,
                             proxy_out=proxy, full_rate_vf=_scdet(sink)))

    assert len(got) == len(plain) == 4
    for (_, a), (_, b) in zip(got, plain):
        assert np.array_equal(a, b)
    assert proxy.exists() and proxy.stat().st_size > 0
    assert display_dims(proxy) is not None
    blocks = _blocks(sink)
    assert len(blocks) == native
    assert blocks[-1]["frame"] == native - 1
