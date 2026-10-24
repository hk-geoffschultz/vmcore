# vmcore

Input-agnostic video plumbing. Every function here answers a question about
*media* — what are this file's dimensions, how do I get frames out of it,
what does Premiere need this XML to look like, have I rendered this exact
segment before. Nothing here knows what a good clip is, what a montage is
for, or who is in frame.

That boundary is the point. It is what lets two unrelated edit pipelines
share the same code without either dragging its domain model into the
other, and it is enforced by a test rather than by good intentions:
`vmcore` may import nothing outside the standard library and numpy.

## What's in it

| module | what it does |
|---|---|
| `probe` | ffprobe metadata, with two rotation readers on purpose. `ClipInfo.rotation` and the width/height swap in `probe_clip` honour the display matrix and then the legacy `tags.rotate`. `display_dims`, which sizes the raw pipe in `frames`, honours the display matrix only, because that is all ffmpeg autorotates from, so a tag-only file is described as portrait and decoded as landscape. Plus audio peak: `audio_peak_db` is -91.0 for silence and -99.0 when nothing was measured |
| `frames` | decode a clip once, yield evenly sampled frames as BGR arrays, sized through the same rotation and byte-alignment rules as `probe`. `full_rate_vf` rides a filter fragment on that decode ahead of the `fps=` thinning (a scene-change detector, a `metadata=...:file=` sink), where it sees every native frame; the decode is then drained to EOF rather than cut off, so the file the filter writes is complete when ffmpeg finishes on its own. vmcore splices it in verbatim and never reads what it wrote; a fragment ffmpeg rejects raises `DecodeError` with its reason instead of yielding nothing. The opt-ins `strict`, `check_exit` and `stall_timeout` make an unsizable, a dead or a wedged decode raise. Damage that ffmpeg tolerates stays silent; see the failure table below |
| `signals` | pure functions over frame arrays: brightness, contrast, saturation and hue, sharpness, motion, and a geometric composition centroid. numpy only, no OpenCV. Values depend on the decode `long_edge` and are not comparable with another pipeline's measures; read the module docstring before reusing a threshold |
| `proxy` | HDR-aware, colour-normalised 720p review proxies, tonemapping HLG to limited-range bt709 rather than letting it look wrong with exit code 0. `build_proxy` encodes to a hidden partial beside the output and renames it on success, so the output is never half written. A builder locks its partial while it runs. A caller killed outright leaves its partial, and the next build of the same output removes it when no live builder holds the lock. A path within about 45 bytes of the system's PATH_MAX gets a shorter hidden name. A path within about 30 bytes is built in place |
| `xmeml` | the FCP7/xmeml boilerplate Premiere actually accepts — rate and samplecharacteristics fragments, sequence skeleton, an atomic write with the exact header, one `<rate>/<ntsc>` parser for readers, and interval carving |
| `segcache` | content-addressed cache for rendered segments: a store, plus `segment_key` keyed on source, in-point, frame rate, frame count, speed, HDR-ness, encode profile and crop centre, and not on the ffmpeg binary or any constant spliced into the filter graph. Read its module docstring before choosing it — an enumerated key is structurally behind the command it stands for, and hashing the ffmpeg argv is the alternative that cannot drift |
| `store` | atomic record writes — a temp file named for the writer (`<name>.<pid>.<thread>.tmp`, so concurrent writers of one file cannot tear it), fsync, `os.replace`, and an optional `.bak` that lands the same way — and the stage pattern: a directory of one JSON record per item plus a CSV index written only after a complete walk |
| `thumbs` | evenly spaced JPEGs via ffmpeg, using the same filter chain the analysis pass uses, so framing and scale match. Colour does not match for BT.709 and BT.2020 sources: the JPEG carries no matrix, so decoders read it as BT.601 |
| `media` | which file extensions count as video, audio and images |
| `testing` | tiny synthetic media built from ffmpeg's lavfi sources, plus an xmeml reference builder and the v1-shape validators that pin `xmeml`'s output contract. The validators report every problem as a list and raise `AssertionError` explicitly, so `python -O` cannot switch them off; `tests/test_testing.py` pins each rule's message |

## How each public function fails by default, and which opt-in changes that

Most functions here keep going on bad input. They return an empty value,
`None`, `False` or -99.0 and say nothing, which suits a walk over hundreds
of clips. This table gives each public function's default, the opt-in that
changes it, and the exception that opt-in raises. A function marked pure
cannot fail on a well-formed argument. It covers every module that defines
a function. `media` defines only sets of extensions.

| function | by default | opt-in | new exception |
|---|---|---|---|
| `probe.source_stamp` | Returns `(size, mtime_ns)`. A missing file raises `FileNotFoundError`. | none | none |
| `probe.display_dims` | Returns the post-rotation `(w, h)`. Returns `None` for no video stream, a zero size, a probe timeout or no ffprobe. Never raises. | `sample_frames(strict=True)` raises on `None` | `UnsizableInput` |
| `probe.scaled_dims` | Pure arithmetic. A non-positive size raises `ValueError`. | none | none |
| `probe.audio_peak` | Returns dBFS. Returns -99.0 for no audio, an unreadable file or no level printed. Raises `subprocess.TimeoutExpired` after `timeout` and `FileNotFoundError` with no ffmpeg. | none | none |
| `probe.probe_clip` | Returns a `ClipInfo`. A file ffprobe cannot read becomes a row with `error` set, zeros, and `audio_peak_db` of -99.0. The duration is 0.0 when the container gives none. Raises `FileNotFoundError` with no ffprobe. | pass its duration to `sample_frames(strict=True)` | `UnsizableInput` |
| `probe.ClipInfo.as_row` | Pure. Returns the fields as a dict. | none | none |
| `probe.find_clips` | Returns the sorted video files. Skips dotfiles. A missing folder gives `[]`. | none | none |
| `frames.analysis_vf`, `frames.thinned_fps` | Pure. `thinned_fps` returns NaN for a NaN duration and 0.0 for an infinite one. | none | none |
| `frames.metadata_print_blocks` | Returns a list of dicts. Lines before the first header are ignored. A malformed header raises `KeyError` or `ValueError`. | none | none |
| `frames.decode_cmd` | Returns the argv. The fused form, with `proxy_out`, runs one ffprobe. Raises nothing. | `proxy_ffmpeg` names the fused binary. A bad name raises at once. | `FileNotFoundError`, `IsADirectoryError`, `PermissionError`, `ValueError`, `TypeError` |
| `frames.sample_frames` | A generator, so its argument errors surface at the first `next`. Yields nothing for a file it cannot size and for a duration of zero or less. Yields nothing for a NaN or infinite duration too, unless `full_rate_vf` or `check_exit` is set. Then it raises `DecodeError`, because ffmpeg rejects the rate. An ffmpeg that dies gives a short clip or none. A wedged ffmpeg is waited on for ever. With `full_rate_vf`, a non-zero exit with nothing yielded raises. | `strict`, `check_exit`, `stall_timeout`, and `proxy_ffmpeg` with `proxy_out`. To hear every failure ffmpeg reports: `strict=True, check_exit=True, stall_timeout=<n>`. | `UnsizableInput`, `DecodeError`, `DecodeTimeout`. `ValueError` for a `stall_timeout` that is not positive, `TypeError` for one that is not a number. `FileNotFoundError`, `IsADirectoryError`, `PermissionError`, `ValueError`, `TypeError` for a bad `proxy_ffmpeg`. |
| `signals.luma` | Returns the grey channel. A frame whose last dimension is not 3 raises `ValueError`. It does not check the number of dimensions. `None` raises `AttributeError`. | none | none |
| `signals.sharpness` | Returns 0.0 for a frame under 3 pixels on a side. A frame whose last dimension is not 3 raises `ValueError`. A 2-D array that ends in 3 raises `IndexError`, and a 4-D batch gives 0.0. `None` raises `AttributeError`. | none | none |
| `signals.brightness`, `contrast`, `saturation_and_hue`, `composition_centroid`, `motion` | Take a list of frames. An empty list gives `0.0`, or a pair of zeros, or `(0.5, 0.5)` for the centroid. A frame whose last dimension is not 3 raises `ValueError`, but `saturation_and_hue` does not check the shape. A `None` in the list raises `AttributeError`. `motion` looks at no frame when given fewer than two, so it raises nothing for a bad one. It raises `ValueError` for frames of different sizes. An array in place of a list raises `ValueError`, except in `motion`. A frame with no pixels gives NaN and a `RuntimeWarning`, except in the centroid. | none | none |
| `proxy.full_ffmpeg` | Returns the `tools/ffmpeg/ffmpeg` above the package if it exists, else `ffmpeg`. | `proxy_ffmpeg` names a binary, which is checked. | `FileNotFoundError`, `IsADirectoryError`, `PermissionError`, `ValueError`, `TypeError` |
| `proxy.color_transfer` | Returns the transfer name. Returns '' on any failure, including no ffprobe. That selects the SDR chain. | none | none |
| `proxy.proxy_vf`, `proxy.proxy_encode_args` | Pure. | none | none |
| `proxy.build_proxy` | Returns `True`. Returns `False` when ffmpeg fails or writes nothing, and a previous proxy is removed. Raises `subprocess.TimeoutExpired` after `timeout`, `FileNotFoundError` with no ffmpeg, and `OSError` for an I/O failure such as a name too long. After an exception `out` is as it was, unless the path is so near PATH_MAX that no hidden name fits. An `out` that is a directory raises `IsADirectoryError` before any work. | `proxy_ffmpeg` names the binary. A caller that wants a timeout to count as a failed clip catches `TimeoutExpired`. | `FileNotFoundError`, `IsADirectoryError`, `PermissionError`, `ValueError`, `TypeError` for a bad `proxy_ffmpeg`, raised before anything is written. |
| `thumbs.thumb_paths` | Pure. | none | none |
| `thumbs.extract` | Returns the thumbnails that exist. Returns `[]` for a duration of zero or less, a `count` below 1, or a failed run that wrote none. Raises `subprocess.TimeoutExpired` after `timeout` and `FileNotFoundError` with no ffmpeg. | none | none |
| `store.atomic_write_text`, `store.atomic_write_json` | Return `None`. The target is whole or untouched, and no temp file is left. An I/O failure raises `OSError`. Text that is not a `str` raises `TypeError`, and text the encoding cannot hold raises `UnicodeEncodeError`. JSON that cannot be dumped raises `TypeError`, or `ValueError` for a circular structure, before anything is written. NaN is written as `NaN` unless `allow_nan=False` is passed, which raises `ValueError`. | `backup=True` keeps `<name>.bak` | none |
| `store.Store` | Creating one makes the work directory. A failure raises `OSError`. | none | none |
| `store.Store.read_json`, `store.Store.read_record` | Return `default` for a missing file, invalid JSON or an `OSError`. A file that is not UTF-8 raises `UnicodeDecodeError`. Invalid JSON means ordinary invalid JSON: a file of 100000 open brackets raises `RecursionError`. | none | none |
| `store.Store.records`, `store.Store.read_index` | `records` skips records with invalid JSON and records that are not objects. A record that is not UTF-8 raises `UnicodeDecodeError` while iterating. `read_index` returns `[]` when the index does not exist. | none | none |
| `store.Store.write_json`, `store.Store.write_record`, `store.Store.write_index` | Create directories and write atomically. An I/O failure raises `OSError`. `write_json` and `write_record` raise what `atomic_write_json` raises for a value that cannot be dumped. | none | none |
| `store.Store.stage_dir`, `store.Store.record_path` | Create the stage directory and return it, or a path inside it. A file in the way raises `FileExistsError`. | none | none |
| `store.Store.index_path` | Pure. Builds the path and creates nothing. | none | none |
| `segcache.profile_hash` | Pure. | none | none |
| `segcache.segment_key` | Returns the key. A missing `fps` raises `TypeError`, `cx` without `cy` raises `ValueError`, and a missing source raises `FileNotFoundError`. | none | none |
| `segcache.SegmentCache.path_for` | Pure. Creating a `SegmentCache` makes its directory. | none | none |
| `segcache.SegmentCache.get` | Returns a `Path`. Returns `None` for a miss and for an empty file. | none | none |
| `segcache.SegmentCache.put` | Returns the cache's own path. A failure raises `OSError`, such as `FileNotFoundError` for a missing source, and the temp file is removed. | none | none |
| `xmeml.write_rate`, `write_samplechars`, `sequence_skeleton` | Build elements. No failure mode of their own. | none | none |
| `xmeml.write_xmeml` | Writes atomically with the exact header. An I/O failure raises `OSError` and the target is untouched. | none | none |
| `xmeml.parse_rate` | Returns fps. A missing timebase element gives `default`. Text that is not a number raises `ValueError`. An empty timebase raises `TypeError`. | none | none |
| `xmeml.carve`, `xmeml.overlaps_any` | Pure. | none | none |
| `testing.make_mp4`, `make_cut_mp4`, `make_wav` | Write a tiny file with ffmpeg. Raise `FileNotFoundError` with no ffmpeg and `subprocess.CalledProcessError` when ffmpeg fails, as it does for a missing folder or a negative duration. | none | none |
| `testing.build_xmeml`, `testing.write_xmeml` | Return the XML text, or write it and return the path. A clip without `name`, `start` or `end`, or a marker without `in`, raises `KeyError`. `None` for the tracks raises `TypeError`. `write_xmeml` into a missing folder raises `FileNotFoundError`. | none | none |
| `testing.video_clipitems`, `testing.audio_clipitems` | Return `(track, clipitem)` pairs. `None` raises `AttributeError`. | none | none |
| `testing.v1_shape_problems`, `testing.audio_shape_problems` | Return a list of problems, empty when the shape is right. A broken document is reported and does not raise. `None` raises `AttributeError`, and so does an `ElementTree` passed to `v1_shape_problems`. | none | none |
| `testing.assert_v1_shape`, `testing.assert_audio_shape` | Return `None`, or raise `AssertionError` listing every problem. They raise explicitly, so `python -O` does not switch them off. | none | none |
| `testing.parse_xml` | Returns the root element. Raises `FileNotFoundError` for a missing file and `AssertionError` for an empty file or a wrong header line. A malformed body raises `ParseError` from `xml.etree.ElementTree`. A file that is not UTF-8 raises `UnicodeDecodeError`. | none | none |

The exceptions form one family. `DecodeError` is a `RuntimeError`.
`UnsizableInput` and `DecodeTimeout` are subclasses of it, so a walker that
must continue catches `DecodeError` per clip. `UnsizableInput.reason` is
`"dimensions"` or `"duration"`. `DecodeTimeout.phase` is `"stall"` for a
`stall_timeout` and `"finish"` for an ffmpeg that outlived `timeout` after
its last frame. `build_proxy`, `audio_peak` and `extract` raise
`subprocess.TimeoutExpired`, which is not a `DecodeError`.

To hear every failure that ffmpeg reports, pass three arguments:

```python
for t, frame in sample_frames(path, duration,
                              strict=True, check_exit=True, stall_timeout=60):
    ...
```

Choose `stall_timeout` larger than ffmpeg needs for one sampling interval,
plus its start-up. Frames received before a raise are a prefix of the clip.
A consumer that records a result per clip must discard them.

Damage that ffmpeg tolerates stays silent, with all three set. A clip whose
payload is damaged part way in can decode with exit 0 and give fewer frames
than it should. That is a short clip, not an error. A consumer that needs the
count compares it with what it expected.

## Install

```
pip install "vmcore @ git+https://github.com/hk-geoffschultz/vmcore.git@<sha>"
```

Replace `<sha>` with the commit your consumer measured against. The line
fails as written, on purpose, so a copied example cannot install an older
vmcore than you meant. `git ls-remote https://github.com/hk-geoffschultz/vmcore.git HEAD`
prints the current tip. Pin a commit SHA, not a tag: none exists on origin
(the session git proxy that pushes this repo blocks tag pushes, so
`v0.1.0` lives only in a local clone and an install that names it fails).
A consumer that records which commit it measured its thresholds against
can reproduce that measurement later; one that tracks a branch cannot.
Pin the squash commit of a release, and no commit of an unsquashed series.
The commits behind a release are merged as one squash commit, so each
release has one SHA. `pyproject.toml` states the rule and lists what each
release changed by default.

`vmcore.__version__` names the release. A release number can span several
commits, and 0.3.0 was three, so record the commit you pinned as well.
`pip freeze` prints the full SHA of a git install. That was measured with
pip 24.0 and a `git+file` URL. The build needs setuptools 77 or later. pip
fetches it in an isolated build. With `--no-build-isolation`, upgrade
setuptools first.

For local development against a consumer, an editable install points at a
checkout so edits are live without a reinstall:

```
pip install -e ../vmcore
```

Needs `ffmpeg` and `ffprobe` on PATH. Analysis decodes, `audio_peak` and
thumbnails always run the PATH `ffmpeg`.

Proxies and the HLG colour chain need `zscale`, which slim system builds
often lack, so they run a separate binary that `proxy.full_ffmpeg()` picks.
First match wins:

1. The `proxy_ffmpeg=` argument of `build_proxy`, `decode_cmd` or
   `sample_frames`. The fused decode (`proxy_out` set) uses it; the
   analysis-only decode ignores it.
2. `tools/ffmpeg/ffmpeg` in the directory above the `vmcore` package. That
   is the vmcore checkout for an editable install (`pip install -e`) and
   `site-packages` for a normal one. vmcore ships no `tools/`, so this
   finds a consumer's build only if the package is vendored beside it.
3. `ffmpeg` on PATH.

No environment variable is read. A consumer that wants one reads its own
and passes the value as `proxy_ffmpeg`. A path named in 1 must exist and be
executable, and `ffprobe` must be on PATH. It raises otherwise and does not
fall through to a lower step. An installed consumer should pass its build
explicitly.

## Tests

```
pip install -e ".[dev]"
make test
```

Needs `ffmpeg` and `ffprobe` on PATH, numpy, and pytest, the `dev` extra.
The synthetic clips are ffmpeg lavfi sources, encoded with libx264 and
falling back to mpeg4. Where only `python3` exists, run
`make test PYTHON=python3`.

`make test` runs the fast subset, which is everything not marked `slow`.
`make test-all` runs every test. Run it for a release and after a change
to `frames.py` or `proxy.py`. A test is marked `slow` when it waits on a
clock on purpose: a stall limit, a timeout or a real-time decode.

Every `test_default_` test and the 143 tests of 757f484 are in the fast
subset. The file `tests/conftest.py` refuses to collect a `test_default_`
test marked slow. That is not every pin of the behaviour that 757f484 had.
The pins for the timeout state, the stall clock and the settle wait run only
under `make test-all`. Run that before a release.

Tests skip, and say why in the skip message, in these cases:

- The proxy ffmpeg has no libx264.
- The platform has no `flock`.
- The PATH ffmpeg lacks the `zscale` or `tonemap` filter.
- The platform has no pseudo-terminal.
- The filesystem reports every file as executable.
- ffmpeg is not deterministic from run to run.
- ffmpeg exits non-zero on a half-zeroed file.

The full suite is 516 tests. It took 113 seconds on Python 3.11, 115 on 3.12
and 116 on 3.13. The fast subset is 485 tests and took 62 seconds on 3.11.
Each ran once with ffmpeg 6.1.1. Several tests run real ffmpeg through kills,
timeouts and a pseudo-terminal.

No model downloads, no network, no real footage. Every test runs on
hand-built inputs or tiny synthetic media from `vmcore.testing`.

## Provenance

Extracted from two working pipelines that had independently grown the same
infrastructure: a birthday-montage tool and an event/behind-the-scenes edit
pipeline. Several comments still reference files that live in those repos
rather than this one — `emit.py`, `rough_cut.py`, `preview.py`,
`deliver.py`, and rule numbers from a pipeline's own CLAUDE.md. Those
references are left as they were rather than tidied away, because they
record why a piece of code is exact rather than approximate, and a
paraphrase would lose that.

Some of the reasoning is worth reading before changing anything:
`xmeml.py` on why the boilerplate is byte-exact, `segcache.py` on what is
deliberately *not* cached, and `signals.py` on HSV saturation being
unstable on dark pixels.
