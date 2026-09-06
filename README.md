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
| `probe` | ffprobe metadata, with one rotation parser that handles both display-matrix side data and the legacy `tags.rotate`, plus audio peak |
| `frames` | decode a clip once, yield evenly sampled frames as BGR arrays, sized through the same rotation and byte-alignment rules as `probe` |
| `signals` | pure functions over frame arrays: brightness, contrast, saturation and hue, sharpness, motion, and a geometric composition centroid. numpy only, no OpenCV |
| `proxy` | HDR-aware, colour-normalised 720p review proxies, tonemapping HLG to limited-range bt709 rather than letting it look wrong with exit code 0 |
| `xmeml` | the FCP7/xmeml boilerplate Premiere actually accepts — rate and samplecharacteristics fragments, sequence skeleton, an atomic write with the exact header, one `<rate>/<ntsc>` parser for readers, and interval carving |
| `segcache` | content-addressed cache for rendered segments, keyed on source, in-point, frame count, speed, HDR-ness and encode profile |
| `store` | atomic record writes — temp file, fsync, `os.replace`, optional `.bak` — and the stage pattern: a directory of one JSON record per item plus a CSV index written only after a complete walk |
| `thumbs` | evenly spaced JPEGs via ffmpeg, using the same filter chain the analysis pass uses, so a thumbnail shows what the analysis saw |
| `media` | which file extensions count as video, audio and images |
| `testing` | tiny synthetic media built from ffmpeg's lavfi sources, plus an xmeml reference builder and the v1-shape validators that pin `xmeml`'s output contract |

## Install

```
pip install "vmcore @ git+https://github.com/hk-geoffschultz/vmcore.git@v0.1.0"
```

Pin a tag or a commit SHA. A consumer that records which version it
measured its thresholds against can reproduce that measurement later; one
that tracks a branch cannot.

For local development against a consumer, an editable install points at a
checkout so edits are live without a reinstall:

```
pip install -e ../vmcore
```

Needs `ffmpeg` and `ffprobe` on PATH. `proxy.full_ffmpeg()` prefers a
static build under `tools/ffmpeg/` when one is present, because the HLG
chain needs `zscale` and slim system builds often lack it.

## Tests

```
make test
```

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
