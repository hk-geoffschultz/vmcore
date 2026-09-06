"""Review proxies: 720p H.264, colour-normalised, one per clip.

Two ways to get them:

- Fused into ingest: `sample_frames(..., proxy_out=...)` adds a second
  output branch to the decode ffmpeg the analysis pass already runs, so
  the proxy costs one encode, not a second decode of the whole library.
- Standalone backfill for an already-ingested library: `build_proxy` per
  clip, decode + encode from scratch.

Colour: phone footage regularly mixes HLG HDR and full-range video in the
same shoot. A proxy that ignores either looks wrong with exit code 0, so
every proxy is normalised to limited-range bt709 SDR here (verified
against every HLG clip in the birthday-pipeline library this module was
forked from - 7/431, all iPhone). The HLG chain needs zscale, which the
slim system ffmpeg lacks - use `full_ffmpeg()`, which prefers the static
build in tools/ffmpeg/.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROXY_LONG_EDGE = 1280
PROXY_CRF = 23
PROXY_PRESET = "veryfast"

_ROOT = Path(__file__).resolve().parent.parent

# HLG -> SDR: linearise, hable tonemap, back to bt709 limited. Validated
# against all 7 HLG clips in the library (exit 0, bt709 output).
TONEMAP = ("zscale=transfer=linear:npl=100,tonemap=hable:desat=0,"
           "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv")

HDR_TRANSFERS = ("arib-std-b67", "smpte2084")


def full_ffmpeg() -> str:
    """The full-featured ffmpeg, never the (possibly slim) PATH one.

    The system binary stays untouched for probe/faces (their HEIC/HEVC
    behaviour is load-bearing); the proxy/colour path takes the static
    build when present.
    """
    cand = _ROOT / "tools" / "ffmpeg" / "ffmpeg"
    return str(cand) if cand.exists() else "ffmpeg"


def color_transfer(path: Path) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer", "-of", "json",
             str(path)],
            capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout)["streams"][0].get("color_transfer", "") or ""
    except Exception:
        return ""


def proxy_vf(transfer: str, long_edge: int = PROXY_LONG_EDGE) -> str:
    """Filter chain for the proxy branch of a given source clip."""
    scale = (f"scale='if(gt(iw,ih),{long_edge},-2)':"
             f"'if(gt(iw,ih),-2,{long_edge})':"
             "in_range=auto:out_range=tv")
    head = TONEMAP + "," if transfer in HDR_TRANSFERS else ""
    return f"{head}{scale},format=yuv420p"


def proxy_encode_args(out: Path) -> list[str]:
    return ["-c:v", "libx264", "-preset", PROXY_PRESET, "-crf", str(PROXY_CRF),
            "-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-y", str(out)]


def build_proxy(path: Path, out: Path, timeout: int = 1800) -> bool:
    """Standalone proxy render for one clip. Returns True on success."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = proxy_vf(color_transfer(path))
    r = subprocess.run(
        [full_ffmpeg(), "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-map", "0:v:0", "-vf", vf]
        + proxy_encode_args(out),
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        return False
    return True
