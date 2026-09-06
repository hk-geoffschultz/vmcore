"""Suite-wide setup.

No real footage anywhere: every test runs on hand-built inputs or tiny
synthetic media from `vmcore.testing`, which shells out to ffmpeg's lavfi
sources. ffmpeg must be on PATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_clip(tmp_path) -> Path:
    """One real, tiny (64x36) synthetic mp4 — enough for anything that just
    needs a probe-able file on disk."""
    from vmcore.testing import make_mp4

    p = tmp_path / "clip.mp4"
    make_mp4(p, duration=1.0)
    return p
