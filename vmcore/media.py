"""Which file extensions count as media.

Lives in vmcore rather than in an app's config because these are facts
about file formats, not preferences about a pipeline. Both the events app
and the birthday pipeline want the same answer.
"""
from __future__ import annotations

VIDEO_EXTS = {
    ".mov", ".mp4", ".m4v", ".avi", ".mts", ".m2ts",
    ".mkv", ".mxf", ".3gp", ".webm", ".wmv",
}

AUDIO_EXTS = {".wav", ".mp3", ".aiff", ".aif", ".m4a", ".flac", ".ogg"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}
