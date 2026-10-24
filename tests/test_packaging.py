"""Facts in pyproject.toml that break a build rather than a test.

Nothing else in the suite reads pyproject.toml, so a floor lowered by hand,
or a dependency added without a matching import, passes every other test
and fails on someone else's machine. Measured 2026-10-22: setuptools 68.2.2
through 76.1.0 refuse `license = "MIT"` with an error that names neither
setuptools nor a version; 77.0.1 and later build it.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _setuptools_floor(cfg: dict) -> tuple[int, ...] | None:
    for req in cfg["build-system"]["requires"]:
        if re.match(r"setuptools\b", req):
            m = re.search(r">=\s*(\d+(?:\.\d+)*)", req)
            return tuple(int(x) for x in m.group(1).split(".")) if m else None
    return None


def test_a_pep_639_license_string_has_a_setuptools_that_can_read_it():
    cfg = _config()
    if not isinstance(cfg["project"].get("license"), str):
        return  # the table form builds on old setuptools; no floor needed
    floor = _setuptools_floor(cfg)
    assert floor is not None, "build-system.requires names no setuptools floor"
    assert floor >= (77,), (
        f"license is an SPDX string but setuptools>={'.'.join(map(str, floor))}"
        " cannot read it; 77 is the first release that can")


def test_the_runtime_dependencies_are_numpy_and_nothing_else():
    """pyproject's own comment says test_no_app_imports.py keeps this list
    short. That test checks IMPORTS against stdlib + numpy; it cannot see a
    dependency declared here and never imported. This one can."""
    names = {re.split(r"[<>=!~\s\[;]", dep, maxsplit=1)[0].lower()
             for dep in _config()["project"]["dependencies"]}
    assert names == {"numpy"}
