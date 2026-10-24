"""Which vmcore is this: the release number.

0.3.0 was three commits: ee0767a, d250867 and 757f484. A number alone
cannot say what a consumer measured against. `pip freeze` prints the commit
of a git install. These tests pin the number itself. It is a plain release
number, it matches pyproject.toml, and importing vmcore does not read it
from the installed metadata.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import vmcore

ROOT = Path(__file__).resolve().parent.parent


def test_version_is_a_plain_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", vmcore.__version__)


def test_version_matches_pyproject():
    """Two literals, one release. Without this the one nobody looks at is
    the one that goes stale."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert vmcore.__version__ == cfg["project"]["version"]


def test_importing_the_package_does_not_import_importlib_metadata():
    """It costs 35 to 40 ms per interpreter (measured, python 3.11), and
    nearly every consumer imports vmcore in a short-lived subprocess."""
    code = ("import sys, vmcore, vmcore.store\n"
            "print('importlib.metadata' in sys.modules)")
    out = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
