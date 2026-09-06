"""vmcore must not import an application.

The whole value of this package is that two unrelated pipelines can share
it. One `from bbp import ...` or `from birthday_thing import ...` inside a
module here would silently end that, and it would not show up as a failure
anywhere else — everything still runs fine inside whichever repo made the
mistake. So the rule gets a test rather than a comment.

Stated as an allowlist rather than a denylist. The original version of this
test named the one app it knew about, which would have said nothing about
the second consumer.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

VMCORE = Path(__file__).resolve().parent.parent / "vmcore"
MODULES = sorted(VMCORE.glob("*.py"))

# Everything this package is allowed to depend on. numpy is here because
# frames are handed out as arrays and every consumer wants them that way;
# ffmpeg is an external binary, not an import.
ALLOWED = set(sys.stdlib_module_names) | {"numpy", "vmcore"}


def _imported_roots(path: Path) -> set[str]:
    """Top-level package name of every import in a module."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import: within vmcore by definition.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_there_are_modules_to_check():
    """A glob that quietly matches nothing would make every check below
    pass without testing anything."""
    assert len(MODULES) >= 6


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_module_imports_nothing_outside_the_allowlist(path):
    outside = _imported_roots(path) - ALLOWED
    assert not outside, (
        f"{path.name} imports {sorted(outside)}, which is outside "
        f"stdlib+numpy. vmcore is shared by more than one pipeline: move "
        f"the shared piece down into vmcore, or keep the app-specific "
        f"piece in the app and pass it in as an argument."
    )
