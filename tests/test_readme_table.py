"""The README's failure table names every public function, and only real ones.

A table that says how each function fails is only worth reading if it is
complete. It goes stale the day someone adds a function without a row. This
test reads the table and compares it with the modules in both directions.
It does not check what a row says. The other tests pin the behaviour, and a
row is a summary of them.
"""
from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
# Every module of the package. The heading says "each public function", so
# the list may not leave a module out. `media` defines sets and no function.
MODULES = ("probe", "frames", "signals", "proxy", "thumbs", "store",
           "segcache", "xmeml", "media", "testing")
HEADING = "## How each public function fails by default"


def _table_text() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(HEADING)
    end = text.index("\n## ", start + 1)
    return text[start:end]


def _public(module):
    """Every public function and class defined in `module`, with the public
    methods of each class. Properties are attributes, not calls, and are not
    listed. Dunder methods are not listed either."""
    for name, obj in vars(module).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj):
            yield name
        elif inspect.isclass(obj):
            yield name
            for meth, member in vars(obj).items():
                if not meth.startswith("_") and inspect.isfunction(member):
                    yield f"{name}.{meth}"


def _mentioned(table: str, qualified: str) -> bool:
    """`probe.audio_peak`, or the bare last part in code format, which the
    table uses when a row lists several functions of one module. The bare
    form counts only in a row that also names the module. A function that two
    modules share, such as `write_xmeml`, then cannot be covered by the other
    module's row."""
    module, _, rest = qualified.partition(".")
    last = rest.rsplit(".", 1)[-1]
    for row in (r for r in table.splitlines() if r.startswith("|")):
        if re.search(rf"`{re.escape(module)}\.{re.escape(rest)}`", row):
            return True
        if (re.search(rf"`{re.escape(module)}\.", row)
                and re.search(rf"`(?:[\w.]*\.)?{re.escape(last)}`", row)):
            return True
    return False


def _readme_modules() -> list[str]:
    """The modules the README's "What's in it" table lists."""
    text = README.read_text(encoding="utf-8")
    start = text.index("## What's in it")
    end = text.index("\n## ", start + 1)
    return re.findall(r"^\| `(\w+)` \|", text[start:end], flags=re.M)


def test_the_module_list_is_the_readme_list_and_the_package():
    """The test's own list, the README's module table and the files in the
    package are one set. A new module fails here until it has a row."""
    package = sorted(p.stem for p in (ROOT / "vmcore").glob("*.py")
                     if p.stem != "__init__")
    assert sorted(MODULES) == sorted(_readme_modules()) == package


@pytest.mark.parametrize("name", MODULES)
def test_every_public_function_has_a_row(name):
    module = importlib.import_module(f"vmcore.{name}")
    table = _table_text()
    missing = [f"{name}.{p}" for p in _public(module)
               if not _mentioned(table, f"{name}.{p}")]
    assert missing == [], f"not in the README failure table: {missing}"


def test_every_dotted_name_in_the_table_exists():
    table = _table_text()
    bad = []
    for token in re.findall(r"`((?:%s)\.[\w.]+)`" % "|".join(MODULES), table):
        obj = importlib.import_module(f"vmcore.{token.split('.')[0]}")
        for part in token.split(".")[1:]:
            obj = getattr(obj, part, None)
            if obj is None:
                bad.append(token)
                break
    assert bad == [], f"the table names things that do not exist: {bad}"


def test_the_table_lists_the_all_loud_recipe():
    assert "strict=True, check_exit=True, stall_timeout=<n>" in _table_text()
