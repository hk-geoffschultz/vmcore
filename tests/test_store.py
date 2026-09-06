from __future__ import annotations

import json

import vmcore.store as store_mod
from vmcore.store import Store


def test_store_json_round_trip(tmp_path):
    st = Store(tmp_path / "work")
    assert st.work.is_dir()
    p = st.work / "x.json"
    obj = {"a": [1, 2, {"b": None}], "c": "café", "d": 1.5}
    st.write_json(p, obj)
    assert st.read_json(p) == obj
    assert st.read_json(st.work / "missing.json", default="dflt") == "dflt"
    p.write_text("{not json")
    assert st.read_json(p, default={"fallback": 1}) == {"fallback": 1}


# post-merge: pins A7 (atomic_write_text in bbp/store.py)
def test_atomic_write_text_leaves_no_tmp_and_backs_up(tmp_path):
    p = tmp_path / "state.json"
    store_mod.atomic_write_text(p, "one")
    assert p.read_text() == "one"
    assert not list(tmp_path.glob("*.tmp"))
    assert not (tmp_path / "state.json.bak").exists()

    store_mod.atomic_write_text(p, "two", backup=True)
    assert p.read_text() == "two"
    assert (tmp_path / "state.json.bak").read_text() == "one"
    assert not list(tmp_path.glob("*.tmp"))


# post-merge: pins A7 (atomic_write_json in bbp/store.py)
def test_atomic_write_json_round_trip(tmp_path):
    p = tmp_path / "nested" / "obj.json"
    obj = {"k": [1, 2.5, "s"], "n": None, "u": "é"}
    store_mod.atomic_write_json(p, obj)
    assert json.loads(p.read_text()) == obj
    assert not list(p.parent.glob("*.tmp"))
