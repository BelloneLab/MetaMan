import json
import os

from MetaMan.services.search_service import query_sessions, search_in_project


def _session(root, rel, meta):
    d = os.path.join(root, rel)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _project(tmp_path):
    root = str(tmp_path / "Proj")
    _session(root, "E1/M1/S1", {"Region": "CA1", "Auto: sample rate (Hz)": "30000", "Experimenter": "Ada"})
    _session(root, "E1/M2/S1", {"Region": "CA3", "Auto: sample rate (Hz)": "20000", "Experimenter": "Ada"})
    _session(root, "E2/M3/S1", {"Region": "CA1", "Auto: sample rate (Hz)": "40000", "Experimenter": "Alan"})
    return root


def test_query_equals(tmp_path):
    root = _project(tmp_path)
    res = query_sessions(root, [("Region", "=", "CA1")])
    assert len(res) == 2


def test_query_numeric_gt(tmp_path):
    root = _project(tmp_path)
    res = query_sessions(root, [("Auto: sample rate (Hz)", ">", "25000")])
    assert len(res) == 2  # 30000 and 40000


def test_query_combined_and(tmp_path):
    root = _project(tmp_path)
    res = query_sessions(root, [("Region", "=", "CA1"), ("Auto: sample rate (Hz)", ">", "35000")])
    assert len(res) == 1
    assert res[0]["meta"]["Experimenter"] == "Alan"


def test_query_contains_case_insensitive(tmp_path):
    root = _project(tmp_path)
    assert len(query_sessions(root, [("Experimenter", "contains", "ad")])) == 2


def test_blank_filters_match_all(tmp_path):
    root = _project(tmp_path)
    assert len(query_sessions(root, [("", "=", "")])) == 3


def test_substring_search_still_works(tmp_path):
    root = _project(tmp_path)
    hits = search_in_project(root, "CA3")
    assert any(h["value"] == "CA3" for h in hits)
