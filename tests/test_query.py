"""Tests for the project query/analysis engine (MetaMan.services.query)."""

import json
import os

from MetaMan.services import query as Q
from MetaMan.services.query import ProjectQuery, compare


# ── operator semantics ───────────────────────────────────────────────────────

def test_operators_text_numeric_date():
    assert compare("CA1", "=", "ca1")            # case-insensitive equality
    assert compare("NAc", "contains", "na")
    assert compare("30000", ">", "25000")        # numeric
    assert not compare("30000", "<", "25000")
    assert compare("2026-06-17", "between", "2026-06-01..2026-06-30")  # date-aware
    assert compare("healthy", "in", "healthy,object,sick")
    assert compare("sick", "not in", "healthy,object")
    assert compare("51542", "regex", r"^515\d+$")
    assert compare("obj", "startswith", "obj") and compare("object", "endswith", "ject")
    assert compare("x", "exists", "") and compare("", "missing", "")


# ── dual-dialect, schema-aware discovery ─────────────────────────────────────

def _write(path, name, meta):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _subject_first_project(tmp_path):
    """project -> subject -> experiment -> session, mixing both metadata dialects
    (like the real mPFC-NAc layout)."""
    root = str(tmp_path / "Proj")
    os.makedirs(root, exist_ok=True)
    # subject-first schema sidecar
    schema = {"raw_levels": [
        {"key": "project", "enabled": True, "label": "Project"},
        {"key": "subject", "enabled": True, "label": "Subject"},
        {"key": "experiment", "enabled": True, "label": "Experiment"},
        {"key": "session", "enabled": True, "label": "Session"},
    ]}
    schema["processed_levels"] = list(schema["raw_levels"])
    _write(root, "_metaman_structure.json", schema)

    # canonical metadata.json
    _write(os.path.join(root, "51542/exp1/object"), "metadata.json",
           {"Region": "NAc", "Condition": "object", "DateTime": "2026-06-17"})
    # acquisition dialect with a *deliberately wrong* Subject we expect the folder to override
    _write(os.path.join(root, "51542/exp1/healthy"), "51542_healthy_metadata.json",
           {"animal_id": "WRONG", "condition": "healthy", "date": "2026-06-18", "arena": "Arena 1"})
    _write(os.path.join(root, "51543/exp1/sick"), "51543_sick_metadata.json",
           {"animal_id": "51543", "condition": "sick", "date": "2026-06-19"})
    return root


def test_discovery_reads_both_dialects(tmp_path):
    root = _subject_first_project(tmp_path)
    recs = list(Q.iter_sessions(root))
    assert len(recs) == 3
    # acquisition dialect was read (old metadata.json-only search would miss it)
    files = {r["metadata_file"] for r in recs}
    assert "metadata.json" in files and "51542_healthy_metadata.json" in files


def test_folder_identity_overrides_bad_metadata(tmp_path):
    root = _subject_first_project(tmp_path)
    pq = ProjectQuery(root)
    healthy = pq.where("session", "=", "healthy").records()[0]
    # folder says subject 51542, even though the file said animal_id="WRONG"
    assert healthy["subject"] == "51542"
    assert healthy["meta"]["Subject"] == "51542"


def test_schema_aware_identity(tmp_path):
    root = _subject_first_project(tmp_path)
    pq = ProjectQuery(root)
    assert sorted(pq.values("subject")) == ["51542", "51543"]
    assert pq.values("experiment") == ["exp1"]
    assert sorted(pq.values("session")) == ["healthy", "object", "sick"]


def test_chained_where_and_or(tmp_path):
    root = _subject_first_project(tmp_path)
    pq = ProjectQuery(root)
    assert pq.where("subject", "=", "51542").count() == 2
    assert pq.where("Condition", "in", "object,sick").count() == 2
    # OR branch
    both = pq.or_where("session", "=", "healthy").or_where("session", "=", "sick")
    assert both.count() == 2


def test_dataframe_and_summary(tmp_path):
    root = _subject_first_project(tmp_path)
    pq = ProjectQuery(root)
    df = pq.to_dataframe()
    assert list(df.columns[:4]) == ["project", "subject", "experiment", "session"]
    assert len(df) == 3
    s = pq.summary()
    assert s["sessions"] == 3 and s["subjects_count"] == 2
    assert s["date_range"] == ["2026-06-17", "2026-06-19"]


def test_to_csv_roundtrip(tmp_path):
    root = _subject_first_project(tmp_path)
    out = str(tmp_path / "out.csv")
    ProjectQuery(root).where("subject", "=", "51542").to_csv(out)
    text = open(out, encoding="utf-8").read()
    assert "subject" in text.splitlines()[0]
    assert text.count("\n") >= 3  # header + 2 data rows


def test_group_counts(tmp_path):
    root = _subject_first_project(tmp_path)
    counts = ProjectQuery(root).group_counts("subject")
    assert counts == {"51542": 2, "51543": 1}
