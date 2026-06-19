"""Tests for the Qt-free preprocessing-ops API (MetaMan.services.preprocessing_ops)."""

import json
import os

from MetaMan.services import preprocessing_ops as pp


def _write(path, name, meta):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _project(tmp_path):
    """project -> subject -> experiment -> session, with a canonical session that
    already carries steps and an acquisition-dialect session that has none."""
    root = str(tmp_path / "Proj")
    schema = {"raw_levels": [
        {"key": "project", "enabled": True, "label": "Project"},
        {"key": "subject", "enabled": True, "label": "Subject"},
        {"key": "experiment", "enabled": True, "label": "Experiment"},
        {"key": "session", "enabled": True, "label": "Session"},
    ]}
    schema["processed_levels"] = list(schema["raw_levels"])
    _write(root, "_metaman_structure.json", schema)
    _write(os.path.join(root, "M1/exp1/s1"), "metadata.json", {
        "Recording": "Neuropixels",
        "preprocessing": [
            {"name": "spike_sorting", "status": "completed"},
            {"name": "curation", "status": "planned"},
        ],
    })
    _write(os.path.join(root, "M2/exp1/s1"), "M2_s1_metadata.json",
           {"animal_id": "M2", "Recording": "Neuropixels"})
    return root


def test_status_normalisation():
    assert pp.normalize_status("in_progress") == pp.ONGOING
    assert pp.normalize_status("DONE") == pp.COMPLETED
    assert pp.normalize_status("") == pp.PLANNED


def test_load_and_read(tmp_path):
    root = _project(tmp_path)
    s1 = os.path.join(root, "M1/exp1/s1")
    assert pp.step_status(s1, "spike_sorting") == pp.COMPLETED
    assert pp.step_status(s1, "curation") == pp.PLANNED
    assert pp.step_status(s1, "nonexistent") == ""


def test_progress_summary(tmp_path):
    root = _project(tmp_path)
    summ = pp.progress_summary(root)
    assert summ["steps_total"] == 2
    assert summ["steps_completed"] == 1
    assert summ["percent_complete"] == 50.0
    assert summ["by_step"]["spike_sorting"]["completed"] == 1


def test_set_step_status_writes_canonical(tmp_path):
    root = _project(tmp_path)
    s2 = os.path.join(root, "M2/exp1/s1")          # acquisition-only session
    assert pp.set_step_status(s2, "spike_sorting", "ongoing") is True
    # a canonical metadata.json is created and carries the step
    assert os.path.isfile(os.path.join(s2, "metadata.json"))
    assert pp.step_status(s2, "spike_sorting") == pp.ONGOING


def test_apply_step_template_seeds_defaults(tmp_path):
    root = _project(tmp_path)
    touched = pp.apply_step_template(root)          # NPX steps from Recording hint
    assert touched >= 1
    s2 = os.path.join(root, "M2/exp1/s1")
    names = {s["name"] for s in pp.load_steps(s2)}
    assert {"spike_sorting", "curation", "histology"} <= names


def test_bulk_set_status_with_where(tmp_path):
    root = _project(tmp_path)
    changed = pp.bulk_set_status(root, "curation", "completed", where=("subject", "=", "M1"))
    assert len(changed) == 1
    assert pp.step_status(os.path.join(root, "M1/exp1/s1"), "curation") == pp.COMPLETED
    # M2 untouched by the where-clause
    assert pp.step_status(os.path.join(root, "M2/exp1/s1"), "curation") == ""


def test_status_table_wide(tmp_path):
    root = _project(tmp_path)
    pp.apply_step_template(root)
    t = pp.status_table(root)
    assert "spike_sorting" in t.columns
    assert "pp_percent" in t.columns
    assert len(t) == 2
