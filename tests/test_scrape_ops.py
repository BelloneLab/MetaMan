"""Tests for the project-wide auto-scrape service (MetaMan.services.scrape_ops)."""

import json
import os

from MetaMan.io_ops import load_session_metadata
from MetaMan.services import scrape_ops


def _write(path, name, content):
    os.makedirs(path, exist_ok=True)
    p = os.path.join(path, name)
    with open(p, "w", encoding="utf-8") as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f)
        else:
            f.write(content)
    return p


def _project(tmp_path):
    """subject-first project with an acquisition-only session and a junk one."""
    root = str(tmp_path / "Proj")
    schema = {"raw_levels": [
        {"key": "project", "enabled": True, "label": "Project"},
        {"key": "subject", "enabled": True, "label": "Subject"},
        {"key": "experiment", "enabled": True, "label": "Experiment"},
        {"key": "session", "enabled": True, "label": "Session"},
    ]}
    schema["processed_levels"] = list(schema["raw_levels"])
    _write(root, "_metaman_structure.json", schema)

    s = os.path.join(root, "M1/exp1/s1")
    # acquisition dialect with a deliberately wrong animal_id
    _write(s, "s1_metadata.json", {"animal_id": "WRONG", "notes": "hi"})
    # a fake SpikeGLX meta so modality + sample rate are detected
    _write(s, "rec_t0.imec0.ap.meta", "imSampRate=30000\nnSavedChans=385\n")
    _write(s, "data.txt", "x")

    # junk subject (skipped by default); acquisition-only so "skipped" means no
    # canonical metadata.json gets created for it
    _write(os.path.join(root, "test/exp1/s1"), "s1_metadata.json", {"x": 1})
    return root


def test_scrape_creates_canonical_and_fixes_identity(tmp_path):
    root = _project(tmp_path)
    stats = scrape_ops.scrape_project(root)
    assert stats["updated"] == 1 and stats["skipped"] == 0
    s = os.path.join(root, "M1/exp1/s1")
    meta = load_session_metadata(s)
    assert meta is not None                       # canonical metadata.json created
    assert meta["Subject"] == "M1"                # folder overrides the wrong file
    assert "Neuropixels" in meta.get("Auto: modality", "")
    assert meta.get("Auto: sample rate (Hz)") == "30,000"
    assert "notes" in meta                        # user fields preserved


def test_skips_junk_subjects(tmp_path):
    root = _project(tmp_path)
    scrape_ops.scrape_project(root)
    # the test/ subject session was skipped -> no canonical metadata.json written
    assert load_session_metadata(os.path.join(root, "test/exp1/s1")) is None


def test_idempotent(tmp_path):
    root = _project(tmp_path)
    assert scrape_ops.scrape_project(root)["updated"] == 1
    second = scrape_ops.scrape_project(root)
    assert second["updated"] == 0 and second["skipped"] == 1


def test_force_rescan(tmp_path):
    root = _project(tmp_path)
    scrape_ops.scrape_project(root)
    forced = scrape_ops.scrape_project(root, only_missing=False)
    assert forced["updated"] == 1


def test_stale_when_files_change(tmp_path):
    root = _project(tmp_path)
    scrape_ops.scrape_project(root)
    # add a new content file -> staleness signature changes -> re-scraped
    _write(os.path.join(root, "M1/exp1/s1"), "extra.csv", "a,b\n1,2\n")
    again = scrape_ops.scrape_project(root)
    assert again["updated"] == 1
