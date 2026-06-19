"""Preprocessing tracking: a Qt-free API over a project's preprocessing steps.

Every session's ``metadata.json`` carries a ``preprocessing`` list of step dicts
``{name, status, params, comments, results_dir}``. The GUI edits one session at
a time; this module lets you read and drive *the whole project* from Python:

    from MetaMan.services import preprocessing_ops as pp

    pp.status_table(project_dir)          # wide DataFrame: a column per step
    pp.progress_summary(project_dir)      # % complete per step + overall
    pp.pending_sessions(project_dir, "spike_sorting")   # what is left to do
    pp.bulk_set_status(project_dir, "curation", "completed",
                       where=("subject", "=", "51542"))  # mark many at once
    pp.apply_step_template(project_dir)   # seed default steps where missing

Writes go through the canonical metadata triplet writer, so the ``.json/.csv/.h5``
stay in sync exactly as the GUI would leave them.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..io_ops import load_session_metadata, save_session_triplet
from .query import compare, field_value, iter_sessions, read_session_metadata

# ── status vocabulary ────────────────────────────────────────────────────────
PLANNED, ONGOING, COMPLETED = "planned", "ongoing", "completed"
STATUSES = (PLANNED, ONGOING, COMPLETED)

_STATUS_ALIASES = {
    "planned": PLANNED, "todo": PLANNED, "pending": PLANNED, "": PLANNED,
    "ongoing": ONGOING, "in_progress": ONGOING, "inprogress": ONGOING,
    "running": ONGOING, "wip": ONGOING,
    "completed": COMPLETED, "done": COMPLETED, "finished": COMPLETED, "complete": COMPLETED,
}

# ── default step templates by modality (extends the GUI's three presets) ─────
STEP_TEMPLATES: Dict[str, List[str]] = {
    "neuropixels": ["spike_sorting", "curation", "time_sync", "histology", "dlc"],
    "fiber": ["dff_zscore", "motion_correction", "behavior_scoring"],
    "behavior": ["dlc", "behavior_scoring", "pose_cleanup"],
    "imaging": ["motion_correction", "segmentation", "dff_zscore"],
}


def normalize_status(status: str) -> str:
    """Map any loose status string to one of :data:`STATUSES`."""
    return _STATUS_ALIASES.get(str(status or "").strip().lower(), PLANNED)


def steps_for_modality(modality: str) -> List[str]:
    """Best-guess default step list from a free-text modality / recording hint."""
    h = str(modality or "").lower()
    if any(t in h for t in ("npx", "neuropixel", "spikeglx", "imec", "ephys")):
        return list(STEP_TEMPLATES["neuropixels"])
    if any(t in h for t in ("fiber", "photometry", "doric", "fip")):
        return list(STEP_TEMPLATES["fiber"])
    if any(t in h for t in ("2p", "miniscope", "calcium", "imaging", "suite2p")):
        return list(STEP_TEMPLATES["imaging"])
    return list(STEP_TEMPLATES["behavior"])


# ── per-session read ─────────────────────────────────────────────────────────

def load_steps(session_dir: str) -> List[Dict[str, Any]]:
    """Return the normalised preprocessing steps for one session (any dialect)."""
    meta, _ = read_session_metadata(session_dir)
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for step in meta.get("preprocessing", []):
        if not isinstance(step, dict):
            continue
        name = str(step.get("name", "")).strip()
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        out.append({
            "name": name,
            "status": normalize_status(step.get("status", "")),
            "params": step.get("params", {}) if isinstance(step.get("params"), dict) else {},
            "comments": str(step.get("comments", "") or ""),
            "results_dir": str(step.get("results_dir", "") or ""),
        })
    return out


def step_status(session_dir: str, step: str) -> str:
    """Status of *step* for one session, or ``""`` if the step is absent."""
    for s in load_steps(session_dir):
        if s["name"].lower() == step.strip().lower():
            return s["status"]
    return ""


# ── per-session write ────────────────────────────────────────────────────────

def _save(session_dir: str, meta: Dict[str, Any]) -> None:
    os.makedirs(session_dir, exist_ok=True)
    save_session_triplet(session_dir, meta)


def _load_writable(session_dir: str) -> Dict[str, Any]:
    """Load the canonical metadata.json for writing (preserve all fields). Falls
    back to whatever dialect exists so we never clobber acquisition metadata."""
    meta = load_session_metadata(session_dir)
    if meta:
        return meta
    meta, _ = read_session_metadata(session_dir)
    return dict(meta or {})


def set_step_status(session_dir: str, step: str, status: str,
                    add_if_missing: bool = True) -> bool:
    """Set *step*'s status. Adds the step when missing (if *add_if_missing*).
    Returns True when something was written."""
    status = normalize_status(status)
    meta = _load_writable(session_dir)
    steps = meta.setdefault("preprocessing", [])
    for s in steps:
        if isinstance(s, dict) and str(s.get("name", "")).strip().lower() == step.strip().lower():
            s["status"] = status
            _save(session_dir, meta)
            return True
    if not add_if_missing:
        return False
    steps.append({"name": step.strip(), "status": status, "params": {},
                  "comments": "", "results_dir": ""})
    _save(session_dir, meta)
    return True


def set_step_params(session_dir: str, step: str, params: Dict[str, Any],
                    merge: bool = True) -> bool:
    """Set (or merge) a step's ``params`` dict."""
    meta = _load_writable(session_dir)
    for s in meta.setdefault("preprocessing", []):
        if isinstance(s, dict) and str(s.get("name", "")).strip().lower() == step.strip().lower():
            cur = s.get("params", {}) if isinstance(s.get("params"), dict) else {}
            s["params"] = {**cur, **params} if merge else dict(params)
            _save(session_dir, meta)
            return True
    return False


def add_step(session_dir: str, step: str, *, status: str = PLANNED,
             params: Optional[Dict[str, Any]] = None) -> bool:
    """Add a preprocessing step if it is not already present."""
    if step_status(session_dir, step):
        return False
    meta = _load_writable(session_dir)
    meta.setdefault("preprocessing", []).append({
        "name": step.strip(), "status": normalize_status(status),
        "params": dict(params or {}), "comments": "", "results_dir": "",
    })
    _save(session_dir, meta)
    return True


def apply_step_template(target_dir: str, steps: Optional[List[str]] = None,
                        *, modality: str = "") -> int:
    """Seed default steps into every session under *target_dir* (a project or a
    single session) that is missing them. Returns the number of sessions
    touched. When *steps* is None, the template is chosen per session from its
    detected modality / Recording field."""
    touched = 0
    for rec in _iter_targets(target_dir):
        wanted = steps or steps_for_modality(
            modality or rec["meta"].get("Auto: modality", "") or rec["meta"].get("Recording", ""))
        existing = {s["name"].lower() for s in load_steps(rec["path"])}
        added = False
        for name in wanted:
            if name.lower() not in existing:
                add_step(rec["path"], name)
                existing.add(name.lower())
                added = True
        touched += int(added)
    return touched


# ── project-wide read ────────────────────────────────────────────────────────

def _iter_targets(target_dir: str) -> List[Dict[str, Any]]:
    """Records for a project (many sessions) or a single session folder."""
    recs = list(iter_sessions(target_dir))
    if recs and any(r["path"] != target_dir for r in recs):
        return recs
    meta, src = read_session_metadata(target_dir)
    return [{"path": target_dir, "project": "", "subject": "", "experiment": "",
             "session": os.path.basename(target_dir), "meta": meta,
             "has_metadata": bool(meta), "metadata_file": src}]


def all_step_names(project_dir: str) -> List[str]:
    """Union of every step name seen across the project, first-seen order."""
    names: List[str] = []
    for rec in iter_sessions(project_dir):
        for s in load_steps(rec["path"]):
            if s["name"] not in names:
                names.append(s["name"])
    return names


def status_table(project_dir: str):
    """Wide :class:`pandas.DataFrame`: one row per session, one column per step
    holding its status, plus ``pp_completed`` / ``pp_pending`` / ``pp_percent``."""
    import pandas as pd
    steps = all_step_names(project_dir)
    rows: List[Dict[str, Any]] = []
    for rec in iter_sessions(project_dir):
        st = {s["name"]: s["status"] for s in load_steps(rec["path"])}
        done = sum(1 for v in st.values() if v == COMPLETED)
        row: Dict[str, Any] = {
            "subject": rec["subject"], "experiment": rec["experiment"],
            "session": rec["session"],
        }
        for name in steps:
            row[name] = st.get(name, "")
        row["pp_completed"] = done
        row["pp_pending"] = len(st) - done
        row["pp_percent"] = round(100 * done / len(st), 1) if st else 0.0
        row["path"] = rec["path"]
        rows.append(row)
    return pd.DataFrame(rows)


def long_status_table(project_dir: str):
    """Tidy/long :class:`pandas.DataFrame`: one row per (session, step, status).
    Convenient for ``groupby`` and seaborn."""
    import pandas as pd
    rows: List[Dict[str, Any]] = []
    for rec in iter_sessions(project_dir):
        for s in load_steps(rec["path"]):
            rows.append({
                "subject": rec["subject"], "experiment": rec["experiment"],
                "session": rec["session"], "step": s["name"],
                "status": s["status"], "path": rec["path"],
            })
    return pd.DataFrame(rows)


def progress_summary(project_dir: str) -> Dict[str, Any]:
    """Per-step and overall completion counts for the project."""
    per_step: Dict[str, Dict[str, int]] = {}
    total = done = 0
    sessions = 0
    for rec in iter_sessions(project_dir):
        steps = load_steps(rec["path"])
        if steps:
            sessions += 1
        for s in steps:
            bucket = per_step.setdefault(s["name"], {PLANNED: 0, ONGOING: 0, COMPLETED: 0})
            bucket[s["status"]] += 1
            total += 1
            done += int(s["status"] == COMPLETED)
    return {
        "sessions_with_steps": sessions,
        "steps_total": total,
        "steps_completed": done,
        "percent_complete": round(100 * done / total, 1) if total else 0.0,
        "by_step": {
            name: {**counts,
                   "percent": round(100 * counts[COMPLETED] / sum(counts.values()), 1)
                   if sum(counts.values()) else 0.0}
            for name, counts in per_step.items()
        },
    }


def pending_sessions(project_dir: str, step: str) -> List[str]:
    """Session folders where *step* is missing or not yet completed."""
    out: List[str] = []
    for rec in iter_sessions(project_dir):
        if step_status(rec["path"], step) != COMPLETED:
            out.append(rec["path"])
    return out


def completed_sessions(project_dir: str, step: str) -> List[str]:
    """Session folders where *step* is completed."""
    return [rec["path"] for rec in iter_sessions(project_dir)
            if step_status(rec["path"], step) == COMPLETED]


# ── project-wide write ───────────────────────────────────────────────────────

WhereClause = Tuple[str, str, str]


def bulk_set_status(project_dir: str, step: str, status: str, *,
                    where: Optional[WhereClause | Callable[[Dict[str, Any]], bool]] = None,
                    add_if_missing: bool = True) -> List[str]:
    """Set *step* to *status* for every matching session. *where* is either a
    ``(field, op, value)`` clause or a ``record -> bool`` predicate. Returns the
    list of session paths that were changed."""
    if callable(where):
        match = where
    elif where:
        f, o, v = where
        match = lambda rec: compare(field_value(rec, f), o, v)  # noqa: E731
    else:
        match = lambda rec: True  # noqa: E731

    changed: List[str] = []
    for rec in iter_sessions(project_dir):
        if match(rec) and set_step_status(rec["path"], step, status, add_if_missing=add_if_missing):
            changed.append(rec["path"])
    return changed
