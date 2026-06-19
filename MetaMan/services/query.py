"""Project-wide metadata query + analysis API.

A small, Qt-free layer you can use straight from Python / Jupyter to search and
analyse a MetaMan project. The GUI "Find Sessions" dialog uses the same engine,
so an interactive query and an analysis-script query return identical rows.

Why this exists
---------------
The old search only recognised files literally named ``metadata.json`` and only
offered string/number comparisons. Real projects mix two metadata dialects (the
MetaMan canonical ``metadata.json`` and acquisition-pipeline ``*_metadata.json``
files) and analysts want a *table*, not a list of paths. This module:

* discovers every session by following the project's **structure schema**
  (``project -> subject -> experiment -> session`` etc.), with a robust fallback;
* reads **either** metadata dialect and normalises identity from the folder tree
  (the folder is authoritative, so a stale ``Subject: "rawData"`` can never lie);
* returns a tidy :class:`pandas.DataFrame` (one row per session) you can group,
  plot and merge;
* supports a rich, composable set of query operators.

Quick start
-----------
    from MetaMan.services.query import ProjectQuery

    pq = ProjectQuery(r"B:/NPX/rawData/mPFC-NAc")
    df = pq.to_dataframe()                       # one tidy row per session

    hits = (pq.where("subject", "=", "51542")
              .where("modality", "contains", "Neuropixels")
              .to_dataframe())

    print(pq.summary())                          # aggregate project info
    pq.where("Region", "contains", "NAc").to_csv("nac_sessions.csv")
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from ..config import SETTINGS_FILE
from ..io_ops import load_project_info, load_subject_info
from ..services.structure_schema import (
    default_structure_schema,
    is_marker_level,
    level_meta_key,
    marker_folder_name,
    normalize_structure_schema,
    sublevels,
)

# Acquisition-dialect key -> canonical MetaMan field. The folder tree still wins
# for identity (project/subject/experiment/session); these only fill the gaps.
_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "animal_id": "Subject",
    "animal": "Subject",
    "subject": "Subject",
    "experiment": "Experiment",
    "session": "Session",
    "trial": "Trial",
    "condition": "Condition",
    "arena": "Arena",
    "date": "DateTime",
    "datetime": "DateTime",
    "timestamp": "DateTime",
    "notes": "Comments",
    "comments": "Comments",
    "experimenter": "Experimenter",
    "region": "Region",
    "recording": "Recording",
}

# Fields that are pure machinery / noise in a per-session row. Kept in ``meta``
# but dropped from the flat DataFrame so the table stays readable.
_NOISE_FIELDS = {
    "recording_timing_audit", "audio_recording", "barcode_sw_params",
    "filename_order", "live_rois", "user_flags", "user_flag", "trial_assets",
    "trial_info", "streams",
}

OPERATORS: List[str] = [
    "=", "!=", "contains", "icontains", "startswith", "endswith", "regex",
    ">", ">=", "<", "<=", "in", "not in", "between", "exists", "missing",
]


# ───────────────────────────── comparisons ──────────────────────────────────

def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _as_date(value: Any) -> Optional[datetime]:
    s = str(value or "").strip()
    if not s:
        return None
    s = s.replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d", "%d-%m-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s[: len(fmt) + 8], fmt)
        except ValueError:
            continue
    try:  # tolerate ISO strings with microseconds / timezone
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _ordered_pair(left: Any, right: Any) -> Tuple[float, float] | Tuple[str, str]:
    """Return a comparable (left, right) pair: numbers if both parse as numbers,
    timestamps if both parse as dates, otherwise lower-cased strings."""
    lf, rf = _as_float(left), _as_float(right)
    if lf is not None and rf is not None:
        return lf, rf
    ld, rd = _as_date(left), _as_date(right)
    if ld is not None and rd is not None:
        return ld.timestamp(), rd.timestamp()
    return _as_text(left).strip().lower(), _as_text(right).strip().lower()


def compare(left: Any, op: str, right: Any) -> bool:
    """Apply one operator. The engine and the GUI both route through here so the
    semantics never drift apart."""
    op = (op or "=").strip().lower()
    lt = _as_text(left).strip().lower()
    rt = _as_text(right).strip().lower()

    if op in ("exists", "present"):
        return _as_text(left).strip() != ""
    if op in ("missing", "empty", "absent"):
        return _as_text(left).strip() == ""
    if op in ("=", "==", "eq", "is"):
        return lt == rt
    if op in ("!=", "ne", "<>", "not"):
        return lt != rt
    if op in ("contains", "icontains", "has"):
        return rt in lt
    if op == "startswith":
        return lt.startswith(rt)
    if op == "endswith":
        return lt.endswith(rt)
    if op in ("regex", "~", "matches"):
        try:
            return re.search(str(right), _as_text(left), re.IGNORECASE) is not None
        except re.error:
            return False
    if op == "in":
        opts = [p.strip().lower() for p in re.split(r"[,;|]", _as_text(right)) if p.strip()]
        return lt in opts
    if op in ("not in", "notin"):
        opts = [p.strip().lower() for p in re.split(r"[,;|]", _as_text(right)) if p.strip()]
        return lt not in opts
    if op == "between":
        bounds = [p.strip() for p in re.split(r"\.\.|,|;", _as_text(right)) if p.strip()]
        if len(bounds) != 2:
            return False
        lo_l, lo_r = _ordered_pair(left, bounds[0])
        hi_l, hi_r = _ordered_pair(left, bounds[1])
        return lo_l >= lo_r and hi_l <= hi_r
    if op in (">", ">=", "<", "<="):
        a, b = _ordered_pair(left, right)
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
        if op == "<":
            return a < b
        return a <= b
    return False


# ───────────────────────────── schema access ────────────────────────────────

def _schema_from_settings(project_name: str) -> Optional[Dict[str, Any]]:
    """Read the per-project (or default) structure schema straight from the
    settings file. Read-only: never instantiates ``AppSettings`` (which would
    create folders and rewrite settings on import)."""
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    by_proj = data.get("structure_schemas_by_project") or {}
    if isinstance(by_proj, dict) and by_proj.get(project_name):
        return by_proj[project_name]
    return data.get("structure_schema") or None


def resolve_schema(project_dir: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve the structure schema for *project_dir*: explicit arg, then the
    in-folder ``_metaman_structure.json`` sidecar, then settings, then default."""
    if schema:
        return normalize_structure_schema(schema)
    sidecar = os.path.join(project_dir, "_metaman_structure.json")
    if os.path.isfile(sidecar):
        try:
            return normalize_structure_schema(json.load(open(sidecar, encoding="utf-8")))
        except Exception:
            pass
    from_settings = _schema_from_settings(os.path.basename(os.path.normpath(project_dir)))
    if from_settings:
        return normalize_structure_schema(from_settings)
    return normalize_structure_schema(default_structure_schema())


# ──────────────────────────── metadata reading ──────────────────────────────

def _subdirs(path: str) -> List[str]:
    try:
        return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
    except OSError:
        return []


def read_session_metadata(session_dir: str) -> Tuple[Dict[str, Any], str]:
    """Return ``(meta, source_filename)`` for a session folder, understanding
    both dialects: the canonical ``metadata.json`` is preferred, otherwise the
    first ``*_metadata.json`` acquisition file (matched to the folder name when
    possible). ``({}, "")`` when nothing readable is found."""
    canonical = os.path.join(session_dir, "metadata.json")
    if os.path.isfile(canonical):
        try:
            return json.load(open(canonical, encoding="utf-8")), "metadata.json"
        except Exception:
            pass
    try:
        candidates = [f for f in os.listdir(session_dir)
                      if f.lower().endswith("_metadata.json")]
    except OSError:
        candidates = []
    base = os.path.basename(session_dir).lower()
    candidates.sort(key=lambda f: (base not in f.lower(), f))
    for fname in candidates:
        try:
            return json.load(open(os.path.join(session_dir, fname), encoding="utf-8")), fname
        except Exception:
            continue
    return {}, ""


def normalise_meta(meta: Dict[str, Any], identity: Dict[str, str]) -> Dict[str, Any]:
    """Merge acquisition-dialect aliases into canonical fields and stamp the
    folder-derived identity on top (the folder always wins for identity)."""
    out: Dict[str, Any] = dict(meta or {})
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if alias in out and str(out.get(canonical, "")).strip() == "":
            out[canonical] = out[alias]
    for key, value in identity.items():
        canonical = level_meta_key({"key": key})
        if value:
            out[canonical] = value
    if identity.get("subject"):
        out.setdefault("Animal", identity["subject"])
    return out


# ──────────────────────────── session discovery ─────────────────────────────

def iter_sessions(project_dir: str,
                  schema: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Yield a record per session under *project_dir*, descending the schema's
    sublevels (subject/experiment/session/...). Each record is a dict with
    ``path``, identity keys, ``meta`` (normalised), ``metadata_file`` and
    ``has_metadata``."""
    project_dir = os.path.normpath(project_dir)
    project = os.path.basename(project_dir)
    subs = sublevels(resolve_schema(project_dir, schema), "raw")
    if not subs:  # degenerate schema: treat the project folder itself as one node
        meta, src = read_session_metadata(project_dir)
        yield _make_record(project, project_dir, {}, meta, src)
        return
    leaf = len(subs) - 1

    def walk(cur_dir: str, idx: int, identity: Dict[str, str]) -> Iterator[Dict[str, Any]]:
        level = subs[idx]
        key = str(level.get("key", "")).strip().lower()
        if is_marker_level(key):
            nd = os.path.join(cur_dir, marker_folder_name(level))
            if not os.path.isdir(nd):
                return
            if idx == leaf:
                meta, src = read_session_metadata(nd)
                yield _make_record(project, nd, identity, meta, src)
            else:
                yield from walk(nd, idx + 1, identity)
            return
        for name in _subdirs(cur_dir):
            nd = os.path.join(cur_dir, name)
            ident2 = dict(identity)
            ident2[key] = name
            if idx == leaf:
                meta, src = read_session_metadata(nd)
                yield _make_record(project, nd, ident2, meta, src)
            else:
                yield from walk(nd, idx + 1, ident2)

    yield from walk(project_dir, 0, {})


def _make_record(project: str, session_dir: str, identity: Dict[str, str],
                 meta: Dict[str, Any], source: str) -> Dict[str, Any]:
    ident = {"project": project, **identity}
    ident.setdefault("session", os.path.basename(session_dir))
    norm = normalise_meta(meta, ident)
    return {
        "path": session_dir,
        "project": project,
        "subject": ident.get("subject", ""),
        "experiment": ident.get("experiment", ""),
        "session": ident.get("session", ""),
        "metadata_file": source,
        "has_metadata": bool(meta),
        "meta": norm,
    }


# ─────────────────────────── preprocessing helpers ──────────────────────────

def _pp_steps(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = meta.get("preprocessing", [])
    return [s for s in steps if isinstance(s, dict) and str(s.get("name", "")).strip()]


def _pp_summary(meta: Dict[str, Any]) -> Dict[str, Any]:
    steps = _pp_steps(meta)
    done = sum(1 for s in steps if str(s.get("status", "")).lower() == "completed")
    return {
        "pp_steps": len(steps),
        "pp_completed": done,
        "pp_pending": len(steps) - done,
        "pp_status": {str(s["name"]): str(s.get("status", "planned")) for s in steps},
    }


# ────────────────────────────── value lookup ────────────────────────────────

def field_value(record: Dict[str, Any], field: str) -> Any:
    """Look up *field* on a record. Resolution order, all case-insensitive:
    identity keys -> exact meta key -> case-insensitive meta key -> derived
    virtuals (``modality``, ``size``, ``files``, ``pp_completed`` ...)."""
    f = field.strip()
    low = f.lower()
    if low in ("project", "subject", "experiment", "session", "path",
               "metadata_file", "has_metadata"):
        return record.get(low if low != "path" else "path")
    meta = record["meta"]
    if f in meta:
        return meta[f]
    for k, v in meta.items():
        if k.lower() == low:
            return v
    virtuals = {
        "modality": meta.get("Auto: modality", ""),
        "size": meta.get("Auto: total size", ""),
        "files": meta.get("Auto: file count", ""),
        "sample_rate": meta.get("Auto: sample rate (Hz)", ""),
        "duration": meta.get("Auto: duration", ""),
    }
    if low in virtuals:
        return virtuals[low]
    pp = _pp_summary(meta)
    if low in ("pp_steps", "pp_completed", "pp_pending"):
        return pp[low]
    if low.startswith("pp_"):  # e.g. pp_spike_sorting -> that step's status
        return pp["pp_status"].get(f[3:], "")
    return ""


# ─────────────────────────────── ProjectQuery ───────────────────────────────

class ProjectQuery:
    """Fluent, chainable query over a project's sessions.

    Filters accumulate with AND. Use :meth:`or_where` for an OR branch, or pass
    a predicate to :meth:`filter` for anything the operators cannot express.
    Every terminal method (:meth:`records`, :meth:`paths`, :meth:`to_dataframe`,
    :meth:`summary`) recomputes against the current filter set, so a query
    object is cheap to reuse and branch.
    """

    def __init__(self, project_dir: str, schema: Optional[Dict[str, Any]] = None,
                 scrape: bool = False, deep: bool = False):
        self.project_dir = os.path.normpath(project_dir)
        self._schema = schema
        self._scrape = scrape
        self._deep = deep
        self._and: List[Tuple[str, str, str]] = []
        self._or: List[Tuple[str, str, str]] = []
        self._predicates: List[Callable[[Dict[str, Any]], bool]] = []
        self._cache: Optional[List[Dict[str, Any]]] = None

    # -- construction / chaining ------------------------------------------
    def _clone(self) -> "ProjectQuery":
        q = ProjectQuery(self.project_dir, self._schema, self._scrape, self._deep)
        q._and = list(self._and)
        q._or = list(self._or)
        q._predicates = list(self._predicates)
        q._cache = self._cache  # discovery result is filter-independent; reuse it
        return q

    def where(self, field: str, op: str = "exists", value: str = "") -> "ProjectQuery":
        q = self._clone()
        q._and.append((field, op, value))
        return q

    def or_where(self, field: str, op: str = "exists", value: str = "") -> "ProjectQuery":
        q = self._clone()
        q._or.append((field, op, value))
        return q

    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> "ProjectQuery":
        """Add an arbitrary predicate ``record -> bool`` (record has ``meta``,
        ``path`` and the identity keys)."""
        q = self._clone()
        q._predicates.append(predicate)
        return q

    # -- evaluation --------------------------------------------------------
    def _all_records(self) -> List[Dict[str, Any]]:
        if self._cache is None:
            recs = list(iter_sessions(self.project_dir, self._schema))
            if self._scrape:
                from ..services.metadata_scraper import scrape_session, merge_auto
                for r in recs:
                    try:
                        r["meta"] = merge_auto(r["meta"], scrape_session(r["path"], deep=self._deep))
                    except Exception:
                        pass
            self._cache = recs
        return self._cache

    def _passes(self, record: Dict[str, Any]) -> bool:
        if any(not compare(field_value(record, f), o, v) for f, o, v in self._and):
            return False
        if self._or and not any(compare(field_value(record, f), o, v) for f, o, v in self._or):
            return False
        return all(p(record) for p in self._predicates)

    def records(self) -> List[Dict[str, Any]]:
        return [r for r in self._all_records() if self._passes(r)]

    def paths(self) -> List[str]:
        return [r["path"] for r in self.records()]

    def count(self) -> int:
        return len(self.records())

    def __len__(self) -> int:  # so ``len(pq)`` and ``if pq:`` work
        return self.count()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.records())

    # -- tabular output ----------------------------------------------------
    def rows(self) -> List[Dict[str, Any]]:
        """Flat, JSON-friendly dicts (one per matching session) suitable for a
        DataFrame or ``csv.DictWriter``. Nested/noise fields are dropped; the
        preprocessing list becomes ``pp_*`` summary columns."""
        out: List[Dict[str, Any]] = []
        for r in self.records():
            row: Dict[str, Any] = {
                "project": r["project"], "subject": r["subject"],
                "experiment": r["experiment"], "session": r["session"],
                "has_metadata": r["has_metadata"], "metadata_file": r["metadata_file"],
            }
            for k, v in r["meta"].items():
                if k in _NOISE_FIELDS or k == "preprocessing":
                    continue
                row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            pp = _pp_summary(r["meta"])
            row["pp_steps"] = pp["pp_steps"]
            row["pp_completed"] = pp["pp_completed"]
            row["pp_pending"] = pp["pp_pending"]
            for name, status in pp["pp_status"].items():
                row[f"pp_{name}"] = status
            row["path"] = r["path"]
            out.append(row)
        return out

    def to_dataframe(self):
        """Return a :class:`pandas.DataFrame`, one tidy row per session.
        Identity columns are ordered first."""
        import pandas as pd
        df = pd.DataFrame(self.rows())
        if df.empty:
            return df
        lead = [c for c in ("project", "subject", "experiment", "session") if c in df.columns]
        rest = [c for c in df.columns if c not in lead and c != "path"]
        return df[lead + rest + (["path"] if "path" in df.columns else [])]

    def to_csv(self, path: str) -> str:
        """Write the result table to *path* (CSV) and return the path."""
        import csv
        rows = self.rows()
        cols: List[str] = []
        for row in rows:
            for k in row:
                if k not in cols:
                    cols.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return path

    # -- aggregation -------------------------------------------------------
    def values(self, field: str) -> List[Any]:
        """Distinct, sorted, non-empty values of *field* across matches."""
        seen = {_as_text(field_value(r, field)).strip() for r in self.records()}
        return sorted(v for v in seen if v)

    def group_counts(self, field: str) -> Dict[str, int]:
        """``{value: count}`` for *field* across matches (great for a quick bar
        chart or a sanity check on subject/condition balance)."""
        counts: Dict[str, int] = {}
        for r in self.records():
            key = _as_text(field_value(r, field)).strip() or "(blank)"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def summary(self) -> Dict[str, Any]:
        """Aggregate project info over the current matches: counts, the set of
        subjects/experiments/modalities/experimenters, preprocessing progress
        and the session date range."""
        recs = self.records()
        with_meta = [r for r in recs if r["has_metadata"]]
        modalities, experimenters, dates = set(), set(), []
        steps_total = steps_done = 0
        for r in recs:
            m = r["meta"]
            for token in str(m.get("Auto: modality", "")).split(","):
                if token.strip():
                    modalities.add(token.strip())
            if str(m.get("Experimenter", "")).strip():
                experimenters.add(str(m["Experimenter"]).strip())
            d = _as_date(m.get("DateTime"))
            if d:
                dates.append(d)
            pp = _pp_summary(m)
            steps_total += pp["pp_steps"]
            steps_done += pp["pp_completed"]
        return {
            "project": os.path.basename(self.project_dir),
            "sessions": len(recs),
            "sessions_with_metadata": len(with_meta),
            "subjects": self.values("subject"),
            "experiments": self.values("experiment"),
            "modalities": sorted(modalities),
            "experimenters": sorted(experimenters),
            "date_range": ([min(dates).strftime("%Y-%m-%d"),
                            max(dates).strftime("%Y-%m-%d")] if dates else []),
            "preprocessing_steps_total": steps_total,
            "preprocessing_steps_completed": steps_done,
            "preprocessing_percent_complete": round(100 * steps_done / steps_total, 1) if steps_total else 0.0,
            "subjects_count": len(self.values("subject")),
            "experiments_count": len(self.values("experiment")),
        }

    def project_info(self) -> Dict[str, Any]:
        """The project's ``project_info.json`` (free-form lab metadata)."""
        return load_project_info(self.project_dir)

    def subjects_info(self) -> Dict[str, Dict[str, Any]]:
        """``{subject: subject_info.json}`` for every subject folder."""
        out: Dict[str, Dict[str, Any]] = {}
        subs = sublevels(resolve_schema(self.project_dir, self._schema), "raw")
        if subs and str(subs[0].get("key", "")).lower() == "subject":
            for name in _subdirs(self.project_dir):
                info = load_subject_info(os.path.join(self.project_dir, name))
                if info:
                    out[name] = info
        else:  # subject is not the first sublevel: collect from discovery
            for r in self.records():
                sub = r["subject"]
                if sub and sub not in out:
                    info = load_subject_info(os.path.dirname(r["path"]))
                    if info:
                        out[sub] = info
        return out

    def __repr__(self) -> str:
        return (f"ProjectQuery({os.path.basename(self.project_dir)!r}, "
                f"filters={len(self._and) + len(self._or) + len(self._predicates)}, "
                f"matches={self.count()})")


# ───────────────────────────── module-level API ─────────────────────────────

def query(project_dir: str, filters: Optional[Iterable[Tuple[str, str, str]]] = None,
          *, match: str = "all", scrape: bool = False) -> ProjectQuery:
    """Build a :class:`ProjectQuery` from a list of ``(field, op, value)`` tuples.

    ``match="all"`` ANDs the filters (default); ``match="any"`` ORs them.
    """
    pq = ProjectQuery(project_dir, scrape=scrape)
    for field, op, value in (filters or []):
        pq = pq.or_where(field, op, value) if match == "any" else pq.where(field, op, value)
    return pq


def sessions_dataframe(project_dir: str, *, scrape: bool = False, deep: bool = False):
    """One-liner: tidy DataFrame of every session in a project."""
    return ProjectQuery(project_dir, scrape=scrape, deep=deep).to_dataframe()


def project_summary(project_dir: str, *, scrape: bool = False) -> Dict[str, Any]:
    """One-liner: aggregate summary dict for a whole project."""
    return ProjectQuery(project_dir, scrape=scrape).summary()
