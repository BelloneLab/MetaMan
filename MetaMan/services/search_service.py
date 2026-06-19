"""Backward-compatible search facade over the analysis engine in ``query.py``.

The GUI ("Find Sessions", "Search in Project") imports the three public names
below. They now delegate to :mod:`MetaMan.services.query`, which means the
interactive search and a Python analysis query share one implementation and one
set of operators, and both understand the canonical ``metadata.json`` *and* the
acquisition ``*_metadata.json`` dialect (the old code only saw ``metadata.json``).
"""

import json
from typing import Any, Dict, List, Tuple

from .query import OPERATORS, ProjectQuery, compare, field_value, iter_sessions

# Re-exported so existing callers keep working; the richer operator set now
# flows straight into the Find Sessions dialog's drop-down.
__all__ = ["OPERATORS", "query_sessions", "search_in_project", "compare"]


def query_sessions(project_dir: str, filters: List[Tuple[str, str, str]]) -> List[Dict]:
    """Return sessions under *project_dir* matching ALL ``(field, op, value)``
    filters. Each result is ``{"path", "meta"}`` (``meta`` is identity-normalised
    so a query on ``Subject`` works regardless of metadata dialect)."""
    pq = ProjectQuery(project_dir)
    for field, op, value in filters:
        if str(field).strip():
            pq = pq.where(field, op, value)
    return [{"path": r["path"], "meta": r["meta"]} for r in pq.records()]


def search_in_project(project_dir: str, query: str) -> List[Dict]:
    """Free-text search across every session's metadata (both dialects). Returns
    ``{"path", "key", "value"}`` hits where *query* appears in a key or value."""
    q = (query or "").strip().lower()
    if not q:
        return []
    hits: List[Dict] = []
    for rec in iter_sessions(project_dir):
        for k, v in rec["meta"].items():
            s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            if q in (k.lower() + " " + s.lower()):
                hits.append({
                    "path": rec["path"], "key": k,
                    "value": (s[:200] + "…" if len(s) > 200 else s),
                })
    return hits


# Kept for any external caller that imported the old helper directly.
def _match(meta: Dict[str, Any], field: str, op: str, value: str) -> bool:
    return compare(meta.get(field, ""), op, value)
