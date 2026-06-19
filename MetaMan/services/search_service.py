import json
import os
from typing import Any, Dict, List, Tuple

OPERATORS = ["=", "!=", "contains", ">", ">=", "<", "<="]


def _to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _match(meta: Dict[str, Any], field: str, op: str, value: str) -> bool:
    raw = meta.get(field, "")
    left = "" if raw is None else (json.dumps(raw, ensure_ascii=False) if isinstance(raw, (dict, list)) else str(raw))
    lv = left.strip().lower()
    rv = str(value).strip().lower()
    if op == "=":
        return lv == rv
    if op == "!=":
        return lv != rv
    if op == "contains":
        return rv in lv
    # numeric comparisons fall back to string ordering when not numeric
    a, b = _to_float(left), _to_float(value)
    if a is not None and b is not None:
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
    if op == ">":
        return lv > rv
    if op == ">=":
        return lv >= rv
    if op == "<":
        return lv < rv
    if op == "<=":
        return lv <= rv
    return False


def query_sessions(project_dir: str, filters: List[Tuple[str, str, str]]) -> List[Dict]:
    """Return sessions under *project_dir* whose ``metadata.json`` satisfies ALL
    *(field, op, value)* filters. Each result is ``{"path", "meta"}``.

    Enables structured queries like ``Region = CA1`` AND
    ``Auto: sample rate (Hz) > 30000`` that the plain substring search cannot."""
    filters = [(f, o, v) for (f, o, v) in filters if str(f).strip()]
    results: List[Dict] = []
    for root, _dirs, files in os.walk(project_dir):
        if "metadata.json" not in files:
            continue
        try:
            with open(os.path.join(root, "metadata.json"), "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        if all(_match(meta, f, o, v) for (f, o, v) in filters):
            results.append({"path": root, "meta": meta})
    return results


def search_in_project(project_dir: str, query: str) -> List[Dict]:
    hits: List[Dict] = []
    q = (query or "").lower()
    for root, dirs, files in os.walk(project_dir):
        if "metadata.json" in files:
            p = os.path.join(root, "metadata.json")
            try:
                data = json.loads(open(p, "r", encoding="utf-8").read())
                for k, v in data.items():
                    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                    if q in (k.lower() + " " + s.lower()):
                        hits.append({"path": root, "key": k, "value": (s[:200] + "..." if len(s) > 200 else s)})
            except Exception:
                pass
    return hits
