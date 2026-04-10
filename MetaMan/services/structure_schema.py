import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


SCHEMA_FILE_NAME = "project_structure.json"
SCHEMA_VERSION = 1

CORE_LEVEL_KEYS: Sequence[str] = (
    "project",
    "experiment",
    "subject",
    "session",
)

MARKER_LEVEL_KEYS: Sequence[str] = (
    "raw",
    "processed",
)

KNOWN_LEVEL_KEYS: Sequence[str] = (
    "project",
    "experiment",
    "subject",
    "session",
    "recording",
    "trial",
    "group",
    "raw",
    "processed",
)

DEFAULT_LABELS: Dict[str, str] = {
    "project": "Project",
    "experiment": "Experiment",
    "subject": "Subject",
    "session": "Session",
    "recording": "Recording",
    "trial": "Trial",
    "group": "Group",
    "raw": "raw",
    "processed": "processed",
}

DEFAULT_ENABLED_RAW: Dict[str, bool] = {
    "project": True,
    "experiment": True,
    "subject": True,
    "session": True,
    "recording": False,
    "trial": False,
    "group": False,
    "raw": False,
    "processed": False,
}

DEFAULT_ENABLED_PROCESSED: Dict[str, bool] = {
    "project": True,
    "experiment": True,
    "subject": True,
    "session": True,
    "recording": False,
    "trial": False,
    "group": False,
    "raw": False,
    "processed": False,
}

DEFAULT_RAW_ORDER: Sequence[str] = (
    "project",
    "experiment",
    "subject",
    "session",
    "recording",
    "trial",
    "group",
    "raw",
    "processed",
)

DEFAULT_PROCESSED_ORDER: Sequence[str] = (
    "project",
    "experiment",
    "subject",
    "session",
    "group",
    "recording",
    "trial",
    "raw",
    "processed",
)


@dataclass
class HierarchyEntry:
    values: Dict[str, str]
    paths: Dict[str, str]


def is_marker_level(key: str) -> bool:
    return str(key or "").strip().lower() in MARKER_LEVEL_KEYS


def marker_folder_name(level: Dict[str, Any]) -> str:
    key = str(level.get("key", "")).strip().lower()
    label = str(level.get("label", "")).strip()
    raw = label or DEFAULT_LABELS.get(key, key)
    safe = raw.replace("\\", "_").replace("/", "_").strip()
    return safe or key


def default_structure_schema() -> Dict[str, Any]:
    return normalize_structure_schema(
        {
            "version": SCHEMA_VERSION,
            "raw_levels": [
                {"key": k, "label": DEFAULT_LABELS.get(k, k.title()), "enabled": DEFAULT_ENABLED_RAW.get(k, False)}
                for k in DEFAULT_RAW_ORDER
            ],
            "processed_levels": [
                {"key": k, "label": DEFAULT_LABELS.get(k, k.title()), "enabled": DEFAULT_ENABLED_PROCESSED.get(k, False)}
                for k in DEFAULT_PROCESSED_ORDER
            ],
        }
    )


def normalize_structure_schema(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = dict(schema or {})
    return {
        "version": SCHEMA_VERSION,
        "raw_levels": _normalize_levels(
            data.get("raw_levels"),
            default_order=DEFAULT_RAW_ORDER,
            default_enabled=DEFAULT_ENABLED_RAW,
        ),
        "processed_levels": _normalize_levels(
            data.get("processed_levels"),
            default_order=DEFAULT_PROCESSED_ORDER,
            default_enabled=DEFAULT_ENABLED_PROCESSED,
        ),
    }


def _normalize_levels(
    levels: Optional[Iterable[Dict[str, Any]]],
    default_order: Sequence[str],
    default_enabled: Dict[str, bool],
) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []

    if isinstance(levels, Iterable):
        for raw in levels:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("key", "")).strip().lower()
            if key not in KNOWN_LEVEL_KEYS:
                continue
            if key in by_key:
                continue
            by_key[key] = raw
            ordered_keys.append(key)

    for key in default_order:
        if key not in by_key:
            by_key[key] = {}
            ordered_keys.append(key)

    if "project" in ordered_keys:
        ordered_keys = [k for k in ordered_keys if k != "project"]
        ordered_keys.insert(0, "project")

    out: List[Dict[str, Any]] = []
    for key in ordered_keys:
        raw = by_key.get(key, {})
        label = str(raw.get("label", "")).strip() or DEFAULT_LABELS.get(key, key.title())
        enabled_default = bool(default_enabled.get(key, False))
        enabled = bool(raw.get("enabled", enabled_default))
        if key == "project":
            enabled = True
        out.append(
            {
                "key": key,
                "label": label,
                "enabled": enabled,
            }
        )
    return out


def levels_for_kind(schema: Dict[str, Any], kind: str = "raw", include_disabled: bool = False) -> List[Dict[str, Any]]:
    if kind == "processed":
        levels = schema.get("processed_levels", [])
    else:
        levels = schema.get("raw_levels", [])
    normalized = _normalize_levels(
        levels,
        default_order=DEFAULT_PROCESSED_ORDER if kind == "processed" else DEFAULT_RAW_ORDER,
        default_enabled=DEFAULT_ENABLED_PROCESSED if kind == "processed" else DEFAULT_ENABLED_RAW,
    )
    if include_disabled:
        return normalized
    return [x for x in normalized if bool(x.get("enabled", False))]


def role_enabled(schema: Dict[str, Any], role: str, kind: str = "raw") -> bool:
    for lvl in levels_for_kind(schema, kind=kind, include_disabled=True):
        if lvl.get("key") == role:
            return bool(lvl.get("enabled", False))
    return False


def role_label(schema: Dict[str, Any], role: str, kind: str = "raw") -> str:
    for lvl in levels_for_kind(schema, kind=kind, include_disabled=True):
        if lvl.get("key") == role:
            txt = str(lvl.get("label", "")).strip()
            return txt or DEFAULT_LABELS.get(role, role.title())
    return DEFAULT_LABELS.get(role, role.title())


def schema_file_path(raw_root: str, project_name: str) -> str:
    return os.path.join(raw_root, project_name, SCHEMA_FILE_NAME)


def load_project_schema(raw_root: str, project_name: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not project_name:
        return normalize_structure_schema(fallback or default_structure_schema())
    path = schema_file_path(raw_root, project_name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return normalize_structure_schema(data)
    except Exception:
        return normalize_structure_schema(fallback or default_structure_schema())


def save_project_schema(raw_root: str, processed_root: str, project_name: str, schema: Dict[str, Any]) -> str:
    normalized = normalize_structure_schema(schema)
    raw_project_dir = os.path.join(raw_root, project_name)
    proc_project_dir = os.path.join(processed_root, project_name) if processed_root else ""

    os.makedirs(raw_project_dir, exist_ok=True)
    if proc_project_dir:
        os.makedirs(proc_project_dir, exist_ok=True)

    out_path = os.path.join(raw_project_dir, SCHEMA_FILE_NAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    if proc_project_dir:
        proc_schema = os.path.join(proc_project_dir, SCHEMA_FILE_NAME)
        try:
            with open(proc_schema, "w", encoding="utf-8") as f:
                json.dump(normalized, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    return out_path


def list_directory_names(path: str) -> List[str]:
    try:
        return sorted(
            [
                d
                for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            ]
        )
    except Exception:
        return []


def collect_hierarchy_entries(
    root_dir: str,
    schema: Dict[str, Any],
    kind: str = "raw",
    project_filter: str = "",
) -> List[HierarchyEntry]:
    levels = levels_for_kind(schema, kind=kind, include_disabled=False)
    if not levels:
        return []
    if not os.path.isdir(root_dir):
        return []

    out: List[HierarchyEntry] = []

    def walk(base_path: str, level_index: int, values: Dict[str, str], paths: Dict[str, str]):
        if level_index >= len(levels):
            return

        level = levels[level_index]
        role = str(level.get("key", "")).strip().lower()

        if is_marker_level(role):
            marker = marker_folder_name(level)
            node_path = os.path.join(base_path, marker)
            if not os.path.isdir(node_path):
                return
            next_values = dict(values)
            next_values[role] = marker
            next_paths = dict(paths)
            next_paths[role] = node_path
            out.append(HierarchyEntry(values=next_values, paths=next_paths))
            walk(node_path, level_index + 1, next_values, next_paths)
            return

        names = list_directory_names(base_path)
        if level_index == 0 and project_filter:
            names = [x for x in names if x == project_filter]

        for name in names:
            node_path = os.path.join(base_path, name)
            next_values = dict(values)
            next_values[role] = name
            next_paths = dict(paths)
            next_paths[role] = node_path
            out.append(HierarchyEntry(values=next_values, paths=next_paths))
            walk(node_path, level_index + 1, next_values, next_paths)

    walk(root_dir, 0, {}, {})
    return out


def list_role_values(
    root_dir: str,
    schema: Dict[str, Any],
    role: str,
    filters: Optional[Dict[str, str]] = None,
    kind: str = "raw",
) -> List[str]:
    role = str(role or "").strip().lower()
    if not role:
        return []
    filt = {
        str(k).strip().lower(): str(v).strip()
        for k, v in dict(filters or {}).items()
        if str(v).strip()
    }
    project_filter = filt.get("project", "")
    entries = collect_hierarchy_entries(root_dir, schema, kind=kind, project_filter=project_filter)
    out: Set[str] = set()
    for entry in entries:
        values = entry.values
        if role not in values:
            continue
        ok = True
        for fk, fv in filt.items():
            if fk == role:
                continue
            if values.get(fk) != fv:
                ok = False
                break
        if ok:
            out.add(values.get(role, ""))

    if role == "project" and not out:
        out.update(list_directory_names(root_dir))
    return sorted([x for x in out if str(x).strip()])


def build_role_path(
    root_dir: str,
    schema: Dict[str, Any],
    values: Dict[str, str],
    role: str,
    kind: str = "raw",
) -> str:
    role = str(role or "").strip().lower()
    if not role:
        return ""
    levels = levels_for_kind(schema, kind=kind, include_disabled=False)
    path = os.path.normpath(root_dir)
    have_any = False
    for lvl in levels:
        key = str(lvl.get("key", "")).strip().lower()
        if is_marker_level(key):
            seg = marker_folder_name(lvl)
        else:
            seg = str(values.get(key, "")).strip()
            if not seg:
                return ""
        path = os.path.join(path, seg)
        have_any = True
        if key == role:
            return path
    return path if have_any and role == levels[-1].get("key") else ""


def build_deepest_existing_role_path(
    root_dir: str,
    schema: Dict[str, Any],
    values: Dict[str, str],
    preferred_roles: Sequence[str],
    kind: str = "raw",
) -> str:
    for role in preferred_roles:
        p = build_role_path(root_dir, schema, values, role=role, kind=kind)
        if p:
            return p
    return ""


def extract_values_from_path(
    root_dir: str,
    schema: Dict[str, Any],
    path: str,
    kind: str = "raw",
) -> Dict[str, str]:
    try:
        rel = os.path.relpath(path, root_dir)
    except Exception:
        return {}
    if rel.startswith(".."):
        return {}

    parts = [p for p in rel.split(os.sep) if p and p != "."]
    levels = levels_for_kind(schema, kind=kind, include_disabled=False)
    out: Dict[str, str] = {}
    for idx, part in enumerate(parts):
        if idx >= len(levels):
            break
        lvl = levels[idx]
        key = str(lvl.get("key", "")).strip().lower()
        if key:
            if is_marker_level(key):
                expected = marker_folder_name(lvl)
                if part != expected:
                    break
            out[key] = part
    return out


def resolve_role_path(
    root_dir: str,
    schema: Dict[str, Any],
    role: str,
    filters: Dict[str, str],
    kind: str = "raw",
) -> str:
    role = str(role or "").strip().lower()
    if not role:
        return ""
    filt = {
        str(k).strip().lower(): str(v).strip()
        for k, v in dict(filters or {}).items()
        if str(v).strip()
    }
    project_filter = filt.get("project", "")
    entries = collect_hierarchy_entries(root_dir, schema, kind=kind, project_filter=project_filter)
    matches: Set[str] = set()
    for entry in entries:
        vals = entry.values
        if role not in vals or role not in entry.paths:
            continue
        ok = True
        for k, v in filt.items():
            if vals.get(k) != v:
                ok = False
                break
        if ok:
            matches.add(entry.paths[role])

    if len(matches) == 1:
        return next(iter(matches))
    if matches:
        return sorted(matches)[0]
    return build_role_path(root_dir, schema, filt, role=role, kind=kind)


def preview_path(schema: Dict[str, Any], kind: str = "raw") -> str:
    levels = levels_for_kind(schema, kind=kind, include_disabled=False)
    parts = [kind]
    for lvl in levels:
        key = str(lvl.get("key", "")).strip().lower()
        label = str(lvl.get("label", "")).strip() or str(lvl.get("key", "")).title()
        if is_marker_level(key):
            parts.append(marker_folder_name(lvl))
        else:
            parts.append(f"<{label}>")
    return os.path.join(*parts)
