"""
Structure schema – defines the folder nesting hierarchy used by MetaMan.

Each schema has two chains: ``raw_levels`` and ``processed_levels``.
Every level is either:

* **Core variable** (project, experiment, subject, session) – folder name
  comes from actual data.
* **Optional variable** (recording, trial) – same idea, user-activated.
* **Marker / fixed** (raw, processed, group) – produces a literal folder
  whose name equals the label.

The user can reorder, enable/disable, and rename levels using the
Structure Designer dialog.
"""

from typing import Any, Dict, List

# ── level key classifications ────────────────────────────────────────────

CORE_LEVEL_KEYS: List[str] = ["project", "experiment", "subject", "session"]

MARKER_LEVEL_KEYS: List[str] = ["raw", "processed", "group"]

KNOWN_LEVEL_KEYS: List[str] = [
    "project",
    "raw",
    "experiment",
    "group",
    "recording",
    "subject",
    "session",
    "trial",
    "processed",
]

# ── visual helpers ───────────────────────────────────────────────────────

LEVEL_ICONS: Dict[str, str] = {
    "project":    "\U0001f4c1",   # 📁
    "experiment": "\U0001f9ea",   # 🧪
    "subject":    "\U0001f42d",   # 🐭
    "session":    "\U0001f4c5",   # 📅
    "recording":  "\U0001f3a4",   # 🎤
    "trial":      "\U0001f52c",   # 🔬
    "raw":        "\U0001f4e6",   # 📦
    "processed":  "\u2699\ufe0f", # ⚙️
    "group":      "\U0001f4c2",   # 📂
}

LEVEL_COLORS: Dict[str, str] = {
    "project":    "#4e86d9",
    "experiment": "#e6913e",
    "subject":    "#2c9a5e",
    "session":    "#9b59b6",
    "recording":  "#e74c8b",
    "trial":      "#17a2b8",
    "raw":        "#6c757d",
    "processed":  "#546e7a",
    "group":      "#795548",
}

LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "project":    "Top-level project folder (always first)",
    "experiment": "Experiment or paradigm name",
    "subject":    "Animal / subject ID",
    "session":    "Recording session (date, number, …)",
    "recording":  "Recording run within a session",
    "trial":      "Individual trial folder",
    "raw":        "Fixed 'raw' marker folder",
    "processed":  "Fixed 'processed' marker folder",
    "group":      "Custom fixed grouping folder",
}


# ── query helpers ────────────────────────────────────────────────────────

def is_marker_level(key: str) -> bool:
    """Return *True* if *key* names a fixed-folder (marker) level."""
    return str(key).strip().lower() in MARKER_LEVEL_KEYS


def marker_folder_name(level: Dict[str, Any]) -> str:
    """Return the literal folder name for a marker level."""
    key = str(level.get("key", "")).strip().lower()
    label = str(level.get("label", "")).strip()
    if key in ("raw", "processed"):
        return label if (label and label.lower() != key) else key
    return label or key


# ── defaults / normalisation ─────────────────────────────────────────────

def _default_levels() -> List[Dict[str, Any]]:
    return [
        {"key": "project",    "enabled": True,  "label": "Project"},
        {"key": "experiment", "enabled": True,  "label": "Experiment"},
        {"key": "subject",    "enabled": True,  "label": "Subject"},
        {"key": "session",    "enabled": True,  "label": "Session"},
        {"key": "recording",  "enabled": False, "label": "Recording"},
        {"key": "trial",      "enabled": False, "label": "Trial"},
        {"key": "raw",        "enabled": False, "label": "raw"},
        {"key": "processed",  "enabled": False, "label": "processed"},
        {"key": "group",      "enabled": False, "label": "group"},
    ]


def default_structure_schema() -> Dict[str, Any]:
    return {
        "raw_levels": _default_levels(),
        "processed_levels": _default_levels(),
    }


def normalize_structure_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure *schema* has both chains and every known key is present."""
    if not isinstance(schema, dict):
        return default_structure_schema()

    out: Dict[str, Any] = {}
    for chain_key in ("raw_levels", "processed_levels"):
        levels = schema.get(chain_key, [])
        if not isinstance(levels, list):
            levels = []

        seen: set = set()
        normalized: List[Dict[str, Any]] = []
        for lvl in levels:
            if not isinstance(lvl, dict):
                continue
            key = str(lvl.get("key", "")).strip().lower()
            if key not in KNOWN_LEVEL_KEYS or key in seen:
                continue
            seen.add(key)
            label = str(lvl.get("label", "")).strip() or (
                key if key in MARKER_LEVEL_KEYS else key.title()
            )
            enabled = bool(lvl.get("enabled", False))
            if key == "project":
                enabled = True
            normalized.append({"key": key, "label": label, "enabled": enabled})

        # Append any missing known keys as disabled
        for key in KNOWN_LEVEL_KEYS:
            if key not in seen:
                normalized.append({
                    "key": key,
                    "label": key if key in MARKER_LEVEL_KEYS else key.title(),
                    "enabled": False,
                })
        out[chain_key] = normalized

    return out


# ── preview ──────────────────────────────────────────────────────────────

def preview_path(schema: Dict[str, Any], kind: str = "raw") -> str:
    """Human-readable preview of the folder chain."""
    levels = schema.get(f"{kind}_levels", [])
    parts: List[str] = []
    for lvl in levels:
        if not isinstance(lvl, dict) or not lvl.get("enabled", False):
            continue
        key = str(lvl.get("key", "")).strip().lower()
        label = str(lvl.get("label", "")).strip() or key.title()
        icon = LEVEL_ICONS.get(key, "\U0001f4c1")
        if is_marker_level(key):
            parts.append(f"{icon} {marker_folder_name(lvl)}/")
        else:
            parts.append(f"{icon} <{label}>/")
    return "  \u2192  ".join(parts) if parts else "\u2014"


def enabled_level_keys(schema: Dict[str, Any], kind: str = "raw") -> List[str]:
    """Return *key* names of enabled levels in order."""
    out: List[str] = []
    for lvl in schema.get(f"{kind}_levels", []):
        if not isinstance(lvl, dict) or not lvl.get("enabled", False):
            continue
        out.append(str(lvl.get("key", "")).strip().lower())
    return out
