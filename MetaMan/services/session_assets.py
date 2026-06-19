import csv
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


SESSION_NOTES_FILE = "session_notes.txt"
UPLOAD_DIRNAME = "_metaman_uploads"


def _read_text(path: str) -> str:
    encodings = ("utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin-1")
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as fh:
                return fh.read()
        except UnicodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _norm_header(value: str) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _clean_scalar(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").strip()
    if text.lower() == "nan":
        return ""
    return text


def _parse_key_value_text(text: str) -> Tuple[Dict[str, str], str]:
    fields: Dict[str, str] = {}
    notes: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if notes and notes[-1] != "":
                notes.append("")
            continue
        if line.startswith("#"):
            continue
        split_at = None
        for sep in (":", "=", "\t"):
            if sep in line:
                split_at = sep
                break
        if split_at:
            key, value = line.split(split_at, 1)
            key = _clean_scalar(key)
            if key:
                fields[key] = _clean_scalar(value)
                continue
        notes.append(raw_line.rstrip())
    return fields, "\n".join(notes).strip()


def _sniff_dialect(text: str, ext: str):
    if ext == ".tsv":
        return csv.excel_tab
    try:
        return csv.Sniffer().sniff(text[:4096])
    except Exception:
        return csv.excel


def _parse_delimited_text(text: str, ext: str) -> Dict[str, str]:
    rows = []
    reader = csv.reader(text.splitlines(), dialect=_sniff_dialect(text, ext))
    for row in reader:
        cleaned = [_clean_scalar(cell) for cell in row]
        if any(cleaned):
            rows.append(cleaned)
    if not rows:
        return {}

    header = rows[0]
    norms = [_norm_header(cell) for cell in header]
    key_names = {"key", "field", "name", "metadatafield", "metadata"}
    value_names = {"value", "val", "metadataValue", "metadatavalue"}

    if len(header) >= 2 and norms[0] in key_names and norms[1] in value_names:
        fields: Dict[str, str] = {}
        for row in rows[1:]:
            if len(row) < 2:
                continue
            key = _clean_scalar(row[0])
            if key:
                fields[key] = _clean_scalar(row[1])
        return fields

    if len(header) == 2 and norms[0] not in key_names and norms[1] not in value_names:
        maybe_pairs = all(len(row) >= 2 for row in rows)
        if maybe_pairs and len(rows) > 1:
            fields = {}
            for row in rows:
                key = _clean_scalar(row[0])
                if key:
                    fields[key] = _clean_scalar(row[1])
            return fields

    if len(rows) < 2:
        return {}
    first_data = rows[1]
    return {
        _clean_scalar(key): _clean_scalar(value)
        for key, value in zip(header, first_data)
        if _clean_scalar(key)
    }


def read_metadata_update_file(path: str) -> Tuple[Dict[str, Any], str]:
    """Read a user-supplied metadata update file.

    Returns ``(fields, notes)``. CSV/TSV can be either key/value rows or a
    normal table whose first data row is imported. TXT accepts ``key: value``,
    ``key=value`` and tab-separated pairs; non-pair text is returned as notes.
    JSON must contain an object.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8-sig") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("JSON metadata imports must contain an object.")
        return payload, ""

    text = _read_text(path)
    if ext in (".csv", ".tsv"):
        return _parse_delimited_text(text, ext), ""
    return _parse_key_value_text(text)


def notes_path(session_dir: str) -> str:
    return os.path.join(session_dir, SESSION_NOTES_FILE)


def load_notes(session_dir: str, metadata: Dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    for key in ("Notes", "notes", "Session notes", "Session Notes"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    path = notes_path(session_dir)
    if os.path.isfile(path):
        return _read_text(path).strip()
    return ""


def save_notes(session_dir: str, notes: str) -> str:
    os.makedirs(session_dir, exist_ok=True)
    path = notes_path(session_dir)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(notes or "").rstrip() + ("\n" if str(notes or "").strip() else ""))
    return path


def _safe_segment(value: str) -> str:
    bad = '<>:"/\\|?*'
    cleaned = "".join(ch for ch in str(value or "").strip() if ch not in bad)
    return cleaned or "uncategorized"


def _unique_destination(dest_dir: str, filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        n += 1
    return candidate


def upload_files_to_project(
    project_dir: str,
    file_paths: Iterable[str],
    context_parts: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    """Copy arbitrary files into ``<project>/_metaman_uploads/...``.

    The return value is ready to append to session metadata under
    ``uploaded_files``.
    """
    project_dir = os.path.abspath(project_dir)
    parts = [_safe_segment(part) for part in context_parts if str(part or "").strip()]
    dest_dir = os.path.join(project_dir, UPLOAD_DIRNAME, *parts)
    os.makedirs(dest_dir, exist_ok=True)
    uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records: List[Dict[str, Any]] = []
    for src in file_paths:
        if not src or not os.path.isfile(src):
            continue
        name = os.path.basename(src)
        dst = _unique_destination(dest_dir, name)
        shutil.copy2(src, dst)
        rel = os.path.relpath(dst, project_dir).replace("\\", "/")
        records.append({
            "name": os.path.basename(dst),
            "original_name": name,
            "relative_path": rel,
            "size": os.path.getsize(dst),
            "uploaded_at": uploaded_at,
        })
    return records
