"""
Staging service – manage locally-created recordings that are linked to
server-resident projects.  Recordings live under ``<data_root>/staging/``
and carry a manifest entry that maps them to a target
``server_root/project/experiment/subject/session`` path.

During backup (manual or scheduled) all pending entries are synced to the
server and marked as *synced*.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from ..config import COPY_CHUNK_BYTES, RAW_DIR_NAME, RAW_DIR_ALIASES

STAGING_FOLDER = "staging"
MANIFEST_FILE = "staging_manifest.json"


def _server_raw_candidates() -> List[str]:
    """Subfolder names that may hold projects under a server root, preferred
    first: the standard ``rawData`` then legacy ``raw``/``rawdata``."""
    out: List[str] = []
    for name in [RAW_DIR_NAME, *RAW_DIR_ALIASES]:
        if name and name not in out:
            out.append(name)
    return out

# ── manifest I/O ────────────────────────────────────────────────────────

def _manifest_path(data_root: str) -> str:
    return os.path.join(data_root, STAGING_FOLDER, MANIFEST_FILE)


def load_manifest(data_root: str) -> List[Dict[str, Any]]:
    path = _manifest_path(data_root)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def save_manifest(data_root: str, entries: List[Dict[str, Any]]):
    staging_dir = os.path.join(data_root, STAGING_FOLDER)
    os.makedirs(staging_dir, exist_ok=True)
    path = _manifest_path(data_root)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


# ── scanning server projects ────────────────────────────────────────────

def list_server_projects(server_root: str) -> List[str]:
    """Return project folder names found under *server_root*."""
    if not server_root or not os.path.isdir(server_root):
        return []
    try:
        base = server_raw_root(server_root)
        return sorted(
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d != STAGING_FOLDER
        )
    except Exception:
        return []


def server_raw_root(server_root: str) -> str:
    """Return the raw-data subfolder of *server_root* (``rawData`` or legacy
    ``raw``) if one exists, else *server_root* itself."""
    for name in _server_raw_candidates():
        cand = os.path.join(server_root, name)
        if os.path.isdir(cand):
            return cand
    return server_root


def list_server_experiments(server_root: str, project: str) -> List[str]:
    base = server_raw_root(server_root)
    proj_dir = os.path.join(base, project)
    if not os.path.isdir(proj_dir):
        return []
    try:
        return sorted(
            d for d in os.listdir(proj_dir)
            if os.path.isdir(os.path.join(proj_dir, d))
        )
    except Exception:
        return []


def list_server_subjects(server_root: str, project: str, experiment: str) -> List[str]:
    base = server_raw_root(server_root)
    exp_dir = os.path.join(base, project, experiment)
    if not os.path.isdir(exp_dir):
        return []
    try:
        return sorted(
            d for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d))
        )
    except Exception:
        return []


# ── create a linked recording ────────────────────────────────────────────

def create_linked_recording(
    data_root: str,
    server_root: str,
    project: str,
    experiment: str,
    subject: str,
    session: str,
) -> Dict[str, Any]:
    """
    Create a local recording folder under ``<data_root>/staging/<project>/<experiment>/<subject>/<session>``
    and register it in the staging manifest.

    Returns the new manifest entry.
    """
    staging_base = os.path.join(data_root, STAGING_FOLDER)
    local_dir = os.path.join(staging_base, project, experiment, subject, session)
    os.makedirs(local_dir, exist_ok=True)

    entry_id = str(uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # write initial metadata.json
    meta = {
        "DateTime": now,
        "Project": project,
        "Experiment": experiment,
        "Animal": subject,
        "Subject": subject,
        "Session": session,
        "Trial": "",
        "Condition": "",
        "Recording": "",
        "Region": "",
        "Experimenter": "",
        "Room": "",
        "Box": "",
        "Comments": "",
        "RootDir": data_root,
        "SessionUUID": entry_id,
        "file_list": [],
        "trial_info": {},
        "trial_assets": {},
        "preprocessing": [],
        "_staging": {
            "entry_id": entry_id,
            "server_root": server_root,
            "status": "pending",
            "created_at": now,
        },
    }
    meta_path = os.path.join(local_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    entry = {
        "id": entry_id,
        "server_root": server_root,
        "project": project,
        "experiment": experiment,
        "subject": subject,
        "session": session,
        "local_path": local_dir,
        "status": "pending",
        "created_at": now,
        "synced_at": "",
        "error": "",
    }

    manifest = load_manifest(data_root)
    manifest.append(entry)
    save_manifest(data_root, manifest)
    return entry


# ── sync helpers ─────────────────────────────────────────────────────────

def _needs_copy(src: str, dst: str) -> bool:
    if not os.path.exists(dst):
        return True
    try:
        return os.path.getsize(src) != os.path.getsize(dst)
    except Exception:
        return True


def _copy_file(src: str, dst: str, log: Callable[[str], None]):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not _needs_copy(src, dst):
        log(f"  [skip] {os.path.basename(src)} (already present)")
        return False
    total = os.path.getsize(src)
    copied = 0
    t0 = time.time()
    with open(src, "rb") as fs, open(dst, "wb") as fd:
        while True:
            chunk = fs.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            fd.write(chunk)
            copied += len(chunk)
            dt = max(time.time() - t0, 1e-6)
            mbps = (copied / dt) / (1024 * 1024)
            log(f"  Copying {os.path.basename(src)}: {copied}/{total} ({mbps:.1f} MB/s)")
    try:
        import shutil
        shutil.copystat(src, dst)
    except Exception:
        pass
    return True


def sync_entry(entry: Dict[str, Any], log: Callable[[str], None]) -> bool:
    """
    Copy all files from *entry['local_path']* into the target server path.
    Returns True on success.
    """
    local = entry["local_path"]
    if not os.path.isdir(local):
        log(f"[error] Local folder missing: {local}")
        return False

    srv_raw = server_raw_root(entry["server_root"])
    dest_base = os.path.join(
        srv_raw,
        entry["project"],
        entry["experiment"],
        entry["subject"],
        entry["session"],
    )
    os.makedirs(dest_base, exist_ok=True)

    scope = f"{entry['project']}/{entry['experiment']}/{entry['subject']}/{entry['session']}"
    log(f"[staging sync] {scope}")
    log(f"  Local  : {local}")
    log(f"  Server : {dest_base}")

    n_copied = 0
    for root, dirs, files in os.walk(local):
        rel = os.path.relpath(root, local)
        dst_dir = os.path.join(dest_base, rel) if rel != "." else dest_base
        os.makedirs(dst_dir, exist_ok=True)
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join(dst_dir, fname)
            if _copy_file(src, dst, log):
                n_copied += 1

    log(f"  Done – {n_copied} file(s) copied to server.")
    return True


def sync_pending(data_root: str, log: Callable[[str], None],
                 entry_ids: Optional[List[str]] = None):
    """
    Sync staged recordings to their server targets.

    If *entry_ids* is given, only those entries are synced; otherwise all
    entries with status ``pending`` are synced.
    """
    manifest = load_manifest(data_root)
    if not manifest:
        log("[staging] No staged recordings.")
        return

    to_sync = []
    for e in manifest:
        if entry_ids is not None:
            if e.get("id") in entry_ids:
                to_sync.append(e)
        elif e.get("status") == "pending":
            to_sync.append(e)

    if not to_sync:
        log("[staging] Nothing to sync.")
        return

    log(f"[staging] Syncing {len(to_sync)} recording(s)...")
    for entry in to_sync:
        try:
            ok = sync_entry(entry, log)
            entry["status"] = "synced" if ok else "error"
            entry["synced_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if ok else ""
            if not ok:
                entry["error"] = "sync failed"
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            log(f"[error] {exc}")

    save_manifest(data_root, manifest)
    log("[staging] Sync complete.")


def remove_entry(data_root: str, entry_id: str):
    """Remove a manifest entry (does NOT delete local files)."""
    manifest = load_manifest(data_root)
    manifest = [e for e in manifest if e.get("id") != entry_id]
    save_manifest(data_root, manifest)


def get_staging_dir(data_root: str) -> str:
    return os.path.join(data_root, STAGING_FOLDER)
