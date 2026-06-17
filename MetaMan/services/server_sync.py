import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import COPY_CHUNK_BYTES

# Files matching the same size *and* a destination mtime that is not older than
# the source are treated as unchanged. A 2s tolerance absorbs FAT/network
# filesystem timestamp rounding so we do not needlessly recopy.
_MTIME_TOLERANCE_S = 2.0

# Per-run error cap so a pathological run can never bloat the report.
_MAX_RECORDED_ERRORS = 200


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def new_stats() -> Dict[str, Any]:
    """A fresh, zeroed copy-statistics accumulator."""
    return {
        "files_total": 0,
        "copied": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "bytes_copied": 0,
        "bytes_total": 0,
        "errors": [],          # list of {"path": rel, "message": str}
        "started_at": "",
        "finished_at": "",
        "duration_s": 0.0,
        "avg_mbps": 0.0,
    }


def _copy_disposition(src: str, dst: str) -> str:
    """Return ``copy`` | ``update`` | ``skip`` for *src* -> *dst*.

    A file is *skipped* only when it already exists with the same size and a
    destination mtime at least as new as the source (within tolerance). A
    same-size but newer source is treated as an *update* so in-place edits and
    truncated/restarted writes are not silently missed (the old size-only check
    would have skipped them).
    """
    if not os.path.exists(dst):
        return "copy"
    try:
        if os.path.getsize(src) != os.path.getsize(dst):
            return "update"
        if os.path.getmtime(src) > os.path.getmtime(dst) + _MTIME_TOLERANCE_S:
            return "update"
        return "skip"
    except Exception:
        return "update"


def _needs_copy(src: str, dst: str) -> bool:
    """Back-compat boolean form of :func:`_copy_disposition`."""
    return _copy_disposition(src, dst) != "skip"


def copy_with_progress(src: str, dst: str, log: Callable[[str], None]) -> Tuple[bool, float]:
    """Copy *src* to *dst* if needed, streaming progress to *log*.

    Returns ``(copied, megabytes_per_second)``. ``copied`` is False when the
    destination already matched and the copy was skipped.
    """
    _ensure_dir(os.path.dirname(dst))
    if not _needs_copy(src, dst):
        return (False, 0.0)

    total = os.path.getsize(src)
    start = time.time()
    copied = 0
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            chunk = fsrc.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            fdst.write(chunk)
            copied += len(chunk)
            dt = max(time.time() - start, 1e-6)
            bps = copied / dt
            log(f"Copying {os.path.basename(src)}: {copied}/{total} bytes ({bps / 1024 / 1024:.2f} MB/s)")
    try:
        import shutil

        shutil.copystat(src, dst)
    except Exception:
        pass
    bps = copied / max(time.time() - start, 1e-6)
    return (True, bps)


def mirror_tree(source_dir: str, dest_dir: str, log: Callable[[str], None],
                stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Copy the *contents* of *source_dir* into *dest_dir* (recursive mirror).

    Unlike a basename-based copy, this never invents an extra folder level: the
    children of *source_dir* land directly inside *dest_dir*.

    Returns a statistics dict (see :func:`new_stats`). Pass an existing *stats*
    to accumulate across several calls (e.g. a multi-source backup).
    """
    s = stats if stats is not None else new_stats()
    own_run = stats is None
    started = time.time()
    if own_run and not s["started_at"]:
        s["started_at"] = _now_iso()

    _ensure_dir(dest_dir)
    for root, dirs, files in os.walk(source_dir):
        rel = os.path.relpath(root, source_dir)
        cur_dst = os.path.join(dest_dir, rel) if rel != "." else dest_dir
        _ensure_dir(cur_dst)
        for d in dirs:
            _ensure_dir(os.path.join(cur_dst, d))
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join(cur_dst, fname)
            rel_dst = os.path.relpath(dst, dest_dir)
            s["files_total"] += 1
            try:
                size = os.path.getsize(src)
            except Exception:
                size = 0
            s["bytes_total"] += size
            disposition = _copy_disposition(src, dst)
            try:
                if disposition == "skip":
                    s["skipped"] += 1
                    log(f"[ok] Already present at destination: {rel_dst}")
                    continue
                copied, bps = copy_with_progress(src, dst, log)
                if not copied:
                    s["skipped"] += 1
                    log(f"[ok] Already present at destination: {rel_dst}")
                    continue
                s["bytes_copied"] += size
                if disposition == "update":
                    s["updated"] += 1
                    log(f"[ok] Updated at destination {rel_dst} @ {bps / 1024 / 1024:.2f} MB/s")
                else:
                    s["copied"] += 1
                    log(f"[ok] Copied to destination at {rel_dst} @ {bps / 1024 / 1024:.2f} MB/s")
            except Exception as exc:
                s["failed"] += 1
                if len(s["errors"]) < _MAX_RECORDED_ERRORS:
                    s["errors"].append({"path": rel_dst, "message": str(exc)})
                log(f"[error] Failed to copy {rel_dst}: {exc}")

    if own_run:
        s["finished_at"] = _now_iso()
        dt = max(time.time() - started, 1e-6)
        s["duration_s"] = round(dt, 3)
        s["avg_mbps"] = round((s["bytes_copied"] / dt) / (1024 * 1024), 2)
    return s


def sync_project_to_server(project_dir: str, server_root: str,
                           log: Callable[[str], None], dest_name: str = None) -> Dict[str, Any]:
    """Mirror *project_dir* into ``<server_root>/<dest_name>``.

    *dest_name* is the folder created under *server_root*. Callers should pass
    it explicitly (the project or experiment name) so the destination never
    depends on the source's basename (which may be a shared raw/processed root,
    not the project itself).

    Returns the copy-statistics dict; existing callers that ignore the return
    value are unaffected.
    """
    name = (dest_name or os.path.basename(project_dir.rstrip("/\\"))).strip()
    dst_project = os.path.join(server_root, name)
    stats = mirror_tree(project_dir, dst_project, log)
    stats["destination_path"] = dst_project
    return stats


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
