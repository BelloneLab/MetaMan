import hashlib
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

# Suffix for the temporary file an atomic copy writes to before renaming into
# place. A run never leaves a half-written file at the real destination path.
_PARTIAL_SUFFIX = ".mmpart"


class CopyCancelled(Exception):
    """Raised inside the copy loop when a CancelToken is set."""


class VerificationError(Exception):
    """Raised when a post-copy checksum does not match the source."""


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _is_cancelled(cancel) -> bool:
    return bool(cancel is not None and cancel.is_cancelled())


def new_stats() -> Dict[str, Any]:
    """A fresh, zeroed copy-statistics accumulator."""
    return {
        "files_total": 0,
        "copied": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "verified": 0,
        "verify_failed": 0,
        "pruned": 0,
        "cancelled": False,
        "bytes_copied": 0,
        "bytes_total": 0,
        "errors": [],          # list of {"path": rel, "message": str}
        "started_at": "",
        "finished_at": "",
        "duration_s": 0.0,
        "avg_mbps": 0.0,
    }


def _copy_disposition(src: str, dst: str) -> str:
    """Return ``copy`` | ``update`` | ``skip`` for *src* -> *dst*."""
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


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(COPY_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_with_progress(src: str, dst: str, log: Callable[[str], None],
                       *, cancel=None, verify: bool = False) -> Tuple[bool, float]:
    """Atomically copy *src* to *dst* if needed, streaming progress to *log*.

    The bytes are written to a temporary ``*.mmpart`` file and only renamed into
    place once the whole file is copied, so an interrupted run never leaves a
    truncated file at the real path. Returns ``(copied, megabytes_per_second)``;
    ``copied`` is False when the destination already matched.

    *cancel* (a CancelToken) aborts the copy at the next chunk, raising
    :class:`CopyCancelled`. *verify* re-reads the destination and compares a
    SHA-256 against the source, raising :class:`VerificationError` on mismatch.
    """
    _ensure_dir(os.path.dirname(dst))
    if not _needs_copy(src, dst):
        return (False, 0.0)

    total = os.path.getsize(src)
    start = time.time()
    copied = 0
    tmp = dst + _PARTIAL_SUFFIX
    src_hash = hashlib.sha256() if verify else None
    try:
        with open(src, "rb") as fsrc, open(tmp, "wb") as fdst:
            while True:
                if _is_cancelled(cancel):
                    raise CopyCancelled()
                chunk = fsrc.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                fdst.write(chunk)
                if src_hash is not None:
                    src_hash.update(chunk)
                copied += len(chunk)
                dt = max(time.time() - start, 1e-6)
                bps = copied / dt
                log(f"Copying {os.path.basename(src)}: {copied}/{total} bytes ({bps / 1024 / 1024:.2f} MB/s)")
        try:
            import shutil
            shutil.copystat(src, tmp)
        except Exception:
            pass
        if verify and _hash_file(tmp) != src_hash.hexdigest():
            raise VerificationError(f"checksum mismatch after copy: {os.path.basename(dst)}")
        os.replace(tmp, dst)
    except BaseException:
        # Clean up the partial file on any failure/cancel.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise

    bps = copied / max(time.time() - start, 1e-6)
    return (True, bps)


def _prune_extras(source_dir: str, dest_dir: str, source_rel: set,
                  log: Callable[[str], None], stats: Dict[str, Any], cancel=None):
    """Remove files (and now-empty dirs) under *dest_dir* that are not present
    in *source_rel* (relative paths that exist in the source)."""
    for root, dirs, files in os.walk(dest_dir, topdown=False):
        for fname in files:
            if fname.endswith(_PARTIAL_SUFFIX):
                continue
            dpath = os.path.join(root, fname)
            rel = os.path.relpath(dpath, dest_dir).replace("\\", "/")
            if rel not in source_rel:
                try:
                    os.remove(dpath)
                    stats["pruned"] += 1
                    log(f"[prune] Removed extra file at destination: {rel}")
                except Exception as exc:
                    log(f"[warning] Could not prune {rel}: {exc}")
        # remove empty directories left behind
        if not os.listdir(root) and os.path.normpath(root) != os.path.normpath(dest_dir):
            try:
                os.rmdir(root)
            except Exception:
                pass
        if _is_cancelled(cancel):
            return


def mirror_tree(source_dir: str, dest_dir: str, log: Callable[[str], None],
                stats: Optional[Dict[str, Any]] = None,
                *, cancel=None, verify: bool = False, prune: bool = False) -> Dict[str, Any]:
    """Copy the *contents* of *source_dir* into *dest_dir* (recursive mirror).

    Returns a statistics dict (see :func:`new_stats`). Pass an existing *stats*
    to accumulate across several calls. *cancel*/*verify*/*prune* are forwarded
    per file; *prune* additionally removes destination files with no source
    counterpart (an opt-in true-mirror pass).
    """
    s = stats if stats is not None else new_stats()
    own_run = stats is None
    started = time.time()
    if own_run and not s["started_at"]:
        s["started_at"] = _now_iso()

    _ensure_dir(dest_dir)
    source_rel: set = set()
    cancelled = False
    for root, dirs, files in os.walk(source_dir):
        if _is_cancelled(cancel):
            cancelled = True
            break
        rel = os.path.relpath(root, source_dir)
        cur_dst = os.path.join(dest_dir, rel) if rel != "." else dest_dir
        _ensure_dir(cur_dst)
        for d in dirs:
            _ensure_dir(os.path.join(cur_dst, d))
        for fname in files:
            if _is_cancelled(cancel):
                cancelled = True
                break
            src = os.path.join(root, fname)
            dst = os.path.join(cur_dst, fname)
            rel_dst = os.path.relpath(dst, dest_dir)
            if prune:
                source_rel.add(rel_dst.replace("\\", "/"))
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
                copied, bps = copy_with_progress(src, dst, log, cancel=cancel, verify=verify)
                if not copied:
                    s["skipped"] += 1
                    log(f"[ok] Already present at destination: {rel_dst}")
                    continue
                s["bytes_copied"] += size
                if verify:
                    s["verified"] += 1
                if disposition == "update":
                    s["updated"] += 1
                    log(f"[ok] Updated at destination {rel_dst} @ {bps / 1024 / 1024:.2f} MB/s")
                else:
                    s["copied"] += 1
                    log(f"[ok] Copied to destination at {rel_dst} @ {bps / 1024 / 1024:.2f} MB/s")
            except CopyCancelled:
                cancelled = True
                break
            except VerificationError as exc:
                s["failed"] += 1
                s["verify_failed"] += 1
                if len(s["errors"]) < _MAX_RECORDED_ERRORS:
                    s["errors"].append({"path": rel_dst, "message": str(exc)})
                log(f"[error] {exc}")
            except Exception as exc:
                s["failed"] += 1
                if len(s["errors"]) < _MAX_RECORDED_ERRORS:
                    s["errors"].append({"path": rel_dst, "message": str(exc)})
                log(f"[error] Failed to copy {rel_dst}: {exc}")
        if cancelled:
            break

    if cancelled:
        s["cancelled"] = True
        log("[cancelled] Copy stopped before completion.")
    elif prune:
        _prune_extras(source_dir, dest_dir, source_rel, log, s, cancel=cancel)

    if own_run:
        s["finished_at"] = _now_iso()
        dt = max(time.time() - started, 1e-6)
        s["duration_s"] = round(dt, 3)
        s["avg_mbps"] = round((s["bytes_copied"] / dt) / (1024 * 1024), 2)
    return s


def sync_project_to_server(project_dir: str, server_root: str,
                           log: Callable[[str], None], dest_name: str = None,
                           *, cancel=None, verify: bool = False, prune: bool = False) -> Dict[str, Any]:
    """Mirror *project_dir* into ``<server_root>/<dest_name>``.

    Returns the copy-statistics dict; existing callers that ignore the return
    value are unaffected.
    """
    name = (dest_name or os.path.basename(project_dir.rstrip("/\\"))).strip()
    dst_project = os.path.join(server_root, name)
    stats = mirror_tree(project_dir, dst_project, log, cancel=cancel, verify=verify, prune=prune)
    stats["destination_path"] = dst_project
    return stats


# ── helpers used by callers (free-space precheck, dry-run diff) ────────────

def tree_size(path: str) -> int:
    """Total size in bytes of all files under *path* (0 on error)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except Exception:
                pass
    return total


def free_space(path: str) -> int:
    """Free bytes on the filesystem that would hold *path* (0 if unknown)."""
    import shutil
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or path).free
    except Exception:
        return 0


def diff_tree(source_dir: str, dest_dir: str) -> Dict[str, List[str]]:
    """Compare *source_dir* against *dest_dir* without copying anything.

    Returns relative paths grouped as ``only_source`` (would be copied),
    ``different`` (size/mtime differ, would be updated) and ``only_dest``
    (present at destination only, would be pruned)."""
    out: Dict[str, List[str]] = {"only_source": [], "different": [], "only_dest": []}
    src_rel: set = set()
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            sp = os.path.join(root, fname)
            rel = os.path.relpath(sp, source_dir)
            src_rel.add(rel.replace("\\", "/"))
            dp = os.path.join(dest_dir, rel)
            disp = _copy_disposition(sp, dp)
            if disp == "copy":
                out["only_source"].append(rel)
            elif disp == "update":
                out["different"].append(rel)
    if os.path.isdir(dest_dir):
        for root, _dirs, files in os.walk(dest_dir):
            for fname in files:
                if fname.endswith(_PARTIAL_SUFFIX):
                    continue
                rel = os.path.relpath(os.path.join(root, fname), dest_dir)
                if rel.replace("\\", "/") not in src_rel:
                    out["only_dest"].append(rel)
    return out


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
