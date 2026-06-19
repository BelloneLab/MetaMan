"""Cross-cutting filesystem operations shared by the UI.

Centralises the "open / reveal in the OS file manager" logic that used to be
copy-pasted across main.py, navigation_tab.py and staging_tab.py, plus the
rename / delete / copy helpers used by the Explorer and Browse context menus.
Copy operations reuse the proven primitives in :mod:`server_sync` so there is a
single copy implementation in the app.
"""

import os
import sys
from typing import Any, Callable, Dict, Optional

from .server_sync import (
    CopyCancelled, VerificationError, copy_with_progress, mirror_tree, new_stats, _now_iso,
)

_BAD_NAME_CHARS = '<>:"/\\|?*'


def safe_name(name: str) -> str:
    """Strip characters that are illegal in folder names (matches the rule used
    elsewhere in the app)."""
    return "".join(ch for ch in str(name or "").strip() if ch not in _BAD_NAME_CHARS).strip()


# ── opening / revealing ───────────────────────────────────────────────────

def open_path(path: str) -> None:
    """Open *path* (a folder or a file) in the OS default handler.

    Raises on failure so callers can surface a message box.
    """
    target = os.path.normpath(path)
    if not os.path.exists(target):
        parent = os.path.dirname(target)
        if os.path.isdir(parent):
            target = parent
        else:
            raise FileNotFoundError(target)
    if os.name == "nt":
        try:
            os.startfile(target)  # noqa: S606 - intended desktop integration
            return
        except Exception:
            import subprocess
            subprocess.run(["explorer", target])
            return
    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", target])
    else:
        import subprocess
        subprocess.run(["xdg-open", target])


def reveal_path(path: str) -> None:
    """Reveal *path* in the OS file manager, selecting it where supported."""
    target = os.path.normpath(path)
    if not os.path.exists(target):
        raise FileNotFoundError(target)
    if os.name == "nt":
        import subprocess
        # /select, highlights the item inside its parent folder.
        subprocess.run(["explorer", "/select,", target])
        return
    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", "-R", target])
        return
    # Linux: no portable "reveal"; open the containing folder.
    open_path(os.path.dirname(target) if os.path.isfile(target) else target)


# ── rename / delete ────────────────────────────────────────────────────────

def rename_path(path: str, new_name: str) -> str:
    """Rename *path* to a sibling named *new_name*. Returns the new path.

    Refuses to clobber an existing sibling and sanitises the name.
    """
    path = os.path.normpath(path)
    clean = safe_name(new_name)
    if not clean:
        raise ValueError("The new name is empty after removing illegal characters.")
    parent = os.path.dirname(path)
    new_path = os.path.join(parent, clean)
    if os.path.normcase(new_path) == os.path.normcase(path):
        return path
    if os.path.exists(new_path):
        raise FileExistsError(f"A file or folder named '{clean}' already exists here.")
    os.rename(path, new_path)
    return new_path


def trash_available() -> bool:
    """True if Send2Trash is importable (recycle-bin deletes possible)."""
    try:
        import send2trash  # noqa: F401
        return True
    except Exception:
        return False


def delete_path(path: str, permanent: bool = False) -> str:
    """Delete *path*. Prefers the OS recycle bin via Send2Trash unless
    *permanent* is requested or Send2Trash is unavailable.

    Returns ``"trash"`` or ``"permanent"`` describing what happened.
    """
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if not permanent:
        try:
            import send2trash
            send2trash.send2trash(path)
            return "trash"
        except Exception:
            pass  # fall through to permanent delete
    import shutil
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    return "permanent"


# ── copying (file or folder) ────────────────────────────────────────────────

def copy_into(src: str, dest_dir: str, log: Callable[[str], None],
              stats: Optional[Dict[str, Any]] = None,
              *, cancel=None, verify: bool = False) -> Dict[str, Any]:
    """Copy *src* (a file or folder) into *dest_dir*, preserving its basename.

    Folders are mirrored via :func:`server_sync.mirror_tree`; single files via
    :func:`server_sync.copy_with_progress`. *cancel* (a CancelToken) and
    *verify* (post-copy checksum) are forwarded. Returns the copy-statistics
    dict so callers can build a report identical to the backup path.
    """
    import time

    src = os.path.normpath(src)
    base = os.path.basename(src.rstrip("/\\"))
    target = os.path.join(dest_dir, base)

    if os.path.isdir(src):
        s = stats if stats is not None else new_stats()
        if not s["started_at"]:
            s["started_at"] = _now_iso()
        started = time.time()
        mirror_tree(src, target, log, stats=s, cancel=cancel, verify=verify)
        s["finished_at"] = _now_iso()
        dt = max(time.time() - started, 1e-6)
        s["duration_s"] = round(dt, 3)
        s["avg_mbps"] = round((s["bytes_copied"] / dt) / (1024 * 1024), 2)
        s["destination_path"] = target
        return s

    # single file
    s = stats if stats is not None else new_stats()
    s["started_at"] = s["started_at"] or _now_iso()
    started = time.time()
    os.makedirs(dest_dir, exist_ok=True)
    try:
        size = os.path.getsize(src)
    except Exception:
        size = 0
    s["files_total"] += 1
    s["bytes_total"] += size
    try:
        copied, _bps = copy_with_progress(src, target, log, cancel=cancel, verify=verify)
        if copied:
            s["copied"] += 1
            s["bytes_copied"] += size
            if verify:
                s["verified"] += 1
        else:
            s["skipped"] += 1
    except CopyCancelled:
        s["cancelled"] = True
        log("[cancelled] Copy stopped before completion.")
    except VerificationError as exc:
        s["failed"] += 1
        s["verify_failed"] += 1
        s["errors"].append({"path": base, "message": str(exc)})
        log(f"[error] {exc}")
    except Exception as exc:
        s["failed"] += 1
        s["errors"].append({"path": base, "message": str(exc)})
        log(f"[error] Failed to copy {base}: {exc}")
    s["finished_at"] = _now_iso()
    dt = max(time.time() - started, 1e-6)
    s["duration_s"] = round(dt, 3)
    s["avg_mbps"] = round((s["bytes_copied"] / dt) / (1024 * 1024), 2)
    s["destination_path"] = target
    return s
