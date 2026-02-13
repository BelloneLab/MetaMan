import os
import time
from typing import Callable, Tuple

from ..config import COPY_CHUNK_BYTES


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _needs_copy(src: str, dst: str) -> bool:
    if not os.path.exists(dst):
        return True
    try:
        return os.path.getsize(src) != os.path.getsize(dst)
    except Exception:
        return True


def copy_with_progress(src: str, dst: str, log: Callable[[str], None]) -> Tuple[bool, float]:
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


def sync_project_to_server(project_dir: str, server_root: str, log: Callable[[str], None]):
    # Function name kept for backward compatibility with callers.
    project_name = os.path.basename(project_dir.rstrip("/\\"))
    dst_project = os.path.join(server_root, project_name)
    _ensure_dir(dst_project)

    for root, dirs, files in os.walk(project_dir):
        rel = os.path.relpath(root, project_dir)
        dst_dir = os.path.join(dst_project, rel) if rel != "." else dst_project
        _ensure_dir(dst_dir)
        for d in dirs:
            _ensure_dir(os.path.join(dst_dir, d))
        for fname in files:
            src = os.path.join(root, fname)
            dst = os.path.join(dst_dir, fname)
            copied, bps = copy_with_progress(src, dst, log)
            if not copied:
                log(f"[ok] Already present at destination: {os.path.relpath(dst, server_root)}")
            else:
                log(f"[ok] Copied to destination at {os.path.relpath(dst, server_root)} @ {bps / 1024 / 1024:.2f} MB/s")
