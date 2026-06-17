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


def mirror_tree(source_dir: str, dest_dir: str, log: Callable[[str], None]):
    """Copy the *contents* of *source_dir* into *dest_dir* (recursive mirror).

    Unlike a basename-based copy, this never invents an extra folder level: the
    children of *source_dir* land directly inside *dest_dir*.
    """
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
            copied, bps = copy_with_progress(src, dst, log)
            rel_dst = os.path.relpath(dst, dest_dir)
            if not copied:
                log(f"[ok] Already present at destination: {rel_dst}")
            else:
                log(f"[ok] Copied to destination at {rel_dst} @ {bps / 1024 / 1024:.2f} MB/s")


def sync_project_to_server(project_dir: str, server_root: str,
                           log: Callable[[str], None], dest_name: str = None):
    """Mirror *project_dir* into ``<server_root>/<dest_name>``.

    *dest_name* is the folder created under *server_root*. Callers should pass
    it explicitly (the project or experiment name) so the destination never
    depends on the source's basename (which may be a shared raw/processed root,
    not the project itself).
    """
    name = (dest_name or os.path.basename(project_dir.rstrip("/\\"))).strip()
    dst_project = os.path.join(server_root, name)
    mirror_tree(project_dir, dst_project, log)
