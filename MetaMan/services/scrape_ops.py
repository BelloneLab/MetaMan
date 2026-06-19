"""Automatic, project-wide metadata scraping.

Wraps the per-session scraper (:mod:`MetaMan.services.metadata_scraper`) in an
idempotent pass over a whole project: enrich every session's metadata with the
machine-derived ``Auto:`` fields and the folder-authoritative identity, then
write the canonical ``metadata.json/.csv/.h5`` triplet.

It is cheap to re-run: with ``only_missing=True`` a session is skipped when it
already carries ``Auto:`` fields and its on-disk file count is unchanged, so the
function is safe to fire automatically whenever a project becomes active.

    from MetaMan.services.scrape_ops import scrape_project
    stats = scrape_project(r"B:/NPX/rawData/mPFC-NAc")        # quiet, idempotent
    # {'scanned': 13, 'updated': 0, 'skipped': 13, 'errors': 0}
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Set

from ..io_ops import load_session_metadata, save_session_triplet
from .metadata_scraper import scrape_session
from .query import iter_sessions, read_session_metadata, resolve_schema

# Folders that are obviously scratch/test data, skipped by default.
DEFAULT_SKIP_SUBJECTS: Set[str] = {"test", "test-behavior"}

Progress = Optional[Callable[[str], None]]

# The canonical sidecars we write ourselves; excluded from the staleness count so
# writing them never makes a just-scraped session look "changed" (which would
# stop the idempotent pass from ever converging).
_SIDECARS = {"metadata.json", "metadata.csv", "metadata.h5"}
_SIG_KEY = "Auto: content files"


def _content_file_count(session_dir: str) -> int:
    """Number of files under *session_dir* excluding the canonical metadata
    sidecars MetaMan writes."""
    n = 0
    for _root, _dirs, files in os.walk(session_dir):
        n += sum(1 for f in files if f.lower() not in _SIDECARS)
    return n


def _needs_scrape(session_dir: str, meta: Dict[str, Any]) -> bool:
    """True when a session has no Auto fields yet, or its content-file count
    changed since the last scrape (so the Auto fields are stale)."""
    if not meta.get("Auto: modality") and _SIG_KEY not in meta:
        return True
    try:
        prev = int(str(meta.get(_SIG_KEY, "")).replace(",", ""))
    except ValueError:
        return True
    return _content_file_count(session_dir) != prev


def scrape_session_into(session_dir: str, identity: Optional[Dict[str, str]] = None,
                        *, deep: bool = False) -> Dict[str, Any]:
    """Scrape one session and write its canonical triplet, refreshing the
    ``Auto:`` fields (machine-derived, always overwritten) while preserving every
    user-entered field. Returns the written metadata dict."""
    canonical = load_session_metadata(session_dir)
    meta = dict(canonical) if canonical else dict(read_session_metadata(session_dir)[0] or {})
    if identity:
        for key, value in identity.items():
            if value:
                meta[key] = value
    # Auto: keys are machine-derived; overwrite them so they reflect disk state.
    meta.update(scrape_session(session_dir, deep=deep))
    meta[_SIG_KEY] = _content_file_count(session_dir)  # staleness signature
    save_session_triplet(session_dir, meta)
    return meta


def scrape_project(project_dir: str, *, deep: bool = False, only_missing: bool = True,
                   skip_subjects: Optional[Set[str]] = None, stamp_identity: bool = True,
                   progress: Progress = None) -> Dict[str, int]:
    """Scrape every session under *project_dir*.

    Parameters
    ----------
    deep:
        Run the heavier probes (video via OpenCV, audio via ``wave``).
    only_missing:
        Skip sessions already scraped whose file count is unchanged (idempotent;
        the default, suitable for an automatic pass).
    skip_subjects:
        Subject folder names to ignore (defaults to scratch/test folders).
    stamp_identity:
        Overwrite Project/Subject/Experiment/Session from the folder tree.
    progress:
        Optional ``callback(message)`` for live logging.
    """
    skip = set(skip_subjects) if skip_subjects is not None else set(DEFAULT_SKIP_SUBJECTS)
    log = progress or (lambda _m: None)
    schema = resolve_schema(project_dir)
    stats = {"scanned": 0, "updated": 0, "skipped": 0, "errors": 0}

    for rec in iter_sessions(project_dir, schema):
        if rec["subject"] in skip:
            continue
        stats["scanned"] += 1
        sdir = rec["path"]
        canonical = load_session_metadata(sdir)
        if only_missing and canonical and not _needs_scrape(sdir, canonical):
            stats["skipped"] += 1
            continue
        identity = {
            "Project": rec["project"], "Subject": rec["subject"], "Animal": rec["subject"],
            "Experiment": rec["experiment"], "Session": rec["session"],
        } if stamp_identity else None
        try:
            scrape_session_into(sdir, identity, deep=deep)
            stats["updated"] += 1
            log(f"scraped {rec['subject']}/{rec['experiment']}/{rec['session']}")
        except Exception as exc:  # one bad session must not abort the pass
            stats["errors"] += 1
            log(f"[error] {sdir}: {exc}")
    return stats
