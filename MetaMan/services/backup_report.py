"""Backup run records and reports.

A *record* is a plain dict describing one backup run (one project/scope to one
destination). It is persisted in app settings for the in-app history and also
written next to the data as JSON + a human-readable text report so the record
travels with the backup.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

REPORT_DIRNAME = "_metaman_backup"


def human_size(n: int) -> str:
    f = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if f < 1024 or unit == "PB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.2f} {unit}"
        f /= 1024.0
    return f"{f:.2f} PB"


def build_record(
    *,
    project: str,
    experiment: str,
    destination_kind: str,
    destination_root: str,
    destination_path: str,
    source_path: str,
    trigger: str,
    stats: Dict[str, Any],
    staging_synced: int = 0,
    error: str = "",
) -> Dict[str, Any]:
    """Assemble a backup-run record from copy *stats* and run context."""
    stats = stats or {}
    failed = int(stats.get("failed", 0) or 0)
    if error:
        status = "error"
    elif failed:
        status = "partial"
    else:
        status = "success"

    return {
        "run_id": str(uuid4()),
        "project": project,
        "experiment": experiment or "",
        "scope": "experiment" if experiment else "project",
        "destination_kind": destination_kind,        # "server" | "hdd"
        "destination_root": destination_root,
        "destination_path": destination_path or stats.get("destination_path", ""),
        "source_path": source_path,
        "trigger": trigger,                            # "manual" | "scheduled"
        "started_at": stats.get("started_at", ""),
        "finished_at": stats.get("finished_at", ""),
        "duration_s": float(stats.get("duration_s", 0.0) or 0.0),
        "status": status,
        "files_total": int(stats.get("files_total", 0) or 0),
        "copied": int(stats.get("copied", 0) or 0),
        "updated": int(stats.get("updated", 0) or 0),
        "skipped": int(stats.get("skipped", 0) or 0),
        "failed": failed,
        "bytes_copied": int(stats.get("bytes_copied", 0) or 0),
        "bytes_total": int(stats.get("bytes_total", 0) or 0),
        "avg_mbps": float(stats.get("avg_mbps", 0.0) or 0.0),
        "staging_synced": int(staging_synced or 0),
        "error": error or "",
        "errors": list(stats.get("errors", []) or []),
    }


def _fmt_duration(seconds: float) -> str:
    s = int(round(seconds or 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def format_report_text(record: Dict[str, Any]) -> str:
    """A clean, human-readable report for one backup run."""
    scope = record.get("project", "")
    if record.get("experiment"):
        scope = f"{scope} / {record['experiment']}"
    dest_kind = "External HDD" if record.get("destination_kind") == "hdd" else "Server"
    lines = [
        "MetaMan backup report",
        "=" * 40,
        f"Project        : {record.get('project', '')}",
        f"Scope          : {scope}",
        f"Destination    : {dest_kind}  ->  {record.get('destination_path', '')}",
        f"Source         : {record.get('source_path', '')}",
        f"Trigger        : {record.get('trigger', '')}",
        f"Status         : {record.get('status', '').upper()}",
        f"Started        : {record.get('started_at', '')}",
        f"Finished       : {record.get('finished_at', '')}",
        f"Duration       : {_fmt_duration(record.get('duration_s', 0))}",
        "",
        f"Files scanned  : {record.get('files_total', 0)}",
        f"  copied (new) : {record.get('copied', 0)}",
        f"  updated      : {record.get('updated', 0)}",
        f"  skipped      : {record.get('skipped', 0)}",
        f"  failed       : {record.get('failed', 0)}",
        f"Data copied    : {human_size(record.get('bytes_copied', 0))}"
        f"  (of {human_size(record.get('bytes_total', 0))} scanned)",
        f"Avg throughput : {record.get('avg_mbps', 0)} MB/s",
    ]
    if record.get("staging_synced"):
        lines.append(f"Staged synced  : {record['staging_synced']} recording(s)")
    if record.get("error"):
        lines += ["", f"Run error      : {record['error']}"]
    errs = record.get("errors") or []
    if errs:
        lines += ["", f"Per-file errors ({len(errs)}):"]
        for e in errs[:50]:
            lines.append(f"  - {e.get('path', '')}: {e.get('message', '')}")
        if len(errs) > 50:
            lines.append(f"  … and {len(errs) - 50} more")
    return "\n".join(lines)


def report_dir(destination_root: str, project: str, experiment: str) -> str:
    """The folder that holds this project/scope's reports under *destination_root*."""
    scope_name = project + (f"__{experiment}" if experiment else "")
    # Keep illegal path chars out of the folder name.
    scope_name = "".join(ch for ch in scope_name if ch not in '<>:"/\\|?*').strip() or "project"
    return os.path.join(destination_root, REPORT_DIRNAME, scope_name)


def write_report_files(destination_root: str, record: Dict[str, Any],
                       log=None) -> Optional[str]:
    """Write the run report next to the backup destination.

    Creates ``<destination_root>/_metaman_backup/<project>[__<exp>]/`` with a
    timestamped ``report_<ts>.json`` + ``report_<ts>.txt``, refreshes
    ``last_report.json`` and appends a row to ``history.csv``. Returns the JSON
    report path (or None on failure). Reports live *outside* the mirrored
    project folder so project trees stay pristine.
    """
    try:
        out_dir = report_dir(destination_root, record.get("project", ""), record.get("experiment", ""))
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(out_dir, f"report_{ts}.json")
        txt_path = os.path.join(out_dir, f"report_{ts}.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(format_report_text(record))
        with open(os.path.join(out_dir, "last_report.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        hist_csv = os.path.join(out_dir, "history.csv")
        new_file = not os.path.exists(hist_csv)
        with open(hist_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow([
                    "finished_at", "status", "trigger", "scope",
                    "files_total", "copied", "updated", "skipped", "failed",
                    "bytes_copied", "duration_s", "destination_path",
                ])
            w.writerow([
                record.get("finished_at", ""), record.get("status", ""),
                record.get("trigger", ""), record.get("scope", ""),
                record.get("files_total", 0), record.get("copied", 0),
                record.get("updated", 0), record.get("skipped", 0),
                record.get("failed", 0), record.get("bytes_copied", 0),
                record.get("duration_s", 0), record.get("destination_path", ""),
            ])
        return json_path
    except Exception as exc:
        if log:
            log(f"[warning] Could not write backup report to destination: {exc}")
        return None
