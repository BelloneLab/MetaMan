import json
import logging
import os
import tempfile
import threading
from typing import Any, Dict, Optional
from .config import (
    SETTINGS_FILE, DEFAULT_DATA_ROOT, DEFAULT_RAW_ROOT, DEFAULT_PROCESSED_ROOT,
    RAW_DIR_NAME, PROCESSED_DIR_NAME, RAW_DIR_ALIASES, PROCESSED_DIR_ALIASES,
)

log = logging.getLogger(__name__)

# Bumped when the on-disk settings layout changes; ``_migrate`` upgrades older
# files in place.
SETTINGS_VERSION = 1


def _sanitize_folder_name(name: str) -> str:
    bad = '<>:"/\\|?*'
    return "".join(ch for ch in str(name or "").strip() if ch not in bad).strip()


class AppSettings:
    def __init__(self):
        self._data: Dict = {
            "data_root": DEFAULT_DATA_ROOT,
            "raw_root": DEFAULT_RAW_ROOT,
            "processed_root": DEFAULT_PROCESSED_ROOT,
            "raw_dir_name": RAW_DIR_NAME,            # user-configurable folder name
            "processed_dir_name": PROCESSED_DIR_NAME,  # user-configurable folder name
            "last_opened_project": "",
            "last_opened_experiment": "",
            "last_opened_session_path": "",
            "server_roots_by_project": {},  # {project: server_root}
            "hdd_roots_by_project": {},  # {project: hdd_root}
            "backup_schedules_by_project": {},  # {project: {"enabled": bool, "time": "HH:MM", "last_run_date": "YYYY-MM-DD", "backup_whole_project": bool, "enabled_experiments": [], "run_dates_by_experiment": {exp: "YYYY-MM-DD"}, "destination_mode": "server|hdd|both"}}
            "data_reorganizer_settings": {},
            "recording_tab_settings": {},
            "preprocessing_tab_settings": {},
            "staging_tab_settings": {},
            "explorer_settings": {},  # {"local_root": ..., "server_root": ...}
            "backup_runs": [],  # capped list of backup-run records (newest last)
            "loaded_project": {},
            "structure_schema": {},
            "structure_schemas_by_project": {},
            "_version": SETTINGS_VERSION,
        }
        self._lock = threading.RLock()
        self.load()
        self.ensure_storage_roots()

    def _fallback_data_root(self) -> str:
        return os.path.join(os.path.expanduser("~"), "MetaManData")

    def _path_drive_is_available(self, path: str) -> bool:
        p = str(path or "").strip()
        if not p:
            return False
        if os.name != "nt":
            return True
        drive, _ = os.path.splitdrive(p)
        if drive:
            return os.path.exists(drive + os.sep)
        return True

    # ── configurable folder names ─────────────────────────────────────
    @property
    def raw_dir_name(self) -> str:
        return _sanitize_folder_name(self._data.get("raw_dir_name")) or RAW_DIR_NAME

    @property
    def processed_dir_name(self) -> str:
        return _sanitize_folder_name(self._data.get("processed_dir_name")) or PROCESSED_DIR_NAME

    def _raw_markers(self) -> set:
        """Folder basenames (lowercase) recognised as a raw root."""
        return set(RAW_DIR_ALIASES) | {self.raw_dir_name.lower()}

    def _processed_markers(self) -> set:
        return set(PROCESSED_DIR_ALIASES) | {self.processed_dir_name.lower()}

    def _strip_root_marker(self, path: str) -> str:
        """If *path* points at a raw/processed folder, return its parent.

        Lets users pick either ``B:/NPX`` or ``B:/NPX/rawData`` as the data root
        and end up with the same result: projects are scanned from
        ``<data_root>/<raw_dir_name>``.
        """
        p = str(path or "").strip()
        if not p:
            return p
        base = os.path.basename(os.path.normpath(p)).lower()
        if base in (self._raw_markers() | self._processed_markers()):
            parent = os.path.dirname(os.path.normpath(p))
            if parent:
                return parent
        return p

    def set_folder_names(self, raw_name: str, processed_name: str):
        """Set the raw/processed folder names and recompute the roots."""
        raw_name = _sanitize_folder_name(raw_name) or RAW_DIR_NAME
        processed_name = _sanitize_folder_name(processed_name) or PROCESSED_DIR_NAME
        self._data["raw_dir_name"] = raw_name
        self._data["processed_dir_name"] = processed_name
        data_root = self.data_root
        self._data["raw_root"] = os.path.join(data_root, raw_name)
        self._data["processed_root"] = os.path.join(data_root, processed_name)
        self.save()

    def _coerce_data_root(self) -> str:
        data_root = str(self._data.get("data_root") or "").strip()
        raw_root = str(self._data.get("raw_root") or "").strip()
        proc_root = str(self._data.get("processed_root") or "").strip()

        if not data_root:
            if os.path.basename(raw_root).lower() in self._raw_markers():
                data_root = os.path.dirname(raw_root)
            elif os.path.basename(proc_root).lower() in self._processed_markers():
                data_root = os.path.dirname(proc_root)
            else:
                data_root = DEFAULT_DATA_ROOT

        # If the data root itself points at a raw/processed folder, step up so
        # projects living directly inside it are discovered. This repairs older
        # settings that stored e.g. "B:/NPX/rawData" as the data root.
        data_root = self._strip_root_marker(data_root)

        if not self._path_drive_is_available(data_root):
            data_root = self._fallback_data_root()
        return data_root

    def load(self):
        try:
            if SETTINGS_FILE.exists():
                self._data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
            self._migrate()
            # Ensure folder-style defaults and coerce invalid legacy roots.
            data_root = self._coerce_data_root()
            self._data["data_root"] = data_root
            self._data["raw_root"] = os.path.join(data_root, self.raw_dir_name)
            self._data["processed_root"] = os.path.join(data_root, self.processed_dir_name)
        except Exception:
            log.exception("Failed to load settings from %s; using defaults.", SETTINGS_FILE)

    def _migrate(self):
        """Upgrade an older settings dict in place. No-op for v1; the hook keeps
        future format changes from silently breaking existing installs."""
        ver = int(self._data.get("_version", 0) or 0)
        # (future migrations keyed on *ver* go here)
        self._data["_version"] = SETTINGS_VERSION

    def ensure_storage_roots(self):
        data_root = self._coerce_data_root()
        raw_root = os.path.join(data_root, self.raw_dir_name)
        proc_root = os.path.join(data_root, self.processed_dir_name)
        try:
            os.makedirs(raw_root, exist_ok=True)
            os.makedirs(proc_root, exist_ok=True)
        except Exception:
            data_root = self._fallback_data_root()
            raw_root = os.path.join(data_root, self.raw_dir_name)
            proc_root = os.path.join(data_root, self.processed_dir_name)
            os.makedirs(raw_root, exist_ok=True)
            os.makedirs(proc_root, exist_ok=True)
        self._data["data_root"] = data_root
        self._data["raw_root"] = raw_root
        self._data["processed_root"] = proc_root
        self.save()

    def save(self):
        """Persist settings atomically and thread-safely.

        Writes to a temp file in the same directory then ``os.replace``s it over
        the target, so a crash or concurrent writer can never leave a truncated
        settings file. The lock serialises writers from worker threads (e.g. a
        scheduled-backup mark running while the GUI records a run)."""
        with self._lock:
            try:
                payload = json.dumps(self._data, indent=2)
                target = str(SETTINGS_FILE)
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                fd, tmp = tempfile.mkstemp(
                    prefix=".settings_", suffix=".tmp",
                    dir=os.path.dirname(target) or ".",
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(payload)
                    os.replace(tmp, target)
                finally:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
            except Exception:
                log.exception("Failed to save settings to %s", SETTINGS_FILE)

    @property
    def raw_root(self) -> str:
        return self._data.get("raw_root") or os.path.join(self.data_root, self.raw_dir_name)

    @raw_root.setter
    def raw_root(self, v: str):
        self._data["raw_root"] = v
        if os.path.basename(v).lower() in self._raw_markers():
            self._data["data_root"] = os.path.dirname(v)
        self.save()

    @property
    def processed_root(self) -> str:
        return self._data.get("processed_root") or os.path.join(self.data_root, self.processed_dir_name)

    @processed_root.setter
    def processed_root(self, v: str):
        self._data["processed_root"] = v
        if os.path.basename(v).lower() in self._processed_markers():
            self._data["data_root"] = os.path.dirname(v)
        self.save()

    @property
    def data_root(self) -> str:
        return self._data.get("data_root") or DEFAULT_DATA_ROOT

    @data_root.setter
    def data_root(self, v: str):
        v = self._strip_root_marker(v)
        self._data["data_root"] = v
        self._data["raw_root"] = os.path.join(v, self.raw_dir_name)
        self._data["processed_root"] = os.path.join(v, self.processed_dir_name)
        self.save()

    def _heal_server_root(self, path: str) -> str:
        """Map a stored server root whose last folder is a legacy raw/processed
        name to the renamed folder when only the new one exists on disk.

        e.g. ``.../PROJECTS/raw`` -> ``.../PROJECTS/rawData`` after a rename.
        """
        p = str(path or "").strip()
        if not p or os.path.isdir(p):
            return p
        norm = os.path.normpath(p)
        base = os.path.basename(norm).lower()
        parent = os.path.dirname(norm)
        if not parent:
            return p
        if base in self._raw_markers():
            cand = os.path.join(parent, self.raw_dir_name)
            if os.path.isdir(cand):
                return cand
        if base in self._processed_markers():
            cand = os.path.join(parent, self.processed_dir_name)
            if os.path.isdir(cand):
                return cand
        return p

    def get_server_root_for_project(self, project: str) -> str:
        raw = (self._data.get("server_roots_by_project") or {}).get(project, "")
        return self._heal_server_root(raw)

    def put_server_root_for_project(self, project: str, server_root: str):
        d = self._data.setdefault("server_roots_by_project", {})
        d[project] = server_root
        self.save()

    def get_hdd_root_for_project(self, project: str) -> str:
        return (self._data.get("hdd_roots_by_project") or {}).get(project, "")

    def put_hdd_root_for_project(self, project: str, hdd_root: str):
        d = self._data.setdefault("hdd_roots_by_project", {})
        d[project] = hdd_root
        self.save()

    def get_backup_schedule_for_project(self, project: str) -> Dict[str, Any]:
        schedules = self._data.get("backup_schedules_by_project") or {}
        raw = schedules.get(project, {}) if isinstance(schedules, dict) else {}
        run_dates = raw.get("run_dates_by_experiment", {})
        if not isinstance(run_dates, dict):
            run_dates = {}
        enabled_exps = raw.get("enabled_experiments", [])
        if not isinstance(enabled_exps, list):
            enabled_exps = []
        return {
            "enabled": bool(raw.get("enabled", False)),
            "time": str(raw.get("time", "")),
            "last_run_date": str(raw.get("last_run_date", "")),
            "backup_whole_project": bool(raw.get("backup_whole_project", True)),
            "enabled_experiments": [str(x) for x in enabled_exps if str(x).strip()],
            "run_dates_by_experiment": {str(k): str(v) for k, v in run_dates.items()},
            "destination_mode": str(raw.get("destination_mode", "server") or "server"),
            "verify": bool(raw.get("verify", False)),
            "prune": bool(raw.get("prune", False)),
        }

    def get_all_backup_schedules(self) -> Dict[str, Dict[str, Any]]:
        schedules = self._data.get("backup_schedules_by_project") or {}
        if not isinstance(schedules, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for project, raw in schedules.items():
            if not isinstance(raw, dict):
                continue
            run_dates = raw.get("run_dates_by_experiment", {})
            if not isinstance(run_dates, dict):
                run_dates = {}
            enabled_exps = raw.get("enabled_experiments", [])
            if not isinstance(enabled_exps, list):
                enabled_exps = []
            out[str(project)] = {
                "enabled": bool(raw.get("enabled", False)),
                "time": str(raw.get("time", "")),
                "last_run_date": str(raw.get("last_run_date", "")),
                "backup_whole_project": bool(raw.get("backup_whole_project", True)),
                "enabled_experiments": [str(x) for x in enabled_exps if str(x).strip()],
                "run_dates_by_experiment": {str(k): str(v) for k, v in run_dates.items()},
                "destination_mode": str(raw.get("destination_mode", "server") or "server"),
                "verify": bool(raw.get("verify", False)),
                "prune": bool(raw.get("prune", False)),
            }
        return out

    def put_backup_schedule_for_project(
        self,
        project: str,
        enabled: bool,
        time_hhmm: str,
        backup_whole_project: bool = True,
        enabled_experiments: Optional[list] = None,
        destination_mode: str = "server",
        verify: bool = False,
        prune: bool = False,
    ):
        schedules = self._data.setdefault("backup_schedules_by_project", {})
        prev = schedules.get(project, {}) if isinstance(schedules, dict) else {}
        prev_run_dates = prev.get("run_dates_by_experiment", {})
        if not isinstance(prev_run_dates, dict):
            prev_run_dates = {}
        exps = enabled_experiments if isinstance(enabled_experiments, list) else prev.get("enabled_experiments", [])
        if not isinstance(exps, list):
            exps = []
        schedules[project] = {
            "enabled": bool(enabled),
            "time": str(time_hhmm or ""),
            "last_run_date": str(prev.get("last_run_date", "")),
            "backup_whole_project": bool(backup_whole_project),
            "enabled_experiments": [str(x) for x in exps if str(x).strip()],
            "run_dates_by_experiment": {str(k): str(v) for k, v in prev_run_dates.items()},
            "destination_mode": str(destination_mode or "server"),
            "verify": bool(verify),
            "prune": bool(prune),
        }
        self.save()

    def mark_backup_schedule_run(self, project: str, run_date: str, experiment: str = ""):
        schedules = self._data.setdefault("backup_schedules_by_project", {})
        prev = schedules.get(project, {}) if isinstance(schedules, dict) else {}
        run_dates = prev.get("run_dates_by_experiment", {})
        if not isinstance(run_dates, dict):
            run_dates = {}
        if experiment:
            run_dates[str(experiment)] = str(run_date or "")
        exps = prev.get("enabled_experiments", [])
        if not isinstance(exps, list):
            exps = []
        schedules[project] = {
            "enabled": bool(prev.get("enabled", False)),
            "time": str(prev.get("time", "")),
            "last_run_date": str(run_date or ""),
            "backup_whole_project": bool(prev.get("backup_whole_project", True)),
            "enabled_experiments": [str(x) for x in exps if str(x).strip()],
            "run_dates_by_experiment": {str(k): str(v) for k, v in run_dates.items()},
            "destination_mode": str(prev.get("destination_mode", "server") or "server"),
            "verify": bool(prev.get("verify", False)),
            "prune": bool(prev.get("prune", False)),
        }
        self.save()

    def get_recording_tab_settings(self) -> Dict[str, Any]:
        raw = self._data.get("recording_tab_settings") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def put_recording_tab_settings(self, data: Dict[str, Any]):
        self._data["recording_tab_settings"] = dict(data or {})
        self.save()

    def get_preprocessing_tab_settings(self) -> Dict[str, Any]:
        raw = self._data.get("preprocessing_tab_settings") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def put_preprocessing_tab_settings(self, data: Dict[str, Any]):
        self._data["preprocessing_tab_settings"] = dict(data or {})
        self.save()

    def get_data_reorganizer_settings(self) -> Dict[str, Any]:
        raw = self._data.get("data_reorganizer_settings") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def put_data_reorganizer_settings(self, data: Dict[str, Any]):
        self._data["data_reorganizer_settings"] = dict(data or {})
        self.save()

    def get_staging_tab_settings(self) -> Dict[str, Any]:
        raw = self._data.get("staging_tab_settings") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def put_staging_tab_settings(self, data: Dict[str, Any]):
        self._data["staging_tab_settings"] = dict(data or {})
        self.save()

    def get_explorer_settings(self) -> Dict[str, Any]:
        raw = self._data.get("explorer_settings") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def put_explorer_settings(self, data: Dict[str, Any]):
        self._data["explorer_settings"] = dict(data or {})
        self.save()

    # ── backup history ────────────────────────────────────────────────
    _MAX_BACKUP_RUNS = 200

    def add_backup_run(self, record: Dict[str, Any]):
        """Append a backup-run record, keeping only the newest entries."""
        runs = self._data.get("backup_runs")
        if not isinstance(runs, list):
            runs = []
        runs.append(dict(record or {}))
        if len(runs) > self._MAX_BACKUP_RUNS:
            runs = runs[-self._MAX_BACKUP_RUNS:]
        self._data["backup_runs"] = runs
        self.save()

    def get_backup_history(self, project: Optional[str] = None,
                           limit: Optional[int] = None) -> list:
        """Return backup-run records newest-first, optionally filtered by
        *project* and capped to *limit*."""
        runs = self._data.get("backup_runs")
        if not isinstance(runs, list):
            return []
        out = [dict(r) for r in runs if isinstance(r, dict)]
        if project:
            out = [r for r in out if str(r.get("project", "")) == project]
        out.reverse()  # newest first
        if limit is not None:
            out = out[:limit]
        return out

    def get_last_backup_for(self, project: str, kind: Optional[str] = None) -> Dict[str, Any]:
        """Return the most recent backup-run record for *project* (optionally a
        specific destination *kind*), or an empty dict."""
        for r in self.get_backup_history(project=project):
            if kind is None or str(r.get("destination_kind", "")) == kind:
                return r
        return {}

    def get_loaded_project(self) -> Dict[str, str]:
        proj = self._data.get("loaded_project") or {}
        return {
            "name": str(proj.get("name", "")),
            "source_path": str(proj.get("source_path", "")),
            "source_type": str(proj.get("source_type", "")),
            "destination_path": str(proj.get("destination_path", "")),
        }

    def put_loaded_project(self, name: str, source_path: str, source_type: str, destination_path: str):
        self._data["loaded_project"] = {
            "name": name, "source_path": source_path,
            "source_type": source_type, "destination_path": destination_path,
        }
        self.save()

    def get_metadata_template(self) -> Dict[str, Any]:
        raw = self._data.get("metadata_template") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def put_metadata_template(self, template: Dict[str, Any]):
        self._data["metadata_template"] = dict(template or {})
        self.save()

    def get_structure_schema(self) -> Dict[str, Any]:
        raw = self._data.get("structure_schema") or {}
        return dict(raw) if isinstance(raw, dict) else {}

    def put_structure_schema(self, schema: Dict[str, Any]):
        self._data["structure_schema"] = dict(schema or {})
        self.save()

    def get_project_structure_schema(self, project: str) -> Dict[str, Any]:
        schemas = self._data.get("structure_schemas_by_project") or {}
        raw = schemas.get(project, {}) if isinstance(schemas, dict) else {}
        return dict(raw) if isinstance(raw, dict) else {}

    def resolve_structure_schema(self, project: str = "") -> Dict[str, Any]:
        """Return the normalized schema that applies to *project*: its own
        per-project schema if set, otherwise the default (all-projects) schema.
        """
        from .services.structure_schema import (
            normalize_structure_schema, default_structure_schema,
        )
        raw = self.get_project_structure_schema(project) if project else {}
        if not raw:
            raw = self.get_structure_schema()
        return normalize_structure_schema(raw or default_structure_schema())

    def put_project_structure_schema(self, project: str, schema: Dict[str, Any]):
        d = self._data.setdefault("structure_schemas_by_project", {})
        d[project] = dict(schema or {})
        self.save()

    @property
    def last_opened_project(self) -> str:
        return self._data.get("last_opened_project", "")

    @last_opened_project.setter
    def last_opened_project(self, v: str):
        self._data["last_opened_project"] = v
        self.save()

    @property
    def last_opened_experiment(self) -> str:
        return self._data.get("last_opened_experiment", "")

    @last_opened_experiment.setter
    def last_opened_experiment(self, v: str):
        self._data["last_opened_experiment"] = v
        self.save()

    @property
    def last_opened_session_path(self) -> str:
        return self._data.get("last_opened_session_path", "")

    @last_opened_session_path.setter
    def last_opened_session_path(self, v: str):
        self._data["last_opened_session_path"] = v
        self.save()


class AppState:
    def __init__(self):
        self.settings = AppSettings()
        self.current_project: str = self.settings.last_opened_project or ""
        self.current_experiment: str = self.settings.last_opened_experiment or ""
        self.current_animal: str = ""
        self.current_session: str = ""
        self.current_session_path: str = self.settings.last_opened_session_path or ""

    def set_current(self, project: Optional[str]=None, experiment: Optional[str]=None, animal: Optional[str]=None,
                    session: Optional[str]=None, session_path: Optional[str]=None):
        if project is not None:
            self.current_project = project
            self.settings.last_opened_project = project
        if experiment is not None:
            self.current_experiment = experiment
            self.settings.last_opened_experiment = experiment
        if animal is not None:
            self.current_animal = animal
        if session is not None:
            self.current_session = session
        if session_path is not None:
            self.current_session_path = session_path
            self.settings.last_opened_session_path = session_path
