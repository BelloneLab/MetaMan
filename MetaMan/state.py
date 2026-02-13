import json
import os
from typing import Any, Dict, Optional
from .config import SETTINGS_FILE, DEFAULT_DATA_ROOT, DEFAULT_RAW_ROOT, DEFAULT_PROCESSED_ROOT

class AppSettings:
    def __init__(self):
        self._data: Dict = {
            "data_root": DEFAULT_DATA_ROOT,
            "raw_root": DEFAULT_RAW_ROOT,
            "processed_root": DEFAULT_PROCESSED_ROOT,
            "last_opened_project": "",
            "last_opened_experiment": "",
            "last_opened_session_path": "",
            "server_roots_by_project": {},  # {project: server_root}
            "hdd_roots_by_project": {},  # {project: hdd_root}
            "backup_schedules_by_project": {},  # {project: {"enabled": bool, "time": "HH:MM", "last_run_date": "YYYY-MM-DD", "backup_whole_project": bool, "enabled_experiments": [], "run_dates_by_experiment": {exp: "YYYY-MM-DD"}, "destination_mode": "server|hdd|both"}}
            "data_reorganizer_settings": {},
            "recording_tab_settings": {},
            "preprocessing_tab_settings": {},
        }
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

    def _coerce_data_root(self) -> str:
        data_root = str(self._data.get("data_root") or "").strip()
        raw_root = str(self._data.get("raw_root") or "").strip()
        proc_root = str(self._data.get("processed_root") or "").strip()

        if not data_root:
            if os.path.basename(raw_root).lower() in ("raw", "rawdata"):
                data_root = os.path.dirname(raw_root)
            elif os.path.basename(proc_root).lower() in ("processed", "processeddata"):
                data_root = os.path.dirname(proc_root)
            else:
                data_root = DEFAULT_DATA_ROOT

        if not self._path_drive_is_available(data_root):
            data_root = self._fallback_data_root()
        return data_root

    def load(self):
        try:
            if SETTINGS_FILE.exists():
                self._data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
            # Ensure folder-style defaults and coerce invalid legacy roots.
            data_root = self._coerce_data_root()
            self._data["data_root"] = data_root
            self._data["raw_root"] = os.path.join(data_root, "raw")
            self._data["processed_root"] = os.path.join(data_root, "processed")
        except Exception:
            pass

    def ensure_storage_roots(self):
        data_root = self._coerce_data_root()
        raw_root = os.path.join(data_root, "raw")
        proc_root = os.path.join(data_root, "processed")
        try:
            os.makedirs(raw_root, exist_ok=True)
            os.makedirs(proc_root, exist_ok=True)
        except Exception:
            data_root = self._fallback_data_root()
            raw_root = os.path.join(data_root, "raw")
            proc_root = os.path.join(data_root, "processed")
            os.makedirs(raw_root, exist_ok=True)
            os.makedirs(proc_root, exist_ok=True)
        self._data["data_root"] = data_root
        self._data["raw_root"] = raw_root
        self._data["processed_root"] = proc_root
        self.save()

    def save(self):
        try:
            SETTINGS_FILE.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception:
            pass

    @property
    def raw_root(self) -> str:
        return self._data.get("raw_root") or os.path.join(self.data_root, "raw")

    @raw_root.setter
    def raw_root(self, v: str):
        self._data["raw_root"] = v
        if os.path.basename(v).lower() in ("raw", "rawdata"):
            self._data["data_root"] = os.path.dirname(v)
        self.save()

    @property
    def processed_root(self) -> str:
        return self._data.get("processed_root") or os.path.join(self.data_root, "processed")

    @processed_root.setter
    def processed_root(self, v: str):
        self._data["processed_root"] = v
        if os.path.basename(v).lower() in ("processed", "processeddata"):
            self._data["data_root"] = os.path.dirname(v)
        self.save()

    @property
    def data_root(self) -> str:
        return self._data.get("data_root") or DEFAULT_DATA_ROOT

    @data_root.setter
    def data_root(self, v: str):
        self._data["data_root"] = v
        self._data["raw_root"] = os.path.join(v, "raw")
        self._data["processed_root"] = os.path.join(v, "processed")
        self.save()

    def get_server_root_for_project(self, project: str) -> str:
        return (self._data.get("server_roots_by_project") or {}).get(project, "")

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
