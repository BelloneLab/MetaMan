import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QTextEdit, QComboBox, QMessageBox, QFileDialog, QSplitter,
    QTableWidget, QTableWidgetItem, QInputDialog
)

from ..state import AppState
from ..config import PROCESSED_DIR_NAME
from ..level_chain import LevelChain
from ..io_ops import (
    list_experiments,
    list_sessions,
    list_subjects,
    save_session_triplet,
    load_session_metadata,
)

# Default step templates
NPX_STEPS = ["spike_sorting", "curation", "time_sync", "histology", "dlc"]
FIBER_STEPS = ["dff_zscore", "behavior_scoring"]
BEHAV_STEPS = ["behavior_scoring", "dlc"]
CUSTOM_STEP_LABEL = "Add custom step..."

STEP_STATUS_PLANNED = "planned"
STEP_STATUS_ONGOING = "ongoing"
STEP_STATUS_COMPLETED = "completed"


def dict_to_table(tbl: QTableWidget, data: Dict[str, Any]):
    tbl.setRowCount(0)
    for k, v in data.items():
        r = tbl.rowCount()
        tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem(str(k)))
        if isinstance(v, (dict, list)):
            tbl.setItem(r, 1, QTableWidgetItem(json.dumps(v, ensure_ascii=False)))
        else:
            tbl.setItem(r, 1, QTableWidgetItem("" if v is None else str(v)))
    tbl.resizeColumnsToContents()


class PreprocessingTab(QWidget):
    """
    Three-pane layout using a horizontal splitter:
      LEFT   : Steps list & controls
      CENTER : Parameters/Comments + Import params + Results folder
      RIGHT  : Session Info (read-only key/value table)

    Uses the loaded project (from Navigation > Project tab) for
    source path & local destination, same as the Recording tab.
    """

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.meta: Dict[str, Any] = {}
        self._loading_session = False
        self._proc_root_user_edited = False
        self._build_ui()
        self._load_settings()
        self._refresh_from_project()
        if self.app_state.current_session_path:
            self._load_from_session(self.app_state.current_session_path)
        else:
            last = self.app_state.settings.get_preprocessing_tab_settings().get("last_session_path", "")
            if last and os.path.isdir(last):
                self._load_from_session(last)

    # ------------------------ UI ------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # The active project is shown in the global project bar; keep a hidden
        # label that the refresh logic still updates.
        self.lbl_project = QLabel("(none)")
        self.lbl_project.setVisible(False)

        row_dest = QHBoxLayout()
        row_dest.addWidget(QLabel("Destination:"))
        self.ed_dest = QLineEdit()
        self.ed_dest.setPlaceholderText("Local folder for recordings")
        row_dest.addWidget(self.ed_dest, 1)
        b_dest = QPushButton("Browse\u2026")
        b_dest.clicked.connect(self._choose_destination)
        row_dest.addWidget(b_dest)
        b_refresh = QPushButton("Refresh")
        b_refresh.clicked.connect(self._refresh_from_project)
        row_dest.addWidget(b_refresh)
        root.addLayout(row_dest)

        # Schema-driven level selector (order/labels follow the project's
        # Structure Designer, like the Recording tab).
        self.levels = LevelChain(self.app_state)
        root.addWidget(self.levels)

        self.lbl_empty = QLabel("Select a project in the bar above to begin.")
        self.lbl_empty.setStyleSheet("color:#93a4c2; padding: 6px 2px;")
        self.lbl_empty.setVisible(False)
        root.addWidget(self.lbl_empty)

        self.levels.changed.connect(self._on_levels_changed)
        self.levels.leaf_changed.connect(self._on_leaf_changed)
        self.ed_dest.textChanged.connect(self._on_dest_changed)

        # Row 2: processed root + create folder
        row = QHBoxLayout()
        row.addWidget(QLabel("Processed root:"))
        self.ed_proc_root = QLineEdit(self.app_state.settings.processed_root)
        self.ed_proc_root.textChanged.connect(self._persist_settings)
        self.ed_proc_root.textEdited.connect(self._on_proc_root_text_edited)
        row.addWidget(self.ed_proc_root, 1)
        b_browse_proc = QPushButton("Browse\u2026")
        b_browse_proc.clicked.connect(self._choose_processed_root)
        row.addWidget(b_browse_proc)
        self.b_new_proc = QPushButton("\u2795  New preprocessing")
        self.b_new_proc.setObjectName("Primary")
        self.b_new_proc.setToolTip("Create the processed-data folder for this session and copy its metadata")
        self.b_new_proc.clicked.connect(self._create_processed_folder)
        row.addWidget(self.b_new_proc)
        root.addLayout(row)

        # Middle: 3 panels in a splitter
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # LEFT: steps & actions
        left = QWidget(); l = QVBoxLayout(left)
        l.addWidget(QLabel("Steps"))
        self.steps = QListWidget(); l.addWidget(self.steps, 1)
        rowL = QHBoxLayout()
        self.cb_step = QComboBox(); rowL.addWidget(self.cb_step)
        b_add = QPushButton("Add step"); b_add.clicked.connect(self._add_step); rowL.addWidget(b_add)
        b_plan = QPushButton("Planned"); b_plan.clicked.connect(self._mark_planned); rowL.addWidget(b_plan)
        b_ongoing = QPushButton("Ongoing"); b_ongoing.clicked.connect(self._mark_ongoing); rowL.addWidget(b_ongoing)
        b_done = QPushButton("Completed"); b_done.clicked.connect(self._mark_completed); rowL.addWidget(b_done)
        b_remove = QPushButton("Remove step"); b_remove.clicked.connect(self._remove_step); rowL.addWidget(b_remove)
        l.addLayout(rowL)
        split.addWidget(left)

        # CENTER: params/comments + import + results dir
        center = QWidget(); c = QVBoxLayout(center)

        c.addWidget(QLabel("Parameters (JSON)"))
        self.txt_params = QTextEdit()
        c.addWidget(self.txt_params, 1)

        rowP = QHBoxLayout()
        b_save_params = QPushButton("Add/Update parameters"); b_save_params.clicked.connect(self._save_params)
        rowP.addWidget(b_save_params)
        b_import = QPushButton("Import params (CSV/JSON)"); b_import.clicked.connect(self._import_params_for_step)
        rowP.addWidget(b_import)
        c.addLayout(rowP)

        c.addWidget(QLabel("Results folder (for this step)"))
        rowR = QHBoxLayout()
        self.ed_results_dir = QLineEdit()
        rowR.addWidget(self.ed_results_dir, 1)
        b_browse_results = QPushButton("Choose\u2026"); b_browse_results.clicked.connect(self._select_results_dir)
        rowR.addWidget(b_browse_results)
        b_open_results = QPushButton("Open folder"); b_open_results.clicked.connect(self._open_step_folder)
        rowR.addWidget(b_open_results)
        b_apply_results = QPushButton("Save results folder"); b_apply_results.clicked.connect(self._apply_results_dir)
        rowR.addWidget(b_apply_results)
        c.addLayout(rowR)

        c.addWidget(QLabel("Comments"))
        self.txt_comments = QTextEdit()
        c.addWidget(self.txt_comments, 1)
        b_save_comment = QPushButton("Save comment"); b_save_comment.clicked.connect(self._save_comment)
        c.addWidget(b_save_comment)

        split.addWidget(center)

        # RIGHT: session info panel (read-only)
        right = QWidget(); r = QVBoxLayout(right)
        r.addWidget(QLabel("Loaded Session Info"))
        self.tbl_session_info = QTableWidget(0, 2)
        self.tbl_session_info.setHorizontalHeaderLabels(["Key", "Value"])
        self.tbl_session_info.horizontalHeader().setStretchLastSection(True)
        r.addWidget(self.tbl_session_info, 1)
        split.addWidget(right)

        split.setSizes([350, 600, 450])

        self.steps.currentRowChanged.connect(self._update_param_comment)
        self._refresh_step_choices()

    # -- project / destination / combo helpers -----------------

    def _choose_destination(self):
        d = QFileDialog.getExistingDirectory(self, "Choose local destination",
                                             self.ed_dest.text() or "")
        if d:
            self.ed_dest.setText(d)

    def _choose_processed_root(self):
        start = self.ed_proc_root.text().strip() or self._default_processed_session_dir()
        d = QFileDialog.getExistingDirectory(self, "Choose processed folder", start)
        if not d:
            return
        self._proc_root_user_edited = True
        self.ed_proc_root.setText(d)
        self._persist_settings()

    def _has_levels(self) -> bool:
        return getattr(self, "levels", None) is not None and self.levels.has_levels()

    def _cur(self, key: str) -> str:
        return self.levels.values_by_key().get(key, "") if self._has_levels() else ""

    def _source_dirs(self) -> List[str]:
        src = self.app_state.settings.get_loaded_project().get("source_path", "")
        return [src] if src else []

    def _on_proc_root_text_edited(self, _txt: str):
        self._proc_root_user_edited = True

    def _default_processed_session_dir(self) -> str:
        dest = self.ed_dest.text().strip()
        proj = self.app_state.settings.get_loaded_project()
        base = dest or proj.get("destination_path", "")
        if not base:
            base = self.app_state.settings.processed_root or ""
        exp = self._cur("experiment") or str(self.meta.get("Experiment", "")).strip()
        sub = self._cur("subject") or str(self.meta.get("Subject", "") or self.meta.get("Animal", "")).strip()
        sess = self._cur("session") or str(self.meta.get("Session", "")).strip()
        proc = os.path.join(os.path.dirname(base), PROCESSED_DIR_NAME) if base else ""
        if proc and exp and sub and sess:
            return os.path.join(proc, exp, sub, sess)
        return proc or ""

    def _update_default_processed_root(self, force: bool = False):
        if self._proc_root_user_edited and not force:
            return
        target = self._default_processed_session_dir()
        if target:
            self.ed_proc_root.blockSignals(True)
            self.ed_proc_root.setText(target)
            self.ed_proc_root.blockSignals(False)

    def _refresh_from_project(self):
        """Rebuild the level chain for the loaded project (order/labels follow
        the project's structure schema)."""
        proj = self.app_state.settings.get_loaded_project()
        name = proj["name"]
        source_type = proj["source_type"]
        dest = proj["destination_path"]

        if name:
            self.lbl_project.setText(f"{name} ({source_type})" if source_type else name)
        else:
            self.lbl_project.setText("(none)")

        if dest and dest != self.ed_dest.text().strip():
            self.ed_dest.blockSignals(True)
            self.ed_dest.setText(dest)
            self.ed_dest.blockSignals(False)

        dest_dir = self.ed_dest.text().strip() or dest
        same_project = (name == getattr(self, "_levels_project", None))
        current = self.levels.values_by_key() if self._has_levels() else {}
        restore = getattr(self, "_pending_level_values", None) or (current if same_project else None)
        self._pending_level_values = None
        self._levels_project = name
        self.levels.configure(name, dest_dir, self._source_dirs())
        if restore:
            self.levels.set_values_by_key(restore)
        has_proj = bool(name)
        self.lbl_empty.setVisible(not has_proj)
        if hasattr(self, "b_new_proc"):
            self.b_new_proc.setEnabled(has_proj)
        self._update_default_processed_root()

    def _on_dest_changed(self, _txt: str = ""):
        if self._has_levels():
            self.levels.set_dirs(self.ed_dest.text().strip(), self._source_dirs())
        self._persist_settings()

    def _on_levels_changed(self):
        self._update_default_processed_root()
        self._persist_settings()

    def _on_leaf_changed(self):
        if self._loading_session:
            return
        self._update_default_processed_root()
        self._persist_settings()
        self._try_load_current_session(show_warning=False)

    def _selected_session_dir(self) -> str:
        """Session dir from the level chain: prefer destination, fall back to source."""
        if not self._has_levels():
            return ""
        parts = self.levels.part_names()
        if not all(parts):
            return ""
        dest = self.ed_dest.text().strip()
        source = self.app_state.settings.get_loaded_project().get("source_path", "")
        for base in (dest, source):
            if base:
                cand = os.path.join(base, *parts)
                if os.path.isdir(cand):
                    return cand
        if dest:
            return os.path.join(dest, *parts)
        if source:
            return os.path.join(source, *parts)
        return ""

    def _try_load_current_session(self, show_warning: bool):
        session_dir = self._selected_session_dir()
        if not session_dir or not os.path.isdir(session_dir):
            if show_warning:
                QMessageBox.warning(self, "Session", f"Session folder not found:\n{session_dir}")
            return
        if self.app_state.current_session_path == session_dir and self.meta:
            return
        self._load_from_session(session_dir)
        self.app_state.set_current(
            project=self.app_state.settings.get_loaded_project()["name"],
            experiment=self._cur("experiment"),
            animal=self._cur("subject"),
            session=self._cur("session"),
            session_path=session_dir,
        )
        self._persist_settings()

    # ------------------------ Session load / step menu ------------------------

    def _determine_step_choices(self) -> List[str]:
        rec = str(self.meta.get("Recording") or "")
        exp = str(self.meta.get("Experiment") or "")
        hint = f"{rec} {exp} {self._cur('experiment')}".lower()
        if "npx" in hint or "neuro" in hint or "neuropixel" in hint:
            return NPX_STEPS
        if "fiber" in hint or "photometry" in hint:
            return FIBER_STEPS
        return BEHAV_STEPS

    def _sanitize_step_name(self, name: str) -> str:
        return str(name or "").strip().replace("\n", " ").replace("\r", " ")

    def _normalize_step_status(self, status: str) -> str:
        s = str(status or "").strip().lower()
        if s in ("in_progress", "inprogress", "ongoing"):
            return STEP_STATUS_ONGOING
        if s in ("completed", "done"):
            return STEP_STATUS_COMPLETED
        return STEP_STATUS_PLANNED

    def _session_path_for_save(self) -> str:
        p = self.app_state.current_session_path
        if p and os.path.isdir(p):
            return p
        return self._selected_session_dir()

    def _save_current_session_meta(self):
        session_dir = self._session_path_for_save()
        if not session_dir:
            return
        os.makedirs(session_dir, exist_ok=True)
        save_session_triplet(session_dir, self.meta)

    def _normalize_preprocessing_steps(self):
        raw_steps = self.meta.get("preprocessing", [])
        if not isinstance(raw_steps, list):
            raw_steps = []
        normalized = []
        seen = set()
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            name = self._sanitize_step_name(step.get("name", ""))
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "name": name,
                    "params": step.get("params", {}) if isinstance(step.get("params", {}), dict) else {},
                    "comments": str(step.get("comments", "") or ""),
                    "status": self._normalize_step_status(str(step.get("status", ""))),
                    "results_dir": str(step.get("results_dir", "") or ""),
                }
            )
        self.meta["preprocessing"] = normalized

    def _ensure_default_steps_present(self):
        defaults = self._determine_step_choices()
        steps = self.meta.setdefault("preprocessing", [])
        existing = {str(s.get("name", "")).strip().lower() for s in steps if isinstance(s, dict)}
        added = False
        for name in defaults:
            key = name.strip().lower()
            if not key or key in existing:
                continue
            steps.append(
                {
                    "name": name,
                    "params": {},
                    "comments": "",
                    "status": STEP_STATUS_PLANNED,
                    "results_dir": "",
                }
            )
            existing.add(key)
            added = True
        if added:
            self._save_current_session_meta()

    def _refresh_step_choices(self):
        choices = self._determine_step_choices() + [CUSTOM_STEP_LABEL]
        keep = self.cb_step.currentText().strip()
        self.cb_step.blockSignals(True)
        self.cb_step.clear()
        self.cb_step.addItems(choices)
        if keep and keep in choices:
            self.cb_step.setCurrentText(keep)
        self.cb_step.blockSignals(False)

    def _load_from_session(self, session_dir: str):
        loaded = load_session_metadata(session_dir) or {}
        if not loaded:
            sub_dir = os.path.dirname(session_dir)
            exp_dir = os.path.dirname(sub_dir)
            project_dir = os.path.dirname(exp_dir)
            subject = os.path.basename(sub_dir)
            loaded = {
                "Project": os.path.basename(project_dir),
                "Experiment": os.path.basename(exp_dir),
                "Subject": subject,
                "Animal": subject,
                "Session": os.path.basename(session_dir),
                "preprocessing": [],
            }
        self.meta = loaded
        self._loading_session = True
        try:
            if self.meta:
                exp = str(self.meta.get("Experiment", "")).strip()
                sub = str(self.meta.get("Subject", "") or self.meta.get("Animal", "")).strip()
                sess = str(self.meta.get("Session", "")).strip()
                self.levels.set_from_metadata(self.meta)
                self.app_state.set_current(
                    project=self.app_state.settings.get_loaded_project()["name"],
                    experiment=exp, animal=sub, session=sess,
                    session_path=session_dir,
                )
        finally:
            self._loading_session = False
        self._normalize_preprocessing_steps()
        self._ensure_default_steps_present()
        self._refresh_steps()
        self._refresh_step_choices()
        dict_to_table(self.tbl_session_info, self.meta)
        self._update_default_processed_root()
        self._persist_settings()

    def _refresh_steps(self):
        self.steps.clear()
        steps = self.meta.get("preprocessing", [])
        if not isinstance(steps, list):
            return
        for s in steps:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", "")).strip()
            if not name:
                continue
            status = self._normalize_step_status(str(s.get("status", "")))
            self.steps.addItem(f"{name} [{status}]")

    def _current_step(self) -> Optional[Dict[str, Any]]:
        row = self.steps.currentRow()
        if row < 0:
            return None
        text = self.steps.item(row).text()
        name = text.split(" [", 1)[0]
        for s in self.meta.get("preprocessing", []):
            if s.get("name") == name:
                return s
        return None

    # ------------------------ Top controls ------------------------

    def _create_processed_folder(self):
        """Create processed folder structure and copy current metadata triplet."""
        exp = self._cur("experiment") or str(self.meta.get("Experiment", "")).strip()
        subject = self._cur("subject") or str(self.meta.get("Subject", "") or self.meta.get("Animal", "")).strip()
        session = self._cur("session") or str(self.meta.get("Session", "")).strip()
        root = self.ed_proc_root.text().strip() or self._default_processed_session_dir()
        if not (exp and subject and session and root):
            QMessageBox.warning(self, "Missing", "Missing processed root or session info.")
            return
        suffix = os.path.normcase(os.path.normpath(os.path.join(exp, subject, session)))
        root_norm = os.path.normcase(os.path.normpath(root))
        if root_norm.endswith(suffix):
            session_proc = root
        else:
            session_proc = os.path.join(root, exp, subject, session)
        os.makedirs(session_proc, exist_ok=True)
        save_session_triplet(session_proc, self.meta)
        QMessageBox.information(self, "Created", f"Processed session created:\n{session_proc}")

    # ------------------------ Step actions ------------------------

    def _add_step(self):
        choice = self.cb_step.currentText().strip()
        if not choice:
            return
        if choice == CUSTOM_STEP_LABEL:
            name, ok = QInputDialog.getText(self, "New step", "Step name:")
            if not ok or not name.strip():
                return
            step_name = self._sanitize_step_name(name)
        else:
            step_name = self._sanitize_step_name(choice)

        if not step_name:
            QMessageBox.warning(self, "Step", "Step name cannot be empty.")
            return

        existing_steps = self.meta.setdefault("preprocessing", [])
        for s in existing_steps:
            if str(s.get("name", "")).strip().lower() == step_name.lower():
                QMessageBox.information(self, "Exists", f"Step '{step_name}' already exists.")
                self._refresh_steps()
                for i in range(self.steps.count()):
                    if self.steps.item(i).text().lower().startswith(step_name.lower() + " ["):
                        self.steps.setCurrentRow(i)
                        break
                return

        step = {"name": step_name, "params": {}, "comments": "", "status": STEP_STATUS_PLANNED, "results_dir": ""}
        existing_steps.append(step)
        self._save_current_session_meta()
        self._refresh_steps()
        self._refresh_step_choices()
        for i in range(self.steps.count()):
            if self.steps.item(i).text().lower().startswith(step_name.lower() + " ["):
                self.steps.setCurrentRow(i)
                break

    def _remove_step(self):
        cur = self._current_step()
        if not cur:
            return
        name = cur.get("name", "")
        steps = self.meta.get("preprocessing", [])
        steps = [s for s in steps if s.get("name") != name]
        self.meta["preprocessing"] = steps
        self._save_current_session_meta()
        self._refresh_steps()
        self._refresh_step_choices()
        self.txt_params.clear()
        self.txt_comments.clear()
        self.ed_results_dir.clear()

    def _set_step_status(self, status: str):
        cur = self._current_step()
        if not cur:
            return
        cur["status"] = self._normalize_step_status(status)
        self._save_current_session_meta()
        selected_name = str(cur.get("name", "")).strip()
        self._refresh_steps()
        for i in range(self.steps.count()):
            if self.steps.item(i).text().lower().startswith(selected_name.lower() + " ["):
                self.steps.setCurrentRow(i)
                break

    def _mark_planned(self):
        self._set_step_status(STEP_STATUS_PLANNED)

    def _mark_ongoing(self):
        self._set_step_status(STEP_STATUS_ONGOING)

    def _mark_completed(self):
        self._set_step_status(STEP_STATUS_COMPLETED)

    def _update_param_comment(self):
        cur = self._current_step()
        if not cur:
            return
        self.txt_params.setText(json.dumps(cur.get("params", {}), indent=2))
        self.txt_comments.setText(cur.get("comments", ""))
        results_dir = str(cur.get("results_dir", "")).strip()
        if not results_dir:
            results_dir = self._default_step_results_dir(cur.get("name", "step"))
        self.ed_results_dir.setText(results_dir)

    # ------------------------ Params & comments ------------------------

    def _save_params(self):
        cur = self._current_step()
        if not cur:
            return
        try:
            params = json.loads(self.txt_params.toPlainText() or "{}")
        except Exception as e:
            QMessageBox.critical(self, "JSON error", str(e))
            return
        cur["params"] = params
        self._save_current_session_meta()

    def _save_comment(self):
        cur = self._current_step()
        if not cur:
            return
        txt = self.txt_comments.toPlainText().strip()
        cur["comments"] = txt
        self._save_current_session_meta()

    # ------------------------ Import params (CSV/JSON) ------------------------

    def _import_params_for_step(self):
        cur = self._current_step()
        if not cur:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose CSV/JSON", "", "Tables (*.csv *.json);;All files (*)")
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cur["params"] = data
            elif ext == ".csv":
                import pandas as pd
                df = None
                for enc in ["utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin-1", "cp1252"]:
                    try:
                        df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                        break
                    except Exception:
                        continue
                if df is None:
                    df = pd.read_csv(path)

                cols_lower = {str(c).strip().lower(): c for c in df.columns}
                if "key" in cols_lower and "value" in cols_lower:
                    kcol, vcol = cols_lower["key"], cols_lower["value"]
                    params = {}
                    for _, row in df.iterrows():
                        k = str(row.get(kcol, "")).strip()
                        if not k:
                            continue
                        v = row.get(vcol, "")
                        try:
                            v = json.loads(v)
                        except Exception:
                            v = "" if (isinstance(v, float) and v != v) else v
                        params[k] = v
                    cur["params"] = params
                else:
                    if len(df) == 1:
                        rec = df.iloc[0].to_dict()
                        for k, v in rec.items():
                            if isinstance(v, float) and v != v:
                                rec[k] = ""
                        cur["params"] = rec
                    else:
                        records = df.to_dict(orient="records")
                        for rec in records:
                            for k, v in list(rec.items()):
                                if isinstance(v, float) and v != v:
                                    rec[k] = ""
                        cur["params"] = records
            else:
                QMessageBox.warning(self, "Format", "Please choose a .csv or .json file.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not import params:\n{e}")
            return

        self.txt_params.setText(json.dumps(cur.get("params", {}), indent=2))
        self._save_current_session_meta()

    # ------------------------ Results folder per step ------------------------

    def _select_results_dir(self):
        cur = self._current_step()
        if not cur:
            return
        d = QFileDialog.getExistingDirectory(self, "Choose results folder", self.ed_results_dir.text() or "")
        if not d:
            return
        self.ed_results_dir.setText(d)

    def _default_step_results_dir(self, step_name: str) -> str:
        base = self.ed_proc_root.text().strip() or self._default_processed_session_dir()
        name = self._sanitize_step_name(step_name).replace(" ", "_")
        return os.path.join(base, name) if name else base

    def _open_step_folder(self):
        cur = self._current_step()
        if not cur:
            QMessageBox.warning(self, "Step", "Select a preprocessing step first.")
            return
        path = self.ed_results_dir.text().strip() or str(cur.get("results_dir", "")).strip()
        if not path:
            path = self._default_step_results_dir(str(cur.get("name", "step")))
            self.ed_results_dir.setText(path)
            cur["results_dir"] = path
            self._save_current_session_meta()
        os.makedirs(path, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            QMessageBox.critical(self, "Open folder", f"Failed to open folder:\n{path}\n\n{e}")

    def _apply_results_dir(self):
        cur = self._current_step()
        if not cur:
            return
        cur["results_dir"] = self.ed_results_dir.text().strip()
        self._save_current_session_meta()

    # -- settings --------------------------------------------------

    def _load_settings(self):
        data = self.app_state.settings.get_preprocessing_tab_settings()
        proj = self.app_state.settings.get_loaded_project()
        dest = data.get("destination_path") or proj.get("destination_path", "")
        if dest:
            self.ed_dest.setText(dest)
        self._proc_root_user_edited = False
        vals = data.get("level_values")
        if not isinstance(vals, dict) or not vals:
            vals = {
                "experiment": str(data.get("experiment", "")),
                "subject": str(data.get("subject", "")),
                "session": str(data.get("session", "")),
            }
        # applied by _refresh_from_project once the levels are built
        self._pending_level_values = {k: v for k, v in vals.items() if str(v).strip()}
        proc = data.get("processed_root", "")
        if proc:
            self._proc_root_user_edited = True
            self.ed_proc_root.setText(proc)
        else:
            self._update_default_processed_root(force=True)

    def _persist_settings(self):
        vals = self.levels.values_by_key() if self._has_levels() else {}
        self.app_state.settings.put_preprocessing_tab_settings(
            {
                "destination_path": self.ed_dest.text().strip(),
                "processed_root": self.ed_proc_root.text().strip(),
                "level_values": vals,
                "experiment": vals.get("experiment", ""),
                "subject": vals.get("subject", ""),
                "session": vals.get("session", ""),
                "last_session_path": self.app_state.current_session_path or self._selected_session_dir(),
            }
        )
