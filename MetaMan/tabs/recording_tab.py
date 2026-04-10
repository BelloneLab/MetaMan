import json
import os
from typing import Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..io_ops import (
    list_projects,
    load_session_metadata,
    save_session_triplet,
)
from ..models import SessionMetadata
from ..services.file_scanner import scan_file_list
from ..services.structure_schema import (
    build_role_path,
    extract_values_from_path,
    list_role_values,
    load_project_schema,
    role_enabled,
    role_label,
)
from ..state import AppState
from ..utils import LogEmitter, run_in_thread


def dict_to_table(tbl: QTableWidget, data: Dict[str, Any]):
    tbl.setRowCount(0)
    for k, v in data.items():
        r = tbl.rowCount()
        tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem(str(k)))
        if isinstance(v, (dict, list)):
            tbl.setItem(r, 1, QTableWidgetItem(json.dumps(v, ensure_ascii=False)))
        else:
            tbl.setItem(r, 1, QTableWidgetItem(str(v)))


def table_to_dict(tbl: QTableWidget) -> Dict[str, str]:
    out = {}
    for r in range(tbl.rowCount()):
        k = (tbl.item(r, 0).text() if tbl.item(r, 0) else "").strip()
        if not k:
            continue
        out[k] = tbl.item(r, 1).text() if tbl.item(r, 1) else ""
    return out


class RecordingTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.meta: Dict[str, Any] = {}
        self._loading_session = False
        self._active_schema = self._default_schema()
        self._build_ui()
        self._load_settings()
        self._refresh_lists()
        if self.app_state.current_session_path and os.path.isdir(self.app_state.current_session_path):
            self.load_session(self.app_state.current_session_path)
        else:
            last = self.app_state.settings.get_recording_tab_settings().get("last_session_path", "")
            if last and os.path.isdir(last):
                self.load_session(last)

    def _build_ui(self):
        root = QHBoxLayout(self)
        main_split = QSplitter(Qt.Horizontal)
        root.addWidget(main_split)

        # Left
        left_panel = QWidget()
        left_vbox = QVBoxLayout(left_panel)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Data root:"))
        self.ed_root = QLineEdit(self.app_state.settings.data_root)
        row1.addWidget(self.ed_root, 1)
        b_browse = QPushButton("Browse...")
        b_browse.clicked.connect(self._choose_root)
        row1.addWidget(b_browse)
        b_reload = QPushButton("Refresh lists")
        b_reload.clicked.connect(self._refresh_lists)
        row1.addWidget(b_reload)
        b_new = QPushButton("New recording")
        b_new.clicked.connect(self.new_recording)
        row1.addWidget(b_new)
        left_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        self.lbl_project = QLabel("Project")
        row2.addWidget(self.lbl_project)
        self.cb_proj = QComboBox()
        self.cb_proj.setEditable(False)
        row2.addWidget(self.cb_proj, 1)
        self.lbl_experiment = QLabel("Experiment")
        row2.addWidget(self.lbl_experiment)
        self.cb_exp = QComboBox()
        self.cb_exp.setEditable(False)
        row2.addWidget(self.cb_exp, 1)
        left_vbox.addLayout(row2)

        row3 = QHBoxLayout()
        self.lbl_subject = QLabel("Subject")
        row3.addWidget(self.lbl_subject)
        self.cb_sub = QComboBox()
        self.cb_sub.setEditable(False)
        row3.addWidget(self.cb_sub, 1)
        b_new_sub = QPushButton("New")
        b_new_sub.clicked.connect(self._prompt_new_subject)
        row3.addWidget(b_new_sub)
        self.lbl_session = QLabel("Session")
        row3.addWidget(self.lbl_session)
        self.cb_sess = QComboBox()
        self.cb_sess.setEditable(False)
        row3.addWidget(self.cb_sess, 1)
        b_new_sess = QPushButton("New")
        b_new_sess.clicked.connect(self._prompt_new_session)
        row3.addWidget(b_new_sess)
        left_vbox.addLayout(row3)

        for cb in (self.cb_proj, self.cb_exp, self.cb_sub, self.cb_sess):
            cb.setMaxVisibleItems(30)

        self.cb_proj.currentIndexChanged.connect(self._on_project_changed)
        self.cb_exp.currentIndexChanged.connect(self._on_experiment_changed)
        self.cb_sub.currentIndexChanged.connect(self._on_subject_changed)
        self.cb_proj.currentIndexChanged.connect(lambda _i: self._persist_settings())
        self.cb_exp.currentIndexChanged.connect(lambda _i: self._persist_settings())
        self.cb_sub.currentIndexChanged.connect(lambda _i: self._persist_settings())
        self.cb_sess.currentIndexChanged.connect(self._on_session_changed)
        self.cb_sess.currentIndexChanged.connect(lambda _i: self._persist_settings())
        self.ed_root.textChanged.connect(self._persist_settings)

        self.tbl_meta = QTableWidget(0, 2)
        self.tbl_meta.setHorizontalHeaderLabels(["Key", "Value"])
        self.tbl_meta.horizontalHeader().setStretchLastSection(True)

        tbl_btns = QHBoxLayout()
        b_add_row = QPushButton("Add row")
        b_add_row.clicked.connect(self._add_meta_row)
        tbl_btns.addWidget(b_add_row)
        b_rm_row = QPushButton("Remove selected")
        b_rm_row.clicked.connect(self._rm_meta_row)
        tbl_btns.addWidget(b_rm_row)
        b_apply = QPushButton("Apply table -> metadata")
        b_apply.clicked.connect(self._apply_table_to_meta)
        tbl_btns.addWidget(b_apply)
        b_save_all = QPushButton("Save metadata (JSON/CSV/H5)")
        b_save_all.clicked.connect(self._save_all)
        tbl_btns.addWidget(b_save_all)

        left_split = QSplitter(Qt.Vertical)
        left_top = QWidget()
        lt_layout = QVBoxLayout(left_top)
        lt_layout.addWidget(self.tbl_meta, 1)
        lt_layout.addLayout(tbl_btns)
        left_split.addWidget(left_top)

        left_bottom = QWidget()
        lb_layout = QVBoxLayout(left_bottom)
        lb_layout.addWidget(QLabel("metadata.json (preview)"))
        self.txt_preview = QTextEdit()
        self.txt_preview.setReadOnly(True)
        mono = QFont("Consolas" if os.name == "nt" else "Monospace")
        mono.setStyleHint(QFont.Monospace)
        self.txt_preview.setFont(mono)
        lb_layout.addWidget(self.txt_preview, 1)
        left_split.addWidget(left_bottom)
        left_split.setSizes([620, 280])
        left_vbox.addWidget(left_split, 1)

        main_split.addWidget(left_panel)

        # Right
        right_panel = QWidget()
        right_vbox = QVBoxLayout(right_panel)

        trial_top = QWidget()
        t_layout = QVBoxLayout(trial_top)
        rowt = QHBoxLayout()
        rowt.addWidget(QLabel("Trial #"))
        self.ed_trial = QLineEdit()
        self.ed_trial.setFixedWidth(70)
        rowt.addWidget(self.ed_trial)
        rowt.addWidget(QLabel("Type"))
        self.ed_trial_type = QLineEdit()
        rowt.addWidget(self.ed_trial_type, 1)
        b_add_trial = QPushButton("Add trial info")
        b_add_trial.clicked.connect(self._add_trial_info)
        rowt.addWidget(b_add_trial)
        b_update = QPushButton("Update file list")
        b_update.clicked.connect(self.update_file_list)
        rowt.addWidget(b_update)
        t_layout.addLayout(rowt)

        rowc = QHBoxLayout()
        rowc.addWidget(QLabel("Comments"))
        self.ed_comments = QLineEdit()
        rowc.addWidget(self.ed_comments, 1)
        b_save_comments = QPushButton("Save comments")
        b_save_comments.clicked.connect(self._save_comments)
        rowc.addWidget(b_save_comments)
        t_layout.addLayout(rowc)

        t_layout.addWidget(QLabel("Trial info"))
        self.list_trials = QListWidget()
        t_layout.addWidget(self.list_trials, 1)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(trial_top)

        log_widget = QWidget()
        lw = QVBoxLayout(log_widget)
        lw.addWidget(QLabel("Log"))
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        lw.addWidget(self.txt_log, 1)
        self.logger = LogEmitter(self.txt_log)
        right_split.addWidget(log_widget)
        right_split.setSizes([300, 320])

        right_vbox.addWidget(right_split, 1)
        main_split.addWidget(right_panel)
        main_split.setSizes([920, 360])

    def _default_schema(self):
        try:
            return self.app_state.settings.get_default_structure_schema()
        except Exception:
            return {}

    def _project_schema(self, project_name: str):
        return load_project_schema(
            self._raw_root(),
            project_name,
            fallback=self._default_schema(),
        )

    def _current_schema(self):
        project = self._current_project()
        if project:
            self._active_schema = self._project_schema(project)
        return self._active_schema

    def _update_role_labels(self):
        schema = self._current_schema()
        self.lbl_project.setText(role_label(schema, "project", kind="raw"))
        self.lbl_experiment.setText(role_label(schema, "experiment", kind="raw"))
        self.lbl_subject.setText(role_label(schema, "subject", kind="raw"))
        self.lbl_session.setText(role_label(schema, "session", kind="raw"))

        exp_enabled = role_enabled(schema, "experiment", kind="raw")
        sub_enabled = role_enabled(schema, "subject", kind="raw")
        sess_enabled = role_enabled(schema, "session", kind="raw")
        self.cb_exp.setEnabled(exp_enabled)
        self.cb_sub.setEnabled(sub_enabled)
        self.cb_sess.setEnabled(sess_enabled)

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose data root", self.ed_root.text() or "")
        if not d:
            return
        normalized = self._normalize_data_root(d)
        self.ed_root.setText(normalized)
        self.app_state.settings.data_root = normalized
        self._ensure_root_dirs()
        self._refresh_lists()
        self._persist_settings()

    def _normalize_data_root(self, path: str) -> str:
        p = os.path.normpath(str(path or "").strip())
        if not p:
            return p
        leaf = os.path.basename(p).lower()
        if leaf in ("raw", "rawdata", "processed", "processeddata"):
            parent = os.path.dirname(p)
            if parent:
                return parent
        return p

    def _effective_data_root(self) -> str:
        root = self.app_state.settings.data_root or self.ed_root.text().strip()
        return self._normalize_data_root(root)

    def _raw_root_candidate(self) -> str:
        return os.path.join(self._effective_data_root(), "raw")

    def _prefer_data_root_as_raw(self) -> bool:
        data_root = self._effective_data_root()
        if not os.path.isdir(data_root):
            return False
        entries = [
            d
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ]
        if not entries:
            return False
        lowered = {d.lower() for d in entries}
        if "raw" in lowered or "processed" in lowered:
            return False
        return True

    def _ensure_root_dirs(self):
        os.makedirs(self._raw_root(), exist_ok=True)
        os.makedirs(self._processed_root(), exist_ok=True)

    def _raw_root(self) -> str:
        candidate = self._raw_root_candidate()
        if os.path.isdir(candidate):
            return candidate
        if self._prefer_data_root_as_raw():
            return self._effective_data_root()
        return candidate

    def _processed_root(self) -> str:
        data_root = self._effective_data_root()
        candidate = os.path.join(data_root, "processed")
        if os.path.isdir(candidate):
            return candidate
        leaf = os.path.basename(data_root).lower()
        if leaf in ("processed", "processeddata"):
            return data_root
        return candidate

    def _current_project(self) -> str:
        return self.cb_proj.currentText().strip()

    def _current_experiment(self) -> str:
        return self.cb_exp.currentText().strip()

    def _current_subject(self) -> str:
        return self.cb_sub.currentText().strip()

    def _current_session(self) -> str:
        return self.cb_sess.currentText().strip()

    def _session_dir(self) -> str:
        schema = self._current_schema()
        values = {
            "project": self._current_project(),
            "experiment": self._current_experiment(),
            "subject": self._current_subject(),
            "session": self._current_session(),
        }
        path = build_role_path(
            self._raw_root(),
            schema,
            values,
            role="session",
            kind="raw",
        )
        if path:
            return path
        return os.path.join(
            self._raw_root(),
            self._current_project(),
            self._current_experiment(),
            self._current_subject(),
            self._current_session(),
        )

    def _set_combo_items(self, cb: QComboBox, items: List[str], keep_text: str):
        cb.blockSignals(True)
        cb.clear()
        cb.addItems(items)
        keep = str(keep_text or "").strip()
        if keep and keep in items:
            cb.setCurrentText(keep)
        elif items:
            cb.setCurrentIndex(0)
        else:
            cb.setCurrentIndex(-1)
            if cb.isEditable():
                cb.setEditText("")
        cb.blockSignals(False)

    def _sanitize_entry_name(self, value: str) -> str:
        bad = '<>:"/\\|?*'
        cleaned = "".join(ch for ch in str(value or "").strip() if ch not in bad)
        return cleaned.strip()

    def _ensure_combo_has_value(self, cb: QComboBox, value: str):
        text = self._sanitize_entry_name(value)
        if not text:
            return
        for i in range(cb.count()):
            if cb.itemText(i).strip().lower() == text.lower():
                cb.setCurrentIndex(i)
                return
        cb.addItem(text)
        cb.setCurrentText(text)

    def _prompt_new_subject(self):
        schema = self._current_schema()
        subject_lbl = role_label(schema, "subject", kind="raw")
        if not role_enabled(schema, "subject", kind="raw"):
            QMessageBox.warning(self, subject_lbl, f"{subject_lbl} level is disabled in this project's schema.")
            return
        project = self._current_project()
        experiment = self._current_experiment()
        if not project or not experiment:
            QMessageBox.warning(self, f"New {subject_lbl}", "Select project and experiment first.")
            return
        text, ok = QInputDialog.getText(self, f"New {subject_lbl}", f"{subject_lbl} name:")
        if not ok:
            return
        subject = self._sanitize_entry_name(text)
        if not subject:
            QMessageBox.warning(self, f"New {subject_lbl}", f"{subject_lbl} cannot be empty.")
            return
        self._ensure_combo_has_value(self.cb_sub, subject)
        self._on_subject_changed()
        self._persist_settings()

    def _prompt_new_session(self):
        schema = self._current_schema()
        session_lbl = role_label(schema, "session", kind="raw")
        if not role_enabled(schema, "session", kind="raw"):
            QMessageBox.warning(self, session_lbl, f"{session_lbl} level is disabled in this project's schema.")
            return
        if not self._current_subject():
            QMessageBox.warning(self, f"New {session_lbl}", "Select or create a subject first.")
            return
        text, ok = QInputDialog.getText(self, f"New {session_lbl}", f"{session_lbl} name:")
        if not ok:
            return
        session = self._sanitize_entry_name(text)
        if not session:
            QMessageBox.warning(self, f"New {session_lbl}", f"{session_lbl} cannot be empty.")
            return
        self._ensure_combo_has_value(self.cb_sess, session)
        self._persist_settings()

    def _refresh_lists(self):
        normalized = self._effective_data_root()
        if normalized:
            self.app_state.settings.data_root = normalized
        if normalized and normalized != self.ed_root.text().strip():
            self.ed_root.blockSignals(True)
            self.ed_root.setText(normalized)
            self.ed_root.blockSignals(False)
        cur_proj = self._current_project()
        projects = list_projects(self._raw_root())
        self._set_combo_items(self.cb_proj, projects, cur_proj)
        self._on_project_changed()

    def _on_project_changed(self, _index: int = -1):
        proj = self._current_project()
        self._active_schema = self._project_schema(proj) if proj else self._default_schema()
        self._update_role_labels()
        exp_keep = self._current_experiment()
        schema = self._current_schema()
        exp_items = (
            list_role_values(
                self._raw_root(),
                schema,
                role="experiment",
                filters={"project": proj},
                kind="raw",
            )
            if proj
            else []
        )
        self._set_combo_items(self.cb_exp, exp_items, exp_keep)
        self._on_experiment_changed()

    def _on_experiment_changed(self, _index: int = -1):
        proj = self._current_project()
        exp = self._current_experiment()
        sub_keep = self._current_subject()
        schema = self._current_schema()
        sub_items = (
            list_role_values(
                self._raw_root(),
                schema,
                role="subject",
                filters={"project": proj, "experiment": exp},
                kind="raw",
            )
            if proj and exp
            else []
        )
        self._set_combo_items(self.cb_sub, sub_items, sub_keep)
        self._on_subject_changed()

    def _on_subject_changed(self, _index: int = -1):
        proj = self._current_project()
        exp = self._current_experiment()
        sub = self._current_subject()
        sess_keep = self._current_session()
        schema = self._current_schema()
        sess_items = (
            list_role_values(
                self._raw_root(),
                schema,
                role="session",
                filters={"project": proj, "experiment": exp, "subject": sub},
                kind="raw",
            )
            if proj and exp and sub
            else []
        )
        self._set_combo_items(self.cb_sess, sess_items, sess_keep)
        self._persist_settings()

    def _on_session_changed(self, _index: int):
        if self._loading_session:
            return
        self._persist_settings()
        self._try_load_current_session(show_warning=False)

    def _try_load_current_session(self, show_warning: bool):
        session_dir = self._session_dir()
        if not os.path.isdir(session_dir):
            if show_warning:
                QMessageBox.warning(self, "Recording", f"Recording folder not found:\n{session_dir}")
            return
        if self.app_state.current_session_path == session_dir and self.meta:
            return
        self.load_session(session_dir)

    def _refresh_preview(self):
        self.txt_preview.setPlainText(json.dumps(self.meta, indent=2, ensure_ascii=False))
        dict_to_table(self.tbl_meta, self.meta)

    def load_session(self, session_dir: str):
        meta = load_session_metadata(session_dir) or {}
        if not meta:
            root = self._raw_root()
            rel_parts = [p for p in os.path.relpath(session_dir, root).split(os.sep) if p and p != "."]
            project = rel_parts[0] if rel_parts else ""
            schema = self._project_schema(project) if project else self._default_schema()
            vals = extract_values_from_path(root, schema, session_dir, kind="raw")
            project = vals.get("project", project)
            experiment = vals.get("experiment", "")
            subject = vals.get("subject", "")
            session = vals.get("session", os.path.basename(session_dir))
            meta = SessionMetadata.new(project, subject, session, self._raw_root()).data
            meta["Experiment"] = experiment
            meta["Subject"] = subject

        self.meta = meta
        subject = str(meta.get("Subject", "") or meta.get("Animal", ""))
        self._loading_session = True
        if self.cb_proj.count() == 0:
            self._refresh_lists()

        try:
            project_name = str(meta.get("Project", ""))
            self.cb_proj.setCurrentText(project_name)
            self._on_project_changed()
            self.cb_exp.setCurrentText(str(meta.get("Experiment", "")))
            self._on_experiment_changed()
            self.cb_sub.setCurrentText(subject)
            self._on_subject_changed()
            self.cb_sess.setCurrentText(str(meta.get("Session", "")))
        finally:
            self._loading_session = False

        self.ed_comments.setText(str(meta.get("Comments", "")))
        self.list_trials.clear()
        for k, v in (meta.get("trial_info") or {}).items():
            self.list_trials.addItem(f"Trial {k}: {v}")

        self.app_state.set_current(
            project=str(meta.get("Project", "")),
            experiment=str(meta.get("Experiment", "")),
            animal=subject,
            session=str(meta.get("Session", "")),
            session_path=session_dir,
        )
        self._refresh_preview()
        self.logger.log(f"Loaded session: {session_dir}")
        self._persist_settings()

    def new_recording(self):
        self._ensure_root_dirs()
        schema = self._current_schema()
        project = self._current_project()
        experiment = self._current_experiment()
        subject = self._current_subject()
        session = self._current_session() or "1"

        required = ["project"]
        if role_enabled(schema, "experiment", kind="raw"):
            required.append("experiment")
        if role_enabled(schema, "subject", kind="raw"):
            required.append("subject")
        if role_enabled(schema, "session", kind="raw"):
            required.append("session")

        values = {
            "project": project,
            "experiment": experiment,
            "subject": subject,
            "session": session,
        }
        missing = [k for k in required if not str(values.get(k, "")).strip()]
        if missing:
            labels = [role_label(schema, k, kind="raw") for k in missing]
            QMessageBox.warning(self, "Missing fields", "Fill: " + ", ".join(labels))
            return

        if role_enabled(schema, "session", kind="raw"):
            target_role = "session"
        elif role_enabled(schema, "subject", kind="raw"):
            target_role = "subject"
        elif role_enabled(schema, "experiment", kind="raw"):
            target_role = "experiment"
        else:
            target_role = "project"

        session_dir = build_role_path(
            self._raw_root(),
            schema,
            values,
            role=target_role,
            kind="raw",
        )
        if not session_dir:
            session_dir = self._session_dir()
        os.makedirs(session_dir, exist_ok=True)

        meta = SessionMetadata.new(project, subject, session, self._raw_root()).data
        meta["Experiment"] = experiment
        meta["Subject"] = subject
        self.meta = meta
        save_session_triplet(session_dir, self.meta, logger=self.logger.log)
        self.load_session(session_dir)
        self._persist_settings()

    def update_file_list(self):
        session_dir = self._session_dir()
        if not os.path.isdir(session_dir):
            QMessageBox.critical(self, "Not found", f"Session directory not found:\n{session_dir}")
            return

        def work():
            file_list = scan_file_list(session_dir)
            self.meta["file_list"] = file_list
            save_session_triplet(session_dir, self.meta, logger=self.logger.log)

            srv_root = self.app_state.settings.get_server_root_for_project(self.meta.get("Project", ""))
            if srv_root and os.path.isdir(srv_root):
                proj = self.meta.get("Project", "")
                server_proj = os.path.join(srv_root, proj)
                local_proj = os.path.join(self._raw_root(), proj)
                for item in self.meta["file_list"]:
                    src_path = item.get("path", "")
                    try:
                        rel = os.path.relpath(src_path, local_proj)
                        spath = os.path.join(server_proj, rel)
                    except Exception:
                        spath = ""
                    item["server_path"] = spath if os.path.exists(spath) else ""
                save_session_triplet(session_dir, self.meta, logger=self.logger.log)
                self.logger.log("Server presence annotated in metadata.")

            self._refresh_preview()
            self.logger.log("File list updated.")

        run_in_thread(work)

    def _add_trial_info(self):
        t = self.ed_trial.text().strip()
        tt = self.ed_trial_type.text().strip()
        if not t:
            QMessageBox.warning(self, "Trial", "Enter trial number.")
            return
        self.meta.setdefault("trial_info", {})[str(int(t))] = tt
        self.list_trials.addItem(f"Trial {t}: {tt}")
        save_session_triplet(self._session_dir(), self.meta, logger=self.logger.log)
        self._refresh_preview()

    def _save_comments(self):
        self.meta["Comments"] = self.ed_comments.text()
        save_session_triplet(self._session_dir(), self.meta, logger=self.logger.log)
        self._refresh_preview()

    def _save_all(self):
        save_session_triplet(self._session_dir(), self.meta, logger=self.logger.log)
        self._refresh_preview()

    def _add_meta_row(self):
        r = self.tbl_meta.rowCount()
        self.tbl_meta.insertRow(r)
        self.tbl_meta.setItem(r, 0, QTableWidgetItem(""))
        self.tbl_meta.setItem(r, 1, QTableWidgetItem(""))

    def _rm_meta_row(self):
        rows = sorted({i.row() for i in self.tbl_meta.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl_meta.removeRow(r)

    def _apply_table_to_meta(self):
        flat = table_to_dict(self.tbl_meta)
        new_meta: Dict[str, Any] = {}
        for k, v in flat.items():
            try:
                new_meta[k] = json.loads(v)
                continue
            except Exception:
                pass
            try:
                if "." in v:
                    new_meta[k] = float(v)
                else:
                    new_meta[k] = int(v)
                continue
            except Exception:
                pass
            new_meta[k] = v
        self.meta = new_meta
        save_session_triplet(self._session_dir(), self.meta, logger=self.logger.log)
        self._refresh_preview()

    def _load_settings(self):
        data = self.app_state.settings.get_recording_tab_settings()
        root = self._normalize_data_root(self.app_state.settings.data_root)
        if root:
            self.ed_root.setText(root)
        proj = str(data.get("project", "")).strip()
        exp = str(data.get("experiment", "")).strip()
        sub = str(data.get("subject", "")).strip()
        sess = str(data.get("session", "")).strip()
        if proj:
            self.cb_proj.setCurrentText(proj)
        if exp:
            self.cb_exp.setCurrentText(exp)
        if sub:
            self.cb_sub.setCurrentText(sub)
        if sess:
            self.cb_sess.setCurrentText(sess)

    def _persist_settings(self):
        root = self._normalize_data_root(self.app_state.settings.data_root)
        data = {
            "data_root": root,
            "project": self._current_project(),
            "experiment": self._current_experiment(),
            "subject": self._current_subject(),
            "session": self._current_session(),
            "last_session_path": self.app_state.current_session_path or self._session_dir(),
        }
        self.app_state.settings.put_recording_tab_settings(data)
