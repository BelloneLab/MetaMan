import json
import os
from datetime import datetime
from typing import Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
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
    list_experiments,
    list_sessions,
    list_subjects,
    load_session_metadata,
    save_session_triplet,
)
from ..models import SessionMetadata
from ..services.file_scanner import scan_file_list
from ..state import AppState
from ..utils import LogEmitter, run_in_thread
from ..side_panel import SidePanelLayout


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
        self._build_ui()
        self._load_settings()
        self._refresh_from_project()
        if self.app_state.current_session_path and os.path.isdir(self.app_state.current_session_path):
            self.load_session(self.app_state.current_session_path)
        else:
            last = self.app_state.settings.get_recording_tab_settings().get("last_session_path", "")
            if last and os.path.isdir(last):
                self.load_session(last)

    # ── UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._side = SidePanelLayout()
        root.addWidget(self._side, 1)

        # Panel 0 – Setup ─────────────────────────────────────────
        setup = QWidget()
        sl = QVBoxLayout(setup)

        # Loaded project (read-only)
        row_proj = QHBoxLayout()
        row_proj.addWidget(QLabel("Project:"))
        self.lbl_project = QLabel("(none)")
        self.lbl_project.setStyleSheet("font-weight: 600;")
        row_proj.addWidget(self.lbl_project, 1)
        b_refresh = QPushButton("Refresh")
        b_refresh.clicked.connect(self._refresh_from_project)
        row_proj.addWidget(b_refresh)
        sl.addLayout(row_proj)

        # Destination path
        row_dest = QHBoxLayout()
        row_dest.addWidget(QLabel("Destination:"))
        self.ed_dest = QLineEdit()
        self.ed_dest.setPlaceholderText("Local folder for recordings")
        row_dest.addWidget(self.ed_dest, 1)
        b_dest = QPushButton("Browse\u2026")
        b_dest.clicked.connect(self._choose_destination)
        row_dest.addWidget(b_dest)
        sl.addLayout(row_dest)

        # Experiment
        row_exp = QHBoxLayout()
        row_exp.addWidget(QLabel("Experiment"))
        self.cb_exp = QComboBox(); self.cb_exp.setEditable(True)
        self.cb_exp.setMaxVisibleItems(30)
        row_exp.addWidget(self.cb_exp, 1)
        sl.addLayout(row_exp)

        # Subject + Session
        row_sub = QHBoxLayout()
        row_sub.addWidget(QLabel("Subject"))
        self.cb_sub = QComboBox(); self.cb_sub.setEditable(True)
        self.cb_sub.setMaxVisibleItems(30)
        row_sub.addWidget(self.cb_sub, 1)
        row_sub.addWidget(QLabel("Session"))
        self.cb_sess = QComboBox(); self.cb_sess.setEditable(True)
        self.cb_sess.setMaxVisibleItems(30)
        row_sub.addWidget(self.cb_sess, 1)
        sl.addLayout(row_sub)

        b_new_rec = QPushButton("New recording")
        b_new_rec.clicked.connect(self.new_recording)
        sl.addWidget(b_new_rec)

        sl.addStretch(1)

        self.cb_exp.currentIndexChanged.connect(self._on_experiment_changed)
        self.cb_sub.currentIndexChanged.connect(self._on_subject_changed)
        self.cb_exp.currentTextChanged.connect(lambda _: self._persist_settings())
        self.cb_sub.currentTextChanged.connect(lambda _: self._persist_settings())
        self.cb_sess.currentIndexChanged.connect(self._on_session_changed)
        self.cb_sess.currentTextChanged.connect(lambda _: self._persist_settings())
        self.ed_dest.textChanged.connect(self._persist_settings)

        self._side.add_panel("\U0001f3af", "Setup", setup)

        # Panel 1 – Metadata editor (default) ─────────────────────
        meta_panel = QWidget()
        ml = QVBoxLayout(meta_panel)

        self.tbl_meta = QTableWidget(0, 2)
        self.tbl_meta.setHorizontalHeaderLabels(["Key", "Value"])
        self.tbl_meta.horizontalHeader().setStretchLastSection(True)

        tbl_btns = QHBoxLayout()
        b_add = QPushButton("Add row"); b_add.clicked.connect(self._add_meta_row); tbl_btns.addWidget(b_add)
        b_rm = QPushButton("Remove selected"); b_rm.clicked.connect(self._rm_meta_row); tbl_btns.addWidget(b_rm)
        b_apply = QPushButton("Apply table \u2192 metadata"); b_apply.clicked.connect(self._apply_table_to_meta); tbl_btns.addWidget(b_apply)
        b_save = QPushButton("Save metadata (JSON/CSV/H5)"); b_save.clicked.connect(self._save_all); tbl_btns.addWidget(b_save)

        meta_split = QSplitter(Qt.Vertical)
        top_w = QWidget(); tl = QVBoxLayout(top_w)
        tl.addWidget(self.tbl_meta, 1); tl.addLayout(tbl_btns)
        meta_split.addWidget(top_w)

        bot_w = QWidget(); bl = QVBoxLayout(bot_w)
        bl.addWidget(QLabel("metadata.json (preview)"))
        self.txt_preview = QTextEdit(); self.txt_preview.setReadOnly(True)
        mono = QFont("Consolas" if os.name == "nt" else "Monospace")
        mono.setStyleHint(QFont.Monospace)
        self.txt_preview.setFont(mono)
        bl.addWidget(self.txt_preview, 1)
        meta_split.addWidget(bot_w)
        meta_split.setSizes([620, 280])

        ml.addWidget(meta_split, 1)

        self._side.add_panel("\U0001f4dd", "Metadata", meta_panel, default=True)

        # Panel 2 – Trials & comments ─────────────────────────────
        trials_panel = QWidget()
        trl = QVBoxLayout(trials_panel)

        rowt = QHBoxLayout()
        rowt.addWidget(QLabel("Trial #"))
        self.ed_trial = QLineEdit(); self.ed_trial.setFixedWidth(70); rowt.addWidget(self.ed_trial)
        rowt.addWidget(QLabel("Type"))
        self.ed_trial_type = QLineEdit(); rowt.addWidget(self.ed_trial_type, 1)
        b_add_trial = QPushButton("Add trial info"); b_add_trial.clicked.connect(self._add_trial_info); rowt.addWidget(b_add_trial)
        trl.addLayout(rowt)

        rowc = QHBoxLayout()
        rowc.addWidget(QLabel("Comments"))
        self.ed_comments = QLineEdit(); rowc.addWidget(self.ed_comments, 1)
        b_save_c = QPushButton("Save comments"); b_save_c.clicked.connect(self._save_comments); rowc.addWidget(b_save_c)
        trl.addLayout(rowc)

        trl.addWidget(QLabel("Trial info"))
        self.list_trials = QListWidget()
        trl.addWidget(self.list_trials, 1)

        self._side.add_panel("\U0001f3ac", "Trials", trials_panel)

        # Panel 3 – Files & Log ───────────────────────────────────
        files_panel = QWidget()
        fl = QVBoxLayout(files_panel)

        b_update = QPushButton("Update file list")
        b_update.clicked.connect(self.update_file_list)
        fl.addWidget(b_update)

        fl.addWidget(QLabel("Log"))
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        fl.addWidget(self.txt_log, 1)
        self.logger = LogEmitter(self.txt_log)

        self._side.add_panel("\U0001f4c1", "Files & Log", files_panel)

    # ── helpers ─────────────────────────────────────────────────────

    def _choose_destination(self):
        d = QFileDialog.getExistingDirectory(self, "Choose destination folder",
                                             self.ed_dest.text() or "")
        if d:
            self.ed_dest.setText(d)
            self._persist_settings()

    def _current_experiment(self) -> str:
        return self.cb_exp.currentText().strip()

    def _current_subject(self) -> str:
        return self.cb_sub.currentText().strip()

    def _current_session(self) -> str:
        return self.cb_sess.currentText().strip()

    def _session_dir(self) -> str:
        return os.path.join(
            self.ed_dest.text().strip(),
            self._current_experiment(),
            self._current_subject(),
            self._current_session(),
        )

    def _set_combo_items(self, cb: QComboBox, items: List[str], keep_text: str):
        cb.blockSignals(True)
        cb.clear(); cb.addItems(items)
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
        return "".join(ch for ch in str(value or "").strip() if ch not in bad).strip()

    def _ensure_combo_has_value(self, cb: QComboBox, value: str):
        text = self._sanitize_entry_name(value)
        if not text:
            return
        for i in range(cb.count()):
            if cb.itemText(i).strip().lower() == text.lower():
                cb.setCurrentIndex(i); return
        cb.addItem(text); cb.setCurrentText(text)

    def _refresh_from_project(self):
        """Populate experiments from the loaded project source, subjects/sessions from destination."""
        proj = self.app_state.settings.get_loaded_project()
        name = proj["name"]
        source = proj["source_path"]
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

        # Experiments: union of source + destination
        exps = set()
        if source and os.path.isdir(source):
            exps.update(list_experiments(source))
        if dest and os.path.isdir(dest):
            exps.update(list_experiments(dest))
        self._set_combo_items(self.cb_exp, sorted(exps), self._current_experiment())
        self._on_experiment_changed()

    def _on_experiment_changed(self, _index: int = -1):
        exp = self._current_experiment()
        dest = self.ed_dest.text().strip()
        proj = self.app_state.settings.get_loaded_project()
        source = proj["source_path"]

        subjects = set()
        if exp and dest and os.path.isdir(os.path.join(dest, exp)):
            subjects.update(list_subjects(os.path.join(dest, exp)))
        if exp and source and os.path.isdir(os.path.join(source, exp)):
            subjects.update(list_subjects(os.path.join(source, exp)))
        self._set_combo_items(self.cb_sub, sorted(subjects), self._current_subject())
        self._on_subject_changed()

    def _on_subject_changed(self, _index: int = -1):
        exp = self._current_experiment()
        sub = self._current_subject()
        dest = self.ed_dest.text().strip()
        proj = self.app_state.settings.get_loaded_project()
        source = proj["source_path"]

        sessions = set()
        if exp and sub and dest:
            sub_dir = os.path.join(dest, exp, sub)
            if os.path.isdir(sub_dir):
                sessions.update(list_sessions(sub_dir))
        if exp and sub and source:
            src_sub_dir = os.path.join(source, exp, sub)
            if os.path.isdir(src_sub_dir):
                sessions.update(list_sessions(src_sub_dir))
        self._set_combo_items(self.cb_sess, sorted(sessions), self._current_session())
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
            sub_dir = os.path.dirname(session_dir)
            exp_dir = os.path.dirname(sub_dir)
            dest = self.ed_dest.text().strip()
            meta = SessionMetadata.new("", os.path.basename(sub_dir),
                                       os.path.basename(session_dir), dest).data
            meta["Experiment"] = os.path.basename(exp_dir)
            meta["Subject"] = os.path.basename(sub_dir)

        self.meta = meta
        subject = str(meta.get("Subject", "") or meta.get("Animal", ""))
        self._loading_session = True
        try:
            self._ensure_combo_has_value(self.cb_exp, str(meta.get("Experiment", "")))
            self._on_experiment_changed()
            self._ensure_combo_has_value(self.cb_sub, subject)
            self._on_subject_changed()
            self._ensure_combo_has_value(self.cb_sess, str(meta.get("Session", "")))
        finally:
            self._loading_session = False

        self.ed_comments.setText(str(meta.get("Comments", "")))
        self.list_trials.clear()
        for k, v in (meta.get("trial_info") or {}).items():
            self.list_trials.addItem(f"Trial {k}: {v}")

        proj = self.app_state.settings.get_loaded_project()
        self.app_state.set_current(
            project=proj["name"],
            experiment=str(meta.get("Experiment", "")),
            animal=subject,
            session=str(meta.get("Session", "")),
            session_path=session_dir,
        )
        self._refresh_preview()
        self.logger.log(f"Loaded session: {session_dir}")
        self._persist_settings()

    def new_recording(self):
        dest = self.ed_dest.text().strip()
        experiment = self._current_experiment()
        subject = self._current_subject()
        session = self._current_session()
        proj = self.app_state.settings.get_loaded_project()
        project_name = proj["name"]

        if not dest:
            QMessageBox.warning(self, "Missing", "Set a Destination path."); return
        if not all([experiment, subject, session]):
            QMessageBox.warning(self, "Missing fields", "Fill Experiment, Subject, and Session."); return

        session_dir = os.path.join(dest, experiment, subject, session)
        os.makedirs(session_dir, exist_ok=True)

        meta = self._new_recording_metadata(project_name, experiment, subject, session)
        self.meta = meta
        save_session_triplet(session_dir, self.meta, logger=self.logger.log)
        self.load_session(session_dir)
        self._refresh_from_project()  # update combos
        self._persist_settings()

    def _new_recording_metadata(self, project, experiment, subject, session):
        dest = self.ed_dest.text().strip()
        meta = SessionMetadata.new(project, subject, session, dest).data
        keep_keys = {"DateTime", "Project", "Experiment", "Animal", "Subject", "Session",
                     "RootDir", "SessionUUID", "file_list", "trial_info", "trial_assets", "preprocessing"}
        for k in list(meta.keys()):
            if k in keep_keys:
                continue
            v = meta.get(k)
            if isinstance(v, list): meta[k] = []
            elif isinstance(v, dict): meta[k] = {}
            else: meta[k] = ""
        meta.update({
            "Project": project, "Experiment": experiment,
            "Animal": subject, "Subject": subject,
            "Session": str(session), "RootDir": dest,
            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_list": [], "trial_info": {}, "trial_assets": {}, "preprocessing": [],
        })
        return meta

    def update_file_list(self):
        session_dir = self._session_dir()
        if not os.path.isdir(session_dir):
            QMessageBox.critical(self, "Not found", f"Session directory not found:\n{session_dir}"); return
        def work():
            file_list = scan_file_list(session_dir)
            self.meta["file_list"] = file_list
            save_session_triplet(session_dir, self.meta, logger=self.logger.log)
            srv_root = self.app_state.settings.get_server_root_for_project(self.meta.get("Project", ""))
            if srv_root and os.path.isdir(srv_root):
                proj = self.meta.get("Project", ""); exp = self.meta.get("Experiment", "")
                subject = self.meta.get("Subject", "") or self.meta.get("Animal", "")
                session = self.meta.get("Session", "")
                server_proj = os.path.join(srv_root, proj)
                for item in self.meta["file_list"]:
                    spath = item["path"].replace(session_dir, os.path.join(server_proj, exp, subject, session))
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
            QMessageBox.warning(self, "Trial", "Enter trial number."); return
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
        r = self.tbl_meta.rowCount(); self.tbl_meta.insertRow(r)
        self.tbl_meta.setItem(r, 0, QTableWidgetItem("")); self.tbl_meta.setItem(r, 1, QTableWidgetItem(""))

    def _rm_meta_row(self):
        for r in sorted({i.row() for i in self.tbl_meta.selectedIndexes()}, reverse=True):
            self.tbl_meta.removeRow(r)

    def _apply_table_to_meta(self):
        flat = table_to_dict(self.tbl_meta)
        new_meta: Dict[str, Any] = {}
        for k, v in flat.items():
            try:
                new_meta[k] = json.loads(v); continue
            except Exception: pass
            try:
                new_meta[k] = float(v) if "." in v else int(v); continue
            except Exception: pass
            new_meta[k] = v
        self.meta = new_meta
        save_session_triplet(self._session_dir(), self.meta, logger=self.logger.log)
        self._refresh_preview()

    def _load_settings(self):
        data = self.app_state.settings.get_recording_tab_settings()
        proj = self.app_state.settings.get_loaded_project()
        dest = data.get("destination_path") or proj.get("destination_path", "")
        if dest:
            self.ed_dest.setText(dest)
        for attr, key in [("cb_exp", "experiment"), ("cb_sub", "subject"), ("cb_sess", "session")]:
            val = str(data.get(key, "")).strip()
            if val:
                getattr(self, attr).setCurrentText(val)

    def _persist_settings(self):
        self.app_state.settings.put_recording_tab_settings({
            "destination_path": self.ed_dest.text().strip(),
            "experiment": self._current_experiment(),
            "subject": self._current_subject(),
            "session": self._current_session(),
            "last_session_path": self.app_state.current_session_path or self._session_dir(),
        })
