import json
import os
from datetime import datetime
from typing import Any, Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
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
from ..level_chain import LevelChain
from ..services.structure_schema import sublevels, level_meta_key, is_marker_level
from ..services.metadata_scraper import scrape_session, merge_auto, is_probably_local


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
        setup = QGroupBox("Recording setup")
        sl = QVBoxLayout(setup)

        # The active project is shown in the global project bar; keep a hidden
        # label that the refresh logic still updates.
        self.lbl_project = QLabel("(none)")
        self.lbl_project.setVisible(False)

        # Destination path
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
        sl.addLayout(row_dest)

        # Schema-driven level selector: the order and labels of the levels
        # below the project (experiment / subject / session / ...) follow this
        # project's Structure Designer layout, so projects that nest
        # Subject -> Experiment show exactly that.
        self.levels = LevelChain(self.app_state)
        sl.addWidget(self.levels)

        self.lbl_empty = QLabel("Select a project in the bar above to begin a recording.")
        self.lbl_empty.setStyleSheet("color:#93a4c2; padding: 6px 2px;")
        self.lbl_empty.setVisible(False)
        sl.addWidget(self.lbl_empty)

        self.b_new_rec = QPushButton("➕  New recording")
        self.b_new_rec.setObjectName("Primary")
        self.b_new_rec.clicked.connect(self.new_recording)
        sl.addWidget(self.b_new_rec)

        self.levels.changed.connect(self._persist_settings)
        self.levels.leaf_changed.connect(self._on_leaf_changed)
        self.ed_dest.textChanged.connect(self._on_dest_changed)

        # (Setup is fused with the metadata editor into one "Session" panel below.)

        # Metadata editor ─────────────────────────────────────────
        meta_panel = QGroupBox("Session metadata")
        ml = QVBoxLayout(meta_panel)

        self.tbl_meta = QTableWidget(0, 2)
        self.tbl_meta.setHorizontalHeaderLabels(["Key", "Value"])
        self.tbl_meta.horizontalHeader().setStretchLastSection(True)

        tbl_btns = QHBoxLayout()
        b_add = QPushButton("Add row"); b_add.clicked.connect(self._add_meta_row); tbl_btns.addWidget(b_add)
        b_rm = QPushButton("Remove selected"); b_rm.clicked.connect(self._rm_meta_row); tbl_btns.addWidget(b_rm)
        b_apply = QPushButton("Apply table \u2192 metadata"); b_apply.clicked.connect(self._apply_table_to_meta); tbl_btns.addWidget(b_apply)
        b_scrape = QPushButton("\ud83d\udd0e  Auto-scrape"); b_scrape.setToolTip("Read metadata from the recording files (SpikeGLX, video, file inventory)"); b_scrape.clicked.connect(self._auto_scrape); tbl_btns.addWidget(b_scrape)
        b_save = QPushButton("Save metadata (JSON/CSV/H5)"); b_save.clicked.connect(self._save_all); tbl_btns.addWidget(b_save)
        b_tmpl = QPushButton("Save as default template"); b_tmpl.setToolTip("Save current metadata keys/values as the default for new recordings"); b_tmpl.clicked.connect(self._save_as_default_template); tbl_btns.addWidget(b_tmpl)

        # Ctrl+C/V/X and right-click cut/copy/paste on the metadata table
        self.tbl_meta.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_meta.customContextMenuRequested.connect(self._meta_table_context_menu)
        self.tbl_meta.installEventFilter(self)

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

        # Fuse setup + metadata into a single, natural "Session" panel.
        session_panel = QWidget()
        session_layout = QVBoxLayout(session_panel)
        session_layout.setContentsMargins(0, 0, 0, 0)
        session_layout.setSpacing(10)
        setup.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        session_layout.addWidget(setup)
        session_layout.addWidget(meta_panel, 1)
        self._side.add_panel("\U0001f4dd", "Session", session_panel, default=True)

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

    def _has_levels(self) -> bool:
        return getattr(self, "levels", None) is not None and self.levels.has_levels()

    def _current_experiment(self) -> str:
        return self.levels.values_by_key().get("experiment", "") if self._has_levels() else ""

    def _current_subject(self) -> str:
        return self.levels.values_by_key().get("subject", "") if self._has_levels() else ""

    def _current_session(self) -> str:
        return self.levels.values_by_key().get("session", "") if self._has_levels() else ""

    def _session_dir(self) -> str:
        return self.levels.leaf_path() if self._has_levels() else self.ed_dest.text().strip()

    def _source_dirs(self) -> List[str]:
        src = self.app_state.settings.get_loaded_project().get("source_path", "")
        return [src] if src else []

    def _sublevels(self) -> List[dict]:
        name = self.app_state.settings.get_loaded_project().get("name", "")
        schema = self.app_state.settings.resolve_structure_schema(name)
        return sublevels(schema, "raw")

    def _refresh_from_project(self):
        """Rebuild the level chain for the loaded project. The level order and
        labels follow that project's Structure Designer schema."""
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
        # Only carry the combo values across a refresh of the *same* project;
        # switching projects starts clean so values never leak between projects.
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
        self.b_new_rec.setEnabled(has_proj)
        if not has_proj:
            self.meta = {}
            self._refresh_preview()

    def _on_dest_changed(self, _txt: str = ""):
        if self._has_levels():
            self.levels.set_dirs(self.ed_dest.text().strip(), self._source_dirs())
        self._persist_settings()

    def _on_leaf_changed(self):
        if self._loading_session:
            return
        self._persist_settings()
        self._try_load_current_session(show_warning=False)

    def _try_load_current_session(self, show_warning: bool):
        if not self._has_levels() or not self.levels.all_filled():
            return
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

    def _path_identity(self, session_dir: str) -> Dict[str, Any]:
        """Structural identity (Project / Subject / Experiment / Session …) taken
        strictly from the folder path + the project's schema. The folder tree is
        authoritative, so this overrides any stale or swapped values that an old
        metadata.json may carry (e.g. Subject="rawData")."""
        out: Dict[str, Any] = {}
        proj = self.app_state.settings.get_loaded_project().get("name", "")
        out["Project"] = proj
        base = self.ed_dest.text().strip()
        parts: List[str] = []
        try:
            if base and os.path.normpath(session_dir).startswith(os.path.normpath(base)):
                rel = os.path.relpath(session_dir, base)
                parts = [] if rel in (".", "") else rel.replace("\\", "/").split("/")
        except Exception:
            parts = []
        subs = self._sublevels()
        values = dict(zip(
            [level_meta_key(l) for l in subs if not is_marker_level(str(l.get("key", "")).lower())],
            [p for p in parts],
        ))
        for lvl in subs:
            key = str(lvl.get("key", "")).lower()
            if is_marker_level(key):
                continue
            mk = level_meta_key(lvl)
            out[mk] = values.get(mk, "")
        out["Animal"] = out.get("Subject", "")
        return out

    def load_session(self, session_dir: str):
        meta = load_session_metadata(session_dir) or {}
        if not meta:
            meta = self._derive_meta_from_path(session_dir)
        # Folder structure is authoritative for identity: overwrite any stale or
        # swapped structural fields from the path (the file may predate the
        # schema-aware writer).
        meta.update(self._path_identity(session_dir))
        # Auto-enrich with whatever we can read from the files (non-destructive).
        # Skip network paths here so navigation never blocks; the Auto-scrape
        # button works on any path on demand.
        if is_probably_local(session_dir):
            try:
                meta = merge_auto(meta, scrape_session(session_dir, deep=True))
            except Exception:
                pass

        self.meta = meta
        subject = str(meta.get("Subject", "") or meta.get("Animal", ""))
        self._loading_session = True
        try:
            self.levels.set_from_metadata(meta)
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

    def _derive_meta_from_path(self, session_dir: str) -> Dict[str, Any]:
        """Build a minimal metadata dict from folder names, mapping each path
        segment to the project's schema level (so Subject/Experiment land in the
        right field whatever the nesting order)."""
        dest = self.ed_dest.text().strip()
        meta = SessionMetadata.new("", "", os.path.basename(session_dir), dest).data
        try:
            base = dest if (dest and os.path.normpath(session_dir).startswith(os.path.normpath(dest))) \
                else os.path.dirname(os.path.dirname(session_dir))
            rel = os.path.relpath(session_dir, base)
            parts = [p for p in rel.replace("\\", "/").split("/") if p and p != "."]
        except Exception:
            parts = [os.path.basename(session_dir)]
        subs = self._sublevels()
        n = min(len(parts), len(subs))
        for lvl, val in zip(subs[-n:], parts[-n:]):
            if is_marker_level(str(lvl.get("key", "")).lower()):
                continue
            meta[level_meta_key(lvl)] = val
        return meta

    def new_recording(self):
        dest = self.ed_dest.text().strip()
        project_name = self.app_state.settings.get_loaded_project()["name"]

        if not dest:
            QMessageBox.warning(self, "Missing", "Set a Destination path."); return
        if not self._has_levels() or not self.levels.all_filled():
            missing = ", ".join(self.levels.missing_labels()) if self._has_levels() else "the levels"
            QMessageBox.warning(self, "Missing fields", f"Fill: {missing}."); return

        session_dir = self.levels.leaf_path()
        os.makedirs(session_dir, exist_ok=True)

        meta = self._new_recording_metadata(project_name)
        self.meta = meta
        save_session_triplet(session_dir, self.meta, logger=self.logger.log)
        self.load_session(session_dir)
        self._refresh_from_project()  # refresh combos with the new folders
        self._persist_settings()

    def _new_recording_metadata(self, project):
        dest = self.ed_dest.text().strip()
        level_meta = self.levels.metadata()  # {"Experiment":.., "Subject":.., "Session":..}
        subject = level_meta.get("Subject", "")
        session = level_meta.get("Session", "")
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
            "Project": project,
            "Animal": subject, "Subject": subject,
            "Session": str(session), "RootDir": dest,
            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_list": [], "trial_info": {}, "trial_assets": {}, "preprocessing": [],
        })
        meta.update(level_meta)  # apply all schema level values in their own fields
        # Apply saved default template (custom keys/values for new recordings)
        template = self.app_state.settings.get_metadata_template()
        for k, v in template.items():
            if k not in meta:
                meta[k] = v
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

    def _auto_scrape(self):
        session_dir = self._session_dir()
        if not os.path.isdir(session_dir):
            QMessageBox.warning(self, "Auto-scrape", f"Session folder not found:\n{session_dir}")
            return
        scraped = scrape_session(session_dir, deep=True)
        before = set(self.meta.keys())
        self.meta = merge_auto(self.meta, scraped)
        added = len(set(self.meta.keys()) - before)
        self._refresh_preview()
        self.logger.log(f"Auto-scrape: detected {len(scraped)} field(s), added {added} new.")

    def _copy_meta_rows(self):
        """Copy selected rows from tbl_meta as JSON to clipboard."""
        selected_rows = sorted({idx.row() for idx in self.tbl_meta.selectedIndexes()})
        if not selected_rows:
            selected_rows = list(range(self.tbl_meta.rowCount()))
        if not selected_rows:
            return
        from PySide6.QtGui import QGuiApplication
        data = {}
        for r in selected_rows:
            ki = self.tbl_meta.item(r, 0)
            vi = self.tbl_meta.item(r, 1)
            if ki:
                data[ki.text()] = vi.text() if vi else ""
        QGuiApplication.clipboard().setText(json.dumps(data, indent=2, ensure_ascii=False))

    def _cut_meta_rows(self):
        """Cut selected rows from tbl_meta (copy + remove)."""
        selected_rows = sorted({idx.row() for idx in self.tbl_meta.selectedIndexes()})
        if not selected_rows:
            return
        self._copy_meta_rows()
        for r in reversed(selected_rows):
            self.tbl_meta.removeRow(r)

    def _paste_meta_rows(self):
        """Paste key-value metadata from clipboard (JSON dict) into the table."""
        from PySide6.QtGui import QGuiApplication
        text = QGuiApplication.clipboard().text().strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return

        existing_keys: Dict[str, int] = {}
        for r in range(self.tbl_meta.rowCount()):
            ki = self.tbl_meta.item(r, 0)
            if ki:
                existing_keys[ki.text()] = r

        for k, v in data.items():
            val_str = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            if k in existing_keys:
                row = existing_keys[k]
                self.tbl_meta.setItem(row, 1, QTableWidgetItem(val_str))
            else:
                row = self.tbl_meta.rowCount()
                self.tbl_meta.insertRow(row)
                self.tbl_meta.setItem(row, 0, QTableWidgetItem(str(k)))
                self.tbl_meta.setItem(row, 1, QTableWidgetItem(val_str))

    def _meta_table_context_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Copy", self._copy_meta_rows)
        menu.addAction("Cut", self._cut_meta_rows)
        menu.addAction("Paste", self._paste_meta_rows)
        menu.exec(self.tbl_meta.viewport().mapToGlobal(pos))

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.tbl_meta and event.type() == QEvent.KeyPress:
            from PySide6.QtGui import QKeySequence
            if event.matches(QKeySequence.Copy):
                self._copy_meta_rows()
                return True
            if event.matches(QKeySequence.Cut):
                self._cut_meta_rows()
                return True
            if event.matches(QKeySequence.Paste):
                self._paste_meta_rows()
                return True
        return super().eventFilter(obj, event)

    def _save_as_default_template(self):
        """Save current metadata table as the default template for new recordings."""
        flat = table_to_dict(self.tbl_meta)
        if not flat:
            QMessageBox.warning(self, "Template", "No metadata to save as template.")
            return
        # Exclude recording-specific fields that change every time
        skip = {"DateTime", "SessionUUID", "file_list", "trial_info", "trial_assets", "preprocessing"}
        template = {k: v for k, v in flat.items() if k not in skip}
        self.app_state.settings.put_metadata_template(template)
        QMessageBox.information(self, "Template saved",
                                f"Default metadata template saved ({len(template)} entries).\n"
                                "New recordings will include these fields.")

    def _load_settings(self):
        data = self.app_state.settings.get_recording_tab_settings()
        proj = self.app_state.settings.get_loaded_project()
        dest = data.get("destination_path") or proj.get("destination_path", "")
        if dest:
            self.ed_dest.setText(dest)
        vals = data.get("level_values")
        if not isinstance(vals, dict) or not vals:
            vals = {
                "experiment": str(data.get("experiment", "")),
                "subject": str(data.get("subject", "")),
                "session": str(data.get("session", "")),
            }
        # applied by _refresh_from_project once the levels are built
        self._pending_level_values = {k: v for k, v in vals.items() if str(v).strip()}

    def _persist_settings(self):
        vals = self.levels.values_by_key() if self._has_levels() else {}
        self.app_state.settings.put_recording_tab_settings({
            "destination_path": self.ed_dest.text().strip(),
            "level_values": vals,
            "experiment": vals.get("experiment", ""),
            "subject": vals.get("subject", ""),
            "session": vals.get("session", ""),
            "last_session_path": self.app_state.current_session_path or self._session_dir(),
        })
