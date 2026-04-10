import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..io_ops import (
    list_projects,
    list_experiments,
    list_subjects,
    list_sessions,
    load_project_info,
    save_project_info,
    load_experiment_info,
    save_experiment_info,
    load_subject_info,
    save_subject_info,
    load_session_metadata,
    save_session_triplet,
)
from ..state import AppState
from ..utils import run_in_thread
from ..services.server_sync import sync_project_to_server


# ── helpers ────────────────────────────────────────────────────────

def _norm_header(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def read_tabular_any(path: str):
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_excel(path)
    encodings = ["utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("Cannot read table")


def ensure_id_header(df):
    def has_id(frame) -> bool:
        norms = [_norm_header(c) for c in frame.columns]
        wanted = {"id", "animalid", "animal_id", "mouseid", "subject", "subjectid"}
        if any(n in wanted for n in norms):
            return True
        return any(n.endswith("id") and len(n) >= 2 for n in norms)
    if has_id(df):
        return df
    if any(str(c).lower().startswith("unnamed") or isinstance(c, (int, float)) for c in df.columns):
        for ridx in range(min(5, len(df))):
            vals = list(df.iloc[ridx].astype(str))
            if sum((v.strip() != "" and v.lower() != "nan") for v in vals) >= 2:
                new_cols = [v.replace("\ufeff", "").strip() for v in vals]
                new_df = df.iloc[ridx + 1:].copy()
                new_df.columns = new_cols
                if has_id(new_df):
                    return new_df.reset_index(drop=True)
                break
    return df


def find_id_column_smart(df) -> Optional[str]:
    cmap = {_norm_header(str(c)): str(c) for c in df.columns}
    for key in ("id", "animalid", "animal_id", "mouseid", "subject", "subjectid"):
        if key in cmap:
            return cmap[key]
    for norm, orig in cmap.items():
        if norm.endswith("id") and len(norm) >= 2:
            return orig
    return None


def table_to_dict(tbl: QTableWidget) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in range(tbl.rowCount()):
        kitem = tbl.item(r, 0)
        vitem = tbl.item(r, 1)
        k = (kitem.text() if kitem else "").strip()
        if not k:
            continue
        out[k] = vitem.text() if vitem else ""
    return out


def dict_to_table(tbl: QTableWidget, data: Dict[str, Any]):
    tbl.setRowCount(0)
    for k, v in data.items():
        r = tbl.rowCount()
        tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem(str(k)))
        tbl.setItem(r, 1, QTableWidgetItem(v if isinstance(v, str) else str(v)))
    tbl.resizeColumnsToContents()


def human_size(n: int) -> str:
    if n is None:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


def canon_path(path: str) -> str:
    p = os.path.normpath(path)
    if os.name == "nt":
        p = p.replace("/", "\\")
    return p


# ── Sync emitter (thread-safe signals) ────────────────────────────

class _SyncEmitter(QObject):
    log_line = Signal(str)
    finished = Signal(bool, str)   # ok, message


# ── NavigationTab ──────────────────────────────────────────────────

class NavigationTab(QWidget):
    DUMMY_CHILD_TEXT = "..."

    def __init__(self, app_state: AppState, on_load_session):
        super().__init__()
        self.app_state = app_state
        self.on_load_session = on_load_session
        self._sync_running = False
        self._sync_emitter = _SyncEmitter()
        self._sync_emitter.log_line.connect(self._on_sync_log)
        self._sync_emitter.finished.connect(self._on_sync_finished)
        self._build_ui()
        self.refresh_tree(collapsed=True, lazy=True)

    # ── UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # ── Left: tree browser ────────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        row_root = QHBoxLayout()
        row_root.addWidget(QLabel("Data root:"))
        self.ed_root = QLineEdit(self.app_state.settings.data_root)
        row_root.addWidget(self.ed_root, 1)
        b_browse = QPushButton("Browse\u2026")
        b_browse.clicked.connect(self._choose_root)
        row_root.addWidget(b_browse)
        b_reload = QPushButton("Reload")
        b_reload.clicked.connect(lambda: self.refresh_tree(collapsed=True, lazy=True))
        row_root.addWidget(b_reload)
        ll.addLayout(row_root)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.tree.itemExpanded.connect(self._on_item_expanded)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        ll.addWidget(self.tree, 1)

        btns = QHBoxLayout()
        for label, slot in [
            ("Open folder", self._open_selected_folder),
            ("Copy path", self._copy_selected_path),
            ("Load in Recording / Preprocessing", self._load_selected_session),
        ]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btns.addWidget(b)
        ll.addLayout(btns)

        splitter.addWidget(left)

        # ── Right: tabbed details + server link ───────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.right_tabs = QTabWidget()
        rl.addWidget(self.right_tabs)

        self.tab_proj = QWidget()
        self._build_info_tab(self.tab_proj, "Project", "project")
        self.right_tabs.addTab(self.tab_proj, "Project Info")

        self.tab_exp = QWidget()
        self._build_info_tab(self.tab_exp, "Experiment", "experiment")
        self.right_tabs.addTab(self.tab_exp, "Experiment Info")

        self.tab_sub = QWidget()
        self._build_info_tab(self.tab_sub, "Subject", "subject")
        self.right_tabs.addTab(self.tab_sub, "Subject Info")

        self.tab_session = QWidget()
        self._build_session_tab(self.tab_session)
        self.right_tabs.addTab(self.tab_session, "Session Metadata")

        # Project tab (load / destination / backup)
        self.tab_project = QWidget()
        self._build_project_tab(self.tab_project)
        self.right_tabs.addTab(self.tab_project, "Project")

        splitter.addWidget(right)
        splitter.setSizes([460, 850])

    # ── sub-builders ──────────────────────────────────────────────

    def _build_info_tab(self, w, label, key):
        lay = QVBoxLayout(w)
        form = QFormLayout()
        lbl_name = QLabel("-")
        lbl_path = QLabel("-")
        form.addRow(f"{label}:", lbl_name)
        form.addRow("Path:", lbl_path)
        lay.addLayout(form)

        tbl = QTableWidget(0, 2)
        tbl.setHorizontalHeaderLabels(["Key", "Value"])
        tbl.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(tbl, 1)

        row = QHBoxLayout()
        b_add = QPushButton("Add row")
        b_add.clicked.connect(lambda: self._add_row(tbl))
        row.addWidget(b_add)
        b_rm = QPushButton("Remove selected")
        b_rm.clicked.connect(lambda: self._remove_selected(tbl))
        row.addWidget(b_rm)
        b_save = QPushButton(f"Save {label.lower()} info")
        b_save.clicked.connect(lambda: self._save_info_for_kind(key))
        row.addWidget(b_save)
        if key == "subject":
            b_csv = QPushButton("Load subject infos from CSV\u2026")
            b_csv.clicked.connect(self._load_subject_csv)
            row.addWidget(b_csv)
        lay.addLayout(row)

        if key == "project":
            self.lbl_proj = lbl_name; self.lbl_proj_path = lbl_path; self.tbl_proj = tbl
        elif key == "experiment":
            self.lbl_exp = lbl_name; self.lbl_exp_path = lbl_path; self.tbl_exp = tbl
        else:
            self.lbl_sub = lbl_name; self.lbl_sub_path = lbl_path; self.tbl_sub = tbl

    def _build_session_tab(self, w):
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.lbl_session = QLabel("-")
        self.lbl_session_path = QLabel("-")
        form.addRow("Session:", self.lbl_session)
        form.addRow("Path:", self.lbl_session_path)
        lay.addLayout(form)
        self.tbl_session = QTableWidget(0, 2)
        self.tbl_session.setHorizontalHeaderLabels(["Key", "Value"])
        self.tbl_session.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.tbl_session, 1)
        row = QHBoxLayout()
        b_add = QPushButton("Add row")
        b_add.clicked.connect(lambda: self._add_row(self.tbl_session))
        row.addWidget(b_add)
        b_rm = QPushButton("Remove selected")
        b_rm.clicked.connect(lambda: self._remove_selected(self.tbl_session))
        row.addWidget(b_rm)
        b_save = QPushButton("Save session metadata")
        b_save.clicked.connect(self._save_session_metadata)
        row.addWidget(b_save)
        lay.addLayout(row)

    def _build_project_tab(self, w):
        lay = QVBoxLayout(w)

        # ── Load project ──────────────────────────────────────────
        grp_load = QGroupBox("Load Project")
        gl = QVBoxLayout(grp_load)

        btn_row = QHBoxLayout()
        b_local = QPushButton("Load Local Project\u2026")
        b_local.setToolTip("Browse to a local project folder")
        b_local.clicked.connect(self._load_local_project)
        btn_row.addWidget(b_local)
        b_server = QPushButton("Load Server Project\u2026")
        b_server.setToolTip("Browse to a project on a server / network drive")
        b_server.clicked.connect(self._load_server_project)
        btn_row.addWidget(b_server)
        gl.addLayout(btn_row)

        form = QFormLayout()
        self.lbl_loaded_name = QLabel("-")
        self.lbl_loaded_name.setStyleSheet("font-weight: 600; font-size: 12px;")
        form.addRow("Project:", self.lbl_loaded_name)
        self.lbl_loaded_source = QLabel("-")
        self.lbl_loaded_source.setWordWrap(True)
        form.addRow("Source:", self.lbl_loaded_source)
        self.lbl_loaded_type = QLabel("-")
        form.addRow("Type:", self.lbl_loaded_type)
        gl.addLayout(form)
        lay.addWidget(grp_load)

        # ── Destination ───────────────────────────────────────────
        grp_dest = QGroupBox("Local Destination")
        gd = QVBoxLayout(grp_dest)
        gd.addWidget(QLabel("Where new recordings are stored locally:"))
        dest_row = QHBoxLayout()
        self.ed_proj_dest = QLineEdit()
        self.ed_proj_dest.setPlaceholderText("e.g. D:\\data\\raw\\MyProject")
        dest_row.addWidget(self.ed_proj_dest, 1)
        b_dest = QPushButton("Browse\u2026")
        b_dest.clicked.connect(self._browse_destination)
        dest_row.addWidget(b_dest)
        gd.addLayout(dest_row)
        b_save_dest = QPushButton("Save")
        b_save_dest.clicked.connect(self._save_project_settings)
        gd.addWidget(b_save_dest)
        lay.addWidget(grp_dest)

        # ── Backup ────────────────────────────────────────────────
        grp_backup = QGroupBox("Backup to Server")
        gb = QVBoxLayout(grp_backup)

        self.lbl_backup_status = QLabel("Load a project to configure backup.")
        self.lbl_backup_status.setWordWrap(True)
        gb.addWidget(self.lbl_backup_status)

        srv_row = QHBoxLayout()
        srv_row.addWidget(QLabel("Server path:"))
        self.ed_server_path = QLineEdit()
        self.ed_server_path.setPlaceholderText("\\\\server\\share  or  Z:\\backups")
        srv_row.addWidget(self.ed_server_path, 1)
        b_srv = QPushButton("Browse\u2026")
        b_srv.clicked.connect(self._browse_server_path)
        srv_row.addWidget(b_srv)
        gb.addLayout(srv_row)

        self.btn_sync = QPushButton("Backup project to server")
        self.btn_sync.setStyleSheet(
            "QPushButton { background: #d0e8ff; font-weight: 600; padding: 8px 16px; }"
            "QPushButton:hover { background: #b0d4ff; }"
            "QPushButton:disabled { background: #e8e8e8; color: #999; }"
        )
        self.btn_sync.clicked.connect(self._sync_to_server)
        gb.addWidget(self.btn_sync)

        self.sync_progress = QProgressBar()
        self.sync_progress.setRange(0, 0)
        self.sync_progress.setVisible(False)
        gb.addWidget(self.sync_progress)

        self.txt_sync_log = QTextEdit()
        self.txt_sync_log.setReadOnly(True)
        self.txt_sync_log.setMaximumHeight(120)
        self.txt_sync_log.setPlaceholderText("Backup log...")
        gb.addWidget(self.txt_sync_log)
        lay.addWidget(grp_backup)

        lay.addStretch(1)

        # populate from saved settings
        self._refresh_project_tab()

    # ── project actions ───────────────────────────────────────────

    def _load_local_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose local project folder",
                                             self.app_state.settings.raw_root)
        if not d:
            return
        name = os.path.basename(os.path.normpath(d))
        self.app_state.settings.put_loaded_project(name, d, "local", d)
        self._refresh_project_tab()
        self.refresh_tree(collapsed=True, lazy=True)

    def _load_server_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose server project folder")
        if not d:
            return
        name = os.path.basename(os.path.normpath(d))
        # Default local destination = raw_root / project_name
        dest = os.path.join(self.app_state.settings.raw_root, name)
        # Auto-link server backup path = parent of project folder
        server_parent = os.path.dirname(os.path.normpath(d))
        self.app_state.settings.put_server_root_for_project(name, server_parent)
        self.app_state.settings.put_loaded_project(name, d, "server", dest)
        self._refresh_project_tab()
        self.refresh_tree(collapsed=True, lazy=True)

    def _browse_destination(self):
        d = QFileDialog.getExistingDirectory(self, "Choose local destination",
                                             self.ed_proj_dest.text() or "")
        if d:
            self.ed_proj_dest.setText(d)

    def _browse_server_path(self):
        d = QFileDialog.getExistingDirectory(self, "Choose server folder",
                                             self.ed_server_path.text() or "")
        if d:
            self.ed_server_path.setText(d)

    def _save_project_settings(self):
        proj = self.app_state.settings.get_loaded_project()
        name = proj["name"]
        if not name:
            QMessageBox.warning(self, "Save", "Load a project first.")
            return
        dest = self.ed_proj_dest.text().strip()
        self.app_state.settings.put_loaded_project(
            name, proj["source_path"], proj["source_type"], dest
        )
        srv = self.ed_server_path.text().strip()
        if srv:
            self.app_state.settings.put_server_root_for_project(name, srv)
        self._refresh_project_tab()
        QMessageBox.information(self, "Saved", f"Settings saved for '{name}'.")

    def _refresh_project_tab(self):
        proj = self.app_state.settings.get_loaded_project()
        name = proj["name"]
        if name:
            self.lbl_loaded_name.setText(name)
            self.lbl_loaded_source.setText(proj["source_path"])
            self.lbl_loaded_type.setText(proj["source_type"] or "local")
            self.ed_proj_dest.setText(proj["destination_path"])
            srv = self.app_state.settings.get_server_root_for_project(name)
            self.ed_server_path.setText(srv)
            if srv:
                self.lbl_backup_status.setText(f"Ready: {name} \u2192 {srv}")
            else:
                self.lbl_backup_status.setText("Set a server path for backup.")
        else:
            self.lbl_loaded_name.setText("-")
            self.lbl_loaded_source.setText("-")
            self.lbl_loaded_type.setText("-")
            self.lbl_backup_status.setText("Load a project first.")

    # ── backup (sync local → server) ──────────────────────────────

    def _sync_to_server(self):
        proj = self.app_state.settings.get_loaded_project()
        name = proj["name"]
        dest = proj["destination_path"]
        if not name or not dest:
            QMessageBox.warning(self, "Backup", "Load a project first.")
            return
        if not os.path.isdir(dest):
            QMessageBox.critical(self, "Backup", f"Local destination not found:\n{dest}")
            return
        server_root = self.ed_server_path.text().strip()
        if not server_root:
            QMessageBox.warning(self, "Backup", "Set a server path first.")
            return
        if not os.path.isdir(server_root):
            QMessageBox.critical(
                self, "Backup",
                f"Server path not accessible:\n{server_root}\n\n"
                "Check that the network drive is mounted."
            )
            return

        # Save server path if changed
        self.app_state.settings.put_server_root_for_project(name, server_root)

        self._sync_running = True
        self.btn_sync.setEnabled(False)
        self.btn_sync.setText(f"Backing up {name}\u2026")
        self.sync_progress.setVisible(True)
        self.txt_sync_log.clear()
        self.lbl_backup_status.setText(f"Syncing {name} \u2192 {server_root}")

        emitter = self._sync_emitter

        def work():
            try:
                sync_project_to_server(
                    dest, server_root,
                    lambda msg: emitter.log_line.emit(msg),
                )
                emitter.finished.emit(True, f"Backup complete: {name}")
            except Exception as e:
                emitter.finished.emit(False, f"Backup failed: {e}")

        run_in_thread(work)

    def _on_sync_log(self, msg: str):
        self.txt_sync_log.append(msg)

    def _on_sync_finished(self, ok: bool, message: str):
        self._sync_running = False
        self.btn_sync.setEnabled(True)
        self.btn_sync.setText("Backup project to server")
        self.sync_progress.setVisible(False)
        self.lbl_backup_status.setText(message)
        if ok:
            self.txt_sync_log.append("\n--- Backup completed successfully ---")
        else:
            self.txt_sync_log.append(f"\n--- {message} ---")
            QMessageBox.critical(self, "Backup failed", message)

    # ── tree ──────────────────────────────────────────────────────

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose data root", self.ed_root.text() or "")
        if not d:
            return
        d = canon_path(d)
        self.ed_root.setText(d)
        self.app_state.settings.data_root = d
        os.makedirs(self.app_state.settings.raw_root, exist_ok=True)
        os.makedirs(self.app_state.settings.processed_root, exist_ok=True)
        self.refresh_tree(collapsed=True, lazy=True)

    def refresh_tree(self, collapsed=True, lazy=True):
        self.tree.clear()
        self.app_state.settings.ensure_storage_roots()
        raw_root = self.app_state.settings.raw_root
        try:
            os.makedirs(raw_root, exist_ok=True)
        except Exception:
            self.app_state.settings.ensure_storage_roots()
            raw_root = self.app_state.settings.raw_root

        for proj in list_projects(raw_root):
            proj_dir = canon_path(os.path.join(raw_root, proj))
            pitem = QTreeWidgetItem([proj])
            pitem.setData(0, Qt.UserRole, ("project", proj_dir))
            self.tree.addTopLevelItem(pitem)
            if lazy:
                pitem.addChild(QTreeWidgetItem([self.DUMMY_CHILD_TEXT]))
            else:
                self._populate_project_children(pitem, proj_dir, lazy)

        if collapsed:
            self.tree.collapseAll()

        last = self.app_state.settings.last_opened_project
        if last:
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                if item.text(0) == last:
                    self.tree.setCurrentItem(item)
                    self._on_select()
                    break

    def _on_item_expanded(self, item):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, path = data
        if item.childCount() == 1 and item.child(0).text(0) == self.DUMMY_CHILD_TEXT:
            item.takeChild(0)
            if kind == "project":
                self._populate_project_children(item, path, lazy=True)
            elif kind == "experiment":
                self._populate_experiment_children(item, path, lazy=True)
            elif kind == "subject":
                self._populate_subject_children(item, path)

    def _populate_project_children(self, pitem, proj_dir, lazy=True):
        for exp in list_experiments(proj_dir):
            exp_dir = canon_path(os.path.join(proj_dir, exp))
            eitem = QTreeWidgetItem([exp])
            eitem.setData(0, Qt.UserRole, ("experiment", exp_dir))
            pitem.addChild(eitem)
            if lazy:
                eitem.addChild(QTreeWidgetItem([self.DUMMY_CHILD_TEXT]))
            else:
                self._populate_experiment_children(eitem, exp_dir, lazy=False)

    def _populate_experiment_children(self, eitem, exp_dir, lazy=True):
        for subject in list_subjects(exp_dir):
            sub_dir = canon_path(os.path.join(exp_dir, subject))
            sitem = QTreeWidgetItem([subject])
            sitem.setData(0, Qt.UserRole, ("subject", sub_dir))
            eitem.addChild(sitem)
            if lazy:
                sitem.addChild(QTreeWidgetItem([self.DUMMY_CHILD_TEXT]))
            else:
                self._populate_subject_children(sitem, sub_dir)

    def _populate_subject_children(self, sitem, sub_dir):
        for sess in list_sessions(sub_dir):
            sess_dir = canon_path(os.path.join(sub_dir, sess))
            sess_item = QTreeWidgetItem([sess])
            sess_item.setData(0, Qt.UserRole, ("session", sess_dir))
            sitem.addChild(sess_item)

    def _get_selected(self) -> Optional[Tuple[str, str, str]]:
        items = self.tree.selectedItems()
        if not items:
            return None
        item = items[0]
        data = item.data(0, Qt.UserRole)
        if not data:
            return None
        kind, path = data
        return kind, path, item.text(0)

    def _get_selected_nodes(self) -> List[Tuple[str, str, str]]:
        out: List[Tuple[str, str, str]] = []
        for item in self.tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if not data:
                continue
            kind, path = data
            out.append((str(kind), str(path), item.text(0)))
        return out

    def _on_select(self):
        sel = self._get_selected()
        if not sel:
            return
        kind, path, text = sel

        # keep project tab in sync — walk up tree to find project
        proj_name = ""
        if kind == "project":
            proj_name = text
        else:
            item = self.tree.selectedItems()[0] if self.tree.selectedItems() else None
            while item and item.parent():
                item = item.parent()
            proj_name = item.text(0) if item else ""

        if kind == "project":
            self.app_state.settings.last_opened_project = text
            self.app_state.set_current(project=text, experiment="", animal="", session="", session_path="")
            info = self._without_stat_keys(load_project_info(path))
            dict_to_table(self.tbl_proj, {**self._project_stats(path), **info})
            self.lbl_proj.setText(text); self.lbl_proj_path.setText(path)
            self.right_tabs.setCurrentIndex(0)
            return

        if kind == "experiment":
            project = os.path.basename(os.path.dirname(path))
            self.app_state.set_current(project=project, experiment=text, animal="", session="", session_path="")
            info = self._without_stat_keys(load_experiment_info(path))
            dict_to_table(self.tbl_exp, {**self._experiment_stats(path), **info})
            self.lbl_exp.setText(text); self.lbl_exp_path.setText(path)
            self.right_tabs.setCurrentIndex(1)
            return

        if kind == "subject":
            exp_dir = os.path.dirname(path)
            project = os.path.basename(os.path.dirname(exp_dir))
            experiment = os.path.basename(exp_dir)
            self.app_state.set_current(project=project, experiment=experiment, animal=text, session="", session_path="")
            info = self._without_stat_keys(load_subject_info(path))
            dict_to_table(self.tbl_sub, {**self._subject_stats(path), **info})
            self.lbl_sub.setText(text); self.lbl_sub_path.setText(path)
            self.right_tabs.setCurrentIndex(2)
            return

        if kind == "session":
            sub_dir = os.path.dirname(path)
            exp_dir = os.path.dirname(sub_dir)
            project = os.path.basename(os.path.dirname(exp_dir))
            experiment = os.path.basename(exp_dir)
            subject = os.path.basename(sub_dir)
            session = os.path.basename(path)
            self.app_state.set_current(project=project, experiment=experiment, animal=subject, session=session, session_path=path)
            meta = load_session_metadata(path) or {}
            self.lbl_session.setText(session); self.lbl_session_path.setText(path)
            dict_to_table(self.tbl_session, meta)
            self.right_tabs.setCurrentIndex(3)

    # ── stats ─────────────────────────────────────────────────────

    def _project_stats(self, proj_dir):
        exp_count = subject_count = session_count = total_files = total_size = 0
        experiments, experimenters, dt_list = set(), set(), []
        for exp in list_experiments(proj_dir):
            exp_count += 1
            exp_dir = os.path.join(proj_dir, exp)
            for sub in list_subjects(exp_dir):
                subject_count += 1
                sub_dir = os.path.join(exp_dir, sub)
                for sess in list_sessions(sub_dir):
                    session_count += 1
                    sm = load_session_metadata(os.path.join(sub_dir, sess)) or {}
                    if sm.get("Experiment"): experiments.add(str(sm["Experiment"]))
                    if sm.get("Experimenter"): experimenters.add(str(sm["Experimenter"]))
                    if sm.get("DateTime"): dt_list.append(str(sm["DateTime"]))
                    for it in sm.get("file_list", []):
                        total_files += 1
                        if isinstance(it, dict) and isinstance(it.get("size"), int): total_size += int(it["size"])
        return {
            "stats_experiments_count": exp_count, "stats_subjects_count": subject_count,
            "stats_sessions_total": session_count,
            "stats_experiments": ", ".join(sorted(experiments)) if experiments else "",
            "stats_experimenters": ", ".join(sorted(experimenters)) if experimenters else "",
            "stats_first_session": min(dt_list) if dt_list else "",
            "stats_last_session": max(dt_list) if dt_list else "",
            "stats_total_files": total_files, "stats_total_size_bytes": total_size,
            "stats_total_size_human": human_size(total_size),
        }

    def _experiment_stats(self, exp_dir):
        subjects = list_subjects(exp_dir)
        sessions_total = total_files = total_size = 0
        rec_types, dt_list = set(), []
        for sub in subjects:
            sub_dir = os.path.join(exp_dir, sub)
            for sess in list_sessions(sub_dir):
                sessions_total += 1
                sm = load_session_metadata(os.path.join(sub_dir, sess)) or {}
                rt = str(sm.get("Recording") or "").strip()
                if rt: rec_types.add(rt)
                if sm.get("DateTime"): dt_list.append(str(sm["DateTime"]))
                for it in sm.get("file_list", []):
                    total_files += 1
                    if isinstance(it, dict) and isinstance(it.get("size"), int): total_size += int(it["size"])
        return {
            "stats_subjects_count": len(subjects), "stats_sessions_total": sessions_total,
            "stats_recording_types": ", ".join(sorted(rec_types)) if rec_types else "",
            "stats_first_session": min(dt_list) if dt_list else "",
            "stats_last_session": max(dt_list) if dt_list else "",
            "stats_files_count": total_files, "stats_size_bytes": total_size,
            "stats_size_human": human_size(total_size),
        }

    def _subject_stats(self, subject_dir):
        sessions = list_sessions(subject_dir)
        rec_types, dt_list = set(), []
        total_files = total_size = 0
        for sess in sessions:
            sm = load_session_metadata(os.path.join(subject_dir, sess)) or {}
            rt = str(sm.get("Recording") or "").strip()
            if rt: rec_types.add(rt)
            if sm.get("DateTime"): dt_list.append(str(sm["DateTime"]))
            for it in sm.get("file_list", []):
                total_files += 1
                if isinstance(it, dict) and isinstance(it.get("size"), int): total_size += int(it["size"])
        return {
            "stats_sessions_total": len(sessions),
            "stats_recording_types": ", ".join(sorted(rec_types)) if rec_types else "",
            "stats_first_session": min(dt_list) if dt_list else "",
            "stats_last_session": max(dt_list) if dt_list else "",
            "stats_files_count": total_files, "stats_size_bytes": total_size,
            "stats_size_human": human_size(total_size),
        }

    def _without_stat_keys(self, data):
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if not str(k).startswith("_stat_") and not str(k).startswith("stats_")}

    # ── actions ───────────────────────────────────────────────────

    def _open_selected_folder(self):
        sel = self._get_selected()
        if not sel:
            return
        target = canon_path(sel[1])
        if not os.path.exists(target):
            parent = canon_path(os.path.dirname(target))
            if os.path.exists(parent):
                target = parent
            else:
                QMessageBox.critical(self, "Path not found", f"Cannot find:\n{target}")
                return
        try:
            if os.name == "nt":
                try:
                    os.startfile(target)
                except Exception:
                    import subprocess; subprocess.run(["explorer", target])
            elif sys.platform == "darwin":
                import subprocess; subprocess.run(["open", target])
            else:
                import subprocess; subprocess.run(["xdg-open", target])
        except Exception as e:
            QMessageBox.critical(self, "Open error", f"Failed to open:\n{target}\n\n{e}")

    def _copy_selected_path(self):
        sel = self._get_selected()
        if not sel:
            return
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText(canon_path(sel[1]))
        QMessageBox.information(self, "Path copied", canon_path(sel[1]))

    def _load_selected_session(self):
        sel = self._get_selected()
        if not sel:
            return
        kind, path, _ = sel
        if kind != "session":
            QMessageBox.warning(self, "Select session", "Please select a session node.")
            return
        self.on_load_session(path)

    def _on_item_double_clicked(self, item, _column):
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, path = data
        if kind == "session":
            self.on_load_session(path)

    # ── table helpers ─────────────────────────────────────────────

    def _add_row(self, tbl):
        r = tbl.rowCount(); tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem("")); tbl.setItem(r, 1, QTableWidgetItem(""))

    def _remove_selected(self, tbl):
        for r in sorted({i.row() for i in tbl.selectedIndexes()}, reverse=True):
            tbl.removeRow(r)

    def _save_info_for_kind(self, kind):
        sel = self._get_selected()
        if not sel:
            return
        skind, path, _ = sel
        if skind != kind:
            QMessageBox.warning(self, "Selection", f"Select a {kind} node first.")
            return
        if kind == "project":
            data = table_to_dict(self.tbl_proj)
        elif kind == "experiment":
            data = table_to_dict(self.tbl_exp)
        elif kind == "subject":
            data = table_to_dict(self.tbl_sub)
        else:
            return
        for k in list(data.keys()):
            if k.startswith("_stat_") or k.startswith("stats_"):
                data.pop(k, None)
        if kind == "project":
            save_project_info(path, data)
        elif kind == "experiment":
            save_experiment_info(path, data)
        elif kind == "subject":
            save_subject_info(path, data)
        QMessageBox.information(self, "Saved", f"{kind.capitalize()} info saved.")

    def _save_session_metadata(self):
        sel = self._get_selected()
        if not sel:
            return
        kind, path, _ = sel
        if kind != "session":
            QMessageBox.warning(self, "Selection", "Select a session node first.")
            return
        save_session_triplet(path, table_to_dict(self.tbl_session))
        QMessageBox.information(self, "Saved", "Session metadata saved.")

    # ── CSV import ────────────────────────────────────────────────

    def _load_subject_csv(self):
        nodes = self._get_selected_nodes()
        if not nodes:
            return
        projects = [(p, t) for k, p, t in nodes if k == "project"]
        subjects = [(p, t) for k, p, t in nodes if k == "subject"]
        others = [(k, t) for k, _, t in nodes if k not in ("project", "subject")]
        if others:
            QMessageBox.warning(self, "Selection", "Select only subject node(s), or one project node."); return
        if projects and subjects:
            QMessageBox.warning(self, "Selection", "Select either subject node(s) or one project node, not both."); return
        if projects and len(projects) > 1:
            QMessageBox.warning(self, "Selection", "Select only one project node."); return
        if not projects and not subjects:
            QMessageBox.warning(self, "Selection", "Select subject node(s) or one project node."); return

        csv_path, _ = QFileDialog.getOpenFileName(self, "Choose CSV/Excel", "", "Tables (*.csv *.xlsx *.xls *.xlsm);;All files (*)")
        if not csv_path:
            return
        try:
            df = read_tabular_any(csv_path)
            df = ensure_id_header(df)
            id_col = find_id_column_smart(df)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to read table:\n{e}"); return
        if not id_col:
            QMessageBox.critical(self, "CSV format", "No ID-like column found for subject matching."); return

        def last5(x):
            s = str(x); return s[-5:] if len(s) >= 5 else s

        df["_ID5"] = df[id_col].map(last5)

        def row_to_info(row):
            info = {}
            for k, v in row.items():
                if k == "_ID5": continue
                try:
                    if isinstance(v, float) and v != v: v = ""
                except Exception: pass
                info[k] = v
            return info

        if projects:
            path, _ = projects[0]
            updated = 0
            for exp in list_experiments(path):
                exp_dir = os.path.join(path, exp)
                for subject in list_subjects(exp_dir):
                    matches = df[df["_ID5"] == last5(subject)]
                    if matches.empty: continue
                    sub_dir = os.path.join(exp_dir, subject)
                    existing = load_subject_info(sub_dir)
                    existing.update(row_to_info(matches.iloc[0].to_dict()))
                    save_subject_info(sub_dir, existing); updated += 1
            QMessageBox.information(self, "Import complete", f"Updated {updated} subjects in project.")
        else:
            updated = 0; unmatched: List[str] = []
            for path, text in subjects:
                matches = df[df["_ID5"] == last5(text)]
                if matches.empty: unmatched.append(text); continue
                existing = load_subject_info(path)
                existing.update(row_to_info(matches.iloc[0].to_dict()))
                save_subject_info(path, existing); updated += 1
            total = len(subjects)
            if updated == 0:
                QMessageBox.warning(self, "No match", f"No selected subjects matched CSV IDs ({total} selected)."); return
            if unmatched:
                preview = ", ".join(unmatched[:8])
                suffix = "..." if len(unmatched) > 8 else ""
                QMessageBox.information(self, "Import complete", f"Updated {updated}/{total} selected subjects.\nUnmatched ({len(unmatched)}): {preview}{suffix}")
            else:
                QMessageBox.information(self, "Import complete", f"Updated {updated}/{total} selected subjects.")
