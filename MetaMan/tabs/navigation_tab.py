import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
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
    load_structure_sidecar,
)
from ..state import AppState
from ..utils import run_in_thread
from ..services import fs_ops
from ..services.staging_service import server_raw_root
from ..services.structure_schema import (
    sublevels, is_marker_level, marker_folder_name, level_meta_key, level_label,
    normalize_structure_schema, LEVEL_COLORS, META_KEY_FOR_LEVEL,
)
from ..services.metadata_scraper import scrape_session, merge_auto, classify_modality, is_probably_local
from ..services.session_assets import (
    load_notes,
    read_metadata_update_file,
    save_notes,
    upload_files_to_project,
)
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

# ── level dot icons (cached) ───────────────────────────────────────────
_DOT_CACHE: Dict[str, QIcon] = {}


def _level_dot(kind: str) -> QIcon:
    """A small filled circle coloured by level type, for tree rows."""
    key = (kind or "project").strip().lower()
    if key in _DOT_CACHE:
        return _DOT_CACHE[key]
    color = LEVEL_COLORS.get(key, "#7aa2e8")
    pm = QPixmap(14, 14)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(2, 2, 10, 10)
    p.end()
    icon = QIcon(pm)
    _DOT_CACHE[key] = icon
    return icon


# Humanised labels for the cryptic ``stats_*`` keys, in display order.
_STAT_ORDER = [
    ("stats_experiments_count", "Experiments"),
    ("stats_subjects_count", "Subjects"),
    ("stats_recordings_count", "Recordings"),
    ("stats_trials_count", "Trials"),
    ("stats_sessions_total", "Sessions"),
    ("stats_modalities", "Modalities"),
    ("stats_files_count", "Files"),
    ("stats_size_human", "Total size"),
    ("stats_first_session", "First session"),
    ("stats_last_session", "Last session"),
    ("stats_recording_types", "Recording types"),
    ("stats_experimenters", "Experimenters"),
]


def humanize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the raw ``stats_*`` dict into a clean, ordered, labelled summary."""
    out: Dict[str, Any] = {}
    for key, label in _STAT_ORDER:
        if key in stats and str(stats[key]).strip() not in ("", "0"):
            out[label] = stats[key]
    return out


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
        value = vitem.text() if vitem else ""
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                out[k] = json.loads(stripped)
                continue
            except Exception:
                pass
        out[k] = value
    return out


def dict_to_table(tbl: QTableWidget, data: Dict[str, Any]):
    tbl.setRowCount(0)
    for k, v in data.items():
        r = tbl.rowCount()
        tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem(str(k)))
        if isinstance(v, (dict, list)):
            text = json.dumps(v, ensure_ascii=False)
        else:
            text = v if isinstance(v, str) else str(v)
        tbl.setItem(r, 1, QTableWidgetItem(text))
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


class _StatsEmitter(QObject):
    ready = Signal(int, str, dict)  # seq, node_path, stats


# ── NavigationTab ──────────────────────────────────────────────────

class NavigationTab(QWidget):
    DUMMY_CHILD_TEXT = "..."

    def __init__(self, app_state: AppState, on_load_session, on_activate_project=None,
                 on_local_changed=None):
        super().__init__()
        self.app_state = app_state
        self.on_load_session = on_load_session
        self.on_activate_project = on_activate_project
        self.on_local_changed = on_local_changed
        # Server tab + local-copy worker state
        self._server_display_root = ""
        self._server_base = ""
        self._copy_running = False
        self._copy_emitter = _SyncEmitter()
        self._copy_emitter.log_line.connect(self._on_copy_log)
        self._copy_emitter.finished.connect(self._on_copy_finished)
        # Background stats: node selection never blocks on a full session walk.
        self._stats_cache = {}
        self._stats_seq = 0
        self._pending_stats = None
        self._stats_emitter = _StatsEmitter()
        self._stats_emitter.ready.connect(self._on_stats_ready)
        self._notes_editors: Dict[str, QTextEdit] = {}
        self._build_ui()
        self.refresh_tree(collapsed=True, lazy=True)
        saved = self.app_state.settings.get_explorer_settings().get("server_root", "")
        if saved:
            self._set_server_root(saved)

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

        # Data root + project loading now live in the global project bar; this
        # tab focuses on browsing. Keep a hidden ed_root for compatibility with
        # callers that still set its text.
        self.ed_root = QLineEdit(self.app_state.settings.data_root)
        self.ed_root.setVisible(False)

        # Local | Server tabs: both are schema-driven dataset trees. They share
        # the right-hand metadata panels; ``self.tree`` always points at the
        # active one.
        self._lr_tabs = QTabWidget()

        # Local page
        local_page = QWidget()
        lp = QVBoxLayout(local_page)
        lp.setContentsMargins(0, 0, 0, 0)
        row_top = QHBoxLayout()
        row_top.addWidget(QLabel("Local projects"))
        row_top.addStretch(1)
        b_reload = QPushButton("Reload")
        b_reload.clicked.connect(lambda: self.refresh_tree(collapsed=True, lazy=True))
        row_top.addWidget(b_reload)
        lp.addLayout(row_top)
        self.local_tree = self._make_tree()
        lp.addWidget(self.local_tree, 1)
        self._lr_tabs.addTab(local_page, "Local")

        # Server page
        server_page = QWidget()
        sp = QVBoxLayout(server_page)
        sp.setContentsMargins(0, 0, 0, 0)
        srv_row = QHBoxLayout()
        self.ed_server_root = QLineEdit()
        self.ed_server_root.setPlaceholderText("Server share that holds the projects…")
        self.ed_server_root.returnPressed.connect(
            lambda: self._set_server_root(self.ed_server_root.text().strip()))
        srv_row.addWidget(self.ed_server_root, 1)
        b_srv_browse = QPushButton("Browse…")
        b_srv_browse.clicked.connect(self._browse_server_root)
        srv_row.addWidget(b_srv_browse)
        b_srv_reload = QPushButton("Reload")
        b_srv_reload.clicked.connect(self._refresh_server_tree)
        srv_row.addWidget(b_srv_reload)
        sp.addLayout(srv_row)
        self.server_tree = self._make_tree()
        sp.addWidget(self.server_tree, 1)
        self._lr_tabs.addTab(server_page, "Server")

        self._lr_tabs.currentChanged.connect(self._on_lr_tab_changed)
        self.tree = self.local_tree  # active tree
        ll.addWidget(self._lr_tabs, 1)

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

        self.lbl_copy = QLabel("")
        self.lbl_copy.setObjectName("Hint")
        self.lbl_copy.setWordWrap(True)
        ll.addWidget(self.lbl_copy)

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

        # (Project load / destination / backup / schedule moved to the global
        # project bar and the Transfer tab.)

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
        self._add_scope_notes_editor(lay, key, label)

        row = QHBoxLayout()
        b_add = QPushButton("Add row")
        b_add.clicked.connect(lambda: self._add_row(tbl))
        row.addWidget(b_add)
        b_rm = QPushButton("Remove selected")
        b_rm.clicked.connect(lambda: self._remove_selected(tbl))
        row.addWidget(b_rm)
        b_save = QPushButton(f"Save {label.lower()} info")
        b_save.setToolTip("Saves table fields and notes.")
        b_save.clicked.connect(lambda: self._save_info_for_kind(key))
        row.addWidget(b_save)
        self._add_scope_file_actions(row, key)
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
        self._add_scope_notes_editor(lay, "session", "Session")
        row = QHBoxLayout()
        b_add = QPushButton("Add row")
        b_add.clicked.connect(lambda: self._add_row(self.tbl_session))
        row.addWidget(b_add)
        b_rm = QPushButton("Remove selected")
        b_rm.clicked.connect(lambda: self._remove_selected(self.tbl_session))
        row.addWidget(b_rm)
        b_save = QPushButton("Save session metadata")
        b_save.setToolTip("Saves table fields and notes.")
        b_save.clicked.connect(self._save_session_metadata)
        row.addWidget(b_save)
        self._add_scope_file_actions(row, "session")
        lay.addLayout(row)

        # Ctrl+C and right-click copy on the session metadata table
        self.tbl_session.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_session.customContextMenuRequested.connect(self._session_table_context_menu)
        self.tbl_session.installEventFilter(self)

    def _add_scope_notes_editor(self, lay: QVBoxLayout, kind: str, label: str):
        notes_label = QLabel(f"{label} notes")
        notes_label.setObjectName("Hint")
        lay.addWidget(notes_label)

        editor = QTextEdit()
        editor.setAcceptRichText(False)
        editor.setMinimumHeight(96)
        editor.setMaximumHeight(160)
        editor.setPlaceholderText(
            f"Write notes for this {label.lower()} here."
        )
        lay.addWidget(editor)
        self._notes_editors[kind] = editor

    def _add_scope_file_actions(self, row: QHBoxLayout, kind: str):
        b_import_meta = QPushButton("Import metadata...")
        b_import_meta.setToolTip(
            "Import CSV/TXT/JSON fields into the selected "
            f"{self._scope_label(kind).lower()} metadata."
        )
        b_import_meta.clicked.connect(lambda _checked=False, k=kind: self._import_metadata_file_for_scope(k))
        row.addWidget(b_import_meta)

        b_upload = QPushButton("Upload files...")
        b_upload.setToolTip(
            "Copy arbitrary files into the project upload folder and record them "
            f"on the selected {self._scope_label(kind).lower()}."
        )
        b_upload.clicked.connect(lambda _checked=False, k=kind: self._upload_files_for_scope(k))
        row.addWidget(b_upload)

    # ── tree ──────────────────────────────────────────────────────

    def _make_tree(self) -> QTreeWidget:
        """A configured dataset tree wired to the shared handlers. Used for both
        the Local and Server tabs."""
        t = QTreeWidget()
        t.setHeaderHidden(True)
        t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        t.itemSelectionChanged.connect(self._on_select)
        t.itemExpanded.connect(self._on_item_expanded)
        t.itemDoubleClicked.connect(self._on_item_double_clicked)
        t.setContextMenuPolicy(Qt.CustomContextMenu)
        t.customContextMenuRequested.connect(self._tree_context_menu)
        return t

    def _on_lr_tab_changed(self, idx: int):
        self.tree = self.server_tree if idx == 1 else self.local_tree
        self._on_select()

    def _active_base(self) -> str:
        """Folder that directly contains project folders for the active tab."""
        if self.tree is self.server_tree:
            return self._server_base
        return self.app_state.settings.raw_root

    def _server_active(self) -> bool:
        return self.tree is self.server_tree

    # ── server tab ─────────────────────────────────────────────────

    def _browse_server_root(self):
        start = self._server_display_root or self.ed_server_root.text().strip() or ""
        d = QFileDialog.getExistingDirectory(self, "Choose the server projects share", start)
        if d:
            self._set_server_root(d)

    def _set_server_root(self, path: str):
        path = canon_path(path) if path else ""
        self._server_display_root = path
        self._server_base = server_raw_root(path) if (path and os.path.isdir(path)) else ""
        self.ed_server_root.setText(path)
        self.app_state.settings.put_explorer_settings({"server_root": path})
        self._build_server_tree()

    def _refresh_server_tree(self):
        self._set_server_root(self.ed_server_root.text().strip())

    def _build_server_tree(self):
        self._stats_cache.clear()
        self.server_tree.clear()
        base = self._server_base
        if not base or not os.path.isdir(base):
            return
        for proj in self._subdirs(base):
            proj_dir = canon_path(os.path.join(base, proj))
            pitem = QTreeWidgetItem([proj])
            pitem.setIcon(0, _level_dot("project"))
            pitem.setData(0, Qt.UserRole, ("project", proj_dir))
            pitem.addChild(QTreeWidgetItem([self.DUMMY_CHILD_TEXT]))
            self.server_tree.addTopLevelItem(pitem)

    def focus_server_tab(self):
        """Switch to the Server tab, pointing it at the active project's server
        root if it has none yet (used by the menu shortcut)."""
        self._lr_tabs.setCurrentIndex(1)
        if not self._server_base:
            active = (self.app_state.current_project
                      or self.app_state.settings.get_loaded_project().get("name", "")).strip()
            srv = self.app_state.settings.get_server_root_for_project(active) if active else ""
            if srv and os.path.isdir(srv):
                self._set_server_root(srv)

    def _refresh_active_tree(self):
        if self._server_active():
            self._build_server_tree()
        else:
            self.refresh_tree(collapsed=True, lazy=True)

    # ── make local copy (server → canonical rawData) ───────────────

    def _make_local_copy(self, ctx):
        if self._copy_running:
            QMessageBox.information(self, "Make local copy", "A copy is already in progress.")
            return
        server_path = ctx["path"]
        base = self._server_base
        raw_root = self.app_state.settings.raw_root

        dest_dir = raw_root
        mapped = False
        if base:
            try:
                rel = os.path.relpath(server_path, base)
            except Exception:
                rel = ""
            if rel and not rel.startswith(".."):
                dest_dir = os.path.join(raw_root, os.path.dirname(rel))
                mapped = True
        if not mapped:
            ans = QMessageBox.question(
                self, "Make local copy",
                "This item is outside the server's project tree, so its position "
                "can't be reconstructed.\n\n"
                f"Copy it directly under the local rawData root?\n{raw_root}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
            dest_dir = raw_root

        self._copy_running = True
        self.lbl_copy.setText(f"Copying {os.path.basename(server_path)} → local…")
        emitter = self._copy_emitter

        def work():
            ok = True
            try:
                stats = fs_ops.copy_into(server_path, dest_dir, lambda m: emitter.log_line.emit(m))
                msg = (f"Local copy complete: {stats['copied'] + stats['updated']} copied, "
                       f"{stats['skipped']} unchanged, {stats['failed']} failed.")
                if stats["failed"]:
                    ok = False
            except Exception as e:
                ok = False
                msg = f"Local copy failed: {e}"
            emitter.finished.emit(ok, msg)

        run_in_thread(work)

    def _on_copy_log(self, msg: str):
        self.lbl_copy.setText(msg if len(msg) < 120 else msg[:117] + "…")

    def _on_copy_finished(self, ok: bool, msg: str):
        self._copy_running = False
        self.lbl_copy.setText(msg)
        self.refresh_tree(collapsed=True, lazy=True)  # local tree now holds the copy
        if self.on_local_changed:
            self.on_local_changed()
        if not ok:
            QMessageBox.warning(self, "Make local copy", msg)

    def refresh_tree(self, collapsed=True, lazy=True):
        """Rebuild the *local* projects tree from the raw root."""
        self._stats_cache.clear()
        self.local_tree.clear()
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
            pitem.setIcon(0, _level_dot("project"))
            pitem.setData(0, Qt.UserRole, ("project", proj_dir))
            self.local_tree.addTopLevelItem(pitem)
            if lazy:
                pitem.addChild(QTreeWidgetItem([self.DUMMY_CHILD_TEXT]))
            else:
                self._populate_children(pitem, proj_dir, lazy)

        if collapsed:
            self.local_tree.collapseAll()

        last = self.app_state.settings.last_opened_project
        if last and self.tree is self.local_tree:
            for i in range(self.local_tree.topLevelItemCount()):
                item = self.local_tree.topLevelItem(i)
                if item.text(0) == last:
                    self.local_tree.setCurrentItem(item)
                    self._on_select()
                    break

    def select_project(self, name: str):
        """Highlight the given top-level project in the local tree (no rebuild)."""
        name = (name or "").strip()
        if not name:
            return
        for i in range(self.local_tree.topLevelItemCount()):
            it = self.local_tree.topLevelItem(i)
            if it.text(0) == name:
                self.local_tree.blockSignals(True)
                self.local_tree.setCurrentItem(it)
                self.local_tree.blockSignals(False)
                return

    def _on_item_expanded(self, item):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        if item.childCount() == 1 and item.child(0).text(0) == self.DUMMY_CHILD_TEXT:
            item.takeChild(0)
            self._populate_children(item, data[1], lazy=True)

    # ── schema-driven traversal ───────────────────────────────────

    def _project_sublevels(self, project: str) -> List[dict]:
        # Prefer a structure sidecar inside the project folder (so a server
        # project renders with its real hierarchy, not the local default); fall
        # back to the per-name schema from settings.
        proj_dir = os.path.join(self._active_base(), project)
        sidecar = load_structure_sidecar(proj_dir)
        if sidecar:
            return sublevels(normalize_structure_schema(sidecar), "raw")
        schema = self.app_state.settings.resolve_structure_schema(project)
        return sublevels(schema, "raw")

    def _node_depth(self, item) -> int:
        """0 for a project (top-level) node, 1 for its children, etc."""
        d = 0
        p = item.parent()
        while p is not None:
            d += 1
            p = p.parent()
        return d

    @staticmethod
    def _subdirs(path: str) -> List[str]:
        try:
            return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))
        except Exception:
            return []

    def _populate_children(self, item, path, lazy=True):
        """Populate *item*'s children using the project's structure schema:
        the folder level at this depth comes from the ordered sublevels."""
        project = self._owning_project(item)
        subs = self._project_sublevels(project)
        idx = self._node_depth(item)  # children correspond to sublevels[idx]
        if idx >= len(subs):
            return
        lvl = subs[idx]
        kind = str(lvl.get("key", "")).strip().lower()
        is_leaf = (idx == len(subs) - 1)

        if is_marker_level(kind):
            names = [marker_folder_name(lvl)]
            names = [n for n in names if os.path.isdir(os.path.join(path, n))]
        else:
            names = self._subdirs(path)

        for name in names:
            child_dir = canon_path(os.path.join(path, name))
            citem = QTreeWidgetItem([name])
            citem.setIcon(0, _level_dot(kind))
            citem.setData(0, Qt.UserRole, (kind, child_dir))
            item.addChild(citem)
            if not is_leaf:
                if lazy:
                    citem.addChild(QTreeWidgetItem([self.DUMMY_CHILD_TEXT]))
                else:
                    self._populate_children(citem, child_dir, lazy=False)

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

    def _selected_context(self) -> Optional[Dict[str, Any]]:
        items = self.tree.selectedItems()
        if not items:
            return None
        item = items[0]
        data = item.data(0, Qt.UserRole)
        if not data:
            return None
        kind, path = data
        project = self._owning_project(item)
        proj_dir = canon_path(os.path.join(self._active_base(), project))
        subs = self._project_sublevels(project)
        try:
            rel = os.path.relpath(path, proj_dir)
            parts = [] if rel in (".", "") else rel.replace("\\", "/").split("/")
        except Exception:
            parts = []
        values: Dict[str, str] = {}
        for lvl, part in zip(subs, parts):
            values[str(lvl.get("key", "")).strip().lower()] = part
        is_leaf = (kind != "project") and len(subs) > 0 and len(parts) >= len(subs)
        return {"item": item, "kind": kind, "path": path, "project": project,
                "proj_dir": proj_dir, "subs": subs, "values": values,
                "parts": parts, "is_leaf": is_leaf}

    def _on_select(self):
        snd = self.sender()
        if isinstance(snd, QTreeWidget):
            self.tree = snd
        ctx = self._selected_context()
        if not ctx:
            return
        item, kind, path, project = ctx["item"], ctx["kind"], ctx["path"], ctx["project"]
        values = ctx["values"]
        experiment = values.get("experiment", "")
        subject = values.get("subject", "")
        idx = self._node_depth(item) - 1  # this node's sublevel index (project -> -1)

        if kind == "project":
            if not self._server_active():
                self.app_state.settings.last_opened_project = project
            self._browse_set_current(project=project, experiment="", animal="", session="", session_path="")
            info = self._without_stat_keys(load_project_info(path))
            self._load_scope_notes_panel("project", path, info)
            info = self._without_note_keys(info)
            self.lbl_proj.setText(project); self.lbl_proj_path.setText(path)
            self.right_tabs.setCurrentIndex(0)
            self._fill_info_with_stats(self.tbl_proj, info, project, path, 0)
            return

        if ctx["is_leaf"]:
            session = os.path.basename(path)
            self._browse_set_current(project=project, experiment=experiment, animal=subject,
                                     session=session, session_path=path)
            meta = load_session_metadata(path) or {}
            # Folder structure is authoritative for identity (fixes stale/swapped
            # values in old metadata.json, e.g. Subject="rawData").
            meta["Project"] = project
            for k, v in values.items():
                mk = META_KEY_FOR_LEVEL.get(k)
                if mk:
                    meta[mk] = v
            if values.get("subject"):
                meta["Animal"] = values["subject"]
            if is_probably_local(path):  # avoid blocking on slow network shares
                try:
                    meta = merge_auto(meta, scrape_session(path, deep=True))
                except Exception:
                    pass
            self.lbl_session.setText(session); self.lbl_session_path.setText(path)
            self._load_scope_notes_panel("session", path, meta)
            dict_to_table(self.tbl_session, self._without_note_keys(meta))
            self.right_tabs.setCurrentIndex(3)
            return

        if kind == "experiment":
            self._browse_set_current(project=project, experiment=os.path.basename(path),
                                     animal="", session="", session_path="")
            info = self._without_stat_keys(load_experiment_info(path))
            self._load_scope_notes_panel("experiment", path, info)
            info = self._without_note_keys(info)
            self.lbl_exp.setText(os.path.basename(path)); self.lbl_exp_path.setText(path)
            self.right_tabs.setCurrentIndex(1)
            self._fill_info_with_stats(self.tbl_exp, info, project, path, idx + 1)
            return

        if kind == "subject":
            self._browse_set_current(project=project, experiment=experiment,
                                     animal=os.path.basename(path), session="", session_path="")
            info = self._without_stat_keys(load_subject_info(path))
            self._load_scope_notes_panel("subject", path, info)
            info = self._without_note_keys(info)
            self.lbl_sub.setText(os.path.basename(path)); self.lbl_sub_path.setText(path)
            self.right_tabs.setCurrentIndex(2)
            self._fill_info_with_stats(self.tbl_sub, info, project, path, idx + 1)
            return

        # Intermediate non-core level (recording / trial / marker): track context only.
        self._browse_set_current(project=project, experiment=experiment, animal=subject,
                                 session="", session_path="")
        self._clear_scope_notes_panels()

    def _browse_set_current(self, **kwargs):
        """Update the active dataset context, but only while browsing Local –
        browsing the Server tab must never hijack the active local project."""
        if self._server_active():
            return
        self.app_state.set_current(**kwargs)

    def _fill_info_with_stats(self, table, info, project, node_path, start_index):
        """Show *info* immediately and compute the (potentially expensive) stats
        off the UI thread, caching the result. Selecting a project with thousands
        of sessions no longer freezes the window."""
        key = (node_path, self._active_base())
        cached = self._stats_cache.get(key)
        if cached is not None:
            dict_to_table(table, {**humanize_stats(cached), **info})
            return
        dict_to_table(table, {**{"Stats": "computing…"}, **info})
        self._stats_seq += 1
        seq = self._stats_seq
        self._pending_stats = {"seq": seq, "key": key, "table": table, "info": info}
        emitter = self._stats_emitter

        def work():
            try:
                stats = self._stats_for(project, node_path, start_index)
            except Exception:
                stats = {}
            emitter.ready.emit(seq, node_path, stats)

        run_in_thread(work)

    def _on_stats_ready(self, seq: int, node_path: str, stats: dict):
        p = self._pending_stats
        if not p or p["seq"] != seq:
            return  # selection moved on; drop this stale result
        self._stats_cache[p["key"]] = stats
        dict_to_table(p["table"], {**humanize_stats(stats), **p["info"]})

    # ── stats (schema-driven) ─────────────────────────────────────

    def _iter_sessions_under(self, project: str, node_dir: str, start_index: int):
        """Yield (values_by_key, session_dir, metadata) for every session below
        *node_dir*, descending the schema sublevels from *start_index*."""
        subs = self._project_sublevels(project)
        if not subs:
            return
        leaf = len(subs) - 1

        def rec(cur_dir, idx, acc):
            lvl = subs[idx]
            key = str(lvl.get("key", "")).strip().lower()
            if is_marker_level(key):
                nd = os.path.join(cur_dir, marker_folder_name(lvl))
                if not os.path.isdir(nd):
                    return
                if idx == leaf:
                    yield acc, nd, (load_session_metadata(nd) or {})
                else:
                    yield from rec(nd, idx + 1, acc)
                return
            for name in self._subdirs(cur_dir):
                nd = os.path.join(cur_dir, name)
                acc2 = dict(acc); acc2[key] = name
                if idx == leaf:
                    yield acc2, nd, (load_session_metadata(nd) or {})
                else:
                    yield from rec(nd, idx + 1, acc2)

        if start_index > leaf:
            return
        yield from rec(node_dir, start_index, {})

    def _stats_for(self, project: str, node_dir: str, start_index: int) -> Dict[str, Any]:
        per_key: Dict[str, set] = {}
        sessions = files = size = 0
        experimenters, rec_types, dt_list, modalities = set(), set(), [], set()
        for values, _sdir, sm in self._iter_sessions_under(project, node_dir, start_index):
            sessions += 1
            for k, v in values.items():
                per_key.setdefault(k, set()).add(v)
            if sm.get("Experimenter"): experimenters.add(str(sm["Experimenter"]))
            if sm.get("DateTime"): dt_list.append(str(sm["DateTime"]))
            rt = str(sm.get("Recording") or "").strip()
            if rt: rec_types.add(rt)
            # cheap modality inference from the saved file list (no disk walk)
            exts, names = {}, []
            for it in sm.get("file_list", []):
                files += 1
                if isinstance(it, dict) and isinstance(it.get("size"), int):
                    size += int(it["size"])
                fpath = it.get("path", "") if isinstance(it, dict) else str(it)
                base = os.path.basename(fpath)
                if base:
                    names.append(base)
                    e = os.path.splitext(base)[1].lower()
                    exts[e] = exts.get(e, 0) + 1
            if sm.get("Auto: modality"):
                modalities.update(str(sm["Auto: modality"]).split(", "))
            else:
                m = classify_modality(exts, names)
                if m:
                    modalities.update(m.split(", "))
        out: Dict[str, Any] = {
            "stats_sessions_total": sessions,
            "stats_files_count": files,
            "stats_size_bytes": size,
            "stats_size_human": human_size(size),
            "stats_first_session": min(dt_list) if dt_list else "",
            "stats_last_session": max(dt_list) if dt_list else "",
            "stats_recording_types": ", ".join(sorted(rec_types)) if rec_types else "",
            "stats_experimenters": ", ".join(sorted(experimenters)) if experimenters else "",
            "stats_modalities": ", ".join(sorted(m for m in modalities if m)) if modalities else "",
        }
        for k, vals in per_key.items():
            out[f"stats_{k}s_count"] = len(vals)
        return out

    def _iter_level_dirs(self, project: str, target_key: str):
        """Yield directories at the sublevel whose key == *target_key*."""
        subs = self._project_sublevels(project)
        target_idx = next((i for i, l in enumerate(subs)
                           if str(l.get("key", "")).strip().lower() == target_key), -1)
        if target_idx < 0:
            return
        proj_dir = os.path.join(self.app_state.settings.raw_root, project)

        def rec(cur, idx):
            lvl = subs[idx]
            key = str(lvl.get("key", "")).strip().lower()
            names = [marker_folder_name(lvl)] if is_marker_level(key) else self._subdirs(cur)
            for name in names:
                nd = os.path.join(cur, name)
                if not os.path.isdir(nd):
                    continue
                if idx == target_idx:
                    yield nd
                elif idx < target_idx:
                    yield from rec(nd, idx + 1)

        yield from rec(proj_dir, 0)

    def _without_stat_keys(self, data):
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if not str(k).startswith("_stat_") and not str(k).startswith("stats_")}

    def _without_note_keys(self, data):
        if not isinstance(data, dict):
            return {}
        note_keys = {"notes", "sessionnotes"}
        return {
            k: v for k, v in data.items()
            if _norm_header(str(k)) not in note_keys
        }

    # ── actions ───────────────────────────────────────────────────

    def _open_selected_folder(self):
        sel = self._get_selected()
        if not sel:
            return
        target = canon_path(sel[1])
        try:
            fs_ops.open_path(target)
        except Exception as e:
            QMessageBox.critical(self, "Open error", f"Failed to open:\n{target}\n\n{e}")

    def _reveal_selected(self):
        sel = self._get_selected()
        if not sel:
            return
        try:
            fs_ops.reveal_path(canon_path(sel[1]))
        except Exception as e:
            QMessageBox.critical(self, "Reveal", f"Could not reveal:\n{sel[1]}\n\n{e}")

    # ── tree context menu (dataset management) ─────────────────────────

    def _tree_context_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        snd = self.sender()
        if isinstance(snd, QTreeWidget):
            self.tree = snd
        item = self.tree.itemAt(pos)
        if item is not None and item not in self.tree.selectedItems():
            self.tree.setCurrentItem(item)
        ctx = self._selected_context()
        if not ctx:
            return
        server = self._server_active()
        menu = QMenu(self)
        menu.addAction("Open folder", self._open_selected_folder)
        menu.addAction("Reveal in file manager", self._reveal_selected)
        menu.addAction("Copy path", self._copy_selected_path)
        if ctx["is_leaf"]:
            menu.addAction("Load in Record / Process", self._load_selected_session)
        if server:
            menu.addSeparator()
            scope = {"project": "project", "experiment": "experiment",
                     "subject": "subject", "session": "session"}.get(ctx["kind"], "item")
            menu.addAction(f"⬇  Make local copy of this {scope}", lambda: self._make_local_copy(ctx))
        menu.addSeparator()
        child = self._child_level(ctx)
        if child is not None:
            menu.addAction(f"New {level_label(child)}…", lambda c=child: self._tree_new_child(ctx, c))
        menu.addAction("Rename…", lambda: self._tree_rename(ctx))
        menu.addAction("Delete…", lambda: self._tree_delete(ctx))
        menu.addSeparator()
        menu.addAction("Refresh", self._refresh_active_tree)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _child_level(self, ctx) -> Optional[dict]:
        """The schema level that a *new child* of the selected node would be, or
        None if the node is a leaf or its child level is a fixed marker."""
        subs = ctx["subs"]
        child_idx = len(ctx["parts"])  # project node → 0 → first sublevel
        if child_idx >= len(subs):
            return None
        lvl = subs[child_idx]
        if is_marker_level(str(lvl.get("key", "")).strip().lower()):
            return None
        return lvl

    def _tree_new_child(self, ctx, lvl):
        from PySide6.QtWidgets import QInputDialog
        label = level_label(lvl)
        name, ok = QInputDialog.getText(self, f"New {label}", f"{label} name:")
        if not ok:
            return
        clean = fs_ops.safe_name(name)
        if not clean:
            QMessageBox.warning(self, "New", "Name is empty after removing illegal characters.")
            return
        target = os.path.join(ctx["path"], clean)
        if os.path.exists(target):
            QMessageBox.warning(self, "New", f"'{clean}' already exists here.")
            return
        try:
            os.makedirs(target)
        except Exception as e:
            QMessageBox.critical(self, "New", str(e))
            return
        self._refresh_active_tree()

    def _tree_rename(self, ctx):
        from PySide6.QtWidgets import QInputDialog
        path = ctx["path"]
        old = os.path.basename(path)
        new, ok = QInputDialog.getText(self, "Rename", "New name:", text=old)
        if not ok or not new.strip():
            return
        try:
            fs_ops.rename_path(path, new)
        except Exception as e:
            QMessageBox.critical(self, "Rename", str(e))
            return
        self._refresh_active_tree()

    def _tree_delete(self, ctx):
        from PySide6.QtWidgets import QInputDialog
        path = ctx["path"]
        name = os.path.basename(path)
        via = "the Recycle Bin" if fs_ops.trash_available() else "PERMANENT deletion (Send2Trash not installed)"
        text, ok = QInputDialog.getText(
            self, "Confirm delete",
            f"This will delete:\n{path}\n\nDeletion uses {via}.\n\n"
            f"Type the name '{name}' to confirm:",
        )
        if not (ok and text.strip() == name):
            return
        try:
            how = fs_ops.delete_path(path)
        except Exception as e:
            QMessageBox.critical(self, "Delete", f"Could not delete:\n{path}\n\n{e}")
            return
        self._refresh_active_tree()
        if not self._server_active():
            self.app_state.set_current(project=self.app_state.current_project, experiment="",
                                       animal="", session="", session_path="")
        QMessageBox.information(
            self, "Delete",
            f"{'Recycled' if how == 'trash' else 'Permanently deleted'}:\n{path}",
        )

    def _copy_session_metadata_rows(self):
        """Copy selected rows (or all rows) from tbl_session as JSON to clipboard."""
        tbl = self.tbl_session
        selected_rows = sorted({idx.row() for idx in tbl.selectedIndexes()})
        if not selected_rows:
            selected_rows = list(range(tbl.rowCount()))
        if not selected_rows:
            return
        import json as _json
        from PySide6.QtGui import QGuiApplication
        data = {}
        for r in selected_rows:
            ki = tbl.item(r, 0)
            vi = tbl.item(r, 1)
            if ki:
                data[ki.text()] = vi.text() if vi else ""
        text = _json.dumps(data, indent=2, ensure_ascii=False)
        QGuiApplication.clipboard().setText(text)

    def _session_table_context_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Copy", self._copy_session_metadata_rows)
        menu.exec(self.tbl_session.viewport().mapToGlobal(pos))

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.tbl_session and event.type() == QEvent.KeyPress:
            from PySide6.QtGui import QKeySequence
            if event.matches(QKeySequence.Copy):
                self._copy_session_metadata_rows()
                return True
        return super().eventFilter(obj, event)

    def _copy_selected_path(self):
        sel = self._get_selected()
        if not sel:
            return
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText(canon_path(sel[1]))
        QMessageBox.information(self, "Path copied", canon_path(sel[1]))

    def _owning_project(self, item) -> str:
        """Return the top-level (project) node text for *item*."""
        while item and item.parent():
            item = item.parent()
        return item.text(0) if item else ""

    def _load_selected_session(self):
        sel = self._get_selected()
        if not sel:
            return
        kind, path, text = sel
        items = self.tree.selectedItems()
        project = self._owning_project(items[0]) if items else (text if kind == "project" else "")
        # Activate the owning project so Recording / Preprocessing follow the
        # selection (instead of staying on a previously loaded project).
        if project and self.on_activate_project:
            self.on_activate_project(project)
        if kind == "session":
            self.on_load_session(path)

    def _on_item_double_clicked(self, item, _column):
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, path = data
        project = self._owning_project(item)
        if project and self.on_activate_project:
            self.on_activate_project(project)
        if kind == "session":
            self.on_load_session(path)

    # ── table helpers ─────────────────────────────────────────────

    def _add_row(self, tbl):
        r = tbl.rowCount(); tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem("")); tbl.setItem(r, 1, QTableWidgetItem(""))

    def _remove_selected(self, tbl):
        for r in sorted({i.row() for i in tbl.selectedIndexes()}, reverse=True):
            tbl.removeRow(r)

    def _scope_label(self, kind: str) -> str:
        return {
            "project": "Project",
            "experiment": "Experiment",
            "subject": "Subject",
            "session": "Session",
        }.get(str(kind or "").strip().lower(), "Scope")

    def _scope_table(self, kind: str):
        if kind == "project":
            return self.tbl_proj
        if kind == "experiment":
            return self.tbl_exp
        if kind == "subject":
            return self.tbl_sub
        if kind == "session":
            return self.tbl_session
        return None

    def _selected_scope_context(self, kind: str) -> Optional[Dict[str, Any]]:
        ctx = self._selected_context()
        label = self._scope_label(kind).lower()
        if not ctx:
            QMessageBox.warning(self, "Selection", f"Select a {label} node first.")
            return None
        actual = "session" if ctx.get("is_leaf") else str(ctx.get("kind", ""))
        if actual != kind:
            QMessageBox.warning(self, "Selection", f"Select a {label} node first.")
            return None
        path = str(ctx.get("path", "") or "")
        if not path or not os.path.isdir(path):
            QMessageBox.warning(self, "Selection", f"The selected {label} folder could not be found.")
            return None
        return ctx

    def _metadata_from_scope_table(self, kind: str) -> Dict[str, Any]:
        table = self._scope_table(kind)
        data = table_to_dict(table) if table is not None else {}
        data = self._without_note_keys(data)
        if kind != "session":
            data = self._without_stat_keys(data)
        return data

    def _current_scope_metadata(self, kind: str) -> Dict[str, Any]:
        return self._metadata_with_scope_notes(kind, self._metadata_from_scope_table(kind))

    def _scope_stat_fields(self, kind: str) -> Dict[str, Any]:
        table = self._scope_table(kind)
        if table is None or kind == "session":
            return {}
        data = table_to_dict(table)
        return {k: v for k, v in data.items() if str(k).startswith("_stat_") or str(k).startswith("stats_")}

    def _write_scope_metadata(self, kind: str, path: str, metadata: Dict[str, Any]):
        if kind == "project":
            data = self._without_stat_keys(metadata)
            save_project_info(path, data)
            dict_to_table(self.tbl_proj, {**self._scope_stat_fields(kind), **self._without_note_keys(data)})
        elif kind == "experiment":
            data = self._without_stat_keys(metadata)
            save_experiment_info(path, data)
            dict_to_table(self.tbl_exp, {**self._scope_stat_fields(kind), **self._without_note_keys(data)})
        elif kind == "subject":
            data = self._without_stat_keys(metadata)
            save_subject_info(path, data)
            dict_to_table(self.tbl_sub, {**self._scope_stat_fields(kind), **self._without_note_keys(data)})
        elif kind == "session":
            self._write_session_metadata(path, metadata)

    def _notes_editor(self, kind: str) -> Optional[QTextEdit]:
        return self._notes_editors.get(kind)

    def _scope_notes_text(self, kind: str) -> str:
        editor = self._notes_editor(kind)
        return editor.toPlainText().strip() if editor is not None else ""

    def _set_scope_notes(self, kind: str, text: str):
        editor = self._notes_editor(kind)
        if editor is not None:
            editor.setPlainText(str(text or ""))

    def _append_scope_notes(self, kind: str, notes: str):
        clean_notes = str(notes or "").strip()
        if not clean_notes:
            return
        current = self._scope_notes_text(kind)
        self._set_scope_notes(kind, f"{current}\n\n{clean_notes}".strip() if current else clean_notes)

    def _load_scope_notes_panel(self, kind: str, path: str, metadata: Dict[str, Any]):
        if kind == "session":
            notes = load_notes(path, metadata)
        else:
            notes = str(metadata.get("Notes") or metadata.get("notes") or "")
        self._clear_scope_notes_panels(keep=kind)
        self._set_scope_notes(kind, notes)

    def _clear_scope_notes_panels(self, keep: str = ""):
        for kind, editor in self._notes_editors.items():
            if kind != keep:
                editor.clear()

    def _metadata_with_scope_notes(self, kind: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(metadata)
        notes = self._scope_notes_text(kind)
        if notes:
            data["Notes"] = notes
        else:
            data.pop("Notes", None)
            data.pop("notes", None)
        return data

    @staticmethod
    def _append_uploaded_records(metadata: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
        data = dict(metadata)
        uploaded = data.get("uploaded_files", [])
        if isinstance(uploaded, str):
            try:
                uploaded = json.loads(uploaded)
            except Exception:
                uploaded = []
        if not isinstance(uploaded, list):
            uploaded = []
        data["uploaded_files"] = uploaded + records
        return data

    def _save_info_for_kind(self, kind):
        ctx = self._selected_scope_context(kind)
        if not ctx:
            return
        self._write_scope_metadata(kind, ctx["path"], self._current_scope_metadata(kind))
        QMessageBox.information(self, "Saved", f"{self._scope_label(kind)} info saved.")

    def _save_session_metadata(self):
        path = self._selected_session_path()
        if not path:
            return
        self._write_session_metadata(path, self._current_scope_metadata("session"))
        QMessageBox.information(self, "Saved", "Session metadata saved.")

    def _selected_session_path(self) -> str:
        ctx = self._selected_context()
        if ctx and ctx.get("is_leaf") and os.path.isdir(ctx["path"]):
            return ctx["path"]
        path = str(getattr(self.app_state, "current_session_path", "") or "")
        if path and os.path.isdir(path):
            return path
        QMessageBox.warning(self, "Selection", "Select a session node first.")
        return ""

    def _write_session_metadata(self, session_dir: str, metadata: Dict[str, Any]):
        data = dict(metadata)
        save_session_triplet(session_dir, data)
        save_notes(session_dir, data.get("Notes", ""))
        dict_to_table(self.tbl_session, self._without_note_keys(data))

    def _save_session_notes(self):
        path = self._selected_session_path()
        if not path:
            return
        self._write_session_metadata(path, self._current_scope_metadata("session"))
        QMessageBox.information(self, "Saved", "Session notes saved.")

    def _import_metadata_file_for_scope(self, kind: str):
        ctx = self._selected_scope_context(kind)
        if not ctx:
            return
        label = self._scope_label(kind)
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {label.lower()} metadata file",
            "",
            "Metadata files (*.csv *.tsv *.txt *.json);;All files (*)",
        )
        if not path:
            return
        try:
            fields, notes = read_metadata_update_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import metadata", f"Could not read metadata file:\n{exc}")
            return

        if not fields and not notes:
            QMessageBox.information(self, "Import metadata", "No metadata fields or notes were found.")
            return

        metadata = self._current_scope_metadata(kind)
        metadata.update(fields)
        if notes:
            self._append_scope_notes(kind, notes)
            metadata = self._metadata_with_scope_notes(kind, metadata)
        self._write_scope_metadata(kind, ctx["path"], metadata)
        QMessageBox.information(
            self,
            "Import metadata",
            f"Imported {len(fields)} {label.lower()} field(s)" + (" and appended notes." if notes else "."),
        )

    def _import_session_metadata_file(self):
        self._import_metadata_file_for_scope("session")

    def _upload_files_for_scope(self, kind: str):
        ctx = self._selected_scope_context(kind)
        if not ctx:
            return
        label = self._scope_label(kind)
        project_dir = ctx.get("proj_dir", "")
        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(self, "Upload files", "Project folder could not be resolved.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Upload files to project",
            "",
            "Common metadata files (*.csv *.tsv *.txt *.json *.xlsx *.xls);;All files (*)",
        )
        if not files:
            return
        context_parts = list(ctx.get("parts", []) or [])
        try:
            records = upload_files_to_project(project_dir, files, context_parts=context_parts)
        except Exception as exc:
            QMessageBox.critical(self, "Upload files", f"Upload failed:\n{exc}")
            return
        if not records:
            QMessageBox.information(self, "Upload files", "No files were uploaded.")
            return

        metadata = self._append_uploaded_records(self._current_scope_metadata(kind), records)
        self._write_scope_metadata(kind, ctx["path"], metadata)
        QMessageBox.information(
            self,
            "Upload files",
            f"Uploaded {len(records)} file(s) for this {label.lower()} into:\n"
            f"{os.path.join(project_dir, '_metaman_uploads')}",
        )

    def _upload_files_to_project(self):
        self._upload_files_for_scope("session")

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

        def norm(x):
            return str(x).strip().lower()

        df["_IDFULL"] = df[id_col].map(norm)
        df["_ID5"] = df[id_col].map(last5)

        def find_match(name):
            """Prefer an exact (normalised) full-ID match; only fall back to the
            last-5-characters heuristic when it is unambiguous."""
            exact = df[df["_IDFULL"] == norm(name)]
            if not exact.empty:
                return exact.iloc[0].to_dict()
            cand = df[df["_ID5"] == last5(name)]
            if len(cand) == 1:
                return cand.iloc[0].to_dict()
            return None  # no match, or an ambiguous last-5 collision

        def row_to_info(row):
            info = {}
            for k, v in row.items():
                if k in ("_ID5", "_IDFULL"): continue
                try:
                    if isinstance(v, float) and v != v: v = ""
                except Exception: pass
                info[k] = v
            return info

        if projects:
            path, project_name = projects[0]
            updated = 0
            for sub_dir in self._iter_level_dirs(project_name, "subject"):
                subject = os.path.basename(sub_dir)
                row = find_match(subject)
                if row is None: continue
                existing = load_subject_info(sub_dir)
                existing.update(row_to_info(row))
                save_subject_info(sub_dir, existing); updated += 1
            QMessageBox.information(self, "Import complete", f"Updated {updated} subjects in project.")
        else:
            updated = 0; unmatched: List[str] = []
            for path, text in subjects:
                row = find_match(text)
                if row is None: unmatched.append(text); continue
                existing = load_subject_info(path)
                existing.update(row_to_info(row))
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
