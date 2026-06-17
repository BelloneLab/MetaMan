"""
Staging tab – create new recordings linked to server-resident projects,
view their sync status, and trigger manual or batch sync.
"""

import json
import os
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..services.staging_service import (
    create_linked_recording,
    get_staging_dir,
    list_server_experiments,
    list_server_projects,
    list_server_subjects,
    load_manifest,
    remove_entry,
    save_manifest,
    sync_pending,
)
from ..state import AppState
from ..utils import LogEmitter, run_in_thread

# status → colour mapping for the table
_STATUS_COLORS = {
    "pending": QColor("#e8a83e"),
    "synced": QColor("#3ba55d"),
    "error": QColor("#ed4245"),
}


class StagingTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self._syncing = False
        self._build_ui()
        self._load_settings()
        QTimer.singleShot(300, self._refresh_manifest_table)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter)

        # ── top: create linked recording ─────────────────────────────────
        top = QGroupBox("Link new recording to a server project")
        top_lay = QVBoxLayout(top)

        # server root row
        row_srv = QHBoxLayout()
        row_srv.addWidget(QLabel("Server root:"))
        self.ed_server_root = QLineEdit()
        self.ed_server_root.setPlaceholderText("Browse to the root of the server data share…")
        row_srv.addWidget(self.ed_server_root, 1)
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._browse_server_root)
        row_srv.addWidget(b_browse)
        b_scan = QPushButton("Scan projects")
        b_scan.setToolTip("Read project & experiment lists from the server root")
        b_scan.clicked.connect(self._scan_server)
        row_srv.addWidget(b_scan)
        top_lay.addLayout(row_srv)

        # project / experiment row
        row_pe = QHBoxLayout()
        row_pe.addWidget(QLabel("Project:"))
        self.cb_project = QComboBox()
        self.cb_project.setMinimumWidth(180)
        self.cb_project.currentIndexChanged.connect(self._on_project_changed)
        row_pe.addWidget(self.cb_project, 1)

        row_pe.addWidget(QLabel("Experiment:"))
        self.cb_experiment = QComboBox()
        self.cb_experiment.setEditable(True)
        self.cb_experiment.setMinimumWidth(180)
        self.cb_experiment.currentIndexChanged.connect(self._on_experiment_changed)
        row_pe.addWidget(self.cb_experiment, 1)
        top_lay.addLayout(row_pe)

        # subject / session row
        row_ss = QHBoxLayout()
        row_ss.addWidget(QLabel("Subject:"))
        self.cb_subject = QComboBox()
        self.cb_subject.setEditable(True)
        self.cb_subject.setMinimumWidth(140)
        row_ss.addWidget(self.cb_subject, 1)
        row_ss.addWidget(QLabel("Session:"))
        self.ed_session = QLineEdit()
        self.ed_session.setPlaceholderText("e.g. 2026-04-10_01")
        row_ss.addWidget(self.ed_session, 1)
        top_lay.addLayout(row_ss)

        # create button
        row_btn = QHBoxLayout()
        row_btn.addStretch(1)
        self.b_create = QPushButton("  Create linked recording  ")
        self.b_create.setObjectName("Success")
        self.b_create.clicked.connect(self._create_linked)
        row_btn.addWidget(self.b_create)
        self.b_open_folder = QPushButton("Open staging folder")
        self.b_open_folder.clicked.connect(self._open_staging_folder)
        row_btn.addWidget(self.b_open_folder)
        row_btn.addStretch(1)
        top_lay.addLayout(row_btn)

        splitter.addWidget(top)

        # ── middle: manifest table ───────────────────────────────────────
        mid = QGroupBox("Staged recordings")
        mid_lay = QVBoxLayout(mid)

        self.tbl = QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels([
            "Status", "Project", "Experiment", "Subject", "Session",
            "Created", "Synced", "Local path",
        ])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setAlternatingRowColors(True)
        mid_lay.addWidget(self.tbl, 1)

        tbl_btns = QHBoxLayout()
        b_refresh = QPushButton("Refresh")
        b_refresh.clicked.connect(self._refresh_manifest_table)
        tbl_btns.addWidget(b_refresh)

        b_sync_sel = QPushButton("Sync selected")
        b_sync_sel.clicked.connect(self._sync_selected)
        tbl_btns.addWidget(b_sync_sel)

        b_sync_all = QPushButton("Sync all pending")
        b_sync_all.setObjectName("Primary")
        b_sync_all.clicked.connect(self._sync_all_pending)
        tbl_btns.addWidget(b_sync_all)

        b_open_local = QPushButton("Open local folder")
        b_open_local.clicked.connect(self._open_selected_local)
        tbl_btns.addWidget(b_open_local)

        b_remove = QPushButton("Remove entry")
        b_remove.clicked.connect(self._remove_selected)
        tbl_btns.addWidget(b_remove)

        b_resync = QPushButton("Mark as pending")
        b_resync.setToolTip("Re-queue a synced or errored entry for another sync pass")
        b_resync.clicked.connect(self._mark_pending)
        tbl_btns.addWidget(b_resync)

        tbl_btns.addStretch(1)
        mid_lay.addLayout(tbl_btns)

        splitter.addWidget(mid)

        # ── bottom: log ──────────────────────────────────────────────────
        bot = QGroupBox("Log")
        bot_lay = QVBoxLayout(bot)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        mono = QFont("Consolas" if os.name == "nt" else "Monospace")
        mono.setStyleHint(QFont.Monospace)
        self.txt_log.setFont(mono)
        bot_lay.addWidget(self.txt_log, 1)
        self.logger = LogEmitter(self.txt_log)
        splitter.addWidget(bot)

        splitter.setSizes([260, 300, 180])

    # ── server browsing ──────────────────────────────────────────────────

    def _browse_server_root(self):
        start = self.ed_server_root.text().strip() or ""
        d = QFileDialog.getExistingDirectory(self, "Choose server data root", start)
        if d:
            self.ed_server_root.setText(d)
            self._scan_server()

    def _scan_server(self):
        srv = self.ed_server_root.text().strip()
        if not srv or not os.path.isdir(srv):
            QMessageBox.warning(self, "Server root", "Choose an existing server root first.")
            return
        projects = list_server_projects(srv)
        self.cb_project.blockSignals(True)
        self.cb_project.clear()
        self.cb_project.addItems(projects)
        self.cb_project.blockSignals(False)
        if projects:
            self.cb_project.setCurrentIndex(0)
            self._on_project_changed()
        self.logger.log(f"Scanned server: {len(projects)} project(s) found.")
        self._persist_settings()

    def _on_project_changed(self, _index: int = 0):
        srv = self.ed_server_root.text().strip()
        proj = self.cb_project.currentText().strip()
        exps = list_server_experiments(srv, proj) if (srv and proj) else []
        self.cb_experiment.blockSignals(True)
        self.cb_experiment.clear()
        self.cb_experiment.addItems(exps)
        self.cb_experiment.blockSignals(False)
        if exps:
            self.cb_experiment.setCurrentIndex(0)
            self._on_experiment_changed()
        else:
            self.cb_subject.clear()
        self._persist_settings()

    def _on_experiment_changed(self, _index: int = 0):
        srv = self.ed_server_root.text().strip()
        proj = self.cb_project.currentText().strip()
        exp = self.cb_experiment.currentText().strip()
        subjects = list_server_subjects(srv, proj, exp) if (srv and proj and exp) else []
        self.cb_subject.blockSignals(True)
        self.cb_subject.clear()
        self.cb_subject.addItems(subjects)
        self.cb_subject.blockSignals(False)
        self._persist_settings()

    # ── create linked recording ──────────────────────────────────────────

    def _safe_name(self, text: str) -> str:
        bad = '<>:"/\\|?*'
        return "".join(ch for ch in str(text or "").strip() if ch not in bad).strip()

    def _create_linked(self):
        srv = self.ed_server_root.text().strip()
        proj = self._safe_name(self.cb_project.currentText())
        exp = self._safe_name(self.cb_experiment.currentText())
        sub = self._safe_name(self.cb_subject.currentText())
        sess = self._safe_name(self.ed_session.text())
        if not srv:
            QMessageBox.warning(self, "Server root", "Set a server root first.")
            return
        if not proj:
            QMessageBox.warning(self, "Project", "Select or type a project name.")
            return
        if not exp:
            QMessageBox.warning(self, "Experiment", "Select or type an experiment name.")
            return
        if not sub:
            QMessageBox.warning(self, "Subject", "Enter a subject / animal ID.")
            return
        if not sess:
            QMessageBox.warning(self, "Session", "Enter a session ID.")
            return

        data_root = self.app_state.settings.data_root
        entry = create_linked_recording(data_root, srv, proj, exp, sub, sess)
        self.logger.log(
            f"Created linked recording: {proj}/{exp}/{sub}/{sess}\n"
            f"  Local: {entry['local_path']}"
        )
        self._refresh_manifest_table()
        self._persist_settings()
        # auto-clear session field for next quick entry
        self.ed_session.clear()

    # ── manifest table ───────────────────────────────────────────────────

    def _refresh_manifest_table(self):
        data_root = self.app_state.settings.data_root
        entries = load_manifest(data_root)
        self.tbl.setRowCount(0)
        for entry in entries:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            status = entry.get("status", "pending")
            status_item = QTableWidgetItem(status.upper())
            color = _STATUS_COLORS.get(status, QColor("#888"))
            status_item.setForeground(color)
            status_item.setFont(QFont("", -1, QFont.Bold))
            # stash entry id for later retrieval
            status_item.setData(Qt.UserRole, entry.get("id", ""))
            self.tbl.setItem(r, 0, status_item)
            self.tbl.setItem(r, 1, QTableWidgetItem(entry.get("project", "")))
            self.tbl.setItem(r, 2, QTableWidgetItem(entry.get("experiment", "")))
            self.tbl.setItem(r, 3, QTableWidgetItem(entry.get("subject", "")))
            self.tbl.setItem(r, 4, QTableWidgetItem(entry.get("session", "")))
            self.tbl.setItem(r, 5, QTableWidgetItem(entry.get("created_at", "")))
            self.tbl.setItem(r, 6, QTableWidgetItem(entry.get("synced_at", "")))
            self.tbl.setItem(r, 7, QTableWidgetItem(entry.get("local_path", "")))
        self.tbl.resizeColumnsToContents()

    def _selected_entry_ids(self) -> List[str]:
        ids = []
        for idx in self.tbl.selectionModel().selectedRows():
            item = self.tbl.item(idx.row(), 0)
            if item:
                eid = item.data(Qt.UserRole)
                if eid:
                    ids.append(str(eid))
        return ids

    # ── sync actions ─────────────────────────────────────────────────────

    def _sync_selected(self):
        ids = self._selected_entry_ids()
        if not ids:
            QMessageBox.information(self, "Selection", "Select one or more rows to sync.")
            return
        self._do_sync(ids)

    def _sync_all_pending(self):
        self._do_sync(None)

    def _do_sync(self, entry_ids):
        if self._syncing:
            QMessageBox.information(self, "Busy", "A sync is already in progress.")
            return
        self._syncing = True
        data_root = self.app_state.settings.data_root

        def work():
            try:
                sync_pending(data_root, self.logger.log, entry_ids=entry_ids)
            finally:
                self._syncing = False
                # schedule table refresh on the main thread
                QTimer.singleShot(0, self._refresh_manifest_table)

        run_in_thread(work)

    # ── other actions ────────────────────────────────────────────────────

    def _open_selected_local(self):
        ids = self._selected_entry_ids()
        if not ids:
            QMessageBox.information(self, "Selection", "Select a row first.")
            return
        entries = load_manifest(self.app_state.settings.data_root)
        for entry in entries:
            if entry.get("id") in ids:
                local = entry.get("local_path", "")
                if local and os.path.isdir(local):
                    os.startfile(local) if os.name == "nt" else os.system(f'xdg-open "{local}"')  # noqa: S605
                    return
        QMessageBox.warning(self, "Folder", "Local folder not found.")

    def _open_staging_folder(self):
        staging = get_staging_dir(self.app_state.settings.data_root)
        os.makedirs(staging, exist_ok=True)
        if os.name == "nt":
            os.startfile(staging)
        else:
            os.system(f'xdg-open "{staging}"')  # noqa: S605

    def _remove_selected(self):
        ids = self._selected_entry_ids()
        if not ids:
            QMessageBox.information(self, "Selection", "Select a row first.")
            return
        ans = QMessageBox.question(
            self, "Remove",
            f"Remove {len(ids)} entry/entries from the manifest?\n"
            "(Local files are NOT deleted.)",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        data_root = self.app_state.settings.data_root
        for eid in ids:
            remove_entry(data_root, eid)
        self._refresh_manifest_table()
        self.logger.log(f"Removed {len(ids)} manifest entry/entries.")

    def _mark_pending(self):
        ids = self._selected_entry_ids()
        if not ids:
            QMessageBox.information(self, "Selection", "Select a row first.")
            return
        data_root = self.app_state.settings.data_root
        manifest = load_manifest(data_root)
        changed = 0
        for entry in manifest:
            if entry.get("id") in ids and entry.get("status") != "pending":
                entry["status"] = "pending"
                entry["synced_at"] = ""
                entry["error"] = ""
                changed += 1
        save_manifest(data_root, manifest)
        self._refresh_manifest_table()
        self.logger.log(f"Marked {changed} entry/entries as pending.")

    # ── settings persistence ─────────────────────────────────────────────

    def _load_settings(self):
        data = self.app_state.settings.get_staging_tab_settings()
        srv = data.get("server_root", "")
        if srv:
            self.ed_server_root.setText(srv)
            # auto-scan on startup if the server root is accessible
            if os.path.isdir(srv):
                QTimer.singleShot(500, self._scan_server)

    def _persist_settings(self):
        self.app_state.settings.put_staging_tab_settings({
            "server_root": self.ed_server_root.text().strip(),
            "last_project": self.cb_project.currentText().strip(),
            "last_experiment": self.cb_experiment.currentText().strip(),
        })
