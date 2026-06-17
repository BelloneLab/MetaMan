"""Transfer tab – everything about moving data off the acquisition machine:

* **Backup**   – copy the active project to a server and/or external HDD.
* **Staging**  – link locally-created recordings to a server project and sync.
* **Schedule** – daily auto-backup configuration.

All panels operate on the *active project* (the one chosen in the project bar).
Backup/schedule execution reuses the proven pipeline in ``MainWindow``.
"""

import os
from typing import List

from PySide6.QtCore import Qt, QTime
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
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
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ..services import fs_ops
from ..services.backup_report import format_report_text, human_size, report_dir
from ..side_panel import SidePanelLayout
from ..state import AppState
from .staging_tab import StagingTab

_STATUS_COLORS = {
    "success": QColor("#1f9d57"),
    "partial": QColor("#e8a83e"),
    "error": QColor("#ed4245"),
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


class TransferTab(QWidget):
    def __init__(self, app_state: AppState, main_window):
        super().__init__()
        self.app_state = app_state
        self.main = main_window
        self.staging = StagingTab(app_state)
        self._build_ui()
        self.refresh()

    # ── build ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._side = SidePanelLayout()
        root.addWidget(self._side, 1)
        self._side.add_panel("☁️", "Backup", self._build_backup_panel(), default=True)
        self._side.add_panel("\U0001f517", "Staging", self.staging)
        self._side.add_panel("⏰", "Schedule", self._build_schedule_panel())
        self._history_idx = self._side.add_panel("\U0001f9fe", "History", self._build_history_panel())
        self._side.panel_changed.connect(self._on_panel_changed)

    def _on_panel_changed(self, _index: int):
        self.refresh()
        self.refresh_history()

    def show_history_panel(self):
        self._side.switch_to(self._history_idx)
        self.refresh_history()

    def _active_project(self) -> str:
        return (self.app_state.current_project
                or self.app_state.settings.get_loaded_project().get("name", "")).strip()

    # ── Backup panel ──────────────────────────────────────────────────
    def _build_backup_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("Back up the active project")
        form = QFormLayout(grp)
        self.lbl_bk_project = QLabel("—")
        self.lbl_bk_project.setStyleSheet("font-weight: 600;")
        form.addRow("Project:", self.lbl_bk_project)
        self.lbl_bk_source = QLabel("—")
        self.lbl_bk_source.setWordWrap(True)
        form.addRow("Local source:", self.lbl_bk_source)

        self.cb_bk_mode = QComboBox()
        self.cb_bk_mode.addItems(["Server", "External HDD", "Both"])
        self.cb_bk_mode.currentIndexChanged.connect(self._update_bk_fields)
        form.addRow("Destination:", self.cb_bk_mode)

        self.ed_bk_server, self.row_bk_server = self._path_row(self._pick_bk_server)
        form.addRow("Server root:", self.row_bk_server)
        self.ed_bk_hdd, self.row_bk_hdd = self._path_row(self._pick_bk_hdd)
        form.addRow("External HDD root:", self.row_bk_hdd)

        self.chk_bk_exp = QCheckBox("Back up only the current experiment")
        form.addRow("", self.chk_bk_exp)

        self.btn_bk_run = QPushButton("Back up now")
        self.btn_bk_run.setObjectName("Primary")
        self.btn_bk_run.clicked.connect(self._run_backup)
        form.addRow("", self.btn_bk_run)
        lay.addWidget(grp)

        lay.addWidget(QLabel("Backup log"))
        self.txt_bk_log = QTextEdit()
        self.txt_bk_log.setReadOnly(True)
        self.txt_bk_log.setPlaceholderText("Backup progress appears here…")
        lay.addWidget(self.txt_bk_log, 1)
        return w

    def _path_row(self, on_browse):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        ed = QLineEdit()
        h.addWidget(ed, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(on_browse)
        h.addWidget(b)
        return ed, row

    def _pick_bk_server(self):
        d = QFileDialog.getExistingDirectory(self, "Choose server root", self.ed_bk_server.text().strip() or "")
        if d:
            self.ed_bk_server.setText(d)

    def _pick_bk_hdd(self):
        d = QFileDialog.getExistingDirectory(self, "Choose external HDD root", self.ed_bk_hdd.text().strip() or "")
        if d:
            self.ed_bk_hdd.setText(d)

    def _bk_mode(self) -> str:
        t = self.cb_bk_mode.currentText().strip().lower()
        if t.startswith("external"):
            return "hdd"
        if t == "both":
            return "both"
        return "server"

    def _update_bk_fields(self):
        mode = self._bk_mode()
        self.row_bk_server.setEnabled(mode in ("server", "both"))
        self.row_bk_hdd.setEnabled(mode in ("hdd", "both"))

    def _append_bk_log(self, msg: str):
        self.txt_bk_log.append(msg)

    def _run_backup(self):
        project = self._active_project()
        if not project:
            QMessageBox.warning(self, "Backup", "Select a project first.")
            return
        source = self.main._resolve_project_dir(project)
        if not os.path.isdir(source):
            QMessageBox.critical(self, "Backup", f"Local source not found:\n{source}")
            return
        mode = self._bk_mode()
        dests = []
        server_dir = self.ed_bk_server.text().strip()
        hdd_dir = self.ed_bk_hdd.text().strip()
        if mode in ("server", "both"):
            if not server_dir or not os.path.isdir(server_dir):
                QMessageBox.warning(self, "Server root", "Choose an existing server root.")
                return
            dests.append(("server", server_dir))
            self.app_state.settings.put_server_root_for_project(project, server_dir)
        if mode in ("hdd", "both"):
            if not hdd_dir or not os.path.isdir(hdd_dir):
                QMessageBox.warning(self, "External HDD root", "Choose an existing external HDD root.")
                return
            dests.append(("hdd", hdd_dir))
            self.app_state.settings.put_hdd_root_for_project(project, hdd_dir)

        experiment = ""
        if self.chk_bk_exp.isChecked() and self.app_state.current_experiment:
            experiment = self.app_state.current_experiment

        self.txt_bk_log.clear()
        for kind, dest_root in dests:
            self.main._start_backup_copy(project, experiment=experiment,
                                         destination_root=dest_root, destination_kind=kind,
                                         scheduled=False, log=self._append_bk_log)

    # ── Schedule panel ────────────────────────────────────────────────
    def _build_schedule_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Scheduled auto-backup")
        form = QFormLayout(grp)

        self.lbl_sc_project = QLabel("—")
        self.lbl_sc_project.setStyleSheet("font-weight:600;")
        form.addRow("Project:", self.lbl_sc_project)

        self.chk_sc_enabled = QCheckBox("Enable daily auto-backup")
        form.addRow("", self.chk_sc_enabled)

        self.te_sc_time = QTimeEdit()
        self.te_sc_time.setDisplayFormat("HH:mm")
        self.te_sc_time.setTime(QTime(2, 0))
        form.addRow("Time:", self.te_sc_time)

        self.cb_sc_mode = QComboBox()
        self.cb_sc_mode.addItems(["Server", "External HDD", "Both"])
        form.addRow("Destination:", self.cb_sc_mode)

        self.ed_sc_server, self.row_sc_server = self._path_row(self._pick_sc_server)
        form.addRow("Server root:", self.row_sc_server)
        self.ed_sc_hdd, self.row_sc_hdd = self._path_row(self._pick_sc_hdd)
        form.addRow("External HDD root:", self.row_sc_hdd)

        self.chk_sc_whole = QCheckBox("Back up the whole project (ignore experiment list)")
        self.chk_sc_whole.setChecked(True)
        self.chk_sc_whole.toggled.connect(lambda c: self.tbl_sc_exps.setEnabled(not c))
        form.addRow("", self.chk_sc_whole)
        lay.addWidget(grp)

        lay.addWidget(QLabel("Experiments to back up"))
        self.tbl_sc_exps = QTableWidget(0, 2)
        self.tbl_sc_exps.setHorizontalHeaderLabels(["Backup", "Experiment"])
        self.tbl_sc_exps.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.tbl_sc_exps, 1)

        self.lbl_sc_status = QLabel("")
        self.lbl_sc_status.setWordWrap(True)
        lay.addWidget(self.lbl_sc_status)

        b_save = QPushButton("Save schedule")
        b_save.clicked.connect(self._save_schedule)
        lay.addWidget(b_save)
        return w

    def _pick_sc_server(self):
        d = QFileDialog.getExistingDirectory(self, "Choose server root", self.ed_sc_server.text().strip() or "")
        if d:
            self.ed_sc_server.setText(d)

    def _pick_sc_hdd(self):
        d = QFileDialog.getExistingDirectory(self, "Choose external HDD root", self.ed_sc_hdd.text().strip() or "")
        if d:
            self.ed_sc_hdd.setText(d)

    def _sc_mode(self) -> str:
        t = self.cb_sc_mode.currentText().strip().lower()
        if t.startswith("external"):
            return "hdd"
        if t == "both":
            return "both"
        return "server"

    def _experiment_names(self, project: str) -> List[str]:
        try:
            return sorted({os.path.basename(d) for d in self.main.nav_tab._iter_level_dirs(project, "experiment")})
        except Exception:
            return []

    def _save_schedule(self):
        project = self._active_project()
        if not project:
            QMessageBox.warning(self, "Schedule", "Select a project first.")
            return
        enabled = self.chk_sc_enabled.isChecked()
        mode = self._sc_mode()
        server_dir = self.ed_sc_server.text().strip()
        hdd_dir = self.ed_sc_hdd.text().strip()
        if enabled and mode in ("server", "both") and (not server_dir or not os.path.isdir(server_dir)):
            QMessageBox.warning(self, "Server root", "Choose an existing server root.")
            return
        if enabled and mode in ("hdd", "both") and (not hdd_dir or not os.path.isdir(hdd_dir)):
            QMessageBox.warning(self, "External HDD root", "Choose an existing external HDD root.")
            return
        if server_dir:
            self.app_state.settings.put_server_root_for_project(project, server_dir)
        if hdd_dir:
            self.app_state.settings.put_hdd_root_for_project(project, hdd_dir)

        whole = self.chk_sc_whole.isChecked()
        selected = []
        for r in range(self.tbl_sc_exps.rowCount()):
            chk = self.tbl_sc_exps.item(r, 0)
            name = self.tbl_sc_exps.item(r, 1)
            if chk and name and chk.checkState() == Qt.Checked:
                selected.append(name.text().strip())
        if enabled and not whole and not selected:
            QMessageBox.warning(self, "Experiments", "Select at least one experiment or enable whole-project backup.")
            return

        self.app_state.settings.put_backup_schedule_for_project(
            project, enabled, self.te_sc_time.time().toString("HH:mm"),
            backup_whole_project=whole, enabled_experiments=selected, destination_mode=mode,
        )
        self.refresh()
        self.lbl_sc_status.setText(
            f"Auto-backup {'enabled' if enabled else 'disabled'} for '{project}'."
        )
        try:
            self.main._run_due_scheduled_backups()
        except Exception:
            pass

    # ── refresh from active project ───────────────────────────────────
    def refresh(self):
        project = self._active_project()
        server = self.app_state.settings.get_server_root_for_project(project)
        hdd = self.app_state.settings.get_hdd_root_for_project(project)

        # Backup panel
        self.lbl_bk_project.setText(project or "—")
        self.lbl_bk_source.setText(self.main._resolve_project_dir(project) if project else "—")
        if not self.ed_bk_server.text().strip():
            self.ed_bk_server.setText(server)
        if not self.ed_bk_hdd.text().strip():
            self.ed_bk_hdd.setText(hdd)
        cur_exp = self.app_state.current_experiment
        self.chk_bk_exp.setEnabled(bool(cur_exp))
        self.chk_bk_exp.setText(
            f"Back up only the current experiment ({cur_exp})" if cur_exp
            else "Back up only the current experiment"
        )
        self._update_bk_fields()

        # Schedule panel
        self.lbl_sc_project.setText(project or "—")
        sched = self.app_state.settings.get_backup_schedule_for_project(project) if project else {}
        self.chk_sc_enabled.setChecked(bool(sched.get("enabled", False)))
        hhmm = str(sched.get("time", "")).strip() or "02:00"
        qt = QTime.fromString(hhmm, "HH:mm")
        self.te_sc_time.setTime(qt if qt.isValid() else QTime(2, 0))
        mode = str(sched.get("destination_mode", "server") or "server").lower()
        self.cb_sc_mode.setCurrentText({"hdd": "External HDD", "both": "Both"}.get(mode, "Server"))
        self.ed_sc_server.setText(server)
        self.ed_sc_hdd.setText(hdd)
        whole = bool(sched.get("backup_whole_project", True))
        self.chk_sc_whole.setChecked(whole)
        self.tbl_sc_exps.setEnabled(not whole)
        enabled_exps = set(sched.get("enabled_experiments", []) or [])
        self.tbl_sc_exps.setRowCount(0)
        for name in self._experiment_names(project):
            r = self.tbl_sc_exps.rowCount()
            self.tbl_sc_exps.insertRow(r)
            chk = QTableWidgetItem("")
            chk.setFlags(chk.flags() | Qt.ItemIsUserCheckable)
            chk.setCheckState(Qt.Checked if name in enabled_exps else Qt.Unchecked)
            self.tbl_sc_exps.setItem(r, 0, chk)
            self.tbl_sc_exps.setItem(r, 1, QTableWidgetItem(name))
        if sched.get("enabled"):
            self.lbl_sc_status.setText(f"Enabled at {hhmm}.  Last run: {sched.get('last_run_date', 'never')}")
        else:
            self.lbl_sc_status.setText("Auto-backup is disabled.")

    # ── History panel ──────────────────────────────────────────────────
    def _build_history_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("Last backup")
        gl = QVBoxLayout(grp)
        self.lbl_hist_summary = QLabel("No backups recorded yet.")
        self.lbl_hist_summary.setWordWrap(True)
        self.lbl_hist_summary.setObjectName("BackupSummary")
        gl.addWidget(self.lbl_hist_summary)
        lay.addWidget(grp)

        ctl = QHBoxLayout()
        self.chk_hist_all = QCheckBox("Show all projects")
        self.chk_hist_all.toggled.connect(lambda _c: self.refresh_history())
        ctl.addWidget(self.chk_hist_all)
        ctl.addStretch(1)
        b_refresh = QPushButton("Refresh")
        b_refresh.clicked.connect(self.refresh_history)
        ctl.addWidget(b_refresh)
        lay.addLayout(ctl)

        split = QSplitter(Qt.Vertical)
        self.tbl_hist = QTableWidget(0, 8)
        self.tbl_hist.setHorizontalHeaderLabels([
            "Finished", "Project", "Scope", "Destination", "Status", "Files", "Copied", "Duration",
        ])
        self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl_hist.horizontalHeader().setStretchLastSection(True)
        self.tbl_hist.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_hist.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_hist.setAlternatingRowColors(True)
        self.tbl_hist.itemSelectionChanged.connect(self._on_hist_row_selected)
        split.addWidget(self.tbl_hist)

        detail = QWidget()
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(0, 0, 0, 0)
        self.txt_hist_detail = QTextEdit()
        self.txt_hist_detail.setReadOnly(True)
        mono = QFont("Consolas" if os.name == "nt" else "Monospace")
        mono.setStyleHint(QFont.Monospace)
        self.txt_hist_detail.setFont(mono)
        self.txt_hist_detail.setPlaceholderText("Select a run to see its full report…")
        dl.addWidget(self.txt_hist_detail, 1)
        drow = QHBoxLayout()
        b_report = QPushButton("Open report folder")
        b_report.clicked.connect(self._open_selected_report)
        drow.addWidget(b_report)
        b_dest = QPushButton("Open destination")
        b_dest.clicked.connect(self._open_selected_destination)
        drow.addWidget(b_dest)
        drow.addStretch(1)
        dl.addLayout(drow)
        split.addWidget(detail)
        split.setSizes([260, 220])
        lay.addWidget(split, 1)
        return w

    def refresh_history(self):
        if not hasattr(self, "tbl_hist"):
            return
        project = self._active_project()
        show_all = self.chk_hist_all.isChecked()
        runs = self.app_state.settings.get_backup_history(
            project=None if show_all else (project or None)
        )
        self._render_summary(project)
        self.tbl_hist.setRowCount(0)
        for rec in runs:
            r = self.tbl_hist.rowCount()
            self.tbl_hist.insertRow(r)
            status = str(rec.get("status", ""))
            it0 = QTableWidgetItem(rec.get("finished_at", "") or rec.get("started_at", ""))
            it0.setData(Qt.UserRole, rec)
            self.tbl_hist.setItem(r, 0, it0)
            self.tbl_hist.setItem(r, 1, QTableWidgetItem(rec.get("project", "")))
            self.tbl_hist.setItem(r, 2, QTableWidgetItem(rec.get("experiment", "") or "(whole project)"))
            self.tbl_hist.setItem(
                r, 3, QTableWidgetItem("External HDD" if rec.get("destination_kind") == "hdd" else "Server"))
            st = QTableWidgetItem(status.upper())
            st.setForeground(_STATUS_COLORS.get(status, QColor("#888")))
            st.setFont(QFont("", -1, QFont.Bold))
            self.tbl_hist.setItem(r, 4, st)
            self.tbl_hist.setItem(r, 5, QTableWidgetItem(str(rec.get("files_total", 0))))
            copied = int(rec.get("copied", 0)) + int(rec.get("updated", 0))
            self.tbl_hist.setItem(r, 6, QTableWidgetItem(f"{copied} · {human_size(rec.get('bytes_copied', 0))}"))
            self.tbl_hist.setItem(r, 7, QTableWidgetItem(_fmt_duration(rec.get("duration_s", 0))))
        self.tbl_hist.resizeColumnsToContents()

    def _render_summary(self, project: str):
        if not project:
            self.lbl_hist_summary.setText("Select a project to see its last backup.")
            return
        last = self.app_state.settings.get_last_backup_for(project)
        if not last:
            self.lbl_hist_summary.setText(f"No backups recorded for '{project}' yet.")
            return
        dest = "External HDD" if last.get("destination_kind") == "hdd" else "Server"
        scope = last.get("experiment", "") or "whole project"
        copied = int(last.get("copied", 0)) + int(last.get("updated", 0))
        self.lbl_hist_summary.setText(
            f"<b>{project}</b> → {dest} ({scope})<br>"
            f"{last.get('finished_at', '')} · <b>{last.get('status', '').upper()}</b><br>"
            f"{copied} copied · {last.get('skipped', 0)} unchanged · {last.get('failed', 0)} failed · "
            f"{human_size(last.get('bytes_copied', 0))} in {_fmt_duration(last.get('duration_s', 0))}"
        )

    def _selected_run(self):
        rows = self.tbl_hist.selectionModel().selectedRows()
        row = rows[0].row() if rows else self.tbl_hist.currentRow()
        if row < 0:
            return None
        it = self.tbl_hist.item(row, 0)
        return it.data(Qt.UserRole) if it else None

    def _on_hist_row_selected(self):
        rec = self._selected_run()
        if rec:
            self.txt_hist_detail.setPlainText(format_report_text(rec))

    def _open_selected_report(self):
        rec = self._selected_run()
        if not rec:
            QMessageBox.information(self, "Report", "Select a run first.")
            return
        out_dir = report_dir(rec.get("destination_root", ""), rec.get("project", ""), rec.get("experiment", ""))
        if not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Report", f"Report folder not found (destination unmounted?):\n{out_dir}")
            return
        try:
            fs_ops.open_path(out_dir)
        except Exception as e:
            QMessageBox.critical(self, "Report", str(e))

    def _open_selected_destination(self):
        rec = self._selected_run()
        if not rec:
            QMessageBox.information(self, "Destination", "Select a run first.")
            return
        dest = rec.get("destination_path", "") or rec.get("destination_root", "")
        if not dest or not os.path.exists(dest):
            QMessageBox.warning(self, "Destination", "Destination folder not found / not mounted.")
            return
        try:
            fs_ops.open_path(dest)
        except Exception as e:
            QMessageBox.critical(self, "Destination", str(e))
