import os
from datetime import datetime
from time import time

from PySide6.QtCore import QTime, QTimer, Qt
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QSplashScreen,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QTimeEdit,
    QVBoxLayout, QWidget, QInputDialog
)

from .config import APP_TITLE, WINDOW_GEOMETRY
from .state import AppState
from .tabs.navigation_tab import NavigationTab
from .tabs.recording_tab import RecordingTab
from .tabs.preprocessing_tab import PreprocessingTab
from .tabs.data_reorganizer_tab import DataReorganizerTab
from .services.server_sync import sync_project_to_server
from .services.search_service import search_in_project
from .io_ops import list_experiments, list_projects, load_session_metadata, save_session_triplet
from .utils import run_in_thread


def _resolve_logo_path() -> str:
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "assests", "metaman.png"),
        os.path.join(here, "assets", "metaman.png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _set_windows_app_user_model_id():
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MetaMan.App")
    except Exception:
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(*WINDOW_GEOMETRY)
        logo_path = _resolve_logo_path()
        if logo_path:
            icon = QIcon(logo_path)
            if not icon.isNull():
                self.setWindowIcon(icon)
        self.state = AppState()
        self._backup_jobs_in_progress = set()
        self._schedule_warning_cache = set()
        self._build_ui()
        self._build_menu()
        self._apply_visual_style()
        self._init_backup_scheduler()

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)

        self.tabs = QTabWidget()
        self.nav_tab = NavigationTab(self.state, on_load_session=self._load_session_everywhere)
        self.rec_tab = RecordingTab(self.state)
        self.pre_tab = PreprocessingTab(self.state)
        self.reorg_tab = DataReorganizerTab(self.state)
        self.tabs.addTab(self.nav_tab, "Navigation")
        self.tabs.addTab(self.rec_tab, "Recording")
        self.tabs.addTab(self.pre_tab, "Preprocessing")
        self.tabs.addTab(self.reorg_tab, "Data reorganizer")
        root.addWidget(self.tabs, 1)

        bar = QHBoxLayout()
        b_copy = QPushButton("Backup project...")
        b_copy.clicked.connect(self._backup_project_now)
        bar.addWidget(b_copy)

        b_sched = QPushButton("Schedule backup...")
        b_sched.clicked.connect(self._schedule_backup)
        bar.addWidget(b_sched)

        b_search = QPushButton("Search...")
        b_search.clicked.connect(self._search)
        bar.addWidget(b_search)

        b_animal = QPushButton("Generate subject summary CSV...")
        b_animal.clicked.connect(self._animal_summary)
        bar.addWidget(b_animal)

        bar.addStretch(1)
        self.lbl_status = QLabel("")
        bar.addWidget(self.lbl_status)
        root.addLayout(bar)

    def _build_menu(self):
        menu_file = self.menuBar().addMenu("&File")

        act_new_project = QAction("New Project...", self)
        act_new_project.triggered.connect(self._menu_new_project)
        menu_file.addAction(act_new_project)

        act_add_experiment = QAction("Add Experiment...", self)
        act_add_experiment.triggered.connect(self._menu_add_experiment)
        menu_file.addAction(act_add_experiment)

        menu_file.addSeparator()

        act_set_root = QAction("Set Data Root...", self)
        act_set_root.triggered.connect(self._menu_set_data_root)
        menu_file.addAction(act_set_root)

        act_refresh = QAction("Refresh All Lists", self)
        act_refresh.triggered.connect(self._refresh_everything)
        menu_file.addAction(act_refresh)

        menu_file.addSeparator()

        act_backup_now = QAction("Backup Now...", self)
        act_backup_now.triggered.connect(self._backup_project_now)
        menu_file.addAction(act_backup_now)

        act_backup_sched = QAction("Schedule Backup...", self)
        act_backup_sched.triggered.connect(self._schedule_backup)
        menu_file.addAction(act_backup_sched)

        menu_file.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

    def _apply_visual_style(self):
        self.setStyleSheet(
            """
            QWidget {
                font-size: 11px;
            }
            QTabWidget::pane {
                border: 1px solid #c9ced6;
                border-radius: 6px;
                background: #fbfcfe;
            }
            QTabBar::tab {
                background: #ebeff5;
                border: 1px solid #c9ced6;
                border-bottom: none;
                padding: 7px 12px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #162133;
                font-weight: 600;
            }
            QGroupBox {
                border: 1px solid #d2d8e1;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #1b2b47;
                font-weight: 600;
            }
            QPushButton {
                background: #f2f5fa;
                border: 1px solid #b8c3d3;
                border-radius: 6px;
                padding: 5px 10px;
                min-height: 22px;
            }
            QPushButton:hover {
                background: #e6edf8;
            }
            QPushButton:pressed {
                background: #dce6f5;
            }
            QLineEdit, QComboBox, QTimeEdit, QTextEdit, QTableWidget {
                border: 1px solid #bcc7d6;
                border-radius: 5px;
                background: #ffffff;
            }
            QHeaderView::section {
                background: #edf2f8;
                border: 1px solid #d0d8e4;
                padding: 4px;
                font-weight: 600;
            }
            """
        )

    def _init_backup_scheduler(self):
        self._backup_timer = QTimer(self)
        self._backup_timer.setInterval(30_000)
        self._backup_timer.timeout.connect(self._run_due_scheduled_backups)
        self._backup_timer.start()
        QTimer.singleShot(2_000, self._run_due_scheduled_backups)

    def _load_session_everywhere(self, session_path: str):
        self.rec_tab.load_session(session_path)
        self.pre_tab._load_from_session(session_path)

    def _project_dir(self, project: str) -> str:
        return os.path.join(self.state.settings.raw_root, project)

    def _experiment_dir(self, project: str, experiment: str) -> str:
        return os.path.join(self.state.settings.raw_root, project, experiment)

    def _backup_job_key(self, project: str, experiment: str = "", destination_kind: str = "server") -> str:
        return f"{project}::{experiment or '*'}::{destination_kind}"

    def _annotate_current_session_backup_paths(self, project: str, destination_root: str, destination_kind: str, log):
        sess_path = self.state.current_session_path
        if not sess_path:
            return

        meta = load_session_metadata(sess_path) or {}
        if str(meta.get("Project", "")) != project:
            return

        experiment = meta.get("Experiment", "") or self.state.current_experiment
        if not experiment:
            try:
                sub_dir = os.path.dirname(sess_path)
                exp_dir = os.path.dirname(sub_dir)
                experiment = os.path.basename(exp_dir)
            except Exception:
                experiment = ""
        subject = meta.get("Subject", "") or meta.get("Animal", "")
        session = meta.get("Session", "")
        destination_project = os.path.join(destination_root, project)
        session_dir = os.path.join(self.state.settings.raw_root, project, experiment, subject, session)
        dest_key = "server_path" if destination_kind == "server" else "hdd_path"

        for item in meta.get("file_list", []):
            src_path = item.get("path", "")
            if not src_path:
                continue
            dpath = src_path.replace(session_dir, os.path.join(destination_project, experiment, subject, session))
            item[dest_key] = dpath if os.path.exists(dpath) else ""

        save_session_triplet(sess_path, meta, logger=log)

    def _start_backup_copy(self, project: str, experiment: str, destination_root: str, destination_kind: str, scheduled: bool):
        source_dir = self._experiment_dir(project, experiment) if experiment else self._project_dir(project)
        if not os.path.isdir(source_dir):
            self.rec_tab.logger.log(f"[backup] Source folder not found: {source_dir}")
            return
        destination_scope_root = os.path.join(destination_root, project) if experiment else destination_root

        job_key = self._backup_job_key(project, experiment, destination_kind=destination_kind)
        if job_key in self._backup_jobs_in_progress:
            scope = f"{project}/{experiment}" if experiment else project
            self.rec_tab.logger.log(f"[backup] Backup already in progress for scope: {scope} ({destination_kind})")
            return

        def log(s: str):
            self.rec_tab.logger.log(s)

        scope = f"{project}/{experiment}" if experiment else project
        mode = "scheduled backup" if scheduled else "manual backup"
        destination_label = "external_hdd" if destination_kind == "hdd" else "server"
        started = time()
        self._backup_jobs_in_progress.add(job_key)
        self.lbl_status.setText(f"{mode.capitalize()} running: {scope} -> {destination_label}")

        def work():
            ok = False
            try:
                log(f"[{mode}/{destination_label}] Starting: {scope} -> {destination_scope_root}")
                sync_project_to_server(source_dir, destination_scope_root, log)
                dt = max(time() - started, 1e-6)
                log(f"[{mode}/{destination_label}] Finished in {dt:.1f}s.")
                if not experiment:
                    self._annotate_current_session_backup_paths(project, destination_root, destination_kind, log)
                ok = True
            except Exception as e:
                log(f"[error] {mode}/{destination_label} failed for '{scope}': {e}")
            finally:
                self._backup_jobs_in_progress.discard(job_key)
                if scheduled and ok:
                    self.state.settings.mark_backup_schedule_run(project, datetime.now().strftime("%Y-%m-%d"), experiment=experiment)

        run_in_thread(work)

    def _copy_to_server(self):
        self._backup_project_now()

    def _backup_project_now(self):
        proj = self.state.current_project
        if not proj:
            QMessageBox.warning(self, "No project", "Select or load a session first.")
            return

        project_dir = self._project_dir(proj)
        if not os.path.isdir(project_dir):
            QMessageBox.critical(self, "Missing", f"Project folder not found:\n{project_dir}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Backup Project")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()

        cb_destination = QComboBox()
        cb_destination.addItems(["Server", "External HDD", "Both"])

        row_server = QHBoxLayout()
        ed_server = QLineEdit(self.state.settings.get_server_root_for_project(proj))
        b_server = QPushButton("Browse...")
        row_server.addWidget(ed_server, 1)
        row_server.addWidget(b_server)
        row_server_w = QWidget()
        row_server_w.setLayout(row_server)

        row_hdd = QHBoxLayout()
        ed_hdd = QLineEdit(self.state.settings.get_hdd_root_for_project(proj))
        b_hdd = QPushButton("Browse...")
        row_hdd.addWidget(ed_hdd, 1)
        row_hdd.addWidget(b_hdd)
        row_hdd_w = QWidget()
        row_hdd_w.setLayout(row_hdd)

        chk_current_experiment = QCheckBox("Backup only current experiment")
        if self.state.current_experiment:
            chk_current_experiment.setText(f"Backup only current experiment ({self.state.current_experiment})")

        form.addRow("Project:", QLabel(proj))
        form.addRow("Destination:", cb_destination)
        form.addRow("Server root:", row_server_w)
        form.addRow("External HDD root:", row_hdd_w)
        form.addRow("", chk_current_experiment)
        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        lay.addWidget(btns)

        def pick_server():
            d = QFileDialog.getExistingDirectory(dlg, "Choose server root", ed_server.text().strip() or "")
            if d:
                ed_server.setText(d)

        def pick_hdd():
            d = QFileDialog.getExistingDirectory(dlg, "Choose external HDD root", ed_hdd.text().strip() or "")
            if d:
                ed_hdd.setText(d)

        def destination_mode() -> str:
            txt = cb_destination.currentText().strip().lower()
            if txt.startswith("external"):
                return "hdd"
            if txt == "both":
                return "both"
            return "server"

        def update_destination_fields():
            mode = destination_mode()
            row_server_w.setEnabled(mode in ("server", "both"))
            row_hdd_w.setEnabled(mode in ("hdd", "both"))

        def start_manual_backup():
            mode = destination_mode()
            destinations = []
            server_dir = ed_server.text().strip()
            hdd_dir = ed_hdd.text().strip()
            if mode in ("server", "both"):
                if not server_dir or not os.path.isdir(server_dir):
                    QMessageBox.warning(dlg, "Server root", "Choose an existing server root.")
                    return
                destinations.append(("server", server_dir))
                self.state.settings.put_server_root_for_project(proj, server_dir)
            if mode in ("hdd", "both"):
                if not hdd_dir or not os.path.isdir(hdd_dir):
                    QMessageBox.warning(dlg, "External HDD root", "Choose an existing external HDD root.")
                    return
                destinations.append(("hdd", hdd_dir))
                self.state.settings.put_hdd_root_for_project(proj, hdd_dir)

            experiment = ""
            if chk_current_experiment.isChecked() and self.state.current_experiment:
                experiment = self.state.current_experiment

            for kind, dest_root in destinations:
                self._start_backup_copy(
                    proj,
                    experiment=experiment,
                    destination_root=dest_root,
                    destination_kind=kind,
                    scheduled=False,
                )
            dlg.accept()

        b_server.clicked.connect(pick_server)
        b_hdd.clicked.connect(pick_hdd)
        cb_destination.currentIndexChanged.connect(update_destination_fields)
        btns.accepted.connect(start_manual_backup)
        btns.rejected.connect(dlg.reject)
        update_destination_fields()
        dlg.exec()

    def _warn_schedule_once(self, today: str, project: str, reason: str, experiment: str = ""):
        key = (today, project, experiment, reason)
        if key in self._schedule_warning_cache:
            return
        self._schedule_warning_cache.add(key)
        scope = f"{project}/{experiment}" if experiment else project
        self.rec_tab.logger.log(f"[scheduled backup] Skipped '{scope}': {reason}")

    def _run_due_scheduled_backups(self):
        schedules = self.state.settings.get_all_backup_schedules()
        if not schedules:
            return

        now = datetime.now()
        now_hhmm = now.strftime("%H:%M")
        today = now.strftime("%Y-%m-%d")
        self._schedule_warning_cache = {k for k in self._schedule_warning_cache if k[0] == today}

        for project, sched in schedules.items():
            if not sched.get("enabled", False):
                continue

            time_hhmm = str(sched.get("time", "")).strip()
            qt = QTime.fromString(time_hhmm, "HH:mm")
            if not qt.isValid():
                self._warn_schedule_once(today, project, f"invalid schedule time '{time_hhmm}'")
                continue

            if str(sched.get("last_run_date", "")) == today:
                # for project-wide schedules this avoids duplicates; per-experiment schedules are handled below
                pass
            if time_hhmm > now_hhmm:
                continue

            mode = str(sched.get("destination_mode", "server") or "server").strip().lower()
            destinations = []
            if mode in ("server", "both"):
                server_dir = self.state.settings.get_server_root_for_project(project)
                if server_dir and os.path.isdir(server_dir):
                    destinations.append(("server", server_dir))
                else:
                    self._warn_schedule_once(today, project, "server root is not set or no longer exists")
            if mode in ("hdd", "both", "external_hdd", "external hdd"):
                hdd_dir = self.state.settings.get_hdd_root_for_project(project)
                if hdd_dir and os.path.isdir(hdd_dir):
                    destinations.append(("hdd", hdd_dir))
                else:
                    self._warn_schedule_once(today, project, "external HDD root is not set or no longer exists")

            if not destinations:
                continue

            project_dir = self._project_dir(project)
            if not os.path.isdir(project_dir):
                self._warn_schedule_once(today, project, f"project folder not found: {project_dir}")
                continue

            backup_whole = bool(sched.get("backup_whole_project", True))
            enabled_exps = [str(x) for x in (sched.get("enabled_experiments", []) or []) if str(x).strip()]
            run_dates = sched.get("run_dates_by_experiment", {}) or {}
            if not isinstance(run_dates, dict):
                run_dates = {}

            if backup_whole or not enabled_exps:
                if str(sched.get("last_run_date", "")) == today:
                    continue
                for kind, destination_root in destinations:
                    self._start_backup_copy(
                        project,
                        experiment="",
                        destination_root=destination_root,
                        destination_kind=kind,
                        scheduled=True,
                    )
                continue

            for exp in enabled_exps:
                exp_dir = self._experiment_dir(project, exp)
                if not os.path.isdir(exp_dir):
                    self._warn_schedule_once(today, project, f"experiment folder not found: {exp_dir}", experiment=exp)
                    continue
                if str(run_dates.get(exp, "")) == today:
                    continue
                for kind, destination_root in destinations:
                    self._start_backup_copy(
                        project,
                        experiment=exp,
                        destination_root=destination_root,
                        destination_kind=kind,
                        scheduled=True,
                    )

    def _schedule_server_backup(self):
        self._schedule_backup()

    def _schedule_backup(self):
        projects = list_projects(self.state.settings.raw_root)
        if not projects:
            QMessageBox.warning(self, "No projects", "No projects found in the current root directory.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Schedule Backup (Server / External HDD)")
        lay = QVBoxLayout(dlg)

        form = QFormLayout()

        cb_project = QComboBox()
        cb_project.addItems(projects)

        chk_enabled = QCheckBox("Enable daily backup")
        chk_whole_project = QCheckBox("Backup whole project (ignore experiment list)")
        chk_whole_project.setChecked(True)

        tm = QTimeEdit()
        tm.setDisplayFormat("HH:mm")
        tm.setTime(QTime(2, 0))

        cb_destination = QComboBox()
        cb_destination.addItems(["Server", "External HDD", "Both"])

        row_server = QHBoxLayout()
        ed_server = QLineEdit()
        b_server = QPushButton("Browse...")
        row_server.addWidget(ed_server, 1)
        row_server.addWidget(b_server)

        row_server_w = QWidget()
        row_server_w.setLayout(row_server)

        row_hdd = QHBoxLayout()
        ed_hdd = QLineEdit()
        b_hdd = QPushButton("Browse...")
        row_hdd.addWidget(ed_hdd, 1)
        row_hdd.addWidget(b_hdd)

        row_hdd_w = QWidget()
        row_hdd_w.setLayout(row_hdd)

        form.addRow("Project:", cb_project)
        form.addRow("Time:", tm)
        form.addRow("Destination:", cb_destination)
        form.addRow("Server root:", row_server_w)
        form.addRow("External HDD root:", row_hdd_w)
        form.addRow("", chk_enabled)
        form.addRow("", chk_whole_project)
        lay.addLayout(form)

        tbl_exps = QTableWidget(0, 2)
        tbl_exps.setHorizontalHeaderLabels(["Backup", "Experiment"])
        tbl_exps.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(QLabel("Experiments to backup"))
        lay.addWidget(tbl_exps)

        help_lbl = QLabel("Backups run while this app is open. One run per enabled scope (project or selected experiments) per day.")
        lay.addWidget(help_lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        lay.addWidget(btns)

        def destination_mode() -> str:
            txt = cb_destination.currentText().strip().lower()
            if txt.startswith("external"):
                return "hdd"
            if txt == "both":
                return "both"
            return "server"

        def selected_experiments_from_table():
            out = []
            for r in range(tbl_exps.rowCount()):
                check_item = tbl_exps.item(r, 0)
                exp_item = tbl_exps.item(r, 1)
                if not check_item or not exp_item:
                    continue
                if check_item.checkState() == Qt.Checked:
                    out.append(exp_item.text().strip())
            return [x for x in out if x]

        def fill_experiment_table(project: str, enabled_exps):
            tbl_exps.setRowCount(0)
            exps = list_experiments(self._project_dir(project))
            enabled_set = {str(x).strip() for x in (enabled_exps or []) if str(x).strip()}
            for exp in exps:
                r = tbl_exps.rowCount()
                tbl_exps.insertRow(r)
                chk = QTableWidgetItem("")
                chk.setFlags(chk.flags() | Qt.ItemIsUserCheckable)
                chk.setCheckState(Qt.Checked if exp in enabled_set else Qt.Unchecked)
                tbl_exps.setItem(r, 0, chk)
                tbl_exps.setItem(r, 1, QTableWidgetItem(exp))
            tbl_exps.resizeColumnsToContents()

        def update_exp_table_enabled():
            tbl_exps.setEnabled(not chk_whole_project.isChecked())

        def update_destination_rows():
            mode = destination_mode()
            row_server_w.setEnabled(mode in ("server", "both"))
            row_hdd_w.setEnabled(mode in ("hdd", "both"))

        def load_for_selected_project():
            project = cb_project.currentText().strip()
            sched = self.state.settings.get_backup_schedule_for_project(project)
            hhmm = str(sched.get("time", "")).strip() or "02:00"
            qtime = QTime.fromString(hhmm, "HH:mm")
            if not qtime.isValid():
                qtime = QTime(2, 0)
            tm.setTime(qtime)
            chk_enabled.setChecked(bool(sched.get("enabled", False)))
            chk_whole_project.setChecked(bool(sched.get("backup_whole_project", True)))
            ed_server.setText(self.state.settings.get_server_root_for_project(project))
            ed_hdd.setText(self.state.settings.get_hdd_root_for_project(project))

            mode = str(sched.get("destination_mode", "server") or "server").strip().lower()
            if mode == "hdd":
                cb_destination.setCurrentText("External HDD")
            elif mode == "both":
                cb_destination.setCurrentText("Both")
            else:
                cb_destination.setCurrentText("Server")

            fill_experiment_table(project, sched.get("enabled_experiments", []))
            update_exp_table_enabled()
            update_destination_rows()

        def pick_server_root():
            start = ed_server.text().strip() or self.state.settings.get_server_root_for_project(cb_project.currentText().strip())
            d = QFileDialog.getExistingDirectory(dlg, "Choose server root", start)
            if d:
                ed_server.setText(d)

        def pick_hdd_root():
            start = ed_hdd.text().strip() or self.state.settings.get_hdd_root_for_project(cb_project.currentText().strip())
            d = QFileDialog.getExistingDirectory(dlg, "Choose external HDD root", start)
            if d:
                ed_hdd.setText(d)

        def save_schedule():
            project = cb_project.currentText().strip()
            if not project:
                QMessageBox.warning(dlg, "Project", "Please choose a project.")
                return

            enabled = chk_enabled.isChecked()
            backup_whole_project = chk_whole_project.isChecked()
            mode = destination_mode()
            server_dir = ed_server.text().strip()
            hdd_dir = ed_hdd.text().strip()
            if enabled and mode in ("server", "both") and (not server_dir or not os.path.isdir(server_dir)):
                QMessageBox.warning(dlg, "Server root", "Choose an existing server root for enabled schedules.")
                return
            if enabled and mode in ("hdd", "both") and (not hdd_dir or not os.path.isdir(hdd_dir)):
                QMessageBox.warning(dlg, "External HDD root", "Choose an existing external HDD root for enabled schedules.")
                return
            selected_exps = selected_experiments_from_table()
            if enabled and (not backup_whole_project) and not selected_exps:
                QMessageBox.warning(dlg, "Experiments", "Select at least one experiment or enable whole-project backup.")
                return

            if server_dir:
                self.state.settings.put_server_root_for_project(project, server_dir)
            if hdd_dir:
                self.state.settings.put_hdd_root_for_project(project, hdd_dir)

            time_hhmm = tm.time().toString("HH:mm")
            self.state.settings.put_backup_schedule_for_project(
                project,
                enabled,
                time_hhmm,
                backup_whole_project=backup_whole_project,
                enabled_experiments=selected_exps,
                destination_mode=mode,
            )

            if enabled:
                scope_txt = "whole project" if backup_whole_project else f"experiments: {', '.join(selected_exps)}"
                self.rec_tab.logger.log(f"[scheduled backup] Enabled for '{project}' at {time_hhmm} ({scope_txt}, destination={mode}).")
                self.lbl_status.setText(f"Backup scheduled: {project} @ {time_hhmm} ({mode})")
            else:
                self.rec_tab.logger.log(f"[scheduled backup] Disabled for '{project}'.")
                self.lbl_status.setText(f"Backup schedule disabled: {project}")

            dlg.accept()
            self._run_due_scheduled_backups()

        b_server.clicked.connect(pick_server_root)
        b_hdd.clicked.connect(pick_hdd_root)
        cb_project.currentIndexChanged.connect(load_for_selected_project)
        chk_whole_project.toggled.connect(update_exp_table_enabled)
        cb_destination.currentIndexChanged.connect(update_destination_rows)
        btns.accepted.connect(save_schedule)
        btns.rejected.connect(dlg.reject)

        if self.state.current_project and self.state.current_project in projects:
            cb_project.setCurrentText(self.state.current_project)

        load_for_selected_project()
        dlg.exec()

    def _safe_name(self, name: str) -> str:
        bad = '<>:"/\\|?*'
        out = "".join(ch for ch in str(name or "").strip() if ch not in bad)
        return out.strip()

    def _refresh_everything(self):
        try:
            self.nav_tab.refresh_tree(collapsed=True, lazy=True)
        except Exception:
            pass
        try:
            self.rec_tab._refresh_lists()
        except Exception:
            pass
        try:
            self.pre_tab._refresh_lists()
        except Exception:
            pass
        try:
            self.reorg_tab._refresh_project_list()
        except Exception:
            pass

    def _menu_set_data_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose data root", self.state.settings.data_root or "")
        if not d:
            return
        self.state.settings.data_root = d
        self.state.settings.ensure_storage_roots()
        try:
            self.nav_tab.ed_root.setText(self.state.settings.data_root)
        except Exception:
            pass
        try:
            self.rec_tab.ed_root.setText(self.state.settings.data_root)
        except Exception:
            pass
        try:
            self.pre_tab.ed_data_root.setText(self.state.settings.data_root)
            self.pre_tab.ed_proc_root.setText(self.state.settings.processed_root)
        except Exception:
            pass
        try:
            self.reorg_tab.ed_target_raw.setText(self.state.settings.raw_root)
            self.reorg_tab.ed_target_proc.setText(self.state.settings.processed_root)
        except Exception:
            pass
        self._refresh_everything()
        self.lbl_status.setText(f"Data root set: {self.state.settings.data_root}")

    def _menu_new_project(self):
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok:
            return
        project = self._safe_name(name)
        if not project:
            QMessageBox.warning(self, "Project", "Project name cannot be empty.")
            return

        raw_project = os.path.join(self.state.settings.raw_root, project)
        proc_project = os.path.join(self.state.settings.processed_root, project)
        os.makedirs(raw_project, exist_ok=True)
        os.makedirs(proc_project, exist_ok=True)
        self.state.set_current(project=project, experiment="", animal="", session="", session_path="")
        self._refresh_everything()
        self.lbl_status.setText(f"Project ready: {project}")

    def _menu_add_experiment(self):
        projects = list_projects(self.state.settings.raw_root)
        if not projects:
            QMessageBox.warning(self, "Projects", "Create a project first.")
            return

        default_project = self.state.current_project if self.state.current_project in projects else projects[0]
        project, ok = QInputDialog.getItem(self, "Add Experiment", "Project:", projects, projects.index(default_project), False)
        if not ok:
            return

        exp_name, ok = QInputDialog.getText(self, "Add Experiment", "Experiment name:")
        if not ok:
            return
        experiment = self._safe_name(exp_name)
        if not experiment:
            QMessageBox.warning(self, "Experiment", "Experiment name cannot be empty.")
            return

        raw_exp = os.path.join(self.state.settings.raw_root, project, experiment)
        proc_exp = os.path.join(self.state.settings.processed_root, project, experiment)
        os.makedirs(raw_exp, exist_ok=True)
        os.makedirs(proc_exp, exist_ok=True)
        self.state.set_current(project=project, experiment=experiment)
        self._refresh_everything()
        self.lbl_status.setText(f"Experiment ready: {project}/{experiment}")

    def _search(self):
        proj = self.state.current_project
        if not proj:
            QMessageBox.warning(self, "No project", "Select or load a session first.")
            return

        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, "Search", "Enter text:")
        if not ok or not text:
            return

        project_dir = os.path.join(self.state.settings.raw_root, proj)
        hits = search_in_project(project_dir, text)
        if not hits:
            QMessageBox.information(self, "Search", "No matches.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Search results")
        lay = QVBoxLayout(dlg)

        txt = QTextEdit()
        txt.setReadOnly(True)
        lay.addWidget(txt)
        txt.setPlainText("\n\n".join([f"{i + 1}. {h['path']}\n  {h['key']}: {h['value']}" for i, h in enumerate(hits[:200])]))

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()

    def _animal_summary(self):
        proj = self.state.current_project
        exp = self.state.current_experiment
        subject = self.state.current_animal
        if not (proj and exp and subject):
            QMessageBox.warning(self, "Choose subject", "Select a subject in Navigation first.")
            return

        rows = []
        import json
        import pandas as pd

        subject_dir = os.path.join(self.state.settings.raw_root, proj, exp, subject)
        for sess in sorted([d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]):
            smeta = os.path.join(subject_dir, sess, "metadata.json")
            if os.path.exists(smeta):
                try:
                    data = json.loads(open(smeta, "r", encoding="utf-8").read())
                    row = {
                        "Project": proj,
                        "Experiment": exp,
                        "Subject": subject,
                        "Session": sess,
                        "DateTime": data.get("DateTime", ""),
                        "Recording": data.get("Recording", ""),
                        "Experimenter": data.get("Experimenter", ""),
                        "Condition": data.get("Condition", ""),
                        "Comments": data.get("Comments", ""),
                    }
                    rows.append(row)
                except Exception:
                    pass

        if not rows:
            QMessageBox.information(self, "No data", "No metadata found for this subject.")
            return

        df = pd.DataFrame(rows)
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", f"{subject}_summary.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        df.to_csv(path, index=False)
        QMessageBox.information(self, "Saved", f"Animal summary saved:\n{path}")


def launch():
    _set_windows_app_user_model_id()
    app = QApplication([])
    app.setApplicationName("MetaMan")
    app.setApplicationDisplayName("MetaMan")

    splash = None
    logo_path = _resolve_logo_path()
    if logo_path:
        icon = QIcon(logo_path)
        if not icon.isNull():
            app.setWindowIcon(icon)
        pix = QPixmap(logo_path)
        if not pix.isNull():
            pix = pix.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            splash = QSplashScreen(pix)
            splash.showMessage("Loading MetaMan...", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
            splash.show()
            app.processEvents()

    win = MainWindow()
    win.show()
    if splash is not None:
        splash.finish(win)
    app.exec()
