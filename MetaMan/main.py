import os
import sys
from datetime import datetime
from time import time

# Allow running this file directly (python main.py) as well as via the package
# entry point (python run_app.py / python -m MetaMan.main). When executed as a
# loose script there is no parent package, so relative imports below would fail.
if __package__ in (None, ""):
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    __package__ = "MetaMan"

from PySide6.QtCore import QObject, QTime, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QSplashScreen,
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QTimeEdit,
    QVBoxLayout, QWidget, QInputDialog
)

from .config import APP_TITLE, WINDOW_GEOMETRY
from .state import AppState
from .tabs.navigation_tab import NavigationTab
from .tabs.recording_tab import RecordingTab
from .tabs.preprocessing_tab import PreprocessingTab
from .tabs.data_reorganizer_tab import DataReorganizerTab
from .tabs.transfer_tab import TransferTab
from .project_bar import ProjectContextBar
from .nav_rail import WorkspaceShell
from .services.server_sync import sync_project_to_server
from .services.backup_report import build_record, write_report_files
from .services.staging_service import load_manifest, sync_pending
from .theme import STYLESHEET, FONT_FAMILY
from .services.search_service import search_in_project
from .io_ops import list_experiments, list_projects, load_session_metadata, save_session_triplet
from .utils import run_in_thread
from .structure_designer import StructureDesignerDialog
from .services.structure_schema import default_structure_schema, normalize_structure_schema


def _resolve_logo_path() -> str:
    search_roots = []
    if getattr(sys, "frozen", False):
        search_roots.extend([getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)])
    search_roots.append(os.path.dirname(__file__))

    relative_candidates = [
        os.path.join("assests", "metaman.png"),
        os.path.join("assets", "metaman.png"),
        os.path.join("MetaMan", "assests", "metaman.png"),
        os.path.join("MetaMan", "assets", "metaman.png"),
    ]
    for root in search_roots:
        if not root:
            continue
        for relative_path in relative_candidates:
            path = os.path.join(root, relative_path)
            if os.path.isfile(path):
                return path
    return ""


def _resolve_font_path() -> str:
    search_roots = []
    if getattr(sys, "frozen", False):
        search_roots.extend([getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)])
    search_roots.append(os.path.dirname(__file__))

    relative_candidates = [
        os.path.join("assets", "fonts", "InterVariable.ttf"),
        os.path.join("MetaMan", "assets", "fonts", "InterVariable.ttf"),
    ]
    for root in search_roots:
        if not root:
            continue
        for relative_path in relative_candidates:
            path = os.path.join(root, relative_path)
            if os.path.isfile(path):
                return path
    return ""


def _load_app_font() -> str:
    """Register the bundled Inter font and return its family, or the fallback."""
    path = _resolve_font_path()
    if path:
        from PySide6.QtGui import QFontDatabase

        font_id = QFontDatabase.addApplicationFont(path)
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return families[0]
    return FONT_FAMILY


def _set_windows_app_user_model_id():
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MetaMan.App")
    except Exception:
        pass


class _BackupEmitter(QObject):
    """Thread-safe bridge for completed backup runs (worker → GUI thread)."""

    run_recorded = Signal(dict)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(*WINDOW_GEOMETRY)
        self.setMinimumSize(1100, 720)
        logo_path = _resolve_logo_path()
        if logo_path:
            icon = QIcon(logo_path)
            if not icon.isNull():
                self.setWindowIcon(icon)
        self.state = AppState()
        self._backup_jobs_in_progress = set()
        self._schedule_warning_cache = set()
        self._backup_emitter = _BackupEmitter()
        self._backup_emitter.run_recorded.connect(self._on_backup_run_recorded)
        self._build_ui()
        self._build_menu()
        self._apply_visual_style()
        self._init_backup_scheduler()

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Global active-project bar drives every tab; folded into the nav rail.
        self.project_bar = ProjectContextBar(self.state, compact=True)
        self.project_bar.project_selected.connect(self._activate_project_everywhere)
        self.project_bar.design_structure_requested.connect(self._menu_design_structure)
        self.project_bar.set_data_root_requested.connect(self._menu_set_data_root)

        self.tabs = WorkspaceShell(
            logo_path=_resolve_logo_path(),
            header_widget=self.project_bar,
        )
        self.nav_tab = NavigationTab(
            self.state,
            on_load_session=self._load_session_everywhere,
            on_activate_project=self._activate_project_everywhere,
            on_local_changed=self._on_datasets_changed,
        )
        self.rec_tab = RecordingTab(self.state)
        self.pre_tab = PreprocessingTab(self.state)
        self.reorg_tab = DataReorganizerTab(self.state)
        # Transfer hosts Backup + Staging + Schedule and needs nav_tab to exist.
        self.transfer_tab = TransferTab(self.state, self)
        self.staging_tab = self.transfer_tab.staging  # back-compat alias

        self.tabs.add_page(self.nav_tab, "Browse", "\U0001f5c2️")
        self.tabs.add_page(self.rec_tab, "Record", "⏺️")
        self.tabs.add_page(self.pre_tab, "Process", "⚙️")
        self.tabs.add_page(self.transfer_tab, "Transfer", "☁️")
        self.tabs.add_page(self.reorg_tab, "Import", "\U0001f4e5")
        root.addWidget(self.tabs, 1)

        # Status bar with a permanent active-project indicator.
        self.lbl_status = QLabel("")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.lbl_active_project = QLabel("")
        self.lbl_active_project.setStyleSheet("font-weight: 600; padding-right: 8px;")
        self.statusBar().addPermanentWidget(self.lbl_active_project)
        self._update_active_project_indicator()

    def _build_menu(self):
        bar = self.menuBar()

        def add(menu, label, slot):
            act = QAction(label, self)
            act.triggered.connect(slot)
            menu.addAction(act)
            return act

        # File
        menu_file = bar.addMenu("&File")
        add(menu_file, "New Project…", self._menu_new_project)
        add(menu_file, "Open Local Project…", lambda: self.project_bar._load_local())
        add(menu_file, "Open Server Project…", lambda: self.project_bar._load_server())
        menu_file.addSeparator()
        add(menu_file, "Browse Server Projects", self._menu_make_local_copy)
        add(menu_file, "Refresh All Lists", self._refresh_everything)
        menu_file.addSeparator()
        add(menu_file, "Exit", self.close)

        # Project (acts on the active project)
        menu_proj = bar.addMenu("&Project")
        add(menu_proj, "Add Experiment…", self._menu_add_experiment)
        add(menu_proj, "\U0001f9e9  Design Data Structure…", self._menu_design_structure)
        add(menu_proj, "Open Project Folder", self._menu_open_project_folder)
        add(menu_proj, "\U0001f5c4️  Make Local Copy from Server…", self._menu_make_local_copy)
        menu_proj.addSeparator()
        add(menu_proj, "Search in Project…", self._search)
        add(menu_proj, "Generate Subject Summary CSV…", self._animal_summary)

        # Settings
        menu_set = bar.addMenu("&Settings")
        add(menu_set, "Set Data Root…", self._menu_set_data_root)
        add(menu_set, "Folder Names (rawData / processedData)…", self._menu_folder_names)

        # Transfer
        menu_tr = bar.addMenu("&Transfer")
        add(menu_tr, "Backup Now…", self._backup_project_now)
        add(menu_tr, "Schedule Backup…", self._schedule_backup)
        add(menu_tr, "View Backup History…", self._menu_backup_history)
        menu_tr.addSeparator()
        add(menu_tr, "Open Transfer Tab", lambda: self.tabs.set_current_widget(self.transfer_tab))

        # Help
        menu_help = bar.addMenu("&Help")
        add(menu_help, "About MetaMan", self._menu_about)

    def _apply_visual_style(self):
        self.setStyleSheet(STYLESHEET)

    def _init_backup_scheduler(self):
        self._backup_timer = QTimer(self)
        self._backup_timer.setInterval(30_000)
        self._backup_timer.timeout.connect(self._run_due_scheduled_backups)
        self._backup_timer.start()
        QTimer.singleShot(2_000, self._run_due_scheduled_backups)

    def _load_session_everywhere(self, session_path: str):
        self.rec_tab.load_session(session_path)
        self.pre_tab._load_from_session(session_path)

    def _on_datasets_changed(self):
        """A make-local-copy / rename / delete touched the local tree; refresh
        the project picker and active-project indicator so new datasets show up.
        (The Browse tab refreshes its own local tree.)"""
        for refresh in (
            self.project_bar.refresh,
            self._update_active_project_indicator,
        ):
            try:
                refresh()
            except Exception:
                pass

    def _on_backup_run_recorded(self, record: dict):
        """Persist a finished backup run and refresh the history view. Runs on
        the GUI thread (settings writes + UI stay off the worker thread)."""
        try:
            self.state.settings.add_backup_run(record)
        except Exception:
            pass
        try:
            self.transfer_tab.refresh_history()
        except Exception:
            pass

    def _activate_project_everywhere(self, project_name: str):
        """Make *project_name* the active project across every tab. Selecting a
        project in the bar (or a node in Browse) flows through here so Record,
        Process and Transfer all follow the same project."""
        project_name = (project_name or "").strip()
        if not project_name:
            return
        # Preserve a freshly-loaded server project; otherwise treat as a local
        # project under the raw root.
        loaded = self.state.settings.get_loaded_project()
        if loaded.get("name") != project_name:
            project_dir = self._project_dir(project_name)
            self.state.settings.put_loaded_project(project_name, project_dir, "local", project_dir)
        self.state.set_current(project=project_name, experiment="", animal="", session="", session_path="")

        for refresh in (
            lambda: self.project_bar.refresh(active=project_name),
            lambda: self.nav_tab.select_project(project_name),
            self.rec_tab._refresh_from_project,
            self.pre_tab._refresh_from_project,
            self.transfer_tab.refresh,
        ):
            try:
                refresh()
            except Exception:
                pass
        self._update_active_project_indicator()
        self.lbl_status.setText(f"Active project: {project_name}")

    def _update_active_project_indicator(self):
        name = (self.state.current_project
                or self.state.settings.get_loaded_project().get("name", "")).strip()
        self.lbl_active_project.setText(f"Project: {name}" if name else "No project loaded")

    def _menu_open_project_folder(self):
        project = self.state.current_project
        if not project:
            QMessageBox.warning(self, "No project", "Select a project first.")
            return
        target = self._resolve_project_dir(project)
        if not os.path.isdir(target):
            QMessageBox.critical(self, "Not found", f"Project folder not found:\n{target}")
            return
        try:
            if os.name == "nt":
                os.startfile(target)
            elif sys.platform == "darwin":
                import subprocess; subprocess.run(["open", target])
            else:
                import subprocess; subprocess.run(["xdg-open", target])
        except Exception as e:
            QMessageBox.critical(self, "Open error", f"Failed to open:\n{target}\n\n{e}")

    def _menu_make_local_copy(self):
        """Jump to Browse ▸ Server tab, so the user can pick an experiment /
        session and right-click ▸ Make local copy into the local rawData root."""
        self.tabs.set_current_widget(self.nav_tab)
        try:
            self.nav_tab.focus_server_tab()
        except Exception:
            pass

    def _menu_backup_history(self):
        self.tabs.set_current_widget(self.transfer_tab)
        try:
            self.transfer_tab.show_history_panel()
        except Exception:
            pass

    def _menu_about(self):
        QMessageBox.information(
            self, "About MetaMan",
            "MetaMan — neuroscience data organization.\n\n"
            "Workflow: Browse · Record · Process · Transfer · Import.\n"
            "Each project's folder structure is configurable in "
            "Project ▸ Design Data Structure.",
        )

    def _project_dir(self, project: str) -> str:
        return os.path.join(self.state.settings.raw_root, project)

    def _resolve_project_dir(self, project: str) -> str:
        """Return the local folder for *project* (always ``<raw_root>/<project>``).

        The loaded project's saved destination is only trusted when it really
        points at the project folder (its basename matches the project name).
        Otherwise we ignore it, so a destination that was mistakenly set to the
        raw root (e.g. ``B:/NPX/rawData``) can never be used as the backup
        source. That mistake is what produced the stray ``server/rawData/...``
        tree on the server.
        """
        loaded = self.state.settings.get_loaded_project()
        dest = str(loaded.get("destination_path") or "").strip()
        if loaded.get("name") == project and dest:
            if os.path.basename(os.path.normpath(dest)).lower() == project.strip().lower():
                return dest
        return self._project_dir(project)

    def _experiment_dir(self, project: str, experiment: str) -> str:
        return os.path.join(self._resolve_project_dir(project), experiment)

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

    def _sync_pending_staging_for_project(self, project: str, log) -> int:
        manifest = load_manifest(self.state.settings.data_root)
        entry_ids = [
            str(entry.get("id", "")).strip()
            for entry in manifest
            if str(entry.get("project", "")).strip() == project and entry.get("status") == "pending"
        ]
        entry_ids = [entry_id for entry_id in entry_ids if entry_id]
        if not entry_ids:
            return 0

        log(f"[staging] Auto-syncing {len(entry_ids)} pending recording(s) for {project}.")
        sync_pending(self.state.settings.data_root, log, entry_ids=entry_ids)
        return len(entry_ids)

    def _start_backup_copy(self, project: str, experiment: str, destination_root: str, destination_kind: str, scheduled: bool, log=None):
        sink = log or self.rec_tab.logger.log
        source_dir = self._experiment_dir(project, experiment) if experiment else self._resolve_project_dir(project)
        if not os.path.isdir(source_dir):
            sink(f"[backup] Source folder not found: {source_dir}")
            return
        # Server mirrors local: <server_root>/<project>[/<experiment>]. The
        # destination folder name is explicit so the source basename can never
        # leak in (which previously created server/rawData instead of project).
        if experiment:
            dest_parent = os.path.join(destination_root, project)
            dest_name = experiment
        else:
            dest_parent = destination_root
            dest_name = project
        destination_scope_root = os.path.join(dest_parent, dest_name)

        job_key = self._backup_job_key(project, experiment, destination_kind=destination_kind)
        if job_key in self._backup_jobs_in_progress:
            scope = f"{project}/{experiment}" if experiment else project
            sink(f"[backup] Backup already in progress for scope: {scope} ({destination_kind})")
            return

        def log(s: str):
            sink(s)

        scope = f"{project}/{experiment}" if experiment else project
        mode = "scheduled backup" if scheduled else "manual backup"
        destination_label = "external_hdd" if destination_kind == "hdd" else "server"
        started = time()
        self._backup_jobs_in_progress.add(job_key)
        self.lbl_status.setText(f"{mode.capitalize()} running: {scope} -> {destination_label}")

        def work():
            ok = False
            stats = None
            run_error = ""
            staging_synced = 0
            try:
                log(f"[{mode}/{destination_label}] Starting: {scope} -> {destination_scope_root}")
                if destination_kind == "server":
                    try:
                        staging_synced = self._sync_pending_staging_for_project(project, log)
                    except Exception as exc:
                        log(f"[warning] Staging auto-sync failed: {exc}")
                stats = sync_project_to_server(source_dir, dest_parent, log, dest_name=dest_name)
                dt = max(time() - started, 1e-6)
                log(f"[{mode}/{destination_label}] Finished in {dt:.1f}s.")
                if not experiment:
                    self._annotate_current_session_backup_paths(project, destination_root, destination_kind, log)
                ok = True
            except Exception as e:
                run_error = str(e)
                log(f"[error] {mode}/{destination_label} failed for '{scope}': {e}")
            finally:
                self._backup_jobs_in_progress.discard(job_key)
                if scheduled and ok:
                    self.state.settings.mark_backup_schedule_run(project, datetime.now().strftime("%Y-%m-%d"), experiment=experiment)
                # Build, persist (on the GUI thread) and write out the run report.
                try:
                    record = build_record(
                        project=project,
                        experiment=experiment,
                        destination_kind=destination_kind,
                        destination_root=destination_root,
                        destination_path=destination_scope_root,
                        source_path=source_dir,
                        trigger="scheduled" if scheduled else "manual",
                        stats=stats or {},
                        staging_synced=staging_synced,
                        error=run_error,
                    )
                    write_report_files(destination_root, record, log=log)
                    self._backup_emitter.run_recorded.emit(record)
                except Exception as exc:
                    log(f"[warning] Could not record backup run: {exc}")

        run_in_thread(work)

    def _copy_to_server(self):
        self._backup_project_now()

    def _backup_project_now(self):
        projects = list_projects(self.state.settings.raw_root)
        if not projects:
            QMessageBox.warning(self, "No projects", "No projects found in the data root.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Backup Project")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()

        # Flexible project picker (defaults to the current project if any).
        cb_project = QComboBox()
        cb_project.addItems(projects)
        if self.state.current_project in projects:
            cb_project.setCurrentText(self.state.current_project)

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

        chk_current_experiment = QCheckBox("Backup only current experiment")
        if self.state.current_experiment:
            chk_current_experiment.setText(f"Backup only current experiment ({self.state.current_experiment})")

        form.addRow("Project:", cb_project)
        form.addRow("Destination:", cb_destination)
        form.addRow("Server root:", row_server_w)
        form.addRow("External HDD root:", row_hdd_w)
        form.addRow("", chk_current_experiment)
        lay.addLayout(form)

        def load_roots_for_project():
            p = cb_project.currentText().strip()
            ed_server.setText(self.state.settings.get_server_root_for_project(p))
            ed_hdd.setText(self.state.settings.get_hdd_root_for_project(p))
            has_cur_exp = bool(self.state.current_experiment) and p == self.state.current_project
            chk_current_experiment.setEnabled(has_cur_exp)
            if not has_cur_exp:
                chk_current_experiment.setChecked(False)

        load_roots_for_project()
        cb_project.currentIndexChanged.connect(lambda _=0: load_roots_for_project())

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
            proj = cb_project.currentText().strip()
            if not proj:
                QMessageBox.warning(dlg, "Project", "Choose a project.")
                return
            project_dir = self._resolve_project_dir(proj)
            if not os.path.isdir(project_dir):
                QMessageBox.critical(dlg, "Missing", f"Project folder not found:\n{project_dir}")
                return
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
            if (chk_current_experiment.isChecked() and self.state.current_experiment
                    and proj == self.state.current_project):
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

            project_dir = self._resolve_project_dir(project)
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
            self.rec_tab._refresh_from_project()
        except Exception:
            pass
        try:
            self.pre_tab._refresh_from_project()
        except Exception:
            pass
        try:
            self.staging_tab._refresh_manifest_table()
        except Exception:
            pass
        try:
            self.reorg_tab._refresh_project_list()
        except Exception:
            pass
        try:
            self.transfer_tab.refresh()
        except Exception:
            pass
        try:
            self.project_bar.refresh()
        except Exception:
            pass
        try:
            self._update_active_project_indicator()
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
            self.rec_tab._refresh_from_project()
        except Exception:
            pass
        try:
            self.pre_tab.ed_data_root.setText(self.state.settings.data_root)
            self.pre_tab.ed_proc_root.setText(self.state.settings.processed_root)
        except Exception:
            pass
        try:
            self.staging_tab._refresh_manifest_table()
        except Exception:
            pass
        try:
            self.reorg_tab.ed_target_raw.setText(self.state.settings.raw_root)
            self.reorg_tab.ed_target_proc.setText(self.state.settings.processed_root)
        except Exception:
            pass
        self._refresh_everything()
        self.lbl_status.setText(f"Data root set: {self.state.settings.data_root}")

    def _menu_folder_names(self):
        """Let the user choose the raw/processed folder names (default
        rawData / processedData). Optionally renames the existing folders."""
        s = self.state.settings
        data_root = s.data_root
        cur_raw = s.raw_dir_name
        cur_proc = s.processed_dir_name

        dlg = QDialog(self)
        dlg.setWindowTitle("Folder Names")
        lay = QVBoxLayout(dlg)

        info = QLabel(
            "Choose the folder names used for raw and processed data. These sit "
            "directly under the data root and hold your projects.\n"
            "Default is rawData / processedData; other users can pick their own."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        form = QFormLayout()
        ed_raw = QLineEdit(cur_raw)
        ed_proc = QLineEdit(cur_proc)
        form.addRow("Raw folder name:", ed_raw)
        form.addRow("Processed folder name:", ed_proc)
        lay.addLayout(form)

        lbl_preview = QLabel()
        lbl_preview.setWordWrap(True)
        lbl_preview.setStyleSheet("color:#6f6a7a; padding:8px; background:#f7f6fc; border:1px solid #e6e3f0; border-radius:8px;")
        lay.addWidget(lbl_preview)

        chk_rename = QCheckBox("Also rename the existing folders on disk (when the new name is free)")
        chk_rename.setChecked(True)
        lay.addWidget(chk_rename)

        def refresh_preview():
            r = ed_raw.text().strip() or cur_raw
            p = ed_proc.text().strip() or cur_proc
            lbl_preview.setText(
                f"{data_root}\\{r}\\<project>\\<experiment>\\...\n"
                f"{data_root}\\{p}\\<project>\\<experiment>\\..."
            )
        ed_raw.textChanged.connect(lambda _=0: refresh_preview())
        ed_proc.textChanged.connect(lambda _=0: refresh_preview())
        refresh_preview()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return

        new_raw = ed_raw.text().strip() or cur_raw
        new_proc = ed_proc.text().strip() or cur_proc
        if new_raw == cur_raw and new_proc == cur_proc:
            return

        if chk_rename.isChecked():
            for old_name, new_name in ((cur_raw, new_raw), (cur_proc, new_proc)):
                if old_name == new_name:
                    continue
                old_dir = os.path.join(data_root, old_name)
                new_dir = os.path.join(data_root, new_name)
                if os.path.isdir(old_dir) and not os.path.exists(new_dir):
                    try:
                        os.rename(old_dir, new_dir)
                    except Exception as e:
                        QMessageBox.warning(
                            self, "Rename failed",
                            f"Could not rename:\n{old_dir}\n->\n{new_dir}\n\n{e}\n\n"
                            "The name was still updated; move the data manually if needed."
                        )

        self.state.settings.set_folder_names(new_raw, new_proc)
        self.state.settings.ensure_storage_roots()
        try:
            self.nav_tab.ed_root.setText(self.state.settings.data_root)
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
        self.lbl_status.setText(
            f"Folder names set: {new_raw} / {new_proc}"
        )

    def _menu_design_structure(self):
        project = self.state.current_project

        default_schema = normalize_structure_schema(
            self.state.settings.get_structure_schema() or default_structure_schema()
        )
        project_schema = None
        if project:
            raw_proj = self.state.settings.get_project_structure_schema(project)
            project_schema = normalize_structure_schema(raw_proj) if raw_proj else None

        dlg = StructureDesignerDialog(
            default_schema=default_schema,
            project_schema=project_schema,
            project_name=project,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        result = dlg.schema()
        if dlg.selected_scope() == "project" and project:
            self.state.settings.put_project_structure_schema(project, result)
            self.lbl_status.setText(f"Structure saved for project: {project}")
        else:
            self.state.settings.put_structure_schema(result)
            self.lbl_status.setText("Default structure saved (applies to all projects).")

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
        subject = self.state.current_animal
        if not (proj and subject):
            QMessageBox.warning(self, "Choose subject", "Select a subject in Browse first.")
            return

        import os.path as _osp
        import pandas as pd

        # Schema-aware: walk every session under the project and keep this
        # subject's, whatever the nesting order (subject-before-experiment etc.).
        rows = []
        proj_dir = self._project_dir(proj)
        for values, sdir, meta in self.nav_tab._iter_sessions_under(proj, proj_dir, 0):
            if values.get("subject") != subject:
                continue
            rows.append({
                "Project": proj,
                "Experiment": values.get("experiment", meta.get("Experiment", "")),
                "Subject": subject,
                "Session": values.get("session", _osp.basename(sdir)),
                "DateTime": meta.get("DateTime", ""),
                "Recording": meta.get("Recording", ""),
                "Experimenter": meta.get("Experimenter", ""),
                "Condition": meta.get("Condition", ""),
                "Comments": meta.get("Comments", ""),
            })

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
    app.setStyle("Fusion")
    app.setFont(QFont(_load_app_font(), 10))
    app.setStyleSheet(STYLESHEET)

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


if __name__ == "__main__":
    launch()
