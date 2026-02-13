import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..io_ops import list_projects
from ..state import AppState
from ..utils import run_in_thread
from ..services.data_reorganizer import (
    ColumnAssignment,
    DataReorganizerService,
    MatchPlan,
    MetadataWriter,
    ReorganizerConfig,
    RunSummary,
    ScanResult,
)


def _table_set_headers(tbl: QTableWidget, headers: List[str]):
    tbl.setColumnCount(len(headers))
    tbl.setHorizontalHeaderLabels(headers)
    tbl.horizontalHeader().setStretchLastSection(True)


def _selected_items_text(tbl: QTableWidget, row: int, col: int) -> str:
    item = tbl.item(row, col)
    return item.text() if item else ""


class ReorgEmitter(QObject):
    log_line = Signal(str)
    progress = Signal(str, int)
    scan_done = Signal(object)
    match_done = Signal(object)
    execute_done = Signal(object, object)
    error = Signal(str)
    busy = Signal(bool)


class DataReorganizerTab(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.service = DataReorganizerService()
        self.emitter = ReorgEmitter()
        self._cancel_event = threading.Event()

        self.plan_load_result = None
        self.plan_rows = []
        self.normalized_df = None
        self.scan_result: Optional[ScanResult] = None
        self.match_plan: Optional[MatchPlan] = None
        self.last_actions = []
        self.last_summary: Optional[RunSummary] = None
        self._loaded_datatype_map_override: Dict[str, str] = {}

        self._wire_emitter()
        self._build_ui()
        self._load_prefs()

    def _wire_emitter(self):
        self.emitter.log_line.connect(self._append_log)
        self.emitter.progress.connect(self._on_progress)
        self.emitter.scan_done.connect(self._on_scan_done)
        self.emitter.match_done.connect(self._on_match_done)
        self.emitter.execute_done.connect(self._on_execute_done)
        self.emitter.error.connect(self._on_error)
        self.emitter.busy.connect(self._set_busy)

    def _build_ui(self):
        root = QVBoxLayout(self)
        main_split = QSplitter(Qt.Horizontal)
        main_split.setChildrenCollapsible(False)
        root.addWidget(main_split, 1)

        # Column 1 (scroll): setup + plan import
        col1_widget = QWidget()
        col1_layout = QVBoxLayout(col1_widget)
        col1_layout.setContentsMargins(0, 0, 0, 0)
        col1_layout.setSpacing(8)

        # Step A
        self.grp_a = QGroupBox("Step A - Project and experiment setup")
        ga = QGridLayout(self.grp_a)
        row = 0
        ga.addWidget(QLabel("Project mode"), row, 0)
        self.cb_project_mode = QComboBox()
        self.cb_project_mode.addItems(["Use existing project", "Create new project"])
        ga.addWidget(self.cb_project_mode, row, 1)
        row += 1

        ga.addWidget(QLabel("Existing project"), row, 0)
        self.cb_existing_project = QComboBox()
        ga.addWidget(self.cb_existing_project, row, 1)
        row += 1

        ga.addWidget(QLabel("New project name"), row, 0)
        self.ed_new_project = QLineEdit()
        ga.addWidget(self.ed_new_project, row, 1)
        row += 1

        ga.addWidget(QLabel("Experiment name"), row, 0)
        self.ed_experiment = QLineEdit()
        ga.addWidget(self.ed_experiment, row, 1)
        row += 1

        self.ed_source_raw, b_sr = self._path_row("Source raw folder(s)", ga, row)
        row += 1
        self.ed_source_proc, b_sp = self._path_row("Source processed folder(s)", ga, row)
        row += 1
        self.ed_target_raw, b_tr = self._path_row("Target raw root", ga, row)
        row += 1
        self.ed_target_proc, b_tp = self._path_row("Target processed root", ga, row)
        row += 1

        ga.addWidget(QLabel("Resolved raw experiment path"), row, 0)
        self.lbl_resolved_raw = QLabel("-")
        self.lbl_resolved_raw.setWordWrap(True)
        self.lbl_resolved_raw.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ga.addWidget(self.lbl_resolved_raw, row, 1)
        row += 1

        ga.addWidget(QLabel("Resolved processed experiment path"), row, 0)
        self.lbl_resolved_proc = QLabel("-")
        self.lbl_resolved_proc.setWordWrap(True)
        self.lbl_resolved_proc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ga.addWidget(self.lbl_resolved_proc, row, 1)
        row += 1

        self.chk_preserve_group = QCheckBox("Preserve processed group hierarchy")
        ga.addWidget(self.chk_preserve_group, row, 0, 1, 2)
        row += 1

        ga.addWidget(QLabel("If destination exists"), row, 0)
        self.cb_overwrite = QComboBox()
        self.cb_overwrite.addItems(["skip", "rename", "overwrite"])
        ga.addWidget(self.cb_overwrite, row, 1)
        row += 1

        self.chk_overwrite_confirm = QCheckBox("I confirm overwrite when strategy is overwrite")
        ga.addWidget(self.chk_overwrite_confirm, row, 0, 1, 2)
        row += 1

        self.chk_dry_run = QCheckBox("Dry run (do not copy files)")
        self.chk_dry_run.setChecked(True)
        ga.addWidget(self.chk_dry_run, row, 0, 1, 2)
        row += 1

        self.chk_verify_size = QCheckBox("Verify file sizes after copy")
        ga.addWidget(self.chk_verify_size, row, 0, 1, 2)
        row += 1

        btn_row = QHBoxLayout()
        self.btn_refresh_projects = QPushButton("Refresh projects")
        self.btn_resolve_paths = QPushButton("Validate/Create target folders")
        btn_row.addWidget(self.btn_refresh_projects)
        btn_row.addWidget(self.btn_resolve_paths)
        btn_row.addStretch(1)
        ga.addLayout(btn_row, row, 0, 1, 2)
        col1_layout.addWidget(self.grp_a)

        # Step B
        self.grp_b = QGroupBox("Step B - Metadata plan import")
        gb = QVBoxLayout(self.grp_b)
        fb = QHBoxLayout()
        self.ed_plan_path = QLineEdit()
        self.btn_browse_plan = QPushButton("Choose plan file...")
        self.btn_load_plan = QPushButton("Load plan")
        fb.addWidget(self.ed_plan_path, 1)
        fb.addWidget(self.btn_browse_plan)
        fb.addWidget(self.btn_load_plan)
        gb.addLayout(fb)

        stat = QHBoxLayout()
        self.lbl_delimiter = QLabel("Delimiter: -")
        self.lbl_rows = QLabel("Rows: -")
        stat.addWidget(self.lbl_delimiter)
        stat.addWidget(self.lbl_rows)
        stat.addStretch(1)
        gb.addLayout(stat)

        self.tbl_plan_preview = QTableWidget(0, 0)
        self.tbl_plan_preview.setMinimumHeight(220)
        gb.addWidget(self.tbl_plan_preview, 1)
        col1_layout.addWidget(self.grp_b)
        col1_layout.addStretch(1)

        col1_scroll = QScrollArea()
        col1_scroll.setWidgetResizable(True)
        col1_scroll.setWidget(col1_widget)
        main_split.addWidget(col1_scroll)

        # Column 2 (scroll): column assignment
        col2_widget = QWidget()
        col2_layout = QVBoxLayout(col2_widget)
        col2_layout.setContentsMargins(0, 0, 0, 0)
        col2_layout.setSpacing(8)

        self.grp_c = QGroupBox("Step C - Column assignment")
        gc = QVBoxLayout(self.grp_c)

        self.cb_subject = QComboBox()
        self.cb_session = QComboBox()
        self.cb_trial = QComboBox()
        self.cb_genotype = QComboBox()
        self.cb_condition = QComboBox()
        self.cb_match_mode = QComboBox()
        self.cb_match_mode.addItems(
            [
                "Use only subject_id for matching",
                "Use only trial_id for matching",
                "Use subject_id + session_id",
                "Use custom columns",
            ]
        )
        self.chk_case_sensitive = QCheckBox("Case sensitive matching")
        self.btn_apply_columns = QPushButton("Apply column assignment")

        form = QFormLayout()
        form.addRow("subject_id (required)", self.cb_subject)
        form.addRow("session_id", self.cb_session)
        form.addRow("trial_id", self.cb_trial)
        form.addRow("genotype", self.cb_genotype)
        form.addRow("condition", self.cb_condition)
        form.addRow("Match mode", self.cb_match_mode)
        gc.addLayout(form)
        gc.addWidget(self.chk_case_sensitive)

        self.lst_extra_cols = QTableWidget(0, 1)
        _table_set_headers(self.lst_extra_cols, ["Include", "Additional metadata column"])
        self.lst_extra_cols.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lst_extra_cols.setSelectionMode(QAbstractItemView.NoSelection)

        self.lst_custom_match_cols = QTableWidget(0, 1)
        _table_set_headers(self.lst_custom_match_cols, ["Custom match columns"])
        self.lst_custom_match_cols.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lst_custom_match_cols.setSelectionMode(QAbstractItemView.MultiSelection)

        self.txt_key_preview = QTextEdit()
        self.txt_key_preview.setReadOnly(True)
        self.txt_key_preview.setMinimumHeight(220)

        c_split = QSplitter(Qt.Horizontal)
        c_split.setChildrenCollapsible(False)

        c_extra_w = QWidget()
        c_extra_l = QVBoxLayout(c_extra_w)
        c_extra_l.setContentsMargins(0, 0, 0, 0)
        c_extra_l.addWidget(QLabel("Additional metadata columns"))
        c_extra_l.addWidget(self.lst_extra_cols, 1)
        c_split.addWidget(c_extra_w)

        c_custom_w = QWidget()
        c_custom_l = QVBoxLayout(c_custom_w)
        c_custom_l.setContentsMargins(0, 0, 0, 0)
        c_custom_l.addWidget(QLabel("Custom match columns"))
        c_custom_l.addWidget(self.lst_custom_match_cols, 1)
        c_split.addWidget(c_custom_w)

        c_preview_w = QWidget()
        c_preview_l = QVBoxLayout(c_preview_w)
        c_preview_l.setContentsMargins(0, 0, 0, 0)
        c_preview_l.addWidget(QLabel("Match key preview (first 10 rows)"))
        c_preview_l.addWidget(self.txt_key_preview, 1)
        c_split.addWidget(c_preview_w)

        c_split.setSizes([280, 280, 360])
        gc.addWidget(c_split, 1)
        gc.addWidget(self.btn_apply_columns)
        col2_layout.addWidget(self.grp_c)
        col2_layout.addStretch(1)

        col2_scroll = QScrollArea()
        col2_scroll.setWidgetResizable(True)
        col2_scroll.setWidget(col2_widget)
        main_split.addWidget(col2_scroll)

        # Column 3: scan/match + execution/log
        col3 = QWidget()
        col3_l = QVBoxLayout(col3)
        col3_l.setContentsMargins(0, 0, 0, 0)
        col3_l.setSpacing(8)

        right_split = QSplitter(Qt.Vertical)
        right_split.setChildrenCollapsible(False)
        col3_l.addWidget(right_split, 1)

        # Step D
        self.grp_d = QGroupBox("Step D - File discovery and matching")
        gd = QVBoxLayout(self.grp_d)
        d_top = QHBoxLayout()
        self.btn_scan = QPushButton("Scan source folders")
        self.btn_build_match_plan = QPushButton("Build match plan")
        d_top.addWidget(self.btn_scan)
        d_top.addWidget(self.btn_build_match_plan)
        d_top.addStretch(1)
        gd.addLayout(d_top)

        d_stat = QHBoxLayout()
        self.lbl_raw_files = QLabel("Raw files: -")
        self.lbl_proc_files = QLabel("Processed files: -")
        self.lbl_extensions = QLabel("Extensions: -")
        d_stat.addWidget(self.lbl_raw_files)
        d_stat.addWidget(self.lbl_proc_files)
        d_stat.addWidget(self.lbl_extensions, 1)
        gd.addLayout(d_stat)

        self.tbl_dtype_map = QTableWidget(0, 2)
        _table_set_headers(self.tbl_dtype_map, ["Detected datatype", "Output datatype"])
        self.tbl_dtype_map.setMinimumHeight(140)
        gd.addWidget(self.tbl_dtype_map, 1)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter"))
        self.cb_plan_filter = QComboBox()
        self.cb_plan_filter.addItems(["all rows", "only unmatched", "only conflicts"])
        self.btn_apply_filter = QPushButton("Apply filter")
        filter_row.addWidget(self.cb_plan_filter)
        filter_row.addWidget(self.btn_apply_filter)
        filter_row.addStretch(1)
        gd.addLayout(filter_row)

        d_tables_split = QSplitter(Qt.Vertical)
        d_tables_split.setChildrenCollapsible(False)

        match_w = QWidget()
        match_l = QVBoxLayout(match_w)
        match_l.setContentsMargins(0, 0, 0, 0)
        self.tbl_match_plan = QTableWidget(0, 7)
        _table_set_headers(
            self.tbl_match_plan,
            ["row", "subject_id", "session_id", "match_key", "matched raw", "matched processed", "warnings"],
        )
        self.tbl_match_plan.setMinimumHeight(220)
        match_l.addWidget(self.tbl_match_plan, 1)
        d_tables_split.addWidget(match_w)

        unmatched_w = QWidget()
        unmatched_l = QVBoxLayout(unmatched_w)
        unmatched_l.setContentsMargins(0, 0, 0, 0)
        self.tbl_unmatched = QTableWidget(0, 3)
        _table_set_headers(self.tbl_unmatched, ["source_kind", "source_path", "reason"])
        self.tbl_unmatched.setMinimumHeight(140)
        unmatched_l.addWidget(self.tbl_unmatched, 1)
        d_tables_split.addWidget(unmatched_w)
        d_tables_split.setSizes([320, 180])

        gd.addWidget(d_tables_split, 3)
        right_split.addWidget(self.grp_d)

        # Step E + logs
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)

        self.grp_e = QGroupBox("Step E - Preview and execution")
        ge = QVBoxLayout(self.grp_e)
        e_btn = QHBoxLayout()
        self.btn_execute = QPushButton("Execute copy")
        self.btn_update_metadata = QPushButton("Update metadata")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_save_config = QPushButton("Save config...")
        self.btn_load_config = QPushButton("Load config...")
        self.btn_export_report = QPushButton("Export match report CSV...")
        e_btn.addWidget(self.btn_execute)
        e_btn.addWidget(self.btn_update_metadata)
        e_btn.addWidget(self.btn_cancel)
        e_btn.addWidget(self.btn_save_config)
        e_btn.addWidget(self.btn_load_config)
        e_btn.addWidget(self.btn_export_report)
        e_btn.addStretch(1)
        ge.addLayout(e_btn)

        progress_row = QHBoxLayout()
        self.lbl_stage = QLabel("Stage: idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_row.addWidget(self.lbl_stage)
        progress_row.addWidget(self.progress, 1)
        ge.addLayout(progress_row)

        self.lbl_summary = QLabel("Summary: -")
        ge.addWidget(self.lbl_summary)
        bl.addWidget(self.grp_e)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(160)
        bl.addWidget(self.txt_log, 1)

        right_split.addWidget(bottom)
        right_split.setSizes([700, 260])
        main_split.addWidget(col3)
        main_split.setSizes([640, 560, 920])
        main_split.setStretchFactor(0, 2)
        main_split.setStretchFactor(1, 2)
        main_split.setStretchFactor(2, 3)

        # Table readability/accessibility improvements.
        all_tables = [
            self.tbl_plan_preview,
            self.lst_extra_cols,
            self.lst_custom_match_cols,
            self.tbl_dtype_map,
            self.tbl_match_plan,
            self.tbl_unmatched,
        ]
        for tbl in all_tables:
            tbl.setAlternatingRowColors(True)
            tbl.verticalHeader().setVisible(False)
            tbl.setWordWrap(False)
            tbl.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            tbl.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.lst_extra_cols.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.lst_extra_cols.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        # Keyboard shortcuts.
        self.btn_scan.setShortcut("Ctrl+Shift+S")
        self.btn_build_match_plan.setShortcut("Ctrl+Shift+M")
        self.btn_execute.setShortcut("Ctrl+Return")
        self.btn_cancel.setShortcut("Esc")

        b_sr.clicked.connect(lambda: self._pick_dir(self.ed_source_raw, append=True))
        b_sp.clicked.connect(lambda: self._pick_dir(self.ed_source_proc, append=True))
        b_tr.clicked.connect(lambda: self._pick_dir(self.ed_target_raw))
        b_tp.clicked.connect(lambda: self._pick_dir(self.ed_target_proc))

        self.cb_project_mode.currentIndexChanged.connect(self._update_project_mode_ui)
        self.btn_refresh_projects.clicked.connect(self._refresh_project_list)
        self.btn_resolve_paths.clicked.connect(self._validate_and_prepare_targets)
        self.btn_browse_plan.clicked.connect(self._choose_plan_file)
        self.btn_load_plan.clicked.connect(self._load_plan)
        self.btn_apply_columns.clicked.connect(self._apply_column_assignment)
        self.btn_scan.clicked.connect(self._scan_sources)
        self.btn_build_match_plan.clicked.connect(self._build_match_plan)
        self.btn_apply_filter.clicked.connect(self._apply_plan_filter)
        self.btn_execute.clicked.connect(self._execute)
        self.btn_update_metadata.clicked.connect(self._update_metadata_only)
        self.btn_cancel.clicked.connect(self._request_cancel)
        self.btn_save_config.clicked.connect(self._save_config_file)
        self.btn_load_config.clicked.connect(self._load_config_file)
        self.btn_export_report.clicked.connect(self._export_match_report)

        self.ed_new_project.textChanged.connect(self._update_resolved_paths)
        self.ed_experiment.textChanged.connect(self._update_resolved_paths)
        self.cb_existing_project.currentTextChanged.connect(self._update_resolved_paths)
        self.ed_target_raw.textChanged.connect(self._update_resolved_paths)
        self.ed_target_proc.textChanged.connect(self._update_resolved_paths)

        self._refresh_project_list()
        self._update_project_mode_ui()
        self._update_resolved_paths()
        self._apply_tooltips()
        self._set_busy(False)

    def _path_row(self, label: str, layout: QGridLayout, row: int):
        layout.addWidget(QLabel(label), row, 0)
        ed = QLineEdit()
        b = QPushButton("Browse...")
        row_w = QWidget()
        row_l = QHBoxLayout(row_w)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.addWidget(ed, 1)
        row_l.addWidget(b)
        layout.addWidget(row_w, row, 1)
        return ed, b

    def _append_log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{ts}] {message}")

    def _on_progress(self, stage: str, percent: int):
        self.lbl_stage.setText(f"Stage: {stage}")
        self.progress.setValue(int(percent))

    def _on_error(self, message: str):
        self._append_log(f"[error] {message}")
        QMessageBox.critical(self, "Data reorganizer", message)

    def _set_busy(self, busy: bool):
        self.btn_scan.setEnabled(not busy)
        self.btn_build_match_plan.setEnabled(not busy)
        self.btn_execute.setEnabled(not busy)
        self.btn_update_metadata.setEnabled(not busy)
        self.btn_load_plan.setEnabled(not busy)
        self.btn_apply_columns.setEnabled(not busy)
        self.btn_resolve_paths.setEnabled(not busy)
        self.btn_save_config.setEnabled(not busy)
        self.btn_load_config.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)

    def _apply_tooltips(self):
        self.ed_source_raw.setToolTip("Add one or more raw source folders. Use ';' or new lines to separate paths.")
        self.ed_source_proc.setToolTip("Processed source folders are optional. Use ';' or new lines to separate paths.")
        self.chk_dry_run.setToolTip("Default safe mode. Build the full plan without copying data files.")
        self.cb_overwrite.setToolTip("skip is safest. overwrite requires explicit confirmation.")
        self.chk_overwrite_confirm.setToolTip("Must be checked to allow overwrite strategy.")
        self.cb_match_mode.setToolTip("Choose which metadata columns define the common file-match key.")
        self.chk_case_sensitive.setToolTip("Off by default: matching is case-insensitive.")
        self.tbl_dtype_map.setToolTip("Map detected datatype labels to output folder names.")
        self.chk_preserve_group.setToolTip("When enabled, processed output keeps a top-level group folder.")
        self.btn_scan.setToolTip("Scans source raw/processed folders and indexes files.")
        self.btn_build_match_plan.setToolTip("Builds deterministic plan-row to file matches with conflict detection.")
        self.btn_execute.setToolTip("Runs copy + metadata/report writing using the selected options.")
        self.btn_update_metadata.setToolTip("Rewrites metadata outputs using current plan + last copy actions without copying files.")
        self.btn_cancel.setToolTip("Requests cancellation for ongoing scan/match/execute jobs.")
        self.btn_save_config.setToolTip("Save Data reorganizer configuration (paths, assignments, options) to JSON.")
        self.btn_load_config.setToolTip("Load a saved Data reorganizer configuration from JSON.")

    def _split_paths(self, text: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for part in re.split(r"[;\n\r]+", str(text or "")):
            p = os.path.normpath(part.strip())
            if not p:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _coerce_paths_value(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return self._split_paths(self._join_paths([str(x) for x in value]))
        return self._split_paths(str(value or ""))

    def _join_paths(self, paths: List[str]) -> str:
        return "; ".join([p for p in paths if str(p).strip()])

    def _pick_dir(self, target: QLineEdit, append: bool = False):
        start = target.text().strip() or ""
        d = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        if not d:
            return
        d = os.path.normpath(d)
        if append:
            items = self._split_paths(target.text())
            if d.lower() not in {x.lower() for x in items}:
                items.append(d)
            target.setText(self._join_paths(items))
        else:
            target.setText(d)
            self._persist_settings()

    def _choose_plan_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose metadata plan",
            "",
            "Tables (*.csv *.tsv *.xlsx *.xls *.xlsm);;All files (*)",
        )
        if path:
            self.ed_plan_path.setText(path)
            self._persist_settings()

    def _refresh_project_list(self):
        current = self.cb_existing_project.currentText()
        projects = list_projects(self.app_state.settings.raw_root)
        self.cb_existing_project.blockSignals(True)
        self.cb_existing_project.clear()
        self.cb_existing_project.addItems(projects)
        if current and current in projects:
            self.cb_existing_project.setCurrentText(current)
        elif self.app_state.current_project and self.app_state.current_project in projects:
            self.cb_existing_project.setCurrentText(self.app_state.current_project)
        self.cb_existing_project.blockSignals(False)
        self._update_resolved_paths()

    def _update_project_mode_ui(self):
        use_existing = self.cb_project_mode.currentText().startswith("Use existing")
        self.cb_existing_project.setEnabled(use_existing)
        self.ed_new_project.setEnabled(not use_existing)
        self._update_resolved_paths()

    def _selected_project_name(self) -> str:
        use_existing = self.cb_project_mode.currentText().startswith("Use existing")
        if use_existing:
            return self.cb_existing_project.currentText().strip()
        return self.ed_new_project.text().strip()

    def _update_resolved_paths(self):
        project = self._selected_project_name()
        experiment = self.ed_experiment.text().strip()
        raw_root = self.ed_target_raw.text().strip()
        proc_root = self.ed_target_proc.text().strip()
        if project and experiment and raw_root:
            self.lbl_resolved_raw.setText(os.path.join(raw_root, project, experiment))
        else:
            self.lbl_resolved_raw.setText("-")
        if project and experiment and proc_root:
            self.lbl_resolved_proc.setText(os.path.join(proc_root, project, experiment))
        else:
            self.lbl_resolved_proc.setText("-")

    def _validate_setup_fields(self) -> Optional[str]:
        project = self._selected_project_name()
        experiment = self.ed_experiment.text().strip()
        if not project:
            return "Project name is required."
        if not experiment:
            return "Experiment name is required."
        if not self.ed_target_raw.text().strip():
            return "Target raw root folder is required."
        if not self.ed_target_proc.text().strip():
            return "Target processed root folder is required."
        return None

    def _validate_and_prepare_targets(self):
        msg = self._validate_setup_fields()
        if msg:
            QMessageBox.warning(self, "Setup", msg)
            return
        raw_root = self.ed_target_raw.text().strip()
        proc_root = self.ed_target_proc.text().strip()
        project = self._selected_project_name()
        experiment = self.ed_experiment.text().strip()
        raw_exp = os.path.join(raw_root, project, experiment)
        proc_exp = os.path.join(proc_root, project, experiment)
        os.makedirs(raw_exp, exist_ok=True)
        os.makedirs(proc_exp, exist_ok=True)
        self._append_log(f"Prepared target folders: {raw_exp} and {proc_exp}")
        self._update_resolved_paths()
        self._persist_settings()

    def _populate_columns_ui(self, columns: List[str]):
        opts = ["(none)"] + list(columns)
        for cb in (self.cb_subject, self.cb_session, self.cb_trial, self.cb_genotype, self.cb_condition):
            cb.clear()
            cb.addItems(opts)

        self._set_combo_guess(self.cb_subject, columns, ["subject_id", "subject", "animal_id", "animal", "id"])
        self._set_combo_guess(self.cb_session, columns, ["session_id", "session", "day", "sessionname"])
        self._set_combo_guess(self.cb_trial, columns, ["trial_id", "trial"])
        self._set_combo_guess(self.cb_genotype, columns, ["genotype"])
        self._set_combo_guess(self.cb_condition, columns, ["condition"])

        self.lst_extra_cols.setRowCount(0)
        self.lst_custom_match_cols.setRowCount(0)
        for col in columns:
            r1 = self.lst_extra_cols.rowCount()
            self.lst_extra_cols.insertRow(r1)
            check_item = QTableWidgetItem("")
            check_item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            check_item.setCheckState(Qt.Unchecked)
            self.lst_extra_cols.setItem(r1, 0, check_item)
            self.lst_extra_cols.setItem(r1, 1, QTableWidgetItem(col))

            r2 = self.lst_custom_match_cols.rowCount()
            self.lst_custom_match_cols.insertRow(r2)
            self.lst_custom_match_cols.setItem(r2, 0, QTableWidgetItem(col))

        self.lst_extra_cols.resizeColumnsToContents()
        self.lst_custom_match_cols.resizeColumnsToContents()

    def _set_combo_guess(self, cb: QComboBox, columns: List[str], candidates: List[str]):
        lowered = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in lowered:
                cb.setCurrentText(lowered[cand.lower()])
                return
        cb.setCurrentText("(none)")

    def _table_selected_values(self, tbl: QTableWidget) -> List[str]:
        rows = sorted({i.row() for i in tbl.selectedIndexes()})
        out: List[str] = []
        for r in rows:
            item = tbl.item(r, 0)
            if item and item.text().strip():
                out.append(item.text().strip())
        return out

    def _table_checked_values(self, tbl: QTableWidget) -> List[str]:
        out: List[str] = []
        for r in range(tbl.rowCount()):
            check_item = tbl.item(r, 0)
            name_item = tbl.item(r, 1)
            if not check_item or not name_item:
                continue
            if check_item.checkState() == Qt.Checked:
                name = name_item.text().strip()
                if name:
                    out.append(name)
        return out

    def _table_check_values(self, tbl: QTableWidget, values: List[str]):
        chosen = {str(v).strip() for v in values if str(v).strip()}
        for r in range(tbl.rowCount()):
            check_item = tbl.item(r, 0)
            name_item = tbl.item(r, 1)
            if not check_item or not name_item:
                continue
            if name_item.text().strip() in chosen:
                check_item.setCheckState(Qt.Checked)
            else:
                check_item.setCheckState(Qt.Unchecked)

    def _combo_value(self, cb: QComboBox) -> str:
        val = cb.currentText().strip()
        return "" if val == "(none)" else val

    def _build_assignment(self) -> ColumnAssignment:
        subject_col = self._combo_value(self.cb_subject)
        if not subject_col:
            raise ValueError("Please assign a subject_id column.")
        session_col = self._combo_value(self.cb_session)
        trial_col = self._combo_value(self.cb_trial)
        genotype_col = self._combo_value(self.cb_genotype)
        condition_col = self._combo_value(self.cb_condition)
        extra_cols = self._table_checked_values(self.lst_extra_cols)

        mode = self.cb_match_mode.currentText()
        if mode.startswith("Use only subject_id"):
            match_cols = [subject_col]
        elif mode.startswith("Use only trial_id"):
            if not trial_col:
                raise ValueError("Assign a trial_id column before using trial_id-only matching.")
            match_cols = [trial_col]
        elif mode.startswith("Use subject_id + session_id"):
            match_cols = [subject_col]
            if session_col:
                match_cols.append(session_col)
        else:
            match_cols = self._table_selected_values(self.lst_custom_match_cols)
            if not match_cols:
                match_cols = [subject_col]

        return ColumnAssignment(
            subject_col=subject_col,
            session_col=session_col,
            trial_col=trial_col,
            genotype_col=genotype_col,
            condition_col=condition_col,
            extra_cols=extra_cols,
            match_cols=match_cols,
            case_sensitive=self.chk_case_sensitive.isChecked(),
        )

    def _load_plan(self):
        path = self.ed_plan_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Plan", "Choose a metadata plan file first.")
            return
        try:
            self.plan_load_result = self.service.load_plan(path)
        except Exception as e:
            QMessageBox.critical(self, "Plan load error", str(e))
            return

        self.lbl_delimiter.setText(f"Delimiter: {self.plan_load_result.delimiter!r}")
        self.lbl_rows.setText(f"Rows: {self.plan_load_result.row_count}")
        self._populate_columns_ui(self.plan_load_result.columns)
        self._fill_plan_preview(self.plan_load_result.preview_rows, self.plan_load_result.columns)
        self._append_log(f"Loaded metadata plan: {path} ({self.plan_load_result.row_count} rows)")
        self._persist_settings()

    def _fill_plan_preview(self, rows: List[Dict[str, Any]], columns: List[str]):
        self.tbl_plan_preview.setRowCount(0)
        self.tbl_plan_preview.setColumnCount(len(columns))
        self.tbl_plan_preview.setHorizontalHeaderLabels(columns)
        for data in rows[:40]:
            r = self.tbl_plan_preview.rowCount()
            self.tbl_plan_preview.insertRow(r)
            for c, col in enumerate(columns):
                self.tbl_plan_preview.setItem(r, c, QTableWidgetItem(str(data.get(col, ""))))
        self.tbl_plan_preview.resizeColumnsToContents()

    def _apply_column_assignment(self):
        if self.plan_load_result is None:
            QMessageBox.warning(self, "Columns", "Load a metadata plan first.")
            return
        try:
            assignment = self._build_assignment()
            self.plan_rows, self.normalized_df = self.service.normalize_plan(self.plan_load_result, assignment)
        except Exception as e:
            QMessageBox.critical(self, "Column assignment", str(e))
            return

        preview_lines = []
        for row in self.plan_rows[:10]:
            preview_lines.append(
                f"row={row.row_index} subject={row.subject_id} session={row.session_id} key={row.match_key or '<unmatchable>'}"
            )
        self.txt_key_preview.setPlainText("\n".join(preview_lines))
        self._append_log(f"Column assignment applied to {len(self.plan_rows)} rows.")
        self._persist_settings()

    def _scan_sources(self):
        if self.plan_rows is None or len(self.plan_rows) == 0:
            QMessageBox.warning(self, "Scan", "Load plan and apply column assignment first.")
            return
        source_raw_dirs = self._split_paths(self.ed_source_raw.text())
        source_proc_dirs = self._split_paths(self.ed_source_proc.text())
        if not source_raw_dirs and not source_proc_dirs:
            QMessageBox.warning(self, "Scan", "Provide at least one source folder (raw or processed).")
            return

        self._cancel_event.clear()
        self.emitter.busy.emit(True)
        self._append_log(f"Starting source scan (raw roots={len(source_raw_dirs)}, processed roots={len(source_proc_dirs)})...")
        self._persist_settings()

        def work():
            try:
                result = self.service.scan_sources(
                    source_raw_dirs=source_raw_dirs,
                    source_processed_dirs=source_proc_dirs,
                    case_sensitive=self.chk_case_sensitive.isChecked(),
                    cancel_event=self._cancel_event,
                    log_cb=lambda m: self.emitter.log_line.emit(m),
                    progress_cb=lambda stage, pct: self.emitter.progress.emit(stage, pct),
                )
                self.emitter.scan_done.emit(result)
            except Exception as e:
                self.emitter.error.emit(str(e))
            finally:
                self.emitter.busy.emit(False)

        run_in_thread(work)

    def _on_scan_done(self, result: ScanResult):
        self.scan_result = result
        self.match_plan = None
        self.last_actions = []
        self.last_summary = None
        self.lbl_raw_files.setText(f"Raw files: {len(result.raw_files)}")
        self.lbl_proc_files.setText(f"Processed files: {len(result.processed_files)}")
        ext_summary = self._summarize_extensions(result)
        self.lbl_extensions.setText(f"Extensions: {ext_summary}")
        self._populate_datatype_map(result)
        self.tbl_match_plan.setRowCount(0)
        self.tbl_unmatched.setRowCount(0)
        self._append_log("Scan completed.")

    def _summarize_extensions(self, result: ScanResult) -> str:
        merged: Dict[str, int] = {}
        for k, v in result.raw_ext_counts.items():
            merged[k] = merged.get(k, 0) + v
        for k, v in result.processed_ext_counts.items():
            merged[k] = merged.get(k, 0) + v
        top = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:8]
        if not top:
            return "-"
        return ", ".join([f"{ext or '<noext>'}:{cnt}" for ext, cnt in top])

    def _populate_datatype_map(self, result: ScanResult):
        labels = sorted(
            {
                f.datatype_detected
                for f in (result.raw_files + result.processed_files)
                if f.datatype_detected
            }
        )
        self.tbl_dtype_map.setRowCount(0)
        for label in labels:
            r = self.tbl_dtype_map.rowCount()
            self.tbl_dtype_map.insertRow(r)
            self.tbl_dtype_map.setItem(r, 0, QTableWidgetItem(label))
            self.tbl_dtype_map.setItem(r, 1, QTableWidgetItem(label))
        self._apply_loaded_datatype_map_override()
        self.tbl_dtype_map.resizeColumnsToContents()

    def _collect_datatype_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for r in range(self.tbl_dtype_map.rowCount()):
            src = _selected_items_text(self.tbl_dtype_map, r, 0).strip()
            dst = _selected_items_text(self.tbl_dtype_map, r, 1).strip()
            if src and dst:
                mapping[src] = dst
        return mapping

    def _apply_loaded_datatype_map_override(self):
        if not self._loaded_datatype_map_override:
            return
        for r in range(self.tbl_dtype_map.rowCount()):
            src = _selected_items_text(self.tbl_dtype_map, r, 0).strip()
            if src and src in self._loaded_datatype_map_override:
                self.tbl_dtype_map.setItem(r, 1, QTableWidgetItem(self._loaded_datatype_map_override[src]))

    def _table_select_values(self, tbl: QTableWidget, values: List[str]):
        chosen = {str(v).strip() for v in values if str(v).strip()}
        tbl.clearSelection()
        if not chosen:
            return
        for r in range(tbl.rowCount()):
            item = tbl.item(r, 0)
            if item and item.text().strip() in chosen:
                item.setSelected(True)

    def _collect_exportable_config(self) -> Dict[str, Any]:
        raw_dirs = self._split_paths(self.ed_source_raw.text())
        proc_dirs = self._split_paths(self.ed_source_proc.text())
        return {
            "version": 1,
            "project_mode": self.cb_project_mode.currentText(),
            "existing_project": self.cb_existing_project.currentText().strip(),
            "new_project_name": self.ed_new_project.text().strip(),
            "experiment_name": self.ed_experiment.text().strip(),
            "source_raw_dirs": raw_dirs,
            "source_processed_dirs": proc_dirs,
            "source_raw_dir": self._join_paths(raw_dirs),
            "source_processed_dir": self._join_paths(proc_dirs),
            "target_raw_root": self.ed_target_raw.text().strip(),
            "target_processed_root": self.ed_target_proc.text().strip(),
            "plan_path": self.ed_plan_path.text().strip(),
            "preserve_group_hierarchy": self.chk_preserve_group.isChecked(),
            "overwrite_strategy": self.cb_overwrite.currentText().strip().lower(),
            "overwrite_confirm": self.chk_overwrite_confirm.isChecked(),
            "dry_run": self.chk_dry_run.isChecked(),
            "verify_size": self.chk_verify_size.isChecked(),
            "column_assignment": {
                "subject_col": self._combo_value(self.cb_subject),
                "session_col": self._combo_value(self.cb_session),
                "trial_col": self._combo_value(self.cb_trial),
                "genotype_col": self._combo_value(self.cb_genotype),
                "condition_col": self._combo_value(self.cb_condition),
                "match_mode": self.cb_match_mode.currentText(),
                "case_sensitive": self.chk_case_sensitive.isChecked(),
                "extra_cols": self._table_checked_values(self.lst_extra_cols),
                "custom_match_cols": self._table_selected_values(self.lst_custom_match_cols),
            },
            "datatype_map": self._collect_datatype_map(),
        }

    def _save_config_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save data reorganizer config",
            "data_reorganizer_config.json",
            "JSON (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        payload = self._collect_exportable_config()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "Save config", str(e))
            return
        self._append_log(f"Saved config: {path}")

    def _load_config_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load data reorganizer config",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load config", str(e))
            return
        if not isinstance(data, dict):
            QMessageBox.critical(self, "Load config", "Invalid config format.")
            return

        raw_dirs = data.get("source_raw_dirs", data.get("source_raw_dir", ""))
        proc_dirs = data.get("source_processed_dirs", data.get("source_processed_dir", ""))
        self.ed_source_raw.setText(self._join_paths(self._coerce_paths_value(raw_dirs)))
        self.ed_source_proc.setText(self._join_paths(self._coerce_paths_value(proc_dirs)))
        self.ed_target_raw.setText(str(data.get("target_raw_root", "")))
        self.ed_target_proc.setText(str(data.get("target_processed_root", "")))
        self.ed_plan_path.setText(str(data.get("plan_path", "")))
        self.ed_experiment.setText(str(data.get("experiment_name", "")))
        self.ed_new_project.setText(str(data.get("new_project_name", "")))
        self.chk_dry_run.setChecked(bool(data.get("dry_run", True)))
        self.chk_verify_size.setChecked(bool(data.get("verify_size", False)))
        self.chk_preserve_group.setChecked(bool(data.get("preserve_group_hierarchy", False)))
        self.chk_overwrite_confirm.setChecked(bool(data.get("overwrite_confirm", False)))

        mode = str(data.get("project_mode", "Use existing project"))
        mode_idx = self.cb_project_mode.findText(mode)
        if mode_idx >= 0:
            self.cb_project_mode.setCurrentIndex(mode_idx)

        self._refresh_project_list()
        existing = str(data.get("existing_project", "")).strip()
        if existing:
            self.cb_existing_project.setCurrentText(existing)

        strategy = str(data.get("overwrite_strategy", "skip")).strip().lower()
        if strategy in ("skip", "rename", "overwrite"):
            self.cb_overwrite.setCurrentText(strategy)

        self._update_project_mode_ui()
        self._update_resolved_paths()

        col_cfg = data.get("column_assignment", {})
        if not isinstance(col_cfg, dict):
            col_cfg = {}

        plan_path = self.ed_plan_path.text().strip()
        if plan_path and os.path.exists(plan_path):
            self._load_plan()

            def set_cb(cb: QComboBox, value: str):
                v = str(value or "").strip()
                cb.setCurrentText(v if v else "(none)")

            set_cb(self.cb_subject, str(col_cfg.get("subject_col", "")))
            set_cb(self.cb_session, str(col_cfg.get("session_col", "")))
            set_cb(self.cb_trial, str(col_cfg.get("trial_col", "")))
            set_cb(self.cb_genotype, str(col_cfg.get("genotype_col", "")))
            set_cb(self.cb_condition, str(col_cfg.get("condition_col", "")))

            mm = str(col_cfg.get("match_mode", "")).strip()
            mm_idx = self.cb_match_mode.findText(mm)
            if mm_idx >= 0:
                self.cb_match_mode.setCurrentIndex(mm_idx)

            self.chk_case_sensitive.setChecked(bool(col_cfg.get("case_sensitive", False)))
            self._table_check_values(self.lst_extra_cols, [str(x) for x in (col_cfg.get("extra_cols") or [])])
            self._table_select_values(self.lst_custom_match_cols, [str(x) for x in (col_cfg.get("custom_match_cols") or [])])

            if self._combo_value(self.cb_subject):
                self._apply_column_assignment()

        dtype_map = data.get("datatype_map", {})
        if not isinstance(dtype_map, dict):
            dtype_map = {}
        self._loaded_datatype_map_override = {
            str(k): str(v)
            for k, v in dtype_map.items()
            if str(k).strip() and str(v).strip()
        }
        self._apply_loaded_datatype_map_override()
        self._append_log(f"Loaded config: {path}")
        self._persist_settings()

    def _build_match_plan(self):
        if not self.plan_rows:
            QMessageBox.warning(self, "Match plan", "Load plan and apply column assignment first.")
            return
        if self.scan_result is None:
            QMessageBox.warning(self, "Match plan", "Scan source folders first.")
            return

        self._cancel_event.clear()
        self.emitter.busy.emit(True)
        self._append_log("Building match plan...")
        self._persist_settings()

        def work():
            try:
                assignment = self._build_assignment()
                plan = self.service.build_match_plan(
                    plan_rows=self.plan_rows,
                    scan_result=self.scan_result,
                    case_sensitive=assignment.case_sensitive,
                    cancel_event=self._cancel_event,
                    log_cb=lambda m: self.emitter.log_line.emit(m),
                    progress_cb=lambda stage, pct: self.emitter.progress.emit(stage, pct),
                )
                self.emitter.match_done.emit(plan)
            except Exception as e:
                self.emitter.error.emit(str(e))
            finally:
                self.emitter.busy.emit(False)

        run_in_thread(work)

    def _on_match_done(self, match_plan: MatchPlan):
        self.match_plan = match_plan
        self.last_actions = []
        self.last_summary = None
        self._populate_match_table(match_plan.row_results)
        self._populate_unmatched_table(match_plan)
        self._append_log("Match plan ready.")

    def _populate_match_table(self, row_results):
        self.tbl_match_plan.setRowCount(0)
        for rr in row_results:
            r = self.tbl_match_plan.rowCount()
            self.tbl_match_plan.insertRow(r)
            self.tbl_match_plan.setItem(r, 0, QTableWidgetItem(str(rr.row_index)))
            self.tbl_match_plan.setItem(r, 1, QTableWidgetItem(rr.subject_id))
            self.tbl_match_plan.setItem(r, 2, QTableWidgetItem(rr.session_id))
            self.tbl_match_plan.setItem(r, 3, QTableWidgetItem(rr.match_key))
            self.tbl_match_plan.setItem(r, 4, QTableWidgetItem(str(len(rr.raw_matches))))
            self.tbl_match_plan.setItem(r, 5, QTableWidgetItem(str(len(rr.processed_matches))))
            self.tbl_match_plan.setItem(r, 6, QTableWidgetItem(" | ".join(rr.warnings)))
        self.tbl_match_plan.resizeColumnsToContents()

    def _populate_unmatched_table(self, match_plan: MatchPlan):
        self.tbl_unmatched.setRowCount(0)
        for fid in match_plan.unmatched_raw_file_ids:
            f = match_plan.files_index.get(fid)
            if not f:
                continue
            self._add_unmatched_row(f.source_kind, f.abs_path, "unmatched")

        for fid in match_plan.unmatched_processed_file_ids:
            f = match_plan.files_index.get(fid)
            if not f:
                continue
            self._add_unmatched_row(f.source_kind, f.abs_path, "unmatched")

        for fid in sorted(match_plan.conflict_file_ids):
            f = match_plan.files_index.get(fid)
            if not f:
                continue
            self._add_unmatched_row(f.source_kind, f.abs_path, "conflict")
        self.tbl_unmatched.resizeColumnsToContents()

    def _add_unmatched_row(self, source_kind: str, source_path: str, reason: str):
        r = self.tbl_unmatched.rowCount()
        self.tbl_unmatched.insertRow(r)
        self.tbl_unmatched.setItem(r, 0, QTableWidgetItem(source_kind))
        self.tbl_unmatched.setItem(r, 1, QTableWidgetItem(source_path))
        self.tbl_unmatched.setItem(r, 2, QTableWidgetItem(reason))

    def _apply_plan_filter(self):
        if self.match_plan is None:
            return
        mode = self.cb_plan_filter.currentText()
        if mode == "all rows":
            rows = self.match_plan.row_results
        elif mode == "only unmatched":
            rows = [rr for rr in self.match_plan.row_results if len(rr.raw_matches) == 0 and len(rr.processed_matches) == 0]
        else:
            rows = [rr for rr in self.match_plan.row_results if len(rr.conflict_file_ids) > 0]
        self._populate_match_table(rows)

    def _build_config(self) -> ReorganizerConfig:
        assignment = self._build_assignment()
        raw_dirs = self._split_paths(self.ed_source_raw.text())
        proc_dirs = self._split_paths(self.ed_source_proc.text())
        return ReorganizerConfig(
            project_name=self._selected_project_name(),
            experiment_name=self.ed_experiment.text().strip(),
            source_raw_dir=(raw_dirs[0] if raw_dirs else ""),
            source_processed_dir=(proc_dirs[0] if proc_dirs else ""),
            target_raw_root=self.ed_target_raw.text().strip(),
            target_processed_root=self.ed_target_proc.text().strip(),
            column_assignment=assignment,
            source_raw_dirs=raw_dirs,
            source_processed_dirs=proc_dirs,
            datatype_map=self._collect_datatype_map(),
            dry_run=self.chk_dry_run.isChecked(),
            preserve_group_hierarchy=self.chk_preserve_group.isChecked(),
            verify_size=self.chk_verify_size.isChecked(),
            overwrite_strategy=self.cb_overwrite.currentText().strip().lower(),
            overwrite_confirm=self.chk_overwrite_confirm.isChecked(),
        )

    def _execute(self):
        msg = self._validate_setup_fields()
        if msg:
            QMessageBox.warning(self, "Execute", msg)
            return
        if not self.plan_rows or self.normalized_df is None:
            QMessageBox.warning(self, "Execute", "Load plan and apply columns first.")
            return
        if self.match_plan is None:
            QMessageBox.warning(self, "Execute", "Build a match plan first.")
            return

        try:
            config = self._build_config()
        except Exception as e:
            QMessageBox.critical(self, "Execute", str(e))
            return

        if config.dry_run:
            ans = QMessageBox.question(
                self,
                "Dry run enabled",
                "Dry run is enabled.\nNo files will be copied to destination folders.\n\nContinue with preview run?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ans != QMessageBox.Yes:
                return

        self._cancel_event.clear()
        self.emitter.busy.emit(True)
        self._append_log("Starting execution...")
        self._persist_settings()

        def work():
            try:
                actions, summary = self.service.execute(
                    config=config,
                    plan_rows=self.plan_rows,
                    normalized_df=self.normalized_df,
                    match_plan=self.match_plan,
                    cancel_event=self._cancel_event,
                    log_cb=lambda m: self.emitter.log_line.emit(m),
                    progress_cb=lambda stage, pct: self.emitter.progress.emit(stage, pct),
                )
                self.emitter.execute_done.emit(actions, summary)
            except Exception as e:
                self.emitter.error.emit(str(e))
            finally:
                self.emitter.busy.emit(False)

        run_in_thread(work)

    def _update_metadata_only(self):
        msg = self._validate_setup_fields()
        if msg:
            QMessageBox.warning(self, "Update metadata", msg)
            return
        if not self.plan_rows or self.normalized_df is None:
            QMessageBox.warning(self, "Update metadata", "Load plan and apply columns first.")
            return
        if self.match_plan is None:
            QMessageBox.warning(self, "Update metadata", "Build a match plan first.")
            return

        if not self.last_actions:
            ans = QMessageBox.question(
                self,
                "No copy actions",
                "No previous copy actions were found in memory.\n"
                "MetaMan can rebuild a planned action list from current match plan without copying files.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ans != QMessageBox.Yes:
                return

        try:
            config = self._build_config()
        except Exception as e:
            QMessageBox.critical(self, "Update metadata", str(e))
            return
        config.dry_run = False

        # Re-apply the current column assignment so newly checked metadata columns are included.
        try:
            assignment = self._build_assignment()
            if self.plan_load_result is None:
                raise ValueError("Load a metadata plan first.")
            self.plan_rows, self.normalized_df = self.service.normalize_plan(self.plan_load_result, assignment)
            preview_lines = []
            for row in self.plan_rows[:10]:
                preview_lines.append(
                    f"row={row.row_index} subject={row.subject_id} session={row.session_id} key={row.match_key or '<unmatchable>'}"
                )
            self.txt_key_preview.setPlainText("\n".join(preview_lines))
            self._append_log("Column assignment re-applied before metadata update.")
        except Exception as e:
            QMessageBox.critical(self, "Update metadata", f"Could not re-apply column assignment:\n{e}")
            return

        run_logs = [ln for ln in self.txt_log.toPlainText().splitlines() if ln.strip()]
        self._cancel_event.clear()
        self.emitter.busy.emit(True)
        self._append_log("Updating metadata outputs (without copying files)...")
        self._persist_settings()

        def work():
            try:
                actions_for_metadata = list(self.last_actions or [])
                if not actions_for_metadata:
                    self.emitter.log_line.emit("Rebuilding planned actions from current match plan (no copy)...")
                    actions_for_metadata = self.service.plan_actions(
                        config=config,
                        plan_rows=self.plan_rows,
                        normalized_df=self.normalized_df,
                        match_plan=self.match_plan,
                        cancel_event=self._cancel_event,
                        log_cb=lambda m: self.emitter.log_line.emit(m),
                        progress_cb=lambda stage, pct: self.emitter.progress.emit(stage, pct),
                    )

                if not actions_for_metadata:
                    raise RuntimeError(
                        "No planned/copied files found to write into session metadata.\n"
                        "Build match plan again and ensure files are matched."
                    )

                writer = MetadataWriter()
                writer.write_outputs(
                    config=config,
                    normalized_df=self.normalized_df,
                    plan_rows=self.plan_rows,
                    match_plan=self.match_plan,
                    actions=actions_for_metadata,
                    run_logs=run_logs,
                    cancel_event=self._cancel_event,
                    log_cb=lambda m: self.emitter.log_line.emit(m),
                )
                self.last_actions = actions_for_metadata
                self.emitter.log_line.emit("Metadata update finished.")
            except Exception as e:
                self.emitter.error.emit(str(e))
            finally:
                self.emitter.busy.emit(False)

        run_in_thread(work)

    def _request_cancel(self):
        self._cancel_event.set()
        self._append_log("Cancel requested.")

    def _on_execute_done(self, actions, summary):
        self.last_actions = actions or []
        self.last_summary = summary
        if getattr(summary, "dry_run", False):
            self.lbl_summary.setText(
                "Summary (dry run): "
                f"subjects={summary.total_subjects}, sessions={summary.total_sessions}, "
                f"planned_raw={summary.raw_files_planned}, planned_processed={summary.processed_files_planned}, "
                f"unmatched={summary.unmatched_files}, conflicts={summary.conflict_files}, "
                f"skipped={summary.skipped_files}, errors={summary.error_files}, cancelled={summary.cancelled}"
            )
        else:
            self.lbl_summary.setText(
                "Summary: "
                f"subjects={summary.total_subjects}, sessions={summary.total_sessions}, "
                f"copied_raw={summary.raw_files_copied}, copied_processed={summary.processed_files_copied}, "
                f"unmatched={summary.unmatched_files}, conflicts={summary.conflict_files}, "
                f"skipped={summary.skipped_files}, errors={summary.error_files}, cancelled={summary.cancelled}"
            )
        self._append_log("Execution finished.")

    def _export_match_report(self):
        if self.match_plan is None:
            QMessageBox.warning(self, "Export report", "Build a match plan first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export match report", "match_report.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        writer = MetadataWriter()
        rows = writer._build_match_report_rows(self.match_plan, self.last_actions or [])
        writer._write_dict_rows_csv(rows, path)
        self._append_log(f"Exported match report: {path}")

    def _load_prefs(self):
        data = self.app_state.settings.get_data_reorganizer_settings()
        raw_dirs = data.get("source_raw_dirs", data.get("source_raw_dir", ""))
        proc_dirs = data.get("source_processed_dirs", data.get("source_processed_dir", ""))
        self.ed_source_raw.setText(self._join_paths(self._coerce_paths_value(raw_dirs)))
        self.ed_source_proc.setText(self._join_paths(self._coerce_paths_value(proc_dirs)))
        self.ed_target_raw.setText(data.get("target_raw_root", self.app_state.settings.raw_root))
        self.ed_target_proc.setText(data.get("target_processed_root", self.app_state.settings.processed_root))
        self.ed_plan_path.setText(data.get("plan_path", ""))
        self.ed_experiment.setText(data.get("experiment_name", ""))
        self.ed_new_project.setText(data.get("new_project_name", ""))
        self.chk_dry_run.setChecked(bool(data.get("dry_run", True)))
        self.chk_verify_size.setChecked(bool(data.get("verify_size", False)))
        self.chk_preserve_group.setChecked(bool(data.get("preserve_group_hierarchy", False)))
        strategy = str(data.get("overwrite_strategy", "skip"))
        if strategy in ("skip", "rename", "overwrite"):
            self.cb_overwrite.setCurrentText(strategy)
        self.chk_overwrite_confirm.setChecked(bool(data.get("overwrite_confirm", False)))
        mode = str(data.get("project_mode", "Use existing project"))
        idx = self.cb_project_mode.findText(mode)
        if idx >= 0:
            self.cb_project_mode.setCurrentIndex(idx)
        self._update_project_mode_ui()
        self._update_resolved_paths()

    def _persist_settings(self):
        raw_dirs = self._split_paths(self.ed_source_raw.text())
        proc_dirs = self._split_paths(self.ed_source_proc.text())
        data = {
            "source_raw_dirs": raw_dirs,
            "source_processed_dirs": proc_dirs,
            "source_raw_dir": self._join_paths(raw_dirs),
            "source_processed_dir": self._join_paths(proc_dirs),
            "target_raw_root": self.ed_target_raw.text().strip(),
            "target_processed_root": self.ed_target_proc.text().strip(),
            "plan_path": self.ed_plan_path.text().strip(),
            "experiment_name": self.ed_experiment.text().strip(),
            "new_project_name": self.ed_new_project.text().strip(),
            "project_mode": self.cb_project_mode.currentText(),
            "dry_run": self.chk_dry_run.isChecked(),
            "verify_size": self.chk_verify_size.isChecked(),
            "preserve_group_hierarchy": self.chk_preserve_group.isChecked(),
            "overwrite_strategy": self.cb_overwrite.currentText().strip().lower(),
            "overwrite_confirm": self.chk_overwrite_confirm.isChecked(),
        }
        self.app_state.settings.put_data_reorganizer_settings(data)
