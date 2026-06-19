"""Search workspace tool.

A first-class nav-rail page for searching and analysing the active project. It
drives the same query engine as the menu "Find Sessions" dialog
(:mod:`MetaMan.services.query`), but as a persistent workspace with:

* a quick free-text box + up to four structured ``field / operator / value``
  conditions (AND or ANY),
* an optional "scan files" pass so you can query on derived fields
  (``Auto: modality`` / size / sample rate),
* a results table (double-click loads a session into Record / Process),
* a details pane showing the selected session's full metadata,
* one-click CSV export of the result table,
* a "Scrape project now" button that enriches every session's metadata.

All disk work runs on a worker thread, so the window never freezes on a slow
network share.
"""

import json
import os
from typing import Any, Dict, List

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
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

from ..services.query import OPERATORS, ProjectQuery
from ..services.scrape_ops import scrape_project
from ..state import AppState
from ..utils import run_in_thread

# Editable suggestions for the field boxes; users can type any field name.
_COMMON_FIELDS = [
    "Subject", "Experiment", "Session", "Condition", "Region", "Trial", "Arena",
    "DateTime", "Experimenter", "Recording",
    "Auto: modality", "Auto: total size", "Auto: sample rate (Hz)",
    "Auto: channels", "Auto: audio kind",
]


class _SearchEmitter(QObject):
    results = Signal(list, dict)   # records, summary
    failed = Signal(str)
    scraped = Signal(dict)
    log = Signal(str)


class SearchTab(QWidget):
    def __init__(self, app_state: AppState, main_window):
        super().__init__()
        self.app_state = app_state
        self.main = main_window
        self._records: List[Dict[str, Any]] = []
        self._busy = False
        self._emit = _SearchEmitter()
        self._emit.results.connect(self._on_results)
        self._emit.failed.connect(self._on_failed)
        self._emit.scraped.connect(self._on_scraped)
        self._emit.log.connect(self._set_status)
        self._build_ui()

    # ── active project ────────────────────────────────────────────────
    def _active_project(self) -> str:
        return (self.app_state.current_project
                or self.app_state.settings.get_loaded_project().get("name", "")).strip()

    def _project_dir(self) -> str:
        proj = self._active_project()
        return os.path.join(self.app_state.settings.raw_root, proj) if proj else ""

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("\U0001f50d  Search")
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #5a3fe0;")
        header.addWidget(title)
        header.addStretch(1)
        self.lbl_project = QLabel("")
        self.lbl_project.setObjectName("Hint")
        header.addWidget(self.lbl_project)
        root.addLayout(header)

        self.ed_text = QLineEdit()
        self.ed_text.setPlaceholderText("Quick search across all metadata (e.g. ultrasonic, CA1, 51542)…  press Enter")
        self.ed_text.returnPressed.connect(self._run_search)
        root.addWidget(self.ed_text)

        # structured conditions
        self._cond_rows = []
        for i in range(4):
            row = QHBoxLayout()
            field = QComboBox()
            field.setEditable(True)
            field.addItem("")
            field.addItems(_COMMON_FIELDS)
            field.setCurrentText("")
            op = QComboBox()
            op.addItems(OPERATORS)
            value = QLineEdit()
            value.setPlaceholderText("value")
            value.returnPressed.connect(self._run_search)
            row.addWidget(QLabel("and" if i else "where"), 0)
            row.addWidget(field, 3)
            row.addWidget(op, 1)
            row.addWidget(value, 3)
            root.addLayout(row)
            self._cond_rows.append((field, op, value))

        opts = QHBoxLayout()
        self.chk_any = QCheckBox("Match ANY condition (OR)")
        opts.addWidget(self.chk_any)
        self.chk_scan = QCheckBox("Scan files for modality / size (slower)")
        opts.addWidget(self.chk_scan)
        opts.addStretch(1)
        self.btn_search = QPushButton("Search")
        self.btn_search.setObjectName("Primary")
        self.btn_search.clicked.connect(self._run_search)
        opts.addWidget(self.btn_search)
        self.btn_export = QPushButton("Export CSV…")
        self.btn_export.clicked.connect(self._export_csv)
        opts.addWidget(self.btn_export)
        self.btn_scrape = QPushButton("\U0001f504  Scrape project now")
        self.btn_scrape.setToolTip("Scan every session and refresh its auto-detected metadata (modality, size, probe, audio…)")
        self.btn_scrape.clicked.connect(self._scrape_now)
        opts.addWidget(self.btn_scrape)
        root.addLayout(opts)

        split = QSplitter(Qt.Horizontal)
        self.tbl = QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(
            ["Subject", "Experiment", "Session", "Modality", "Size", "Date", "Source"])
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.tbl.itemSelectionChanged.connect(self._on_row_selected)
        self.tbl.itemDoubleClicked.connect(self._on_double_click)
        split.addWidget(self.tbl)

        self.txt_detail = QTextEdit()
        self.txt_detail.setReadOnly(True)
        self.txt_detail.setPlaceholderText("Select a result to see its full metadata.\nDouble-click to load it into Record / Process.")
        split.addWidget(self.txt_detail)
        split.setSizes([760, 420])
        root.addWidget(split, 1)

        self.lbl_status = QLabel("Pick a project, then Search. Leave conditions blank to list every session.")
        self.lbl_status.setObjectName("Hint")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.refresh()

    # ── public ────────────────────────────────────────────────────────
    def refresh(self):
        proj = self._active_project()
        self.lbl_project.setText(f"Project: {proj}" if proj else "No project loaded")

    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)

    # ── query building ────────────────────────────────────────────────
    def _filters(self):
        out = []
        for field, op, value in self._cond_rows:
            f = field.currentText().strip()
            if f:
                out.append((f, op.currentText(), value.text().strip()))
        return out

    def _build_query(self) -> ProjectQuery:
        pq = ProjectQuery(self._project_dir(), scrape=self.chk_scan.isChecked())
        any_mode = self.chk_any.isChecked()
        for f, o, v in self._filters():
            pq = pq.or_where(f, o, v) if any_mode else pq.where(f, o, v)
        text = self.ed_text.text().strip().lower()
        if text:
            def has_text(rec, t=text):
                for k, val in rec["meta"].items():
                    s = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False) \
                        if isinstance(val, (dict, list)) else str(val)
                    if t in (k.lower() + " " + s.lower()):
                        return True
                return False
            pq = pq.filter(has_text)
        return pq

    # ── search ────────────────────────────────────────────────────────
    def _run_search(self):
        if self._busy:
            return
        project_dir = self._project_dir()
        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(self, "Search", "Select a project first (top bar).")
            return
        self._set_busy(True, "Searching…")
        pq = self._build_query()

        def work():
            try:
                recs = pq.records()
                summary = pq.summary()
                self._emit.results.emit(recs, summary)
            except Exception as exc:
                self._emit.failed.emit(str(exc))

        run_in_thread(work)

    def _on_results(self, records: List[Dict[str, Any]], summary: Dict[str, Any]):
        self._records = records
        self.tbl.setRowCount(0)
        for rec in records:
            m = rec["meta"]
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            cells = [
                rec["subject"], rec["experiment"], rec["session"],
                str(m.get("Auto: modality", "")), str(m.get("Auto: total size", "")),
                str(m.get("DateTime", "")), rec["metadata_file"] or "(none)",
            ]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 0:
                    it.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, c, it)
        self.tbl.resizeColumnsToContents()
        self._set_busy(False)
        subs = summary.get("subjects_count", 0)
        self.lbl_status.setText(
            f"{len(records)} session(s) · {subs} subject(s) · "
            f"modalities: {', '.join(summary.get('modalities', [])) or '—'}. "
            "Double-click a row to load it into Record / Process.")

    def _on_failed(self, msg: str):
        self._set_busy(False)
        QMessageBox.critical(self, "Search failed", msg)

    # ── result interaction ────────────────────────────────────────────
    def _selected_record(self):
        items = self.tbl.selectedItems()
        if not items:
            return None
        return self.tbl.item(items[0].row(), 0).data(Qt.UserRole)

    def _on_row_selected(self):
        rec = self._selected_record()
        if rec:
            self.txt_detail.setPlainText(json.dumps(rec["meta"], indent=2, ensure_ascii=False))

    def _on_double_click(self, _item):
        rec = self._selected_record()
        if not rec:
            return
        try:
            self.main._activate_project_everywhere(rec["project"])
            if rec["has_metadata"]:
                self.main._load_session_everywhere(rec["path"])
            self._set_status(f"Loaded {rec['subject']}/{rec['experiment']}/{rec['session']} into Record / Process.")
        except Exception as exc:
            QMessageBox.warning(self, "Load", f"Could not load session:\n{exc}")

    # ── export ────────────────────────────────────────────────────────
    def _export_csv(self):
        if not self._records:
            QMessageBox.information(self, "Export", "Run a search first.")
            return
        proj = self._active_project() or "project"
        path, _ = QFileDialog.getSaveFileName(self, "Export results", f"{proj}_sessions.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            self._build_query().to_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export", f"Could not write CSV:\n{exc}")
            return
        self._set_status(f"Exported {len(self._records)} session(s) to {path}")

    # ── scrape now ────────────────────────────────────────────────────
    def _scrape_now(self):
        if self._busy:
            return
        project_dir = self._project_dir()
        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(self, "Scrape", "Select a project first (top bar).")
            return
        self._set_busy(True, "Scraping project metadata…")

        def work():
            try:
                stats = scrape_project(project_dir, deep=True, only_missing=False,
                                       progress=lambda m: self._emit.log.emit(f"[scrape] {m}"))
                self._emit.scraped.emit(stats)
            except Exception as exc:
                self._emit.failed.emit(str(exc))

        run_in_thread(work)

    def _on_scraped(self, stats: Dict[str, int]):
        self._set_busy(False)
        self._set_status(
            f"Scrape complete: {stats.get('updated', 0)} updated, "
            f"{stats.get('skipped', 0)} unchanged, {stats.get('errors', 0)} error(s).")
        try:
            self.main.nav_tab.refresh_tree(collapsed=True, lazy=True)
        except Exception:
            pass
        self._run_search()

    # ── busy / status ─────────────────────────────────────────────────
    def _set_busy(self, busy: bool, message: str = ""):
        self._busy = busy
        for w in (self.btn_search, self.btn_scrape, self.btn_export):
            w.setEnabled(not busy)
        if message:
            self._set_status(message)

    def _set_status(self, message: str):
        self.lbl_status.setText(message)
