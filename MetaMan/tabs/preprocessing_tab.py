import json
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QListWidget, QTextEdit, QComboBox, QMessageBox, QFileDialog, QSplitter,
    QTableWidget, QTableWidgetItem, QInputDialog
)

from ..state import AppState
from ..io_ops import (
    list_experiments,
    list_projects,
    list_sessions,
    list_subjects,
    save_session_triplet,
    load_session_metadata,
)

# Step menus (kept as requested)
NPX_STEPS = ["spike_sorting", "curation", "histology", "time_sync", "dlc", "add_new_step"]
FIBER_STEPS = ["artefact_removal", "delta_F/F", "time_sync", "dlc", "add_new_step"]
BEHAV_STEPS = ["manual_scoring", "DLC", "lisbet", "dlc", "add_new_step"]


def dict_to_table(tbl: QTableWidget, data: Dict[str, Any]):
    tbl.setRowCount(0)
    for k, v in data.items():
        r = tbl.rowCount()
        tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem(str(k)))
        if isinstance(v, (dict, list)):
            tbl.setItem(r, 1, QTableWidgetItem(json.dumps(v, ensure_ascii=False)))
        else:
            tbl.setItem(r, 1, QTableWidgetItem("" if v is None else str(v)))
    tbl.resizeColumnsToContents()


class PreprocessingTab(QWidget):
    """
    Three-pane layout using a horizontal splitter:
      LEFT   : Steps list & controls
      CENTER : Parameters/Comments + Import params + Results folder
      RIGHT  : Session Info (read-only key/value table)

    - 'add_new_step' lets the user name a custom step.
    - 'Import params (CSV/JSON)' loads that file into the selected step's params.
    - 'Results folder' lets you store a path per step as 'results_dir'.
    """

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.meta: Dict[str, Any] = {}
        self._loading_session = False
        self._build_ui()
        self._load_settings()
        self._refresh_lists()
        if self.app_state.current_session_path:
            self._load_from_session(self.app_state.current_session_path)
        else:
            last = self.app_state.settings.get_preprocessing_tab_settings().get("last_session_path", "")
            if last and os.path.isdir(last):
                self._load_from_session(last)

    # ------------------------ UI ------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Selector row
        row_sel0 = QHBoxLayout()
        row_sel0.addWidget(QLabel("Data root:"))
        self.ed_data_root = QLineEdit(self.app_state.settings.data_root)
        row_sel0.addWidget(self.ed_data_root, 1)
        b_browse_data = QPushButton("Browse...")
        b_browse_data.clicked.connect(self._choose_data_root)
        row_sel0.addWidget(b_browse_data)
        b_refresh = QPushButton("Refresh lists")
        b_refresh.clicked.connect(self._refresh_lists)
        row_sel0.addWidget(b_refresh)
        root.addLayout(row_sel0)

        row_sel1 = QHBoxLayout()
        row_sel1.addWidget(QLabel("Project"))
        self.cb_proj = QComboBox(); self.cb_proj.setEditable(False); row_sel1.addWidget(self.cb_proj, 1)
        row_sel1.addWidget(QLabel("Experiment"))
        self.cb_exp = QComboBox(); self.cb_exp.setEditable(False); row_sel1.addWidget(self.cb_exp, 1)
        row_sel1.addWidget(QLabel("Subject"))
        self.cb_sub = QComboBox(); self.cb_sub.setEditable(False); row_sel1.addWidget(self.cb_sub, 1)
        row_sel1.addWidget(QLabel("Session"))
        self.cb_sess = QComboBox(); self.cb_sess.setEditable(False); row_sel1.addWidget(self.cb_sess, 1)
        root.addLayout(row_sel1)

        for cb in (self.cb_proj, self.cb_exp, self.cb_sub, self.cb_sess):
            cb.setMaxVisibleItems(30)

        self.cb_proj.currentIndexChanged.connect(self._on_project_changed)
        self.cb_exp.currentIndexChanged.connect(self._on_experiment_changed)
        self.cb_sub.currentIndexChanged.connect(self._on_subject_changed)
        self.cb_proj.currentIndexChanged.connect(lambda _i: self._persist_settings())
        self.cb_exp.currentIndexChanged.connect(lambda _i: self._persist_settings())
        self.cb_sub.currentIndexChanged.connect(lambda _i: self._persist_settings())
        self.cb_sess.currentIndexChanged.connect(self._on_session_changed)
        self.cb_sess.currentIndexChanged.connect(lambda _i: self._persist_settings())
        self.ed_data_root.textChanged.connect(self._persist_settings)

        # Top bar: processed root + create folder (kept behavior)
        row = QHBoxLayout()
        row.addWidget(QLabel("Processed root:"))
        self.ed_proc_root = QLineEdit(self.app_state.settings.processed_root)
        self.ed_proc_root.textChanged.connect(self._persist_settings)
        row.addWidget(self.ed_proc_root, 1)
        b_create = QPushButton("Create folder")
        b_create.clicked.connect(self._create_processed_folder)
        row.addWidget(b_create)
        b_new = QPushButton("New preprocessing")
        b_new.clicked.connect(self._create_processed_folder)  # alias
        row.addWidget(b_new)
        root.addLayout(row)

        # Middle: 3 panels in a splitter
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # LEFT: steps & actions
        left = QWidget(); l = QVBoxLayout(left)
        l.addWidget(QLabel("Steps"))
        self.steps = QListWidget(); l.addWidget(self.steps, 1)
        rowL = QHBoxLayout()
        self.cb_step = QComboBox(); self.cb_step.addItems(BEHAV_STEPS); rowL.addWidget(self.cb_step)
        b_add = QPushButton("Add step"); b_add.clicked.connect(self._add_step); rowL.addWidget(b_add)
        b_done = QPushButton("Completed"); b_done.clicked.connect(self._mark_completed); rowL.addWidget(b_done)
        b_remove = QPushButton("Remove step"); b_remove.clicked.connect(self._remove_step); rowL.addWidget(b_remove)
        l.addLayout(rowL)
        split.addWidget(left)

        # CENTER: params/comments + import + results dir
        center = QWidget(); c = QVBoxLayout(center)

        # Params
        c.addWidget(QLabel("Parameters (JSON)"))
        self.txt_params = QTextEdit()
        c.addWidget(self.txt_params, 1)

        rowP = QHBoxLayout()
        b_save_params = QPushButton("Add/Update parameters"); b_save_params.clicked.connect(self._save_params)
        rowP.addWidget(b_save_params)
        b_import = QPushButton("Import params (CSV/JSON)"); b_import.clicked.connect(self._import_params_for_step)
        rowP.addWidget(b_import)
        c.addLayout(rowP)

        # Results folder selector per step
        c.addWidget(QLabel("Results folder (for this step)"))
        rowR = QHBoxLayout()
        self.ed_results_dir = QLineEdit()
        rowR.addWidget(self.ed_results_dir, 1)
        b_browse_results = QPushButton("Choose…"); b_browse_results.clicked.connect(self._select_results_dir)
        rowR.addWidget(b_browse_results)
        b_apply_results = QPushButton("Save results folder"); b_apply_results.clicked.connect(self._apply_results_dir)
        rowR.addWidget(b_apply_results)
        c.addLayout(rowR)

        # Comments
        c.addWidget(QLabel("Comments"))
        self.txt_comments = QTextEdit()
        c.addWidget(self.txt_comments, 1)
        b_save_comment = QPushButton("Save comment"); b_save_comment.clicked.connect(self._save_comment)
        c.addWidget(b_save_comment)

        split.addWidget(center)

        # RIGHT: session info panel (read-only)
        right = QWidget(); r = QVBoxLayout(right)
        r.addWidget(QLabel("Loaded Session Info"))
        self.tbl_session_info = QTableWidget(0, 2)
        self.tbl_session_info.setHorizontalHeaderLabels(["Key", "Value"])
        self.tbl_session_info.horizontalHeader().setStretchLastSection(True)
        r.addWidget(self.tbl_session_info, 1)
        split.addWidget(right)

        split.setSizes([350, 600, 450])

        # Hook step selection
        self.steps.currentRowChanged.connect(self._update_param_comment)

    def _choose_data_root(self):
        d = QFileDialog.getExistingDirectory(self, "Choose data root", self.ed_data_root.text() or "")
        if not d:
            return
        normalized = self._normalize_data_root(d)
        self.ed_data_root.setText(normalized)
        self.app_state.settings.data_root = normalized
        self.ed_proc_root.setText(self.app_state.settings.processed_root)
        self._refresh_lists()
        self._persist_settings()

    def _normalize_data_root(self, path: str) -> str:
        p = os.path.normpath(str(path or "").strip())
        if not p:
            return p
        leaf = os.path.basename(p).lower()
        if leaf in ("raw", "rawdata", "processed", "processeddata"):
            parent = os.path.dirname(p)
            if parent:
                return parent
        return p

    def _effective_data_root(self) -> str:
        root = self.app_state.settings.data_root or self.ed_data_root.text().strip()
        return self._normalize_data_root(root)

    def _prefer_data_root_as_raw(self) -> bool:
        data_root = self._effective_data_root()
        if not os.path.isdir(data_root):
            return False
        entries = [
            d
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ]
        if not entries:
            return False
        lowered = {d.lower() for d in entries}
        if "raw" in lowered or "processed" in lowered:
            return False
        return True

    def _raw_root(self) -> str:
        data_root = self._effective_data_root()
        candidate = os.path.join(data_root, "raw")
        if os.path.isdir(candidate):
            return candidate
        if self._prefer_data_root_as_raw():
            return data_root
        return candidate

    def _set_combo_items(self, cb: QComboBox, items: List[str], keep_text: str):
        cb.blockSignals(True)
        cb.clear()
        cb.addItems(items)
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

    def _refresh_lists(self):
        normalized = self._effective_data_root()
        if normalized:
            self.app_state.settings.data_root = normalized
        if normalized and normalized != self.ed_data_root.text().strip():
            self.ed_data_root.blockSignals(True)
            self.ed_data_root.setText(normalized)
            self.ed_data_root.blockSignals(False)
        keep_proj = self.cb_proj.currentText().strip()
        self._set_combo_items(self.cb_proj, list_projects(self._raw_root()), keep_proj)
        self._on_project_changed()

    def _on_project_changed(self, _index: int = -1):
        proj = self.cb_proj.currentText().strip()
        keep_exp = self.cb_exp.currentText().strip()
        items = list_experiments(os.path.join(self._raw_root(), proj)) if proj else []
        self._set_combo_items(self.cb_exp, items, keep_exp)
        self._on_experiment_changed()

    def _on_experiment_changed(self, _index: int = -1):
        proj = self.cb_proj.currentText().strip()
        exp = self.cb_exp.currentText().strip()
        keep_sub = self.cb_sub.currentText().strip()
        items = list_subjects(os.path.join(self._raw_root(), proj, exp)) if (proj and exp) else []
        self._set_combo_items(self.cb_sub, items, keep_sub)
        self._on_subject_changed()

    def _on_subject_changed(self, _index: int = -1):
        proj = self.cb_proj.currentText().strip()
        exp = self.cb_exp.currentText().strip()
        sub = self.cb_sub.currentText().strip()
        keep_sess = self.cb_sess.currentText().strip()
        items = list_sessions(os.path.join(self._raw_root(), proj, exp, sub)) if (proj and exp and sub) else []
        self._set_combo_items(self.cb_sess, items, keep_sess)
        self._persist_settings()

    def _on_session_changed(self, _index: int):
        if self._loading_session:
            return
        self._persist_settings()
        self._try_load_current_session(show_warning=False)

    def _selected_session_dir(self) -> str:
        return os.path.join(
            self._raw_root(),
            self.cb_proj.currentText().strip(),
            self.cb_exp.currentText().strip(),
            self.cb_sub.currentText().strip(),
            self.cb_sess.currentText().strip(),
        )

    def _try_load_current_session(self, show_warning: bool):
        session_dir = self._selected_session_dir()
        if not os.path.isdir(session_dir):
            if show_warning:
                QMessageBox.warning(self, "Session", f"Session folder not found:\n{session_dir}")
            return
        if self.app_state.current_session_path == session_dir and self.meta:
            return
        self._load_from_session(session_dir)
        self.app_state.set_current(
            project=self.cb_proj.currentText().strip(),
            experiment=self.cb_exp.currentText().strip(),
            animal=self.cb_sub.currentText().strip(),
            session=self.cb_sess.currentText().strip(),
            session_path=session_dir,
        )
        self._persist_settings()

    # ------------------------ Session load / step menu ------------------------

    def _determine_step_choices(self) -> List[str]:
        rec = (self.meta.get("Recording") or "").lower()
        if "npx" in rec or "neuro" in rec:
            return NPX_STEPS
        if "fiber" in rec:
            return FIBER_STEPS
        return BEHAV_STEPS

    def _load_from_session(self, session_dir: str):
        self.meta = load_session_metadata(session_dir) or {}
        self._loading_session = True
        try:
            if self.meta:
                proj = str(self.meta.get("Project", "")).strip()
                exp = str(self.meta.get("Experiment", "")).strip()
                sub = str(self.meta.get("Subject", "") or self.meta.get("Animal", "")).strip()
                sess = str(self.meta.get("Session", "")).strip()
                if proj:
                    self.cb_proj.setCurrentText(proj)
                    self._on_project_changed()
                if exp:
                    self.cb_exp.setCurrentText(exp)
                    self._on_experiment_changed()
                if sub:
                    self.cb_sub.setCurrentText(sub)
                    self._on_subject_changed()
                if sess:
                    self.cb_sess.setCurrentText(sess)
                self.app_state.set_current(project=proj, experiment=exp, animal=sub, session=sess, session_path=session_dir)
        finally:
            self._loading_session = False
        self._refresh_steps()
        # update choices
        self.cb_step.clear()
        self.cb_step.addItems(self._determine_step_choices())
        # refresh session info panel
        dict_to_table(self.tbl_session_info, self.meta)
        self._persist_settings()

    def _refresh_steps(self):
        self.steps.clear()
        for s in self.meta.get("preprocessing", []):
            name = s.get("name", "")
            status = s.get("status", "")
            self.steps.addItem(f"{name} [{status}]")

    def _current_step(self) -> Optional[Dict[str, Any]]:
        row = self.steps.currentRow()
        if row < 0:
            return None
        # robust match by name inside brackets
        text = self.steps.item(row).text()
        name = text.split(" [", 1)[0]
        for s in self.meta.get("preprocessing", []):
            if s.get("name") == name:
                return s
        return None

    # ------------------------ Top controls ------------------------

    def _create_processed_folder(self):
        """Create processed folder structure and copy current metadata triplet."""
        proj = self.meta.get("Project", "")
        exp = self.meta.get("Experiment", "")
        subject = self.meta.get("Subject", "") or self.meta.get("Animal", "")
        session = self.meta.get("Session", "")
        root = self.ed_proc_root.text().strip()
        if not (proj and exp and subject and session and root):
            QMessageBox.warning(self, "Missing", "Missing processed root or session info.")
            return
        session_proc = os.path.join(root, proj, exp, subject, session)
        os.makedirs(session_proc, exist_ok=True)
        save_session_triplet(session_proc, self.meta)
        self.app_state.settings.processed_root = root
        QMessageBox.information(self, "Created", f"Processed session created:\n{session_proc}")

    # ------------------------ Step actions ------------------------

    def _add_step(self):
        choice = self.cb_step.currentText().strip()
        if not choice:
            return
        if choice == "add_new_step":
            name, ok = QInputDialog.getText(self, "New step", "Step name:")
            if not ok or not name.strip():
                return
            step_name = name.strip()
        else:
            step_name = choice

        # Avoid duplicates with same name
        for s in self.meta.get("preprocessing", []):
            if s.get("name") == step_name:
                QMessageBox.information(self, "Exists", f"Step '{step_name}' already exists.")
                return

        step = {"name": step_name, "params": {}, "comments": "", "status": "in_progress"}
        self.meta.setdefault("preprocessing", []).append(step)
        save_session_triplet(self.app_state.current_session_path, self.meta)
        self._refresh_steps()
        # select newly added
        self.steps.setCurrentRow(self.steps.count() - 1)

    def _remove_step(self):
        cur = self._current_step()
        if not cur:
            return
        name = cur.get("name", "")
        steps = self.meta.get("preprocessing", [])
        steps = [s for s in steps if s.get("name") != name]
        self.meta["preprocessing"] = steps
        save_session_triplet(self.app_state.current_session_path, self.meta)
        self._refresh_steps()
        self.txt_params.clear()
        self.txt_comments.clear()
        self.ed_results_dir.clear()

    def _mark_completed(self):
        cur = self._current_step()
        if not cur:
            return
        cur["status"] = "completed"
        save_session_triplet(self.app_state.current_session_path, self.meta)
        self._refresh_steps()

    def _update_param_comment(self):
        cur = self._current_step()
        if not cur:
            return
        # Update center widgets from selected step
        self.txt_params.setText(json.dumps(cur.get("params", {}), indent=2))
        self.txt_comments.setText(cur.get("comments", ""))
        self.ed_results_dir.setText(cur.get("results_dir", ""))

    # ------------------------ Params & comments ------------------------

    def _save_params(self):
        cur = self._current_step()
        if not cur:
            return
        try:
            params = json.loads(self.txt_params.toPlainText() or "{}")
        except Exception as e:
            QMessageBox.critical(self, "JSON error", str(e))
            return
        cur["params"] = params
        save_session_triplet(self.app_state.current_session_path, self.meta)

    def _save_comment(self):
        cur = self._current_step()
        if not cur:
            return
        txt = self.txt_comments.toPlainText().strip()
        cur["comments"] = txt
        save_session_triplet(self.app_state.current_session_path, self.meta)

    # ------------------------ Import params (CSV/JSON) ------------------------

    def _import_params_for_step(self):
        """
        Load params for the selected step:
          - JSON: expect a dict (stored as-is) OR a list (stored as-is).
          - CSV : preferred formats:
                * two-column ['key','value'] (case-insensitive)
                * a single-row table -> dict of {col: value}
                * multi-row table -> list of row dicts
        """
        cur = self._current_step()
        if not cur:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose CSV/JSON", "", "Tables (*.csv *.json);;All files (*)")
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cur["params"] = data
            elif ext == ".csv":
                import pandas as pd
                # try robust encodings
                df = None
                for enc in ["utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin-1", "cp1252"]:
                    try:
                        df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
                        break
                    except Exception:
                        continue
                if df is None:
                    df = pd.read_csv(path)  # let it error if truly unreadable

                cols_lower = {str(c).strip().lower(): c for c in df.columns}
                if "key" in cols_lower and "value" in cols_lower:
                    kcol, vcol = cols_lower["key"], cols_lower["value"]
                    params = {}
                    for _, row in df.iterrows():
                        k = str(row.get(kcol, "")).strip()
                        if not k:
                            continue
                        v = row.get(vcol, "")
                        # try JSON parsing per-cell for structured values
                        try:
                            v = json.loads(v)
                        except Exception:
                            v = "" if (isinstance(v, float) and v != v) else v
                        params[k] = v
                    cur["params"] = params
                else:
                    # if single-row -> dict; else list of dicts
                    if len(df) == 1:
                        rec = df.iloc[0].to_dict()
                        # convert NaN → ""
                        for k, v in rec.items():
                            if isinstance(v, float) and v != v:
                                rec[k] = ""
                        cur["params"] = rec
                    else:
                        records = df.to_dict(orient="records")
                        for rec in records:
                            for k, v in list(rec.items()):
                                if isinstance(v, float) and v != v:
                                    rec[k] = ""
                        cur["params"] = records
            else:
                QMessageBox.warning(self, "Format", "Please choose a .csv or .json file.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not import params:\n{e}")
            return

        # Reflect in UI and save
        self.txt_params.setText(json.dumps(cur.get("params", {}), indent=2))
        save_session_triplet(self.app_state.current_session_path, self.meta)

    # ------------------------ Results folder per step ------------------------

    def _select_results_dir(self):
        cur = self._current_step()
        if not cur:
            return
        d = QFileDialog.getExistingDirectory(self, "Choose results folder", self.ed_results_dir.text() or "")
        if not d:
            return
        self.ed_results_dir.setText(d)

    def _apply_results_dir(self):
        cur = self._current_step()
        if not cur:
            return
        cur["results_dir"] = self.ed_results_dir.text().strip()
        save_session_triplet(self.app_state.current_session_path, self.meta)

    def _load_settings(self):
        data = self.app_state.settings.get_preprocessing_tab_settings()
        dr = self._normalize_data_root(self.app_state.settings.data_root)
        if dr:
            self.ed_data_root.setText(dr)
        pr = str(data.get("processed_root", "")).strip()
        if pr:
            self.ed_proc_root.setText(pr)
        for cb, k in (
            (self.cb_proj, "project"),
            (self.cb_exp, "experiment"),
            (self.cb_sub, "subject"),
            (self.cb_sess, "session"),
        ):
            v = str(data.get(k, "")).strip()
            if v:
                cb.setCurrentText(v)

    def _persist_settings(self):
        root = self._normalize_data_root(self.app_state.settings.data_root)
        self.app_state.settings.put_preprocessing_tab_settings(
            {
                "data_root": root,
                "processed_root": self.ed_proc_root.text().strip(),
                "project": self.cb_proj.currentText().strip(),
                "experiment": self.cb_exp.currentText().strip(),
                "subject": self.cb_sub.currentText().strip(),
                "session": self.cb_sess.currentText().strip(),
                "last_session_path": self.app_state.current_session_path or self._selected_session_dir(),
            }
        )
