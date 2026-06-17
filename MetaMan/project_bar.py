"""Global active-project bar.

A persistent header strip above the tabs. It owns the one notion of the
"active project" for the whole app: picking a project here drives Browse,
Record, Process and Transfer. It also surfaces the data root and the project's
folder structure (e.g. ``Subject  >  Experiment  >  Session``).
"""

import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .io_ops import list_projects
from .services.structure_schema import (
    is_marker_level,
    level_label,
    marker_folder_name,
    sublevels,
)


class ProjectContextBar(QWidget):
    """Active-project picker + data root + structure summary."""

    project_selected = Signal(str)        # user chose a project
    design_structure_requested = Signal()
    set_data_root_requested = Signal()

    def __init__(self, app_state, parent=None, compact=False):
        super().__init__(parent)
        self.app_state = app_state
        self._loading = False
        self._compact = compact
        self.setObjectName("ProjectBarCompact" if compact else "ProjectBar")
        if compact:
            self._build_ui_compact()
        else:
            self._build_ui()
        self.refresh()

    # ── build (vertical, folded into the nav rail) ─────────────────────
    def _build_ui_compact(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(6)

        cap = QLabel("PROJECT")
        cap.setObjectName("NavSection")
        lay.addWidget(cap)

        self.cb_project = QComboBox()
        self.cb_project.setObjectName("ProjectBarCombo")
        self.cb_project.setToolTip("Active project — drives Browse, Record, Process and Transfer")
        self.cb_project.currentIndexChanged.connect(self._on_combo_changed)
        lay.addWidget(self.cb_project)

        self.btn_load = QToolButton()
        self.btn_load.setText("Load project  ▾")
        self.btn_load.setToolTip("Load a project from a local or server folder")
        self.btn_load.setPopupMode(QToolButton.InstantPopup)
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        menu = QMenu(self.btn_load)
        menu.addAction("Local folder…", self._load_local)
        menu.addAction("Server folder…", self._load_server)
        self.btn_load.setMenu(menu)
        lay.addWidget(self.btn_load)

        self.lbl_root = QLabel("—")
        self.lbl_root.setObjectName("ProjectBarRoot")
        self.lbl_root.setWordWrap(True)
        self.lbl_root.setToolTip("Data root (click to change)")
        self.lbl_root.setCursor(Qt.PointingHandCursor)
        self.lbl_root.mousePressEvent = lambda _e: self.set_data_root_requested.emit()
        lay.addWidget(self.lbl_root)

        struct_cap = QLabel("STRUCTURE")
        struct_cap.setObjectName("NavSection")
        lay.addWidget(struct_cap)

        self.lbl_structure = QLabel("—")
        self.lbl_structure.setObjectName("ProjectBarStructure")
        self.lbl_structure.setWordWrap(True)
        self.lbl_structure.setToolTip("Folder nesting for this project — click Design to change")
        lay.addWidget(self.lbl_structure)

        b_design = QPushButton("Design structure…")
        b_design.setCursor(Qt.PointingHandCursor)
        b_design.setToolTip("Open the Structure Designer for this project")
        b_design.clicked.connect(lambda: self.design_structure_requested.emit())
        lay.addWidget(b_design)

    # ── build ─────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(8)

        lbl = QLabel("Project:")
        lbl.setObjectName("ProjectBarLabel")
        lay.addWidget(lbl)

        self.cb_project = QComboBox()
        self.cb_project.setMinimumWidth(220)
        self.cb_project.setObjectName("ProjectBarCombo")
        self.cb_project.setToolTip("Active project — drives Browse, Record, Process and Transfer")
        self.cb_project.currentIndexChanged.connect(self._on_combo_changed)
        lay.addWidget(self.cb_project)

        self.btn_load = QToolButton()
        self.btn_load.setText("Load ▾")
        self.btn_load.setToolTip("Load a project from a local or server folder")
        self.btn_load.setPopupMode(QToolButton.InstantPopup)
        self.btn_load.setCursor(Qt.PointingHandCursor)
        menu = QMenu(self.btn_load)
        menu.addAction("Local folder…", self._load_local)
        menu.addAction("Server folder…", self._load_server)
        self.btn_load.setMenu(menu)
        lay.addWidget(self.btn_load)

        lay.addWidget(self._sep())

        self.lbl_root = QLabel("—")
        self.lbl_root.setObjectName("ProjectBarRoot")
        self.lbl_root.setToolTip("Data root (click to change)")
        self.lbl_root.setCursor(Qt.PointingHandCursor)
        self.lbl_root.mousePressEvent = lambda _e: self.set_data_root_requested.emit()
        lay.addWidget(self.lbl_root)

        lay.addWidget(self._sep())

        struct_cap = QLabel("Structure:")
        struct_cap.setObjectName("ProjectBarLabel")
        lay.addWidget(struct_cap)
        self.lbl_structure = QLabel("—")
        self.lbl_structure.setObjectName("ProjectBarStructure")
        self.lbl_structure.setToolTip("Folder nesting for this project — click Design to change")
        lay.addWidget(self.lbl_structure)

        b_design = QPushButton("Design…")
        b_design.setCursor(Qt.PointingHandCursor)
        b_design.setToolTip("Open the Structure Designer for this project")
        b_design.clicked.connect(lambda: self.design_structure_requested.emit())
        lay.addWidget(b_design)

        lay.addStretch(1)

    def _sep(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setObjectName("ProjectBarSep")
        return f

    # ── data ──────────────────────────────────────────────────────────
    def _project_items(self) -> list:
        names = list(list_projects(self.app_state.settings.raw_root))
        loaded = self.app_state.settings.get_loaded_project().get("name", "").strip()
        if loaded and loaded not in names:
            names.append(loaded)
        return sorted(dict.fromkeys(n for n in names if n))

    def _active_name(self) -> str:
        return (self.app_state.current_project
                or self.app_state.settings.get_loaded_project().get("name", "")).strip()

    def _structure_summary(self, project: str) -> str:
        subs = sublevels(self.app_state.settings.resolve_structure_schema(project), "raw")
        parts = []
        for lvl in subs:
            key = str(lvl.get("key", "")).strip().lower()
            parts.append(marker_folder_name(lvl) if is_marker_level(key) else level_label(lvl))
        return "  ▸  ".join(parts) if parts else "—"

    def refresh(self, active: Optional[str] = None):
        self._loading = True
        try:
            items = self._project_items()
            target = (active or self._active_name()).strip()
            self.cb_project.clear()
            self.cb_project.addItems(items)
            if target and target in items:
                self.cb_project.setCurrentText(target)
            elif items:
                self.cb_project.setCurrentIndex(0)
            else:
                self.cb_project.setCurrentIndex(-1)
            self.lbl_root.setText(self.app_state.settings.data_root or "—")
            self.lbl_structure.setText(self._structure_summary(self.cb_project.currentText().strip()))
        finally:
            self._loading = False

    # ── events ────────────────────────────────────────────────────────
    def _on_combo_changed(self, _index: int = -1):
        if self._loading:
            return
        name = self.cb_project.currentText().strip()
        if name:
            self.lbl_structure.setText(self._structure_summary(name))
            self.project_selected.emit(name)

    def _load_local(self):
        d = QFileDialog.getExistingDirectory(self, "Choose local project folder",
                                             self.app_state.settings.raw_root)
        if not d:
            return
        name = os.path.basename(os.path.normpath(d))
        self.app_state.settings.put_loaded_project(name, d, "local", d)
        self.refresh(active=name)
        self.project_selected.emit(name)

    def _load_server(self):
        d = QFileDialog.getExistingDirectory(self, "Choose server project folder")
        if not d:
            return
        name = os.path.basename(os.path.normpath(d))
        dest = os.path.join(self.app_state.settings.raw_root, name)
        self.app_state.settings.put_server_root_for_project(name, os.path.dirname(os.path.normpath(d)))
        self.app_state.settings.put_loaded_project(name, d, "server", dest)
        self.refresh(active=name)
        self.project_selected.emit(name)
