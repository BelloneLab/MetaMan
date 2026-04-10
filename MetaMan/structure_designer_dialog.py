from typing import Any, Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .services.structure_schema import (
    CORE_LEVEL_KEYS,
    KNOWN_LEVEL_KEYS,
    MARKER_LEVEL_KEYS,
    default_structure_schema,
    is_marker_level,
    marker_folder_name,
    normalize_structure_schema,
    preview_path,
)


class BlockChainEditor(QWidget):
    changed = Signal()

    def __init__(self, title: str, accent: str = "#4e86d9", parent=None):
        super().__init__(parent)
        self._accent = accent
        self._title = title
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        header = QLabel(self._title)
        header.setStyleSheet("font-weight: 600;")
        root.addWidget(header)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setAlternatingRowColors(False)
        self.list.setStyleSheet(
            f"""
            QListWidget {{
                border: 1px solid #c7d2e3;
                border-radius: 8px;
                padding: 6px;
                background: #f8fbff;
            }}
            QListWidget::item {{
                margin: 3px;
                padding: 9px 10px;
                border: 1px solid #c7d2e3;
                border-radius: 8px;
                background: #ffffff;
            }}
            QListWidget::item:selected {{
                border: 2px solid {self._accent};
                background: #ecf4ff;
            }}
            """
        )
        root.addWidget(self.list, 1)

        row = QHBoxLayout()
        b_toggle = QPushButton("Toggle selected")
        b_toggle.clicked.connect(self._toggle_selected)
        row.addWidget(b_toggle)

        b_rename = QPushButton("Rename selected")
        b_rename.clicked.connect(self._rename_selected)
        row.addWidget(b_rename)

        b_enable = QPushButton("Enable all")
        b_enable.clicked.connect(self._enable_all)
        row.addWidget(b_enable)

        row.addStretch(1)
        root.addLayout(row)

        self.lbl_chain = QLabel("-")
        self.lbl_chain.setWordWrap(True)
        self.lbl_chain.setStyleSheet("padding: 6px; background: #ffffff; border: 1px solid #d5deea; border-radius: 6px;")
        root.addWidget(self.lbl_chain)

        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list.model().rowsMoved.connect(lambda *_: self._on_rows_moved())

    def set_levels(self, levels: List[Dict[str, Any]]):
        self.list.blockSignals(True)
        self.list.clear()
        seen = set()
        ordered = [x for x in levels if str(x.get("key", "")).strip().lower() in KNOWN_LEVEL_KEYS]
        for key in KNOWN_LEVEL_KEYS:
            if key not in [str(x.get("key", "")).strip().lower() for x in ordered]:
                ordered.append({"key": key, "enabled": False, "label": key})

        for lvl in ordered:
            key = str(lvl.get("key", "")).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            label = str(lvl.get("label", "")).strip() or key.title()
            enabled = bool(lvl.get("enabled", key == "project"))
            if key == "project":
                enabled = True
            item = QListWidgetItem()
            item.setFlags(
                Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
                | Qt.ItemIsDragEnabled
                | Qt.ItemIsUserCheckable
            )
            item.setData(Qt.UserRole, {"key": key, "label": label})
            item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
            self._render_item(item)
            self.list.addItem(item)

        self.list.blockSignals(False)
        self._enforce_project_top()
        self._refresh_chain_preview()

    def levels(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            raw = item.data(Qt.UserRole) or {}
            key = str(raw.get("key", "")).strip().lower()
            if key not in KNOWN_LEVEL_KEYS:
                continue
            label = str(raw.get("label", "")).strip() or key.title()
            enabled = item.checkState() == Qt.Checked
            if key == "project":
                enabled = True
            out.append({"key": key, "label": label, "enabled": enabled})
        return out

    def _render_item(self, item: QListWidgetItem):
        raw = item.data(Qt.UserRole) or {}
        key = str(raw.get("key", "")).strip().lower()
        label = str(raw.get("label", "")).strip() or key.title()
        enabled = item.checkState() == Qt.Checked
        state = "ON" if enabled else "OFF"
        marker = "fixed folder" if is_marker_level(key) else "variable folder"
        if is_marker_level(key):
            display = marker_folder_name({"key": key, "label": label})
            item.setText(f"{state}   [{display}]   ({key}, {marker})")
        else:
            item.setText(f"{state}   [{label}]   ({key}, {marker})")

    def _enforce_project_top(self):
        proj_idx = -1
        for i in range(self.list.count()):
            raw = self.list.item(i).data(Qt.UserRole) or {}
            if str(raw.get("key", "")).strip().lower() == "project":
                proj_idx = i
                break
        if proj_idx > 0:
            it = self.list.takeItem(proj_idx)
            self.list.insertItem(0, it)
        if self.list.count() > 0:
            first = self.list.item(0)
            raw = first.data(Qt.UserRole) or {}
            if str(raw.get("key", "")).strip().lower() == "project":
                first.setCheckState(Qt.Checked)
                self._render_item(first)

    def _on_rows_moved(self):
        self._enforce_project_top()
        self._refresh_chain_preview()
        self.changed.emit()

    def _on_item_changed(self, item: QListWidgetItem):
        raw = item.data(Qt.UserRole) or {}
        if str(raw.get("key", "")).strip().lower() == "project":
            item.setCheckState(Qt.Checked)
        self._render_item(item)
        self._refresh_chain_preview()
        self.changed.emit()

    def _on_item_double_clicked(self, item: QListWidgetItem):
        raw = item.data(Qt.UserRole) or {}
        key = str(raw.get("key", "")).strip().lower()
        if key == "project":
            return
        item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)

    def _selected_item(self) -> QListWidgetItem:
        row = self.list.currentRow()
        if row < 0:
            return None
        return self.list.item(row)

    def _toggle_selected(self):
        item = self._selected_item()
        if not item:
            return
        raw = item.data(Qt.UserRole) or {}
        key = str(raw.get("key", "")).strip().lower()
        if key == "project":
            return
        item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)

    def _rename_selected(self):
        item = self._selected_item()
        if not item:
            return
        raw = dict(item.data(Qt.UserRole) or {})
        key = str(raw.get("key", "")).strip().lower()
        current = str(raw.get("label", "")).strip() or key.title()
        prompt = "Folder name for this fixed block:" if key in MARKER_LEVEL_KEYS else "Display label:"
        text, ok = QInputDialog.getText(self, "Rename block", prompt, text=current)
        if not ok:
            return
        new_label = str(text or "").strip()
        if not new_label:
            return
        raw["label"] = new_label
        item.setData(Qt.UserRole, raw)
        self._render_item(item)
        self._refresh_chain_preview()
        self.changed.emit()

    def _enable_all(self):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setCheckState(Qt.Checked)
            self._render_item(item)
        self.list.blockSignals(False)
        self._enforce_project_top()
        self._refresh_chain_preview()
        self.changed.emit()

    def _refresh_chain_preview(self):
        tokens = []
        for lvl in self.levels():
            if not lvl.get("enabled", False):
                continue
            key = str(lvl.get("key", "")).strip().lower()
            label = str(lvl.get("label", "")).strip() or key.title()
            if is_marker_level(key):
                tokens.append(f"[{marker_folder_name({'key': key, 'label': label})}]")
            else:
                tokens.append(f"[{label}]")
        self.lbl_chain.setText(" -> ".join(tokens) if tokens else "-")


class StructureDesignerDialog(QDialog):
    def __init__(self, schema: Dict[str, Any], project_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Structure Designer")
        self.resize(1040, 760)
        self._schema = normalize_structure_schema(schema or default_structure_schema())
        self._project_name = project_name.strip()
        self._build_ui()
        self._load_schema_to_editors()
        self._refresh_previews()

    def _build_ui(self):
        root = QVBoxLayout(self)
        scope = self._project_name or "Default (new projects)"
        root.addWidget(
            QLabel(
                f"Structure playground for: <b>{scope}</b><br/>"
                "Drag blocks to reorder levels. "
                "Use ON/OFF to include or exclude a block. "
                "You can rename any block. "
                "Add fixed `raw` / `processed` blocks anywhere if your data uses nested splits."
            )
        )

        self.editor_raw = BlockChainEditor("Raw chain blocks", accent="#4e86d9")
        self.editor_proc = BlockChainEditor("Processed chain blocks", accent="#2c9a5e")
        root.addWidget(self.editor_raw, 1)
        root.addWidget(self.editor_proc, 1)

        self.editor_raw.changed.connect(self._refresh_previews)
        self.editor_proc.changed.connect(self._refresh_previews)

        previews = QGroupBox("Filesystem preview")
        p_lay = QVBoxLayout(previews)
        self.lbl_raw_preview = QLabel("-")
        self.lbl_proc_preview = QLabel("-")
        p_lay.addWidget(QLabel("Raw path pattern"))
        p_lay.addWidget(self.lbl_raw_preview)
        p_lay.addWidget(QLabel("Processed path pattern"))
        p_lay.addWidget(self.lbl_proc_preview)
        root.addWidget(previews)

        row = QHBoxLayout()
        b_reset = QPushButton("Reset defaults")
        b_reset.clicked.connect(self._reset_defaults)
        row.addWidget(b_reset)
        row.addStretch(1)
        root.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _load_schema_to_editors(self):
        self.editor_raw.set_levels(self._schema.get("raw_levels", []))
        self.editor_proc.set_levels(self._schema.get("processed_levels", []))

    def _reset_defaults(self):
        self._schema = default_structure_schema()
        self._load_schema_to_editors()
        self._refresh_previews()

    def _refresh_previews(self):
        schema = self.schema()
        self.lbl_raw_preview.setText(preview_path(schema, kind="raw"))
        self.lbl_proc_preview.setText(preview_path(schema, kind="processed"))

    def schema(self) -> Dict[str, Any]:
        return normalize_structure_schema(
            {
                "raw_levels": self.editor_raw.levels(),
                "processed_levels": self.editor_proc.levels(),
            }
        )

    def _on_accept(self):
        schema = self.schema()
        if not self._validate_core_order(schema, "raw_levels"):
            return
        if not self._validate_core_order(schema, "processed_levels"):
            return
        self._schema = schema
        self.accept()

    def _validate_core_order(self, schema: Dict[str, Any], key: str) -> bool:
        levels = [x for x in schema.get(key, []) if bool(x.get("enabled", False))]
        order = [str(x.get("key", "")).strip().lower() for x in levels]
        core = [k for k in CORE_LEVEL_KEYS if k in order]
        if core != sorted(core, key=lambda k: CORE_LEVEL_KEYS.index(k)):
            QMessageBox.warning(
                self,
                "Invalid order",
                "For compatibility with Navigation/Recording/Preprocessing, enabled core blocks must stay ordered as:\n"
                "project -> experiment -> subject -> session.\n"
                "You can still place optional blocks (recording/trial/group/raw/processed) around them.",
            )
            return False
        return True
