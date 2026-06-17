"""
Structure Designer – a fun, visual block-chain editor for building
hierarchical data folder structures.

Blocks can be dragged on top of each other to reorder the nesting.
Each block is colour-coded and shows its role (variable vs fixed folder).
"""

from typing import Any, Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
    LEVEL_COLORS,
    LEVEL_DESCRIPTIONS,
    LEVEL_ICONS,
    MARKER_LEVEL_KEYS,
    default_structure_schema,
    is_marker_level,
    marker_folder_name,
    normalize_structure_schema,
    preview_path,
)


# ── colour helpers ───────────────────────────────────────────────────────

def _tint(hex_color: str, factor: float = 0.12) -> str:
    """Mix *hex_color* with white to get a soft background tint."""
    c = QColor(hex_color)
    r = int(c.red()   + (255 - c.red())   * (1 - factor))
    g = int(c.green() + (255 - c.green()) * (1 - factor))
    b = int(c.blue()  + (255 - c.blue())  * (1 - factor))
    return QColor(r, g, b).name()


def _mix(hex_color: str, other: str, factor: float) -> str:
    """Mix *hex_color* with *other* by *factor* (0 = other, 1 = hex_color)."""
    a, b = QColor(hex_color), QColor(other)
    return QColor(
        int(b.red()   + (a.red()   - b.red())   * factor),
        int(b.green() + (a.green() - b.green()) * factor),
        int(b.blue()  + (a.blue()  - b.blue())  * factor),
    ).name()


# ═════════════════════════════════════════════════════════════════════════
#  Block-chain editor: a drag-and-drop list of level blocks
# ═════════════════════════════════════════════════════════════════════════

class BlockChainEditor(QWidget):
    """Reorderable list of level blocks for one chain (raw or processed)."""

    changed = Signal()

    def __init__(self, title: str, accent: str = "#4e86d9", parent=None):
        super().__init__(parent)
        self._accent = accent
        self._title = title
        self._build_ui()

    # ── build ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        header = QLabel(f"  {self._title}")
        header.setStyleSheet(
            f"font-weight: 700; font-size: 13px; color: {self._accent}; "
            "padding: 4px 0;"
        )
        root.addWidget(header)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.setAlternatingRowColors(False)
        self.list.setSpacing(2)
        self.list.setCursor(Qt.OpenHandCursor)
        self.list.setStyleSheet(
            f"""
            QListWidget {{
                border: 1px solid {self._accent}55;
                border-radius: 10px;
                padding: 8px;
                background: #1b1a24;
            }}
            QListWidget::item {{
                margin: 3px 0px;
                padding: 12px 12px;
                border: 1px solid #3a3548;
                border-left: 5px solid #3a3548;
                border-radius: 10px;
                background: #262232;
                font-size: 12px;
            }}
            QListWidget::item:selected {{
                border: 1px solid {self._accent};
                border-left: 5px solid {self._accent};
                background: #322a4c;
            }}
            QListWidget::item:hover {{
                background: #2c2740;
            }}
            """
        )
        root.addWidget(self.list, 1)

        # Button row
        row = QHBoxLayout()
        for label, slot in [
            ("\u2b06 Move up",   self._move_up),
            ("\u2b07 Move down", self._move_down),
            ("\u2714 Toggle",    self._toggle_selected),
            ("\u270f Rename",    self._rename_selected),
            ("Enable all",       self._enable_all),
        ]:
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)

        # Chain preview
        self.lbl_chain = QLabel("\u2014")
        self.lbl_chain.setWordWrap(True)
        self.lbl_chain.setStyleSheet(
            "padding: 8px 10px; background: #1b1a24; "
            "border: 1px solid #3a3548; border-radius: 8px; "
            "font-size: 12px; color: #cfc7e6;"
        )
        root.addWidget(self.lbl_chain)

        # Signals
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list.model().rowsMoved.connect(lambda *_: self._on_rows_moved())

    # ── public API ───────────────────────────────────────────────────────

    def set_levels(self, levels: List[Dict[str, Any]]):
        self.list.blockSignals(True)
        self.list.clear()

        seen: set = set()
        # Start with user-supplied order
        ordered = list(levels)
        # Append any missing known keys
        existing_keys = {str(x.get("key", "")).strip().lower() for x in ordered}
        for key in KNOWN_LEVEL_KEYS:
            if key not in existing_keys:
                ordered.append({"key": key, "enabled": False, "label": key})

        for lvl in ordered:
            key = str(lvl.get("key", "")).strip().lower()
            if not key or key in seen or key not in KNOWN_LEVEL_KEYS:
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

    # ── rendering ────────────────────────────────────────────────────────

    def _render_item(self, item: QListWidgetItem):
        raw = item.data(Qt.UserRole) or {}
        key = str(raw.get("key", "")).strip().lower()
        label = str(raw.get("label", "")).strip() or key.title()
        enabled = item.checkState() == Qt.Checked

        icon = LEVEL_ICONS.get(key, "\U0001f4c1")
        color = LEVEL_COLORS.get(key, "#666666")
        desc = LEVEL_DESCRIPTIONS.get(key, "")

        if is_marker_level(key):
            display = marker_folder_name({"key": key, "label": label})
            kind = "fixed folder"
        else:
            display = label
            kind = "variable"

        status_icon = "\u2705" if enabled else "\u26aa"  # ✅ / ⚪
        item.setText(f"  \u28ff    {status_icon}  {icon}   {display}      \u2014  {kind}")
        item.setToolTip(f"{desc}\nDrag the \u28ff handle to reorder \u2022 Double-click to toggle")

        # Colour coding (grey/purple): brightened text on a dark, colour-tinted row
        if enabled:
            item.setForeground(QColor(_mix(color, "#ffffff", 0.66)))
            item.setBackground(QColor(_mix(color, "#262232", 0.20)))
        else:
            item.setForeground(QColor("#6f6885"))
            item.setBackground(QColor("#201e2a"))

        font = item.font()
        font.setBold(enabled)
        item.setFont(font)

    # ── internal logic ───────────────────────────────────────────────────

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
        self._rerender_all()
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
        item.setCheckState(
            Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
        )

    def _rerender_all(self):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self._render_item(self.list.item(i))
        self.list.blockSignals(False)

    def _selected_item(self):
        row = self.list.currentRow()
        return self.list.item(row) if row >= 0 else None

    # ── button actions ───────────────────────────────────────────────────

    def _move_up(self):
        row = self.list.currentRow()
        if row <= 0:
            return
        item = self.list.takeItem(row)
        self.list.insertItem(row - 1, item)
        self.list.setCurrentRow(row - 1)
        self._enforce_project_top()
        self._rerender_all()
        self._refresh_chain_preview()
        self.changed.emit()

    def _move_down(self):
        row = self.list.currentRow()
        if row < 0 or row >= self.list.count() - 1:
            return
        item = self.list.takeItem(row)
        self.list.insertItem(row + 1, item)
        self.list.setCurrentRow(row + 1)
        self._enforce_project_top()
        self._rerender_all()
        self._refresh_chain_preview()
        self.changed.emit()

    def _toggle_selected(self):
        item = self._selected_item()
        if not item:
            return
        raw = item.data(Qt.UserRole) or {}
        if str(raw.get("key", "")).strip().lower() == "project":
            return
        item.setCheckState(
            Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
        )

    def _rename_selected(self):
        item = self._selected_item()
        if not item:
            return
        raw = dict(item.data(Qt.UserRole) or {})
        key = str(raw.get("key", "")).strip().lower()
        current = str(raw.get("label", "")).strip() or key.title()
        prompt = (
            "Folder name for this fixed block:"
            if key in MARKER_LEVEL_KEYS
            else "Display label:"
        )
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

    # ── chain preview ────────────────────────────────────────────────────

    def _refresh_chain_preview(self):
        tokens: List[str] = []
        for lvl in self.levels():
            if not lvl.get("enabled", False):
                continue
            key = str(lvl.get("key", "")).strip().lower()
            label = str(lvl.get("label", "")).strip() or key.title()
            icon = LEVEL_ICONS.get(key, "\U0001f4c1")
            if is_marker_level(key):
                tokens.append(f"{icon} {marker_folder_name(lvl)}/")
            else:
                tokens.append(f"{icon} <{label}>/")
        self.lbl_chain.setText(
            "  \u2192  ".join(tokens) if tokens else "\u2014"
        )


# ═════════════════════════════════════════════════════════════════════════
#  Structure Designer dialog
# ═════════════════════════════════════════════════════════════════════════

class StructureDesignerDialog(QDialog):
    """Full-screen-ish dialog with two BlockChainEditors (raw + processed)."""

    def __init__(self, default_schema: Dict[str, Any], project_schema: Dict[str, Any] = None,
                 project_name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("\U0001f9e9  Data Structure Designer")
        self.resize(1060, 800)
        self._project_name = (project_name or "").strip()
        # Two independent schemas the user can switch between and edit.
        self._schemas: Dict[str, Dict[str, Any]] = {
            "default": normalize_structure_schema(default_schema or default_structure_schema()),
        }
        if self._project_name:
            self._schemas["project"] = normalize_structure_schema(
                project_schema or default_schema or default_structure_schema()
            )
        self._scope = "default"
        self._build_ui()
        self._load_scope("default")
        self._refresh_previews()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # Header
        hdr = QLabel(
            "<span style='font-size:14px; font-weight:700;'>"
            "\U0001f3d7\ufe0f  Structure playground</span><br/>"
            "<span style='color:#93a4c2; font-size:11px;'>"
            "Drag blocks up/down to reorder nesting levels \u2022 "
            "Check/uncheck to enable \u2022 "
            "Rename labels for custom folder names \u2022 "
            "Add fixed <code>rawData</code> / <code>processedData</code> / "
            "<code>group</code> blocks anywhere."
            "</span>"
        )
        hdr.setWordWrap(True)
        hdr.setStyleSheet(
            "padding: 10px 12px; background: #221c33; color: #eae9f2; "
            "border: 1px solid #3a3548; border-radius: 10px;"
        )
        root.addWidget(hdr)

        # Scope selector: default (all projects) vs the loaded project
        scope_row = QHBoxLayout()
        scope_lbl = QLabel("Apply this structure to:")
        scope_lbl.setStyleSheet("font-weight: 600;")
        scope_row.addWidget(scope_lbl)
        self.cb_scope = QComboBox()
        self.cb_scope.addItem("\U0001f310  Default \u2013 all projects", "default")
        if self._project_name:
            self.cb_scope.addItem(
                f"\U0001f4c1  This project only \u2013 {self._project_name}", "project"
            )
        self.cb_scope.setCursor(Qt.PointingHandCursor)
        self.cb_scope.currentIndexChanged.connect(self._on_scope_changed)
        scope_row.addWidget(self.cb_scope, 1)
        root.addLayout(scope_row)

        # Two editors side by side
        editors = QHBoxLayout()
        self.editor_raw = BlockChainEditor(
            "\U0001f4e6  Raw chain blocks", accent="#8b5cf6"
        )
        self.editor_proc = BlockChainEditor(
            "\u2699\ufe0f  Processed chain blocks", accent="#22a565"
        )
        editors.addWidget(self.editor_raw, 1)
        editors.addWidget(self.editor_proc, 1)
        root.addLayout(editors, 1)

        self.editor_raw.changed.connect(self._refresh_previews)
        self.editor_proc.changed.connect(self._refresh_previews)

        # Previews
        previews = QGroupBox("\U0001f50d  Filesystem preview")
        p_lay = QVBoxLayout(previews)
        p_lay.addWidget(QLabel("<b>Raw path pattern:</b>"))
        self.lbl_raw_preview = QLabel("\u2014")
        self.lbl_raw_preview.setWordWrap(True)
        self.lbl_raw_preview.setStyleSheet(
            "padding: 6px; background: #1b1a24; color:#cfc7e6; border:1px solid #3a3548; border-radius: 4px;"
        )
        p_lay.addWidget(self.lbl_raw_preview)
        p_lay.addWidget(QLabel("<b>Processed path pattern:</b>"))
        self.lbl_proc_preview = QLabel("\u2014")
        self.lbl_proc_preview.setWordWrap(True)
        self.lbl_proc_preview.setStyleSheet(
            "padding: 6px; background: #1b1a24; color:#cfc7e6; border:1px solid #3a3548; border-radius: 4px;"
        )
        p_lay.addWidget(self.lbl_proc_preview)
        root.addWidget(previews)

        # Bottom buttons
        row = QHBoxLayout()
        b_reset = QPushButton("\U0001f504  Reset defaults")
        b_reset.setCursor(Qt.PointingHandCursor)
        b_reset.clicked.connect(self._reset_defaults)
        row.addWidget(b_reset)
        row.addStretch(1)
        root.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ── load / reset / scope ─────────────────────────────────────────────

    def _load_scope(self, scope: str):
        schema = self._schemas.get(scope) or default_structure_schema()
        self.editor_raw.set_levels(schema.get("raw_levels", []))
        self.editor_proc.set_levels(schema.get("processed_levels", []))

    def _on_scope_changed(self, _index: int = 0):
        # Stash edits for the scope we are leaving, then load the new one.
        self._schemas[self._scope] = self.schema()
        self._scope = self.cb_scope.currentData() or "default"
        self._load_scope(self._scope)
        self._refresh_previews()

    def selected_scope(self) -> str:
        return self._scope

    def _reset_defaults(self):
        self._schemas[self._scope] = default_structure_schema()
        self._load_scope(self._scope)
        self._refresh_previews()

    def _refresh_previews(self):
        schema = self.schema()
        self.lbl_raw_preview.setText(preview_path(schema, kind="raw"))
        self.lbl_proc_preview.setText(preview_path(schema, kind="processed"))

    # ── result ───────────────────────────────────────────────────────────

    def schema(self) -> Dict[str, Any]:
        return normalize_structure_schema({
            "raw_levels": self.editor_raw.levels(),
            "processed_levels": self.editor_proc.levels(),
        })

    # ── validation ───────────────────────────────────────────────────────

    def _on_accept(self):
        schema = self.schema()
        if not self._validate_core_order(schema, "raw_levels"):
            return
        if not self._validate_core_order(schema, "processed_levels"):
            return
        self._schemas[self._scope] = schema
        self.accept()

    def _validate_core_order(self, schema: Dict[str, Any], key: str) -> bool:
        levels = [
            x for x in schema.get(key, []) if bool(x.get("enabled", False))
        ]
        order = [str(x.get("key", "")).strip().lower() for x in levels]
        core = [k for k in CORE_LEVEL_KEYS if k in order]
        expected = sorted(core, key=lambda k: CORE_LEVEL_KEYS.index(k))
        if core != expected:
            chain_label = "Raw" if "raw" in key else "Processed"
            QMessageBox.warning(
                self,
                "Invalid block order",
                f"In the <b>{chain_label}</b> chain, enabled core blocks "
                "must keep their natural order:\n\n"
                "\U0001f4c1 Project  \u2192  \U0001f9ea Experiment  "
                "\u2192  \U0001f42d Subject  \u2192  \U0001f4c5 Session\n\n"
                "Optional blocks (recording, trial, group, raw, processed) "
                "can be placed around them.",
            )
            return False
        return True
