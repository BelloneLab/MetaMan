"""Schema-driven level selector.

A reusable row of labeled combo boxes for the folder levels that sit *under* a
project (experiment / subject / session / recording / trial), in the order
defined by the project's Structure Designer schema. Marker levels
(rawData / processedData / group) contribute a fixed folder name.

Both the Recording and Preprocessing tabs embed this so they automatically
follow whatever order the user configured per project (e.g. some projects nest
Subject -> Experiment, others Experiment -> Subject).
"""

import os
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QWidget

from .services.structure_schema import (
    is_marker_level,
    level_label,
    level_meta_key,
    marker_folder_name,
    sublevels,
)


class LevelChain(QWidget):
    """Ordered, schema-driven combos for a project's sub-folders."""

    changed = Signal()       # any level value changed (typing or selection)
    leaf_changed = Signal()  # the deepest level changed by selection

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._project = ""
        self._dest_dir = ""
        self._source_dirs: List[str] = []
        self._rows: List[Dict] = []
        self._building = False
        self._columns = 2  # lay level fields out across this many columns
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(8)

    # ── configuration ────────────────────────────────────────────────
    def configure(self, project: str, dest_dir: str, source_dirs: Optional[List[str]] = None):
        """Rebuild the level rows for *project* and set the listing roots."""
        self._project = (project or "").strip()
        self._dest_dir = (dest_dir or "").strip()
        self._source_dirs = [d for d in (source_dirs or []) if d]
        self._rebuild()

    def set_dirs(self, dest_dir: str, source_dirs: Optional[List[str]] = None):
        """Update the listing roots without changing the project/levels."""
        self._dest_dir = (dest_dir or "").strip()
        self._source_dirs = [d for d in (source_dirs or []) if d]
        self._repopulate_from(0)

    def _schema_sublevels(self) -> List[Dict]:
        schema = self.app_state.settings.resolve_structure_schema(self._project)
        return sublevels(schema, "raw")

    # ── build ────────────────────────────────────────────────────────
    def _clear_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._rows = []

    def _rebuild(self):
        self._building = True
        self._clear_grid()
        cols = max(1, self._columns)
        for i, lvl in enumerate(self._schema_sublevels()):
            key = str(lvl.get("key", "")).strip().lower()
            label = level_label(lvl)
            grid_row = i // cols
            base_col = (i % cols) * 2          # each cell = [label][field]
            self._grid.setColumnStretch(base_col + 1, 1)
            self._grid.addWidget(QLabel(label), grid_row, base_col)
            if is_marker_level(key):
                folder = marker_folder_name(lvl)
                val = QLabel(folder)
                val.setStyleSheet("color:#8b8694; font-style:italic;")
                self._grid.addWidget(val, grid_row, base_col + 1)
                self._rows.append({"key": key, "label": label, "is_marker": True,
                                   "folder": folder, "combo": None})
            else:
                cb = QComboBox()
                cb.setEditable(True)
                cb.setMaxVisibleItems(30)
                idx = len(self._rows)
                cb.currentIndexChanged.connect(lambda _=0, i=idx: self._on_index_changed(i))
                cb.currentTextChanged.connect(lambda _="", i=idx: self._on_text_changed(i))
                self._grid.addWidget(cb, grid_row, base_col + 1)
                self._rows.append({"key": key, "label": label, "is_marker": False,
                                   "folder": None, "combo": cb})
        self._building = False
        self._repopulate_from(0)

    # ── cascade / listing ─────────────────────────────────────────────
    def _parts_before(self, idx: int) -> List[str]:
        parts: List[str] = []
        for row in self._rows[:idx]:
            if row["is_marker"]:
                parts.append(row["folder"])
            else:
                parts.append(row["combo"].currentText().strip())
        return parts

    def _list_union(self, rel_parts: List[str]) -> List[str]:
        names = set()
        for root in [self._dest_dir, *self._source_dirs]:
            if not root:
                continue
            d = os.path.join(root, *rel_parts) if rel_parts else root
            if os.path.isdir(d):
                try:
                    names.update(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))
                except Exception:
                    pass
        return sorted(names)

    def _repopulate_from(self, idx: int):
        was = self._building
        self._building = True
        try:
            for i in range(idx, len(self._rows)):
                row = self._rows[i]
                if row["is_marker"]:
                    continue
                # If a preceding variable level is empty the deeper list is moot.
                parts = self._parts_before(i)
                items = self._list_union(parts)
                cb = row["combo"]
                keep = cb.currentText().strip()
                cb.blockSignals(True)
                cb.clear()
                cb.addItems(items)
                if keep:
                    if cb.findText(keep) < 0:
                        cb.addItem(keep)
                    cb.setCurrentText(keep)
                elif items:
                    cb.setCurrentIndex(0)
                else:
                    cb.setCurrentIndex(-1)
                    cb.setEditText("")
                cb.blockSignals(False)
        finally:
            self._building = was

    def _last_var_index(self) -> int:
        for i in range(len(self._rows) - 1, -1, -1):
            if not self._rows[i]["is_marker"]:
                return i
        return -1

    def _on_index_changed(self, i: int):
        if self._building:
            return
        self._repopulate_from(i + 1)
        self.changed.emit()
        if i == self._last_var_index():
            self.leaf_changed.emit()

    def _on_text_changed(self, _i: int):
        if self._building:
            return
        self.changed.emit()

    # ── values ────────────────────────────────────────────────────────
    def has_levels(self) -> bool:
        return bool(self._rows)

    def part_names(self) -> List[str]:
        return self._parts_before(len(self._rows))

    def leaf_path(self) -> str:
        if not self._dest_dir:
            return ""
        return os.path.join(self._dest_dir, *self.part_names())

    def values_by_key(self) -> Dict[str, str]:
        return {row["key"]: row["combo"].currentText().strip()
                for row in self._rows if not row["is_marker"]}

    def metadata(self) -> Dict[str, str]:
        return {level_meta_key(row): row["combo"].currentText().strip()
                for row in self._rows if not row["is_marker"]}

    def all_filled(self) -> bool:
        return all(row["is_marker"] or row["combo"].currentText().strip()
                   for row in self._rows)

    def missing_labels(self) -> List[str]:
        return [row["label"] for row in self._rows
                if not row["is_marker"] and not row["combo"].currentText().strip()]

    def ensure_value(self, key: str, value: str):
        text = str(value or "").strip()
        if not text:
            return
        for row in self._rows:
            if row["is_marker"] or row["key"] != key:
                continue
            cb = row["combo"]
            cb.blockSignals(True)
            if cb.findText(text) < 0:
                cb.addItem(text)
            cb.setCurrentText(text)
            cb.blockSignals(False)

    def set_values_by_key(self, values: Dict[str, str]):
        self._building = True
        try:
            for row in self._rows:
                if row["is_marker"]:
                    continue
                val = str(values.get(row["key"], "") or "").strip()
                if val:
                    cb = row["combo"]
                    if cb.findText(val) < 0:
                        cb.addItem(val)
                    cb.setCurrentText(val)
        finally:
            self._building = False
        self._repopulate_from(0)

    def set_from_metadata(self, meta: Dict):
        values: Dict[str, str] = {}
        for row in self._rows:
            if row["is_marker"]:
                continue
            mk = level_meta_key(row)
            val = str(meta.get(mk, "") or "")
            if not val and row["key"] == "subject":
                val = str(meta.get("Animal", "") or "")
            values[row["key"]] = val
        self.set_values_by_key(values)
