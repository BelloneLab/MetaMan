"""Reusable side-panel layout: narrow icon toolbar on the left, stacked content on the right."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class SidePanelLayout(QWidget):
    """Narrow icon sidebar on the left, active tool panel filling the remaining space."""

    panel_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._bar = QWidget()
        self._bar.setObjectName("SidePanelBar")
        self._bar_layout = QVBoxLayout(self._bar)
        self._bar_layout.setContentsMargins(4, 8, 4, 8)
        self._bar_layout.setSpacing(4)
        self._bar_layout.addStretch(1)

        self._stack = QStackedWidget()

        root.addWidget(self._bar)
        root.addWidget(self._stack, 1)

        self._buttons: list = []
        self._default_idx = 0

    # ------------------------------------------------------------------
    def add_panel(self, icon: str, label: str, widget: QWidget, *, default: bool = False) -> int:
        """Add a panel. *icon* is shown on the sidebar button, *label* as tooltip."""
        idx = self._stack.count()
        self._stack.addWidget(widget)

        btn = QPushButton(label)
        btn.setToolTip(label)
        btn.setCheckable(True)
        btn.setFixedHeight(36)
        btn.setObjectName("SidePanelButton")
        btn.clicked.connect(lambda _checked, i=idx: self.switch_to(i))

        # Insert before the stretch
        self._bar_layout.insertWidget(self._bar_layout.count() - 1, btn)
        self._buttons.append(btn)

        if default:
            self._default_idx = idx

        if len(self._buttons) == 1 or default:
            self.switch_to(idx)

        return idx

    def switch_to(self, idx: int):
        if 0 <= idx < self._stack.count():
            self._stack.setCurrentIndex(idx)
            for i, b in enumerate(self._buttons):
                b.setChecked(i == idx)
            self.panel_changed.emit(idx)

    def current_index(self) -> int:
        return self._stack.currentIndex()

    # ------------------------------------------------------------------
    @staticmethod
    def style_sheet() -> str:
        return """
        #SidePanelBar {
            background: #faf9fd;
            border-right: 1px solid #e6e3f0;
            min-width: 72px;
            max-width: 72px;
        }
        #SidePanelButton {
            font-size: 10px;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            background: transparent;
            padding: 6px 2px;
            color: #8b8694;
        }
        #SidePanelButton:hover {
            background: #ece7fe;
            color: #5a3fe0;
        }
        #SidePanelButton:checked {
            background: #7c5cfc;
            color: #ffffff;
        }
        """
