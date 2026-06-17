"""Left navigation rail + content stack (workspace shell).

Replaces the top tab bar with a vertical sidebar in the style of modern
collaboration apps: a brand block at the top, a small section caption, then a
column of icon+label items. The active item gets a soft lavender card with a
violet accent. The right side is a ``QStackedWidget`` that holds the pages.
"""

from typing import List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class WorkspaceShell(QWidget):
    """Vertical nav rail on the left, active page filling the rest."""

    page_changed = Signal(int)

    def __init__(self, logo_path: str = "", header_widget: Optional[QWidget] = None, parent=None):
        super().__init__(parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── rail ──────────────────────────────────────────────────────
        self._rail = QWidget()
        self._rail.setObjectName("NavRail")
        rail_lay = QVBoxLayout(self._rail)
        rail_lay.setContentsMargins(14, 16, 14, 16)
        rail_lay.setSpacing(6)

        # brand block (logo + wordmark)
        brand = QWidget()
        brand.setObjectName("NavBrand")
        brand_lay = QHBoxLayout(brand)
        brand_lay.setContentsMargins(6, 2, 6, 2)
        brand_lay.setSpacing(9)
        if logo_path:
            pix = QPixmap(logo_path)
            if not pix.isNull():
                logo = QLabel()
                logo.setPixmap(
                    pix.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                logo.setFixedSize(24, 24)
                brand_lay.addWidget(logo)
        wordmark = QLabel("MetaMan")
        wordmark.setObjectName("NavBrandText")
        brand_lay.addWidget(wordmark)
        brand_lay.addStretch(1)
        rail_lay.addWidget(brand)

        # optional project/context header folded into the rail
        if header_widget is not None:
            rail_lay.addWidget(self._divider())
            rail_lay.addWidget(header_widget)

        rail_lay.addWidget(self._divider())
        caption = QLabel("WORKSPACE")
        caption.setObjectName("NavSection")
        rail_lay.addWidget(caption)

        self._rail_lay = rail_lay
        self._rail_lay.addStretch(1)

        # ── content stack ─────────────────────────────────────────────
        self._stack = QStackedWidget()

        root.addWidget(self._rail)
        root.addWidget(self._stack, 1)

        self._buttons: List[QPushButton] = []

    # ------------------------------------------------------------------
    def _divider(self) -> QFrame:
        line = QFrame()
        line.setObjectName("NavDivider")
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        return line

    def add_page(self, widget: QWidget, label: str, icon: str = "") -> int:
        """Add *widget* as a page with a rail item showing *icon* + *label*."""
        idx = self._stack.addWidget(widget)

        text = f"   {label}"
        btn = QPushButton(f"{icon}{text}" if icon else label)
        btn.setObjectName("NavItem")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(44)
        btn.setIconSize(QSize(18, 18))
        btn.clicked.connect(lambda _checked, i=idx: self.set_current_index(i))

        # Extra breathing room between workspace items (independent of the
        # tighter spacing used by the brand/project header above).
        if self._buttons:
            self._rail_lay.insertSpacing(self._rail_lay.count() - 1, 16)
        # insert before the trailing stretch
        self._rail_lay.insertWidget(self._rail_lay.count() - 1, btn)
        self._buttons.append(btn)

        if len(self._buttons) == 1:
            self.set_current_index(0)
        return idx

    def set_current_index(self, idx: int):
        if 0 <= idx < self._stack.count():
            self._stack.setCurrentIndex(idx)
            for i, b in enumerate(self._buttons):
                b.setChecked(i == idx)
            self.page_changed.emit(idx)

    def set_current_widget(self, widget: QWidget):
        idx = self._stack.indexOf(widget)
        if idx >= 0:
            self.set_current_index(idx)

    def current_index(self) -> int:
        return self._stack.currentIndex()

    def current_widget(self) -> Optional[QWidget]:
        return self._stack.currentWidget()
