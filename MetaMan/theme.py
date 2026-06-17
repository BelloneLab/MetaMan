"""Centralised application theme.

A clean, light "workspace" theme inspired by modern SaaS collaboration tools:
near-white surfaces, a faint-lavender chrome, hairline borders, soft rounded
cards, pill badges and a violet primary accent with a green success accent.
Buttons opt into accents with ``setObjectName("Primary")`` /
``setObjectName("Success")``.
"""

# ── palette (kept here for reference; the QSS below uses literal hex) ──────
BG0 = "#f4f3f9"          # window / menu / status (faint lavender)
BG1 = "#ffffff"          # panels, cards, tab pane
BG2 = "#f7f6fc"          # raised: headers, selected tab track, hover fills
INPUT = "#ffffff"        # editable fields, tables, trees
BORDER = "#e6e3f0"       # standard hairline border
BORDER_SOFT = "#efedf6"  # softer divider
TEXT = "#232026"         # primary text (warm near-black)
MUTED = "#8b8694"        # secondary text
ACCENT = "#7c5cfc"       # violet
ACCENT_HOVER = "#6b48f2"
ACCENT_SOFT = "#ece7fe"  # tinted violet fill (selection, pills)
SUCCESS = "#1f9d57"      # green
SUCCESS_SOFT = "#e3f4ea"
SEL = "#ece7fe"          # selected row fill (subtle lavender)

# Preferred UI font is bundled Inter (assets/fonts/InterVariable.ttf); this is
# the fallback used when Inter is not available at runtime.
FONT_FAMILY = "Segoe UI"

STYLESHEET = """
/* ── base ─────────────────────────────────────────────────────────── */
QWidget {
    background-color: #f4f3f9;
    color: #232026;
    font-size: 12px;
    selection-background-color: #d8ccfd;
    selection-color: #1f1b2e;
}
QMainWindow, QDialog { background-color: #f4f3f9; }
QLabel { background: transparent; color: #232026; }
QToolTip {
    background-color: #2b2733; color: #f3f1f9;
    border: none; padding: 5px 8px; border-radius: 6px;
}

/* ── tabs ─────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #e6e3f0; border-radius: 12px;
    background: #ffffff; top: -1px;
}
QTabBar { background: transparent; }
QTabBar::tab {
    background: transparent; color: #8b8694;
    padding: 9px 18px; margin-right: 4px; border: none;
    border-top-left-radius: 9px; border-top-right-radius: 9px;
    font-weight: 600;
}
QTabBar::tab:hover { color: #232026; }
QTabBar::tab:selected {
    color: #5a3fe0; background: #ffffff;
    border-bottom: 2px solid #7c5cfc;
}

/* ── group boxes (cards) ──────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #e6e3f0; border-radius: 12px;
    margin-top: 14px; padding-top: 14px; background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 14px; padding: 0 6px;
    color: #6f6a7a; font-weight: 700;
}

/* ── buttons ──────────────────────────────────────────────────────── */
QPushButton {
    background: #ffffff; color: #3a3545;
    border: 1px solid #e0dcec; border-radius: 9px;
    padding: 7px 13px; min-height: 22px;
}
QPushButton:hover { background: #f3f0fb; border-color: #d4ccef; }
QPushButton:pressed { background: #ebe6f8; }
QPushButton:disabled { color: #b6b1c2; background: #f5f4f9; border-color: #ebe9f2; }
QPushButton#Primary {
    background: #7c5cfc; color: #ffffff; border: none; font-weight: 700;
    padding: 9px 18px;
}
QPushButton#Primary:hover { background: #6b48f2; }
QPushButton#Primary:pressed { background: #5a3fe0; }
QPushButton#Primary:disabled { background: #d9cffb; color: #ffffff; }
QPushButton#Success {
    background: #1f9d57; color: #ffffff; border: none; font-weight: 700;
    padding: 9px 18px;
}
QPushButton#Success:hover { background: #1a8a4c; }
QPushButton#Success:disabled { background: #bfe5cd; color: #ffffff; }

QToolButton {
    background: #ffffff; color: #3a3545;
    border: 1px solid #e0dcec; border-radius: 9px; padding: 6px 11px;
}
QToolButton:hover { background: #f3f0fb; border-color: #d4ccef; }
QToolButton::menu-indicator { image: none; }

/* ── inputs ───────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QTimeEdit, QSpinBox, QDateTimeEdit {
    background: #ffffff; color: #232026;
    border: 1px solid #e0dcec; border-radius: 9px;
    padding: 6px 9px; selection-background-color: #d8ccfd; selection-color: #1f1b2e;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
QTimeEdit:focus, QSpinBox:focus, QDateTimeEdit:focus {
    border: 1px solid #7c5cfc;
}
QLineEdit:read-only { color: #8b8694; background: #f7f6fc; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #ffffff; color: #232026;
    border: 1px solid #e6e3f0; border-radius: 8px; outline: 0;
    selection-background-color: #ece7fe; selection-color: #2a2533;
    padding: 4px;
}

/* ── tables / trees / lists ───────────────────────────────────────── */
QTableWidget, QTreeWidget, QListWidget {
    background: #ffffff; color: #232026;
    alternate-background-color: #faf9fd;
    gridline-color: #efedf6;
    border: 1px solid #e6e3f0; border-radius: 10px;
    outline: 0;
}
QTableWidget::item, QTreeWidget::item, QListWidget::item {
    padding: 4px 5px; border-radius: 6px;
}
QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected {
    background: #ece7fe; color: #2a2533;
}
QTreeWidget::item:hover, QListWidget::item:hover { background: #f3f0fb; }
QHeaderView::section {
    background: #f7f6fc; color: #6f6a7a;
    border: none; border-right: 1px solid #efedf6; border-bottom: 1px solid #e6e3f0;
    padding: 7px 9px; font-weight: 700;
}
QTableCornerButton::section { background: #f7f6fc; border: none; }

/* ── checkboxes ───────────────────────────────────────────────────── */
QCheckBox { color: #232026; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #cfc8e0;
    border-radius: 5px; background: #ffffff;
}
QCheckBox::indicator:hover { border-color: #7c5cfc; }
QCheckBox::indicator:checked { background: #7c5cfc; border-color: #7c5cfc; }

/* ── progress ─────────────────────────────────────────────────────── */
QProgressBar {
    background: #efedf6; border: none;
    border-radius: 8px; text-align: center; color: #3a3545; height: 16px;
}
QProgressBar::chunk { background: #7c5cfc; border-radius: 8px; }

/* ── menus ────────────────────────────────────────────────────────── */
QMenuBar { background: #f4f3f9; color: #232026; border-bottom: 1px solid #e6e3f0; }
QMenuBar::item { background: transparent; padding: 6px 12px; border-radius: 7px; }
QMenuBar::item:selected { background: #ece7fe; color: #5a3fe0; }
QMenu { background: #ffffff; color: #232026; border: 1px solid #e6e3f0; border-radius: 10px; padding: 5px; }
QMenu::item { padding: 7px 22px; border-radius: 6px; }
QMenu::item:selected { background: #ece7fe; color: #5a3fe0; }
QMenu::separator { height: 1px; background: #efedf6; margin: 5px 8px; }

/* ── status bar ───────────────────────────────────────────────────── */
QStatusBar { background: #f4f3f9; color: #8b8694; border-top: 1px solid #e6e3f0; }
QStatusBar::item { border: none; }

/* ── splitter / scrollbars ────────────────────────────────────────── */
QSplitter::handle { background: #e6e3f0; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical { height: 2px; }
QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #d6d1e4; border-radius: 6px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #bdb6d4; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: #d6d1e4; border-radius: 6px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #bdb6d4; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ── project bar ──────────────────────────────────────────────────── */
#ProjectBar {
    background: #ffffff;
    border-bottom: 1px solid #e6e3f0;
}
#ProjectBar QLabel { background: transparent; }
#ProjectBarLabel { color: #8b8694; font-weight: 700; }
#ProjectBarCombo { font-weight: 700; min-height: 26px; }
#ProjectBarRoot { color: #6b48f2; }
#ProjectBarStructure {
    color: #5a3fe0; font-weight: 700;
    background: #ece7fe;
    border: 1px solid #ddd3fd;
    border-radius: 11px; padding: 3px 11px;
}
#ProjectBarSep { color: #e6e3f0; max-width: 1px; }

/* generic helper text */
QLabel#Hint { color: #8b8694; }

/* ── left navigation rail ─────────────────────────────────────────── */
#NavRail {
    background: #faf9fd;
    border-right: 1px solid #e6e3f0;
    min-width: 240px; max-width: 240px;
}
#NavRail QWidget#ProjectBarCompact { background: transparent; }
#NavDivider { background: #ece9f3; border: none; margin: 4px 0; }
#NavBrandText {
    color: #6b48f2; font-weight: 800; font-size: 16px;
}
#NavSection {
    color: #a39ead; font-weight: 700; font-size: 10px;
    padding: 2px 2px; letter-spacing: 1px;
}
#NavItem {
    text-align: left; font-size: 13px; font-weight: 600;
    color: #5a5566; background: transparent;
    border: none; border-radius: 10px; padding: 0 12px;
}
#NavItem:hover { background: #ece7fe; color: #3a3545; }
#NavItem:checked {
    background: #ffffff; color: #5a3fe0; font-weight: 700;
    border: 1px solid #e3dcfb;
}

/* ── side panel (Record / Transfer / Import) ──────────────────────── */
#SidePanelBar {
    background: #faf9fd; border-right: 1px solid #e6e3f0;
    min-width: 86px; max-width: 86px;
}
#SidePanelButton {
    font-size: 10px; font-weight: 700; border: none; border-radius: 10px;
    background: transparent; padding: 8px 4px; color: #8b8694;
}
#SidePanelButton:hover { background: #ece7fe; color: #5a3fe0; }
#SidePanelButton:checked { background: #7c5cfc; color: #ffffff; }

/* ── explorer (QTreeView file browser) ────────────────────────────── */
QTreeView {
    background: #ffffff; color: #232026;
    alternate-background-color: #faf9fd;
    border: 1px solid #e6e3f0; border-radius: 10px; outline: 0;
}
QTreeView::item { padding: 3px 4px; border-radius: 5px; }
QTreeView::item:hover { background: #f3f0fb; }
QTreeView::item:selected { background: #ece7fe; color: #2a2533; }
#ExplorerPanelTitle { font-weight: 800; color: #5a3fe0; font-size: 13px; }
#BackupSummary { color: #3a3545; }

/* ── guarded destructive button ───────────────────────────────────── */
QPushButton#Danger {
    background: #ed4245; color: #ffffff; border: none; font-weight: 700;
    padding: 9px 18px;
}
QPushButton#Danger:hover { background: #d83a3d; }
QPushButton#Danger:disabled { background: #f2b6b7; color: #ffffff; }
"""
