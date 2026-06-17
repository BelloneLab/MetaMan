"""Centralised application theme.

A cohesive grey/purple dark theme: cool charcoal surfaces with a violet accent
and a green success accent. Buttons opt into accents with
``setObjectName("Primary")`` / ``setObjectName("Success")``.
"""

# ── palette (kept here for reference; the QSS below uses literal hex) ──────
BG0 = "#16151d"          # window / menu / status
BG1 = "#211f2b"          # panels, cards, tab pane
BG2 = "#2c2839"          # raised: buttons, selected tab, headers
INPUT = "#1b1a24"        # editable fields, tables, trees
BORDER = "#3a3548"
BORDER_SOFT = "#2c2939"
TEXT = "#eae9f2"
MUTED = "#a79fbb"
ACCENT = "#8b5cf6"       # violet
ACCENT_HOVER = "#9a6ff5"
SUCCESS = "#22a565"
SEL = "#43356e"

FONT_FAMILY = "Segoe UI"

STYLESHEET = """
/* ── base ─────────────────────────────────────────────────────────── */
QWidget {
    background-color: #16151d;
    color: #eae9f2;
    font-size: 12px;
    selection-background-color: #43356e;
    selection-color: #ffffff;
}
QMainWindow, QDialog { background-color: #16151d; }
QLabel { background: transparent; color: #eae9f2; }
QToolTip {
    background-color: #2c2839; color: #eae9f2;
    border: 1px solid #3a3548; padding: 4px 6px; border-radius: 4px;
}

/* ── tabs ─────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #3a3548; border-radius: 10px;
    background: #211f2b; top: -1px;
}
QTabBar::tab {
    background: transparent; color: #a79fbb;
    padding: 9px 18px; margin-right: 2px; border: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
    font-weight: 600;
}
QTabBar::tab:hover { color: #eae9f2; }
QTabBar::tab:selected {
    color: #ffffff; background: #211f2b;
    border-bottom: 2px solid #8b5cf6;
}

/* ── group boxes (cards) ──────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #2c2939; border-radius: 10px;
    margin-top: 12px; padding-top: 12px; background: #211f2b;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: #a79fbb; font-weight: 700;
}

/* ── buttons ──────────────────────────────────────────────────────── */
QPushButton {
    background: #2c2839; color: #ded9ec;
    border: 1px solid #3a3548; border-radius: 7px;
    padding: 6px 12px; min-height: 22px;
}
QPushButton:hover { background: #36314a; border-color: #4a4360; }
QPushButton:pressed { background: #272335; }
QPushButton:disabled { color: #6f6885; background: #211f2b; border-color: #2c2939; }
QPushButton#Primary {
    background: #8b5cf6; color: #ffffff; border: none; font-weight: 700;
    padding: 8px 16px;
}
QPushButton#Primary:hover { background: #9a6ff5; }
QPushButton#Primary:pressed { background: #7a48e6; }
QPushButton#Primary:disabled { background: #322c46; color: #6f6885; }
QPushButton#Success {
    background: #22a565; color: #ffffff; border: none; font-weight: 700;
    padding: 8px 16px;
}
QPushButton#Success:hover { background: #1c8a54; }
QPushButton#Success:disabled { background: #322c46; color: #6f6885; }

QToolButton {
    background: #2c2839; color: #ded9ec;
    border: 1px solid #3a3548; border-radius: 7px; padding: 5px 10px;
}
QToolButton:hover { background: #36314a; }
QToolButton::menu-indicator { image: none; }

/* ── inputs ───────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QTimeEdit, QSpinBox, QDateTimeEdit {
    background: #1b1a24; color: #eae9f2;
    border: 1px solid #3a3548; border-radius: 7px;
    padding: 5px 8px; selection-background-color: #43356e;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QTimeEdit:focus {
    border: 1px solid #8b5cf6;
}
QLineEdit:read-only { color: #a79fbb; background: #1d1c27; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #1d1c27; color: #eae9f2;
    border: 1px solid #3a3548; outline: 0;
    selection-background-color: #8b5cf6; selection-color: #ffffff;
}

/* ── tables / trees / lists ───────────────────────────────────────── */
QTableWidget, QTreeWidget, QListWidget {
    background: #1b1a24; color: #eae9f2;
    alternate-background-color: #211f2c;
    gridline-color: #2c2939;
    border: 1px solid #3a3548; border-radius: 8px;
    outline: 0;
}
QTableWidget::item, QTreeWidget::item, QListWidget::item { padding: 3px 4px; }
QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected {
    background: #43356e; color: #ffffff;
}
QTreeWidget::item:hover, QListWidget::item:hover { background: #2a2740; }
QHeaderView::section {
    background: #2c2839; color: #a79fbb;
    border: none; border-right: 1px solid #2c2939; border-bottom: 1px solid #3a3548;
    padding: 6px 8px; font-weight: 700;
}
QTableCornerButton::section { background: #2c2839; border: none; }

/* ── checkboxes ───────────────────────────────────────────────────── */
QCheckBox { color: #eae9f2; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #3a3548;
    border-radius: 4px; background: #1b1a24;
}
QCheckBox::indicator:checked { background: #8b5cf6; border-color: #8b5cf6; }

/* ── progress ─────────────────────────────────────────────────────── */
QProgressBar {
    background: #1b1a24; border: 1px solid #3a3548;
    border-radius: 7px; text-align: center; color: #eae9f2; height: 14px;
}
QProgressBar::chunk { background: #8b5cf6; border-radius: 6px; }

/* ── menus ────────────────────────────────────────────────────────── */
QMenuBar { background: #16151d; color: #eae9f2; border-bottom: 1px solid #2c2939; }
QMenuBar::item { background: transparent; padding: 6px 12px; }
QMenuBar::item:selected { background: #2c2839; border-radius: 6px; }
QMenu { background: #211f2b; color: #eae9f2; border: 1px solid #3a3548; padding: 4px; }
QMenu::item { padding: 6px 22px; border-radius: 5px; }
QMenu::item:selected { background: #8b5cf6; color: #ffffff; }
QMenu::separator { height: 1px; background: #2c2939; margin: 4px 8px; }

/* ── status bar ───────────────────────────────────────────────────── */
QStatusBar { background: #16151d; color: #a79fbb; border-top: 1px solid #2c2939; }
QStatusBar::item { border: none; }

/* ── splitter / scrollbars ────────────────────────────────────────── */
QSplitter::handle { background: #2c2939; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical { height: 2px; }
QScrollBar:vertical { background: transparent; width: 11px; margin: 0; }
QScrollBar::handle:vertical { background: #463f5c; border-radius: 5px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #5a5176; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 0; }
QScrollBar::handle:horizontal { background: #463f5c; border-radius: 5px; min-width: 28px; }
QScrollBar::handle:horizontal:hover { background: #5a5176; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ── project bar ──────────────────────────────────────────────────── */
#ProjectBar {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                stop:0 #221c33, stop:1 #17151f);
    border-bottom: 1px solid #3a3548;
}
#ProjectBar QLabel { background: transparent; }
#ProjectBarLabel { color: #a79fbb; font-weight: 700; }
#ProjectBarCombo { font-weight: 700; min-height: 26px; }
#ProjectBarRoot { color: #b196f5; }
#ProjectBarStructure {
    color: #c9b6f7; font-weight: 700;
    background: rgba(139,92,246,0.14);
    border: 1px solid rgba(155,124,255,0.35);
    border-radius: 11px; padding: 3px 10px;
}
#ProjectBarSep { color: #3a3548; max-width: 1px; }
#Wordmark {
    color: #b196f5; font-weight: 800; font-size: 15px; padding-right: 4px;
}

/* generic helper text */
QLabel#Hint { color: #a79fbb; }

/* ── side panel (Record / Transfer / Import) ──────────────────────── */
#SidePanelBar {
    background: #1d1b26; border-right: 1px solid #3a3548;
    min-width: 86px; max-width: 86px;
}
#SidePanelButton {
    font-size: 10px; font-weight: 700; border: none; border-radius: 9px;
    background: transparent; padding: 8px 4px; color: #a79fbb;
}
#SidePanelButton:hover { background: #2c2839; color: #eae9f2; }
#SidePanelButton:checked { background: #8b5cf6; color: #ffffff; }
"""
