"""
ui/style.py

Global dark-theme stylesheet and shared visual helpers.
"""

from __future__ import annotations
from pathlib import Path as _Path
from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from qtpy.QtCore import Qt

_ARROW_SVG = str(_Path(__file__).parent / "assets" / "arrow_down.svg").replace("\\", "/")

# ── Palette ───────────────────────────────────────────────────────────────────

C_BG        = "#262930"   # scroll-area / pane background
C_CARD      = "#202020"   # card surface
C_CARD_ALT  = "#1a1a1a"   # slightly recessed (e.g. input fields)
C_BORDER    = "#2e2e2e"   # input borders
C_BORDER_LO = "#252525"   # inner separator lines

C_TEXT      = "#e0e0e0"   # primary text
C_TEXT_DIM  = "#777777"   # secondary / hint text
C_TEXT_MONO = "#cc9944"   # inline code / coordinates

C_GREEN     = "#22c55e"
C_RED       = "#ef4444"
C_BLUE      = "#3b72d4"
C_ORANGE    = "#f59e0b"
C_GRAY      = "#444444"

# ── Master QSS ───────────────────────────────────────────────────────────────
#
# Key design rule: do NOT set background on the generic QWidget selector.
# That would paint the outermost plugin container dark and make it look like
# a black box inside napari's dock frame.  Instead, backgrounds are set only
# on the specific containers that need them (scroll area, tab pane, cards).

STYLESHEET = f"""
/* ── Base: colour and font only — no background on the outer shell ───────── */
QWidget {{
    color: {C_TEXT};
    font-size: 12px;
}}

/* ── Dark background only inside the scroll areas and tab pane ───────────── */
QScrollArea {{
    background: {C_BG};
    border: none;
}}
QScrollArea > QWidget {{
    background: {C_BG};
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: {C_BG};
    border: none;
}}

QScrollBar:vertical {{
    background: {C_BG};
    width: 6px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: #383838;
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Cards (QFrame with objectName "card") ───────────────────────────────── */
QFrame#card {{
    background: {C_CARD};
    border: none;
    border-radius: 8px;
}}

/* Direct content-holder QWidgets inside cards should be transparent so they
   show the card's own C_CARD background rather than the OS / napari palette. */
QFrame#card > QWidget {{
    background: transparent;
}}

/* Magicgui parameter container — needs an explicit transparent override
   because magicgui manages its own native QWidget outside our tree. */
#paramContainer {{
    background: transparent;
    border: none;
}}

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton {{
    background: #2a2a2a;
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 4px 12px;
    min-height: 26px;
}}
QPushButton:hover  {{ background: #333333; border-color: #444; }}
QPushButton:pressed {{ background: #1e1e1e; }}
QPushButton:disabled {{ color: #444; border-color: #252525; }}

QPushButton#runBtn {{
    background: {C_BLUE};
    color: #ffffff;
    font-weight: bold;
    font-size: 14px;
    border: none;
    border-radius: 8px;
    padding: 10px;
    min-height: 42px;
}}
QPushButton#runBtn:hover   {{ background: #4a82e4; }}
QPushButton#runBtn:pressed {{ background: #2a5bb8; }}
QPushButton#runBtn:disabled {{ background: #1e2d50; color: #557; }}

QPushButton#cancelBtn {{
    background: #2a2020;
    color: #cc7777;
    border-color: #4a2a2a;
}}
QPushButton#cancelBtn:hover {{ background: #352020; }}

QPushButton#retryBtn {{
    background: #2d1a1a;
    color: #ff9999;
    border: 1px solid #5a2a2a;
    border-radius: 5px;
    padding: 3px 10px;
}}
QPushButton#retryBtn:hover {{ background: #3d2222; }}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {C_CARD_ALT};
    border: 1px solid {C_BORDER};
    border-radius: 5px;
    padding: 3px 6px;
    color: {C_TEXT};
    selection-background-color: {C_BLUE};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {C_BLUE};
}}

QComboBox {{
    background: {C_CARD_ALT};
    border: 1px solid {C_BORDER};
    border-radius: 5px;
    padding: 3px 6px;
    color: {C_TEXT};
}}
QComboBox:focus {{ border-color: {C_BLUE}; }}
QComboBox::drop-down {{
    border-left: 1px solid {C_BORDER};
    width: 22px;
    background: transparent;
    subcontrol-origin: padding;
    subcontrol-position: right center;
}}
QComboBox::down-arrow {{
    image: url("{_ARROW_SVG}");
    width: 10px;
    height: 6px;
}}
QComboBox QAbstractItemView {{
    background: #252525;
    border: 1px solid {C_BORDER};
    selection-background-color: {C_BLUE};
    color: {C_TEXT};
    outline: none;
}}

/* ── List widget (video sidebar) ─────────────────────────────────────────── */
QListWidget {{
    background: {C_BG};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    outline: none;
}}
QListWidget::item {{
    border: none;
    border-bottom: 1px solid {C_BORDER_LO};
}}
QListWidget::item:selected {{
    background: #2a3d5c;
}}
QListWidget::item:hover:!selected {{
    background: #242424;
}}

/* ── Context menu ────────────────────────────────────────────────────────── */
QMenu {{
    background: #232323;
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    padding: 2px;
}}
QMenu::item {{
    padding: 5px 18px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {C_BLUE};
    color: #ffffff;
}}
QMenu::separator {{
    height: 1px;
    background: {C_BORDER};
    margin: 4px 6px;
}}

/* ── CheckBox ────────────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 6px;
    color: {C_TEXT};
    background: transparent;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid #555;
    border-radius: 3px;
    background: {C_CARD_ALT};
}}
QCheckBox::indicator:checked {{
    background: {C_BLUE};
    border-color: {C_BLUE};
}}
QCheckBox::indicator:hover {{
    border-color: #777;
}}

/* ── Tab widget ─────────────────────────────────────────────────────────── */
QTabWidget::pane {{ border: none; background: {C_BG}; }}
QTabBar           {{ background: transparent; }}
QTabBar::tab {{
    background: transparent;
    color: {C_TEXT_DIM};
    padding: 7px 18px;
    border-bottom: 2px solid transparent;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    color: {C_TEXT};
    border-bottom: 2px solid {C_TEXT};
}}
QTabBar::tab:hover:!selected {{ color: #aaa; }}

/* ── Table ───────────────────────────────────────────────────────────────── */
QTableWidget {{
    background: {C_CARD_ALT};
    border: 1px solid {C_BORDER};
    gridline-color: {C_BORDER};
    color: {C_TEXT};
}}
QTableWidget::item {{
    padding: 3px 6px;
}}
QTableWidget::item:selected {{
    background: {C_BLUE};
    color: #ffffff;
}}
QHeaderView::section {{
    background: {C_CARD};
    color: {C_TEXT_DIM};
    border: none;
    border-bottom: 1px solid {C_BORDER};
    padding: 4px 6px;
    font-size: 11px;
}}
QHeaderView::section:hover {{
    background: #282828;
}}

/* ── ProgressBar (hidden overall bar) ───────────────────────────────────── */
QProgressBar {{
    background: #252525;
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    text-align: center;
    color: {C_TEXT};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background: {C_BLUE};
    border-radius: 3px;
}}

/* ── Tooltip ─────────────────────────────────────────────────────────────── */
QToolTip {{
    background: #2a2a2a;
    color: {C_TEXT};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 4px;
}}
"""


# ── Widget helpers ────────────────────────────────────────────────────────────

def card(parent=None, padding: int = 12) -> QFrame:
    """Return a styled card QFrame.  Styled via QFrame#card in STYLESHEET."""
    f = QFrame(parent)
    f.setFrameShape(QFrame.NoFrame)
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(padding, padding, padding, padding)
    lay.setSpacing(6)
    return f


def step_badge(number: int | str, color: str = C_BLUE) -> QLabel:
    """Return a small circular step-number badge."""
    lbl = QLabel(str(number))
    lbl.setFixedSize(22, 22)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(f"""
        QLabel {{
            background: {color};
            color: #ffffff;
            font-weight: bold;
            font-size: 11px;
            border-radius: 11px;
        }}
    """)
    return lbl


def status_icon(color: str = C_GRAY, size: int = 12) -> QLabel:
    """Return a small square status indicator."""
    lbl = QLabel()
    lbl.setFixedSize(size, size)
    _set_icon_color(lbl, color, size)
    return lbl


def _set_icon_color(lbl: QLabel, color: str, size: int = 12) -> None:
    lbl.setStyleSheet(f"""
        QLabel {{
            background: {color};
            border-radius: {size // 4}px;
        }}
    """)


def add_step_header(lay, number: int, title: str) -> None:
    """Prepend a numbered badge + bold title row to an existing card layout."""
    hdr = QHBoxLayout()
    hdr.addWidget(step_badge(number))
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(
        f"font-weight: bold; color: {C_TEXT}; font-size: 12px;"
    )
    hdr.addWidget(title_lbl)
    hdr.addStretch()
    lay.addLayout(hdr)


def section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; font-weight: bold;")
    return lbl


def dim_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
    return lbl
