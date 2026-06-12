"""
ui/widget.py

Main dock widget containing the Calibrate and Process workflow tabs.
Tabs are constructed lazily on first activation to keep startup fast.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLabel,
)
from qtpy.QtCore import Qt


class CassiopeaWidget(QWidget):
    """
    Top-level dock widget with two tabs:
      - Calibrate: one-time per-animal annotation of rhopalia positions
      - Process:   video processing workflow

    Each tab is constructed on first activation (lazy) so startup cost is
    paid only for the tab the user actually opens first.
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.calib_tab   = None
        self.process_tab = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Add lightweight placeholder widgets; real tabs built on first click.
        self.tabs.addTab(_placeholder("Calibrate"), "Calibrate")
        self.tabs.addTab(_placeholder("Process"),   "Process")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        # Build the first tab immediately so it's ready without a click.
        self._on_tab_changed(0)

    def _on_tab_changed(self, index: int):
        if index == 0 and self.calib_tab is None:
            from .calibration import CalibrationTab
            self.calib_tab = CalibrationTab(self.viewer)
            self.tabs.removeTab(0)
            self.tabs.insertTab(0, self.calib_tab, "Calibrate")
            self.tabs.setCurrentIndex(0)

        elif index == 1 and self.process_tab is None:
            from .processing import ProcessingTab
            self.process_tab = ProcessingTab(self.viewer)
            self.tabs.removeTab(1)
            self.tabs.insertTab(1, self.process_tab, "Process")
            self.tabs.setCurrentIndex(1)


def _placeholder(name: str) -> QWidget:
    w = QWidget()
    lbl = QLabel(f"Loading {name}…")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("color: #888; font-style: italic;")
    lay = QVBoxLayout(w)
    lay.addWidget(lbl)
    return w
