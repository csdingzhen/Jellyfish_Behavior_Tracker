"""
ui/widget.py

Main dock widget containing the Calibrate and Process workflow tabs.
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
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── Tab: Calibrate ────────────────────────────────────────────────────
        from .calibration import CalibrationTab
        self.calib_tab = CalibrationTab(self.viewer)
        self.tabs.addTab(self.calib_tab, "Calibrate")

        # ── Tab: Process ──────────────────────────────────────────────────────
        from .processing import ProcessingTab
        self.process_tab = ProcessingTab(self.viewer)
        self.tabs.addTab(self.process_tab, "Process")
