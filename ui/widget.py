"""
ui/widget.py

Main dock widget containing the Calibrate and Process workflow tabs.
Tabs are constructed lazily on first activation to keep startup fast.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLabel,
)
from qtpy.QtCore import Qt, QTimer, Signal

from .project import ProjectBar
from .hardware import HardwareWidget
from .style import STYLESHEET


class CassiopeaWidget(QWidget):
    """
    Top-level dock widget with:
      - ProjectBar (New / Open / Save + project name label)
      - HardwareWidget (compact GPU status bar + auto-queue toggle)
      - Two lazy tabs: Calibrate and Process

    Signals
    -------
    pipeline_finished(Path, object) — forwarded from ProcessingTab whenever a
    manual run (started via its own "Run pipeline" button, not the sidebar
    queue) finishes. app.py listens for this to keep the sidebar status dot
    and continuity-click propagation in sync for manual runs too.
    """

    pipeline_finished = Signal(Path, object)

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.calib_tab   = None
        self.process_tab = None
        self._pending_video: Path | None = None
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(STYLESHEET)
        # Keep the outer container transparent so napari's dock background
        # shows through instead of creating a black-box effect.
        self.setAttribute(Qt.WA_StyledBackground, False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Project bar at the very top
        self.project_bar = ProjectBar()
        self.project_bar.project_changed.connect(self._on_project_changed)
        layout.addWidget(self.project_bar)

        # Collapsible hardware panel
        self.hw_widget = HardwareWidget()
        layout.addWidget(self.hw_widget)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Placeholder widgets; real tabs built on first activation.
        self.tabs.addTab(_placeholder("Calibrate"), "Calibrate")
        self.tabs.addTab(_placeholder("Process"),   "Process")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        # Defer first-tab build until after the event loop starts so the
        # window appears before the heavy CalibrationTab imports run.
        QTimer.singleShot(0, lambda: self._on_tab_changed(0))

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
            self.process_tab.pipeline_finished.connect(self.pipeline_finished.emit)
            self.tabs.removeTab(1)
            self.tabs.insertTab(1, self.process_tab, "Process")
            self.tabs.setCurrentIndex(1)
            # Sync current project state immediately — project_changed may
            # have already fired (e.g. project opened while still on the
            # Calibrate tab) before this tab existed to receive it.
            if self.project_bar.project is not None:
                self.process_tab.on_project_changed(self.project_bar.project)
            # Deliver any video that was selected before the tab existed
            if self._pending_video is not None:
                self.process_tab.load_video(self._pending_video)
                self._pending_video = None

    def on_video_selected(self, path: Path):
        """Called by the VideoSidebarWidget when the user clicks a video."""
        if self.process_tab is not None:
            self.process_tab.load_video(path)
        else:
            self._pending_video = path

    def _on_project_changed(self, state):
        # Propagate to the sidebar (app.py wires this up via project_bar.project_changed)
        if self.process_tab is not None:
            self.process_tab.on_project_changed(state)


def _placeholder(name: str) -> QWidget:
    w = QWidget()
    lbl = QLabel(f"Loading {name}…")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("color: #888; font-style: italic;")
    lay = QVBoxLayout(w)
    lay.addWidget(lbl)
    return w
