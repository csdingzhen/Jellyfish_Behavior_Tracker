"""
ui/sidebar.py

Video browser sidebar — shows thumbnails for all videos in a folder.
Thumbnails are loaded asynchronously (up to 4 concurrent workers) so the
list stays responsive even with 24+ videos.

Emits video_selected(Path) when the user clicks an item.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, QSize
from qtpy.QtGui import QIcon, QPixmap, QImage, QColor
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QSizePolicy, QLineEdit,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
THUMB_W, THUMB_H = 96, 72

_pool = QThreadPool.globalInstance()
_pool.setMaxThreadCount(4)


# ── Async thumbnail loader ────────────────────────────────────────────────────

class _ThumbSignals(QObject):
    done = Signal(str, object)   # (path_str, rgb ndarray | None)


class _ThumbLoader(QRunnable):
    def __init__(self, video_path: Path, signals: _ThumbSignals):
        super().__init__()
        self._path   = video_path
        self.signals = signals
        self.setAutoDelete(True)

    def run(self):
        from .thumbnails import get_thumbnail
        rgb = get_thumbnail(self._path)
        self.signals.done.emit(str(self._path), rgb)


# ── Sidebar widget ────────────────────────────────────────────────────────────

class VideoSidebarWidget(QWidget):
    """Left-panel video browser with async thumbnail loading."""

    video_selected = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder:   Path | None                   = None
        self._item_map: dict[str, QListWidgetItem]    = {}
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Folder row
        folder_row = QHBoxLayout()
        self._folder_lbl = QLabel("No folder selected")
        self._folder_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._folder_lbl.setStyleSheet("font-size: 10px; color: #aaa;")
        self._folder_lbl.setWordWrap(False)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(72)
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_lbl)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # Filter box
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter videos…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        # Video list
        self._list = QListWidget()
        self._list.setIconSize(QSize(THUMB_W, THUMB_H))
        self._list.setSpacing(2)
        self._list.setUniformItemSizes(True)
        self._list.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

        # Status bar
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: #666;")
        layout.addWidget(self._status)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_folder(self, folder: Path):
        """Populate the list with all videos in *folder*, loading thumbnails async."""
        self._folder = folder
        name = folder.name if len(folder.name) < 32 else f"…{folder.name[-30:]}"
        self._folder_lbl.setText(name)
        self._folder_lbl.setToolTip(str(folder))

        self._item_map.clear()
        self._list.clear()

        videos = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        )
        placeholder = _placeholder_icon()
        for vp in videos:
            item = QListWidgetItem(placeholder, vp.name)
            item.setData(Qt.UserRole, vp)
            item.setToolTip(str(vp))
            self._list.addItem(item)
            self._item_map[str(vp)] = item

            sigs = _ThumbSignals()
            sigs.done.connect(self._on_thumb_done)
            _pool.start(_ThumbLoader(vp, sigs))

        n = len(videos)
        self._status.setText(f"{n} video{'s' if n != 1 else ''}")
        self._apply_filter(self._search.text())

    def select_video(self, path: Path):
        """Programmatically highlight *path* in the list."""
        item = self._item_map.get(str(path))
        if item:
            self._list.blockSignals(True)
            self._list.setCurrentItem(item)
            self._list.blockSignals(False)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _browse(self):
        try:
            from config import VIDEO_DIR
            start = str(self._folder or VIDEO_DIR)
        except Exception:
            start = str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select video folder", start)
        if folder:
            self.load_folder(Path(folder))

    def _on_item_changed(self, current: QListWidgetItem | None, _prev):
        if current is None:
            return
        self.video_selected.emit(current.data(Qt.UserRole))

    def _on_thumb_done(self, path_str: str, rgb):
        item = self._item_map.get(path_str)
        if item is None or rgb is None:
            return
        h, w, c = rgb.shape
        qi   = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        item.setIcon(QIcon(QPixmap.fromImage(qi.copy())))

    def _apply_filter(self, text: str):
        q = text.strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(bool(q) and q not in item.text().lower())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _placeholder_icon() -> QIcon:
    px = QPixmap(THUMB_W, THUMB_H)
    px.fill(QColor("#2a2a2a"))
    return QIcon(px)
