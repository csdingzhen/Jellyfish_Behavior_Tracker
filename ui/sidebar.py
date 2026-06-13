"""
ui/sidebar.py

Left-panel video browser with:
  • Async thumbnail loading (up to 4 concurrent workers)
  • Per-video status badges (recording / queued / processing / done / failed)
  • Sequential auto-queue: completed videos are queued and processed one at a time
  • FolderWatcher integration: new recordings are detected and auto-queued
    when auto-queue mode is enabled

Signals
-------
video_selected(Path)  — user clicked a video (load into viewer)
queue_start(Path)     — next video in queue is ready to process;
                        app.py resolves bell/dye clicks and starts the worker
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, QSize
from qtpy.QtGui import QIcon, QPixmap, QImage, QColor, QPainter, QBrush
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QSizePolicy, QLineEdit, QMenu, QAction,
)

from .watcher import FolderWatcher, VideoStatus, VIDEO_EXTS

THUMB_W, THUMB_H = 96, 72
STATUS_DOT_SIZE  = 10   # px diameter of the status indicator dot

_STATUS_COLORS = {
    VideoStatus.UNKNOWN:    "#555555",
    VideoStatus.RECORDING:  "#cc8833",
    VideoStatus.QUEUED:     "#cccc33",
    VideoStatus.PROCESSING: "#3388cc",
    VideoStatus.DONE:       "#33aa55",
    VideoStatus.FAILED:     "#cc3333",
    VideoStatus.SKIPPED:    "#666666",
}
_STATUS_BG = {
    VideoStatus.UNKNOWN:    None,
    VideoStatus.RECORDING:  "#2a1e00",
    VideoStatus.QUEUED:     "#2a2a00",
    VideoStatus.PROCESSING: "#001a2a",
    VideoStatus.DONE:       "#002a10",
    VideoStatus.FAILED:     "#2a0000",
    VideoStatus.SKIPPED:    None,
}

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
    """Left-panel video browser with async thumbnails, status badges, and auto-queue."""

    video_selected = Signal(Path)   # user clicked a video
    queue_start    = Signal(Path)   # pipeline should start for this video

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder:     Path | None                = None
        self._item_map:   dict[str, QListWidgetItem] = {}   # path_str → item
        self._statuses:   dict[str, VideoStatus]     = {}   # path_str → status
        self._queue:      list[Path]                 = []   # ordered processing queue
        self._processing: Path | None               = None  # currently running video
        self._auto_queue: bool                       = False

        self._watcher = FolderWatcher(self)
        self._watcher.file_appeared.connect(self._on_file_appeared)
        self._watcher.file_ready.connect(self._on_file_ready)

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
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(72)
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_lbl)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # Watch toggle row
        watch_row = QHBoxLayout()
        self._watch_btn = QPushButton("Watch OFF")
        self._watch_btn.setCheckable(True)
        self._watch_btn.setFixedWidth(80)
        self._watch_btn.setToolTip(
            "Monitor this folder for new video files.\n"
            "When a recording finishes writing to disk, it is automatically\n"
            "added to the processing queue (if auto-queue is enabled)."
        )
        self._watch_btn.toggled.connect(self._on_watch_toggled)
        self._watch_lbl = QLabel("Idle")
        self._watch_lbl.setStyleSheet("font-size: 10px; color: #666;")
        watch_row.addWidget(self._watch_btn)
        watch_row.addWidget(self._watch_lbl, stretch=1)
        layout.addLayout(watch_row)

        # Filter
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
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._list)

        # Queue / status footer
        footer_row = QHBoxLayout()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("font-size: 10px; color: #666;")
        self._queue_lbl = QLabel("")
        self._queue_lbl.setStyleSheet("font-size: 10px; color: #aacc88;")
        footer_row.addWidget(self._status_lbl, stretch=1)
        footer_row.addWidget(self._queue_lbl)
        layout.addLayout(footer_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_folder(self, folder: Path):
        """Populate the list with all videos in *folder*."""
        self._folder = folder
        name = folder.name if len(folder.name) < 32 else f"…{folder.name[-30:]}"
        self._folder_lbl.setText(name)
        self._folder_lbl.setToolTip(str(folder))

        self._item_map.clear()
        self._statuses.clear()
        self._list.clear()

        videos = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        )
        placeholder = _placeholder_icon()
        for vp in videos:
            self._add_item(vp, placeholder)

        n = len(videos)
        self._status_lbl.setText(f"{n} video{'s' if n != 1 else ''}")
        self._apply_filter(self._search.text())

        # Restart watcher on new folder if it was active
        if self._watcher.is_active:
            self._watcher.watch(folder)

    def select_video(self, path: Path):
        """Programmatically highlight *path* without emitting video_selected."""
        item = self._item_map.get(str(path))
        if item:
            self._list.blockSignals(True)
            self._list.setCurrentItem(item)
            self._list.blockSignals(False)

    def set_auto_queue(self, enabled: bool):
        self._auto_queue = enabled

    # ── Queue management (called by app.py) ───────────────────────────────────

    def enqueue(self, path: Path):
        """Add *path* to the processing queue and start it if the GPU is free."""
        key = str(path)
        if key not in self._item_map:
            self._add_item(path, _placeholder_icon())
            sigs = _ThumbSignals()
            sigs.done.connect(self._on_thumb_done)
            _pool.start(_ThumbLoader(path, sigs))

        if self._statuses.get(key) in (
            VideoStatus.QUEUED, VideoStatus.PROCESSING, VideoStatus.DONE
        ):
            return   # already handled

        self._set_status(path, VideoStatus.QUEUED)
        self._queue.append(path)
        self._update_queue_label()
        self._try_start_next()

    def mark_processing(self, path: Path):
        self._processing = path
        self._set_status(path, VideoStatus.PROCESSING)
        self._update_queue_label()

    def mark_done(self, path: Path):
        self._set_status(path, VideoStatus.DONE)
        if self._processing == path:
            self._processing = None
        self._update_queue_label()
        self._try_start_next()

    def mark_failed(self, path: Path):
        self._set_status(path, VideoStatus.FAILED)
        if self._processing == path:
            self._processing = None
        self._update_queue_label()
        self._try_start_next()

    # ── Watcher callbacks ─────────────────────────────────────────────────────

    def _on_file_appeared(self, path: Path):
        """New file detected — still being written."""
        key = str(path)
        if key not in self._item_map:
            self._add_item(path, _placeholder_icon())
            sigs = _ThumbSignals()
            sigs.done.connect(self._on_thumb_done)
            _pool.start(_ThumbLoader(path, sigs))
        self._set_status(path, VideoStatus.RECORDING)
        self._watch_lbl.setText(f"Recording: {path.name}")

    def _on_file_ready(self, path: Path):
        """Recording finished — file is fully written."""
        self._watch_lbl.setText(f"Ready: {path.name}")
        if self._auto_queue:
            self.enqueue(path)
        else:
            self._set_status(path, VideoStatus.UNKNOWN)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add_item(self, vp: Path, icon: QIcon):
        item = QListWidgetItem(icon, vp.name)
        item.setData(Qt.UserRole, vp)
        item.setToolTip(str(vp))
        self._list.addItem(item)
        self._item_map[str(vp)] = item
        self._statuses[str(vp)] = VideoStatus.UNKNOWN
        self._apply_status_style(item, VideoStatus.UNKNOWN)

        sigs = _ThumbSignals()
        sigs.done.connect(self._on_thumb_done)
        _pool.start(_ThumbLoader(vp, sigs))

    def _set_status(self, path: Path, status: VideoStatus):
        key  = str(path)
        self._statuses[key] = status
        item = self._item_map.get(key)
        if item:
            self._apply_status_style(item, status)

    def _apply_status_style(self, item: QListWidgetItem, status: VideoStatus):
        bg  = _STATUS_BG.get(status)
        dot = _STATUS_COLORS.get(status, "#555")

        if bg:
            item.setBackground(QBrush(QColor(bg)))
        else:
            item.setBackground(QBrush())   # default

        # Overlay a small colored dot on the existing icon
        existing = item.icon()
        px = existing.pixmap(THUMB_W, THUMB_H)
        if not px.isNull():
            painter = QPainter(px)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(QColor(dot)))
            painter.setPen(Qt.NoPen)
            r = STATUS_DOT_SIZE
            painter.drawEllipse(4, 4, r, r)
            painter.end()
            item.setIcon(QIcon(px))

        item.setToolTip(f"{item.data(Qt.UserRole)}\nStatus: {status.value}")

    def _try_start_next(self):
        if self._processing is not None:
            return   # GPU busy with another video
        while self._queue:
            nxt = self._queue.pop(0)
            if self._statuses.get(str(nxt)) == VideoStatus.QUEUED:
                self.queue_start.emit(nxt)
                return
        self._update_queue_label()

    def _update_queue_label(self):
        n_queued = sum(
            1 for s in self._statuses.values() if s == VideoStatus.QUEUED
        )
        n_done = sum(
            1 for s in self._statuses.values() if s == VideoStatus.DONE
        )
        parts = []
        if self._processing:
            parts.append(f"Processing: {self._processing.name}")
        if n_queued:
            parts.append(f"{n_queued} queued")
        if n_done:
            parts.append(f"{n_done} done")
        self._queue_lbl.setText("  ".join(parts))

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

    def _on_watch_toggled(self, checked: bool):
        if checked:
            if self._folder:
                self._watcher.watch(self._folder)
            self._watch_btn.setText("Watch ON")
            self._watch_btn.setStyleSheet("background: #224433;")
            self._watch_lbl.setText("Watching…")
        else:
            self._watcher.stop()
            self._watch_btn.setText("Watch OFF")
            self._watch_btn.setStyleSheet("")
            self._watch_lbl.setText("Idle")

    def _on_item_changed(self, current: QListWidgetItem | None, _prev):
        if current is None:
            return
        self.video_selected.emit(current.data(Qt.UserRole))

    def _on_thumb_done(self, path_str: str, rgb):
        item = self._item_map.get(path_str)
        if item is None or rgb is None:
            return
        h, w, c = rgb.shape
        qi  = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        new_px = QPixmap.fromImage(qi.copy())
        # Re-apply the status dot on the fresh thumbnail
        status = self._statuses.get(path_str, VideoStatus.UNKNOWN)
        dot = _STATUS_COLORS.get(status, "#555")
        painter = QPainter(new_px)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(dot)))
        painter.setPen(Qt.NoPen)
        r = STATUS_DOT_SIZE
        painter.drawEllipse(4, 4, r, r)
        painter.end()
        item.setIcon(QIcon(new_px))

    def _apply_filter(self, text: str):
        q = text.strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(bool(q) and q not in item.text().lower())

    def _on_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is None:
            return
        path   = item.data(Qt.UserRole)
        status = self._statuses.get(str(path), VideoStatus.UNKNOWN)

        menu = QMenu(self)
        if status not in (VideoStatus.QUEUED, VideoStatus.PROCESSING, VideoStatus.DONE):
            add_act = QAction("Add to queue", self)
            add_act.triggered.connect(lambda: self.enqueue(path))
            menu.addAction(add_act)

        if status == VideoStatus.QUEUED:
            skip_act = QAction("Remove from queue", self)
            skip_act.triggered.connect(
                lambda: self._set_status(path, VideoStatus.SKIPPED)
            )
            menu.addAction(skip_act)

        if status == VideoStatus.DONE:
            redo_act = QAction("Re-queue (reprocess)", self)
            redo_act.triggered.connect(lambda: self.enqueue(path))
            menu.addAction(redo_act)

        if not menu.isEmpty():
            menu.exec(self._list.viewport().mapToGlobal(pos))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _placeholder_icon() -> QIcon:
    px = QPixmap(THUMB_W, THUMB_H)
    px.fill(QColor("#2a2a2a"))
    return QIcon(px)
