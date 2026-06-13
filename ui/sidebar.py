"""
ui/sidebar.py

Left-panel video browser with:
  • Async thumbnail loading (up to 4 concurrent workers)
  • Per-video inline progress bar + status text (via setItemWidget)
  • Status dot overlay on thumbnail so queue state is visible at a glance
  • File-removal detection: items auto-removed when deleted from disk
    (keeps rows whose pipeline has already finished so results remain browsable)
  • Sequential auto-queue: recordings processed one at a time
  • FolderWatcher integration: new recordings appear automatically

Signals
-------
video_selected(Path)  — user clicked a video
queue_start(Path)     — app.py should start the pipeline for this video
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, QSize
from qtpy.QtGui import QIcon, QPixmap, QImage, QColor, QPainter, QBrush
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QSizePolicy, QLineEdit, QMenu, QAction,
    QProgressBar,
)

from .watcher import FolderWatcher, VideoStatus, VIDEO_EXTS

THUMB_W, THUMB_H = 96, 72
DOT_R            = 10   # status-dot diameter on thumbnail

_DOT_COLOR = {
    VideoStatus.UNKNOWN:    "#555555",
    VideoStatus.RECORDING:  "#cc8833",
    VideoStatus.QUEUED:     "#cccc33",
    VideoStatus.PROCESSING: "#3388cc",
    VideoStatus.DONE:       "#33aa55",
    VideoStatus.FAILED:     "#cc3333",
    VideoStatus.SKIPPED:    "#666666",
}
_BAR_COLOR = {
    VideoStatus.UNKNOWN:    "#555555",
    VideoStatus.RECORDING:  "#cc8833",
    VideoStatus.QUEUED:     "#886600",
    VideoStatus.PROCESSING: "#3388cc",
    VideoStatus.DONE:       "#33aa55",
    VideoStatus.FAILED:     "#cc3333",
    VideoStatus.SKIPPED:    "#555555",
}
_STATUS_TEXT = {
    VideoStatus.UNKNOWN:    ("—",           0.0),
    VideoStatus.RECORDING:  ("Recording…",  0.0),
    VideoStatus.QUEUED:     ("Waiting",     0.0),
    VideoStatus.PROCESSING: ("Processing…", 0.0),
    VideoStatus.DONE:       ("Completed",   1.0),
    VideoStatus.FAILED:     ("Failed",      0.0),
    VideoStatus.SKIPPED:    ("Skipped",     0.0),
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


# ── Per-item widget ───────────────────────────────────────────────────────────

class _VideoItemWidget(QWidget):
    """
    Embedded into each QListWidgetItem.  Displays filename, a thin
    progress bar, and a one-line status string.

    The QListWidgetItem keeps its icon (thumbnail with status dot) on the
    left; this widget occupies the text area to the right.
    """

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(THUMB_H)
        self.setAutoFillBackground(True)   # required for correct selection highlight

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 10, 4, 10)
        layout.setSpacing(4)

        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._name_lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(self._bar_style("#555555"))
        layout.addWidget(self._bar)

        self._status_lbl = QLabel("—")
        self._status_lbl.setStyleSheet("font-size: 9px; color: #888;")
        layout.addWidget(self._status_lbl)

        layout.addStretch()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_status(self, status: VideoStatus):
        text, frac = _STATUS_TEXT.get(status, ("—", 0.0))
        color = _BAR_COLOR.get(status, "#555555")
        self._bar.setValue(int(frac * 100))
        self._bar.setStyleSheet(self._bar_style(color))
        self._status_lbl.setText(text)

    def set_progress(self, fraction: float, task_name: str = ""):
        pct = int(fraction * 100)
        self._bar.setValue(pct)
        self._bar.setStyleSheet(self._bar_style("#3388cc"))
        label = f"{task_name}  {pct}%" if task_name else f"Processing  {pct}%"
        self._status_lbl.setText(label)

    def mark_file_removed(self):
        self._status_lbl.setText("File removed from disk")
        self._status_lbl.setStyleSheet("font-size: 9px; color: #886666;")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bar_style(color: str) -> str:
        return (
            "QProgressBar {"
            "  background: #333; border: none; border-radius: 3px;"
            "}"
            f"QProgressBar::chunk {{"
            f"  background: {color}; border-radius: 3px;"
            f"}}"
        )


# ── Sidebar widget ────────────────────────────────────────────────────────────

class VideoSidebarWidget(QWidget):
    video_selected = Signal(Path)
    queue_start    = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder:       Path | None                  = None
        self._item_map:     dict[str, QListWidgetItem]   = {}
        self._widget_map:   dict[str, _VideoItemWidget]  = {}
        self._statuses:     dict[str, VideoStatus]       = {}
        self._clean_thumbs: dict[str, QPixmap]           = {}
        self._queue:        list[Path]                   = []
        self._processing:   Path | None                  = None
        self._auto_queue:   bool                         = False

        self._watcher = FolderWatcher(self)
        self._watcher.file_appeared.connect(self._on_file_appeared)
        self._watcher.file_ready.connect(self._on_file_ready)
        self._watcher.file_removed.connect(self._on_file_removed)

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
        self._watch_btn.setFixedWidth(82)
        self._watch_btn.setToolTip(
            "Monitor this folder for new video files.\n"
            "New recordings appear here automatically when writing finishes."
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
        self._list.setUniformItemSizes(False)
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._list)

        # Footer
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
        self._folder = folder
        name = folder.name if len(folder.name) < 32 else f"…{folder.name[-30:]}"
        self._folder_lbl.setText(name)
        self._folder_lbl.setToolTip(str(folder))

        self._item_map.clear()
        self._widget_map.clear()
        self._statuses.clear()
        self._clean_thumbs.clear()
        self._list.clear()

        videos = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        )
        for vp in videos:
            self._add_item(vp)

        n = len(videos)
        self._status_lbl.setText(f"{n} video{'s' if n != 1 else ''}")
        self._apply_filter(self._search.text())

        if self._watcher.is_active:
            self._watcher.watch(folder)

    def select_video(self, path: Path):
        item = self._item_map.get(str(path))
        if item:
            self._list.blockSignals(True)
            self._list.setCurrentItem(item)
            self._list.blockSignals(False)

    def set_auto_queue(self, enabled: bool):
        self._auto_queue = enabled

    # ── Queue management (called by app.py) ───────────────────────────────────

    def enqueue(self, path: Path):
        key = str(path)
        if key not in self._item_map:
            self._add_item(path)
        if self._statuses.get(key) in (
            VideoStatus.QUEUED, VideoStatus.PROCESSING, VideoStatus.DONE
        ):
            return
        self._set_status(path, VideoStatus.QUEUED)
        self._queue.append(path)
        self._update_footer()
        self._try_start_next()

    def mark_processing(self, path: Path):
        self._processing = path
        self._set_status(path, VideoStatus.PROCESSING)
        self._update_footer()

    def mark_done(self, path: Path):
        self._set_status(path, VideoStatus.DONE)
        if self._processing == path:
            self._processing = None
        self._update_footer()
        self._try_start_next()

    def mark_failed(self, path: Path):
        self._set_status(path, VideoStatus.FAILED)
        if self._processing == path:
            self._processing = None
        self._update_footer()
        self._try_start_next()

    def update_video_progress(self, path: Path, fraction: float, task_name: str = ""):
        """Called from app.py on every ProgressEvent yielded by the worker."""
        w = self._widget_map.get(str(path))
        if w is not None:
            w.set_progress(fraction, task_name)

    # ── Watcher callbacks ─────────────────────────────────────────────────────

    def _on_file_appeared(self, path: Path):
        key = str(path)
        if key not in self._item_map:
            self._add_item(path)
        self._set_status(path, VideoStatus.RECORDING)
        self._watch_lbl.setText(f"Recording: {path.name}")

    def _on_file_ready(self, path: Path):
        self._watch_lbl.setText(f"Ready: {path.name}")
        if self._auto_queue:
            self.enqueue(path)
        else:
            self._set_status(path, VideoStatus.UNKNOWN)

    def _on_file_removed(self, path: Path):
        key    = str(path)
        status = self._statuses.get(key, VideoStatus.UNKNOWN)

        if status == VideoStatus.DONE:
            # Results are already saved — keep the row, mark it
            w = self._widget_map.get(key)
            if w:
                w.mark_file_removed()
            return

        if status == VideoStatus.PROCESSING:
            # Pipeline is already running; it will fail on its own when
            # it tries to read the missing file. Leave the row as-is.
            return

        # Remove from the list and all maps
        item = self._item_map.pop(key, None)
        if item is not None:
            row = self._list.row(item)
            self._list.takeItem(row)
        self._widget_map.pop(key, None)
        self._statuses.pop(key, None)
        self._clean_thumbs.pop(key, None)
        self._queue = [p for p in self._queue if str(p) != key]

        self._update_footer()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add_item(self, vp: Path):
        item = QListWidgetItem(_placeholder_icon(), "")
        item.setData(Qt.UserRole, vp)
        item.setSizeHint(QSize(0, THUMB_H + 4))
        self._list.addItem(item)

        iw = _VideoItemWidget(vp.name)
        self._list.setItemWidget(item, iw)

        self._item_map[str(vp)]   = item
        self._widget_map[str(vp)] = iw
        self._statuses[str(vp)]   = VideoStatus.UNKNOWN

        sigs = _ThumbSignals()
        sigs.done.connect(self._on_thumb_done)
        _pool.start(_ThumbLoader(vp, sigs))

    def _set_status(self, path: Path, status: VideoStatus):
        key = str(path)
        self._statuses[key] = status
        item = self._item_map.get(key)
        iw   = self._widget_map.get(key)
        if item:
            self._apply_dot(item, status)
        if iw:
            iw.set_status(status)

    def _apply_dot(self, item: QListWidgetItem, status: VideoStatus):
        """
        Redraw the thumbnail icon from the clean stored copy, then paint a
        fresh status dot on top.  Never reads the current icon so dots
        cannot accumulate across status changes.
        """
        path_str = str(item.data(Qt.UserRole))
        clean    = self._clean_thumbs.get(path_str)

        if clean is not None:
            px = clean.copy()
        else:
            px = QPixmap(THUMB_W, THUMB_H)
            px.fill(QColor("#2a2a2a"))

        if status != VideoStatus.UNKNOWN:
            color = _DOT_COLOR.get(status, "#555555")
            p = QPainter(px)
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(QBrush(QColor(color)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(4, 4, DOT_R, DOT_R)
            p.end()

        item.setIcon(QIcon(px))

    def _try_start_next(self):
        if self._processing is not None:
            return
        while self._queue:
            nxt = self._queue.pop(0)
            if self._statuses.get(str(nxt)) == VideoStatus.QUEUED:
                self.queue_start.emit(nxt)
                return
        self._update_footer()

    def _update_footer(self):
        n_queued = sum(1 for s in self._statuses.values() if s == VideoStatus.QUEUED)
        n_done   = sum(1 for s in self._statuses.values() if s == VideoStatus.DONE)
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
        qi    = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        clean = QPixmap.fromImage(qi.copy())
        self._clean_thumbs[path_str] = clean
        # Redraw with current status dot on the new clean thumbnail
        status = self._statuses.get(path_str, VideoStatus.UNKNOWN)
        self._apply_dot(item, status)

    def _apply_filter(self, text: str):
        q = text.strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            vp: Path = item.data(Qt.UserRole)
            item.setHidden(bool(q) and q not in vp.name.lower())

    def _on_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is None:
            return
        path   = item.data(Qt.UserRole)
        status = self._statuses.get(str(path), VideoStatus.UNKNOWN)

        menu = QMenu(self)
        if status not in (VideoStatus.QUEUED, VideoStatus.PROCESSING, VideoStatus.DONE):
            act = QAction("Add to queue", self)
            act.triggered.connect(lambda: self.enqueue(path))
            menu.addAction(act)
        if status == VideoStatus.QUEUED:
            act = QAction("Remove from queue", self)
            act.triggered.connect(
                lambda: self._set_status(path, VideoStatus.SKIPPED)
            )
            menu.addAction(act)
        if status == VideoStatus.DONE:
            act = QAction("Re-queue (reprocess)", self)
            act.triggered.connect(lambda: self.enqueue(path))
            menu.addAction(act)
        if not menu.isEmpty():
            menu.exec(self._list.viewport().mapToGlobal(pos))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _placeholder_icon() -> QIcon:
    px = QPixmap(THUMB_W, THUMB_H)
    px.fill(QColor("#2a2a2a"))
    return QIcon(px)
