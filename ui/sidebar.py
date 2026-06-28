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
  • "Queue all in folder" — one-click batch enqueue for already-recorded
    videos, plus multi-select context-menu actions for partial batches

Signals
-------
video_selected(Path)  — user clicked a video
queue_start(Path)     — app.py should start the pipeline for this video
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, QSize, QTimer
from qtpy.QtGui import QIcon, QPixmap, QImage, QColor, QPainter, QBrush
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QSizePolicy, QLineEdit, QMenu, QAction,
    QProgressBar, QAbstractItemView,
)

from .watcher import FolderWatcher, VideoStatus, VIDEO_EXTS
from .style import (
    STYLESHEET, card,
    C_TEXT, C_TEXT_DIM,
    C_BLUE, C_GREEN, C_RED, C_ORANGE, C_GRAY,
)

THUMB_W, THUMB_H = 96, 72
DOT_R            = 10   # status-dot diameter on thumbnail

_DOT_COLOR = {
    VideoStatus.UNKNOWN:     C_GRAY,
    VideoStatus.RECORDING:   C_ORANGE,
    VideoStatus.QUEUED:      "#cccc33",
    VideoStatus.PROCESSING:  C_BLUE,
    VideoStatus.DONE:        C_GREEN,
    VideoStatus.FAILED:      C_RED,
    VideoStatus.SKIPPED:     C_GRAY,
    VideoStatus.NEEDS_INPUT: C_ORANGE,
}
_BAR_COLOR = {
    VideoStatus.UNKNOWN:     C_GRAY,
    VideoStatus.RECORDING:   C_ORANGE,
    VideoStatus.QUEUED:      "#886600",
    VideoStatus.PROCESSING:  C_BLUE,
    VideoStatus.DONE:        C_GREEN,
    VideoStatus.FAILED:      C_RED,
    VideoStatus.SKIPPED:     C_GRAY,
    VideoStatus.NEEDS_INPUT: C_ORANGE,
}
_STATUS_TEXT = {
    VideoStatus.UNKNOWN:     ("—",                   0.0),
    VideoStatus.RECORDING:   ("Recording…",          0.0),
    VideoStatus.QUEUED:      ("Waiting",              0.0),
    VideoStatus.PROCESSING:  ("Processing…",          0.0),
    VideoStatus.DONE:        ("Completed",            1.0),
    VideoStatus.FAILED:      ("Failed",               0.0),
    VideoStatus.SKIPPED:     ("Skipped",              0.0),
    VideoStatus.NEEDS_INPUT: ("Needs annotation",     0.0),
}

# Statuses that should never be re-queued by a bulk "queue all" / re-trigger.
_TERMINAL_OR_ACTIVE = (VideoStatus.QUEUED, VideoStatus.PROCESSING, VideoStatus.DONE)

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
        self._name_lbl.setStyleSheet(f"font-size: 11px; color: {C_TEXT}; background: transparent;")
        layout.addWidget(self._name_lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(self._bar_style(C_GRAY))
        layout.addWidget(self._bar)

        self._status_lbl = QLabel("—")
        self._status_lbl.setStyleSheet(
            f"font-size: 9px; color: {C_TEXT_DIM}; background: transparent;"
        )
        layout.addWidget(self._status_lbl)

        layout.addStretch()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_status(self, status: VideoStatus):
        text, frac = _STATUS_TEXT.get(status, ("—", 0.0))
        color = _BAR_COLOR.get(status, C_GRAY)
        self._bar.setValue(int(frac * 100))
        self._bar.setStyleSheet(self._bar_style(color))
        self._status_lbl.setText(text)

    def set_progress(self, fraction: float, task_name: str = ""):
        pct = int(fraction * 100)
        self._bar.setValue(pct)
        self._bar.setStyleSheet(self._bar_style(C_BLUE))
        label = f"{task_name}  {pct}%" if task_name else f"Processing  {pct}%"
        self._status_lbl.setText(label)

    def mark_file_removed(self):
        self._status_lbl.setText("File removed from disk")
        self._status_lbl.setStyleSheet(
            f"font-size: 9px; color: {C_RED}; background: transparent;"
        )

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
        self._queue_all_mode: str                        = "queue"   # "queue" | "remove"

        self._watcher = FolderWatcher(self)
        self._watcher.file_appeared.connect(self._on_file_appeared)
        self._watcher.file_ready.connect(self._on_file_ready)
        self._watcher.file_removed.connect(self._on_file_removed)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet(STYLESHEET)
        self.setAttribute(Qt.WA_StyledBackground, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Folder + watch card ─────────────────────────────────────────────
        top = card()
        top_lay = top.layout()

        folder_row = QHBoxLayout()
        self._folder_lbl = QLabel("No folder selected")
        self._folder_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._folder_lbl.setStyleSheet(f"font-size: 11px; color: {C_TEXT};")
        browse_btn = QPushButton("Browse…")
        browse_btn.setMinimumWidth(76)
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(self._folder_lbl, stretch=1)
        folder_row.addWidget(browse_btn)
        top_lay.addLayout(folder_row)

        watch_row = QHBoxLayout()
        self._watch_btn = QPushButton("Watch OFF")
        self._watch_btn.setCheckable(True)
        self._watch_btn.setMinimumWidth(82)
        self._watch_btn.setToolTip(
            "Monitor this folder for new video files.\n"
            "New recordings appear here automatically when writing finishes."
        )
        self._watch_btn.toggled.connect(self._on_watch_toggled)
        self._watch_lbl = QLabel("Idle")
        self._watch_lbl.setStyleSheet(f"font-size: 10px; color: {C_TEXT_DIM};")
        watch_row.addWidget(self._watch_btn)
        watch_row.addWidget(self._watch_lbl, stretch=1)
        top_lay.addLayout(watch_row)

        self._queue_all_btn = QPushButton("Queue all in folder")
        self._queue_all_btn.clicked.connect(self._on_queue_all_clicked)
        self._queue_all_btn.setEnabled(False)
        top_lay.addWidget(self._queue_all_btn)

        layout.addWidget(top)

        # ── Filter ───────────────────────────────────────────────────────────
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter videos…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search)

        # ── Video list ───────────────────────────────────────────────────────
        self._list = QListWidget()
        self._list.setIconSize(QSize(THUMB_W, THUMB_H))
        self._list.setSpacing(2)
        self._list.setUniformItemSizes(False)
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._list)

        # ── Footer ───────────────────────────────────────────────────────────
        footer_row = QHBoxLayout()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"font-size: 10px; color: {C_TEXT_DIM};")
        self._queue_lbl = QLabel("")
        self._queue_lbl.setStyleSheet(f"font-size: 10px; color: {C_GREEN};")
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

        self._update_queue_all_button()

    def select_video(self, path: Path):
        item = self._item_map.get(str(path))
        if item:
            self._list.blockSignals(True)
            self._list.setCurrentItem(item)
            self._list.blockSignals(False)

    def set_auto_queue(self, enabled: bool):
        self._auto_queue = enabled

    # ── Queue management (called by app.py) ───────────────────────────────────

    def enqueue(self, path: Path, force: bool = False):
        """
        Add *path* to the processing queue.

        By default, videos that are already queued, processing, or done are
        left untouched (safe to call repeatedly / in a bulk loop).  Pass
        force=True to re-queue a DONE video for reprocessing.
        """
        key = str(path)
        if key not in self._item_map:
            self._add_item(path)
        status = self._statuses.get(key)
        if status in (VideoStatus.QUEUED, VideoStatus.PROCESSING):
            return
        if status == VideoStatus.DONE and not force:
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
        """A per-video pipeline run raised an error. Other queued videos
        are unaffected and the queue keeps advancing."""
        self._set_status(path, VideoStatus.FAILED)
        if self._processing == path:
            self._processing = None
        self._update_footer()
        self._try_start_next()

    def mark_needs_attention(self, path: Path):
        """
        The run couldn't even start because of missing project setup
        (no calibration, or no bell+dye annotation yet) rather than a
        per-video error.  Deliberately does NOT call _try_start_next() —
        the same cause would otherwise repeat for every remaining queued
        video, cascading the whole batch to "failed" in one shot.

        Once the user resolves it (e.g. manually annotates the first
        video via the Process tab), clicking "Queue all in folder" again
        resumes the rest of the batch.
        """
        self._set_status(path, VideoStatus.NEEDS_INPUT)
        if self._processing == path:
            self._processing = None
        self._update_footer()

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
        self._update_queue_all_button()

    def _on_file_ready(self, path: Path):
        self._watch_lbl.setText(f"Ready: {path.name}")
        if self._auto_queue:
            self.enqueue(path)
        else:
            self._set_status(path, VideoStatus.UNKNOWN)
        self._update_queue_all_button()

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
        self._update_queue_all_button()

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
            color = _DOT_COLOR.get(status, C_GRAY)
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
                # Claim the slot synchronously, *before* deferring the emit.
                # enqueue() calls _try_start_next() in a tight loop when
                # called repeatedly (e.g. from _queue_all()) — without
                # claiming here, a second call could see self._processing
                # still None (since mark_processing() only runs once the
                # deferred emit below actually fires) and dispatch a second
                # video concurrently, breaking the "one video at a time"
                # guarantee. mark_processing() re-sets the same value once
                # the run actually starts; mark_failed()/mark_needs_attention()
                # clear it again if the dispatch never gets that far.
                self._processing = nxt
                # Defer the actual emit to the next event-loop tick rather
                # than calling it synchronously.  If the receiving slot
                # fails fast and calls mark_failed() (which calls back into
                # _try_start_next()), a synchronous emit would recurse one
                # Python stack frame per queued video — fine for a handful
                # of videos, but risks hitting the recursion limit on a
                # large "queue all" batch that fails uniformly.  Deferring
                # turns that into flat iteration across event-loop ticks.
                QTimer.singleShot(0, lambda p=nxt: self.queue_start.emit(p))
                self._update_footer()
                return
        self._update_footer()

    def _on_queue_all_clicked(self):
        """Dispatches to queue-all / remove-all / resume depending on the
        button's current mode (kept in sync by _update_queue_all_button)."""
        if self._queue_all_mode == "remove":
            self._remove_all_queued()
        elif self._queue_all_mode == "resume":
            self._try_start_next()
        else:
            self._queue_all()

    def _queue_all(self):
        """Enqueue every currently visible video that isn't already queued,
        processing, or done. (Resuming an already-stalled queue is handled
        separately by the "resume" button mode, which calls
        _try_start_next() directly — see _update_queue_all_button.)"""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.isHidden():
                continue
            path   = item.data(Qt.UserRole)
            status = self._statuses.get(str(path), VideoStatus.UNKNOWN)
            if status not in _TERMINAL_OR_ACTIVE:
                self.enqueue(path)
        self._try_start_next()

    def _remove_all_queued(self):
        """Cancel a batch queued via 'Queue all in folder' before it's been
        processed. Only removes videos still WAITING (status QUEUED) — does
        not touch one that's already PROCESSING; use the Cancel button in
        the Process tab for that."""
        queued_items = [
            item for item in self._item_map.values()
            if self._statuses.get(str(item.data(Qt.UserRole))) == VideoStatus.QUEUED
        ]
        for item in queued_items:
            self._remove_from_queue(item.data(Qt.UserRole))

    def _visible_eligible_count(self) -> int:
        count = 0
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.isHidden():
                continue
            path   = item.data(Qt.UserRole)
            status = self._statuses.get(str(path), VideoStatus.UNKNOWN)
            if status not in _TERMINAL_OR_ACTIVE:
                count += 1
        return count

    def _update_queue_all_button(self):
        """
        Button mode is derived from current state, not tracked separately,
        and distinguishes three situations:

        - Something is QUEUED *and* something is actively PROCESSING: the
          batch is healthily running — offer to cancel what hasn't started.
        - Something is QUEUED but nothing is PROCESSING: the queue stalled
          (mark_needs_attention() deliberately doesn't auto-advance it, to
          avoid cascading the same "no annotation" failure through every
          remaining video) — offer to resume rather than ambiguously
          offering "remove", since the user most likely wants to continue,
          not cancel.
        - Nothing QUEUED: offer to queue whatever's newly eligible.
        """
        n_queued = sum(1 for s in self._statuses.values() if s == VideoStatus.QUEUED)
        if n_queued > 0 and self._processing is None:
            self._queue_all_mode = "resume"
            self._queue_all_btn.setText(f"Resume queue ({n_queued})")
            self._queue_all_btn.setToolTip(
                "The queue paused, likely because a video needed manual\n"
                "annotation or calibration. Resolve that in the Process\n"
                "tab, then click here to continue the rest of the batch."
            )
            self._queue_all_btn.setEnabled(True)
        elif n_queued > 0:
            self._queue_all_mode = "remove"
            self._queue_all_btn.setText(f"Remove all from queue ({n_queued})")
            self._queue_all_btn.setToolTip(
                "Removes every video still waiting in the queue.\n"
                "Does not cancel a video that's already processing —\n"
                "use Cancel in the Process tab for that."
            )
            self._queue_all_btn.setEnabled(True)
        else:
            n = self._visible_eligible_count()
            self._queue_all_mode = "queue"
            self._queue_all_btn.setText(
                f"Queue all in folder ({n})" if n else "Queue all in folder"
            )
            self._queue_all_btn.setToolTip(
                "Enqueue every video currently shown below that isn't\n"
                "already queued, processing, or done — for batches of\n"
                "pre-recorded videos. Processing still runs strictly one\n"
                "video at a time."
            )
            self._queue_all_btn.setEnabled(n > 0)

    def _update_footer(self):
        n_total  = len(self._statuses)
        n_queued = sum(1 for s in self._statuses.values() if s == VideoStatus.QUEUED)
        n_done   = sum(1 for s in self._statuses.values() if s == VideoStatus.DONE)
        n_failed = sum(
            1 for s in self._statuses.values()
            if s in (VideoStatus.FAILED, VideoStatus.NEEDS_INPUT)
        )
        parts = []
        if self._processing:
            parts.append(f"Processing: {self._processing.name}")
        if n_queued:
            parts.append(f"{n_queued} queued")
        if n_done:
            parts.append(f"{n_done}/{n_total} done")
        if n_failed:
            parts.append(f"{n_failed} need attention")
        self._queue_lbl.setText("  ".join(parts))
        self._update_queue_all_button()

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
        self._update_queue_all_button()

    def _remove_from_queue(self, path: Path):
        self._set_status(path, VideoStatus.SKIPPED)
        self._queue = [p for p in self._queue if str(p) != str(path)]
        self._update_footer()

    def _on_context_menu(self, pos):
        clicked = self._list.itemAt(pos)
        if clicked is None:
            return

        selected = self._list.selectedItems()
        if clicked not in selected:
            self._list.setCurrentItem(clicked)
            selected = [clicked]

        entries = [
            (it.data(Qt.UserRole),
             self._statuses.get(str(it.data(Qt.UserRole)), VideoStatus.UNKNOWN))
            for it in selected
        ]

        menu = QMenu(self)

        queueable = [p for p, s in entries if s not in _TERMINAL_OR_ACTIVE]
        if queueable:
            label = "Add to queue" if len(queueable) == 1 else f"Add {len(queueable)} to queue"
            act = QAction(label, self)
            act.triggered.connect(
                lambda: [self.enqueue(p) for p in queueable]
            )
            menu.addAction(act)

        removable = [p for p, s in entries if s == VideoStatus.QUEUED]
        if removable:
            label = ("Remove from queue" if len(removable) == 1
                      else f"Remove {len(removable)} from queue")
            act = QAction(label, self)
            act.triggered.connect(
                lambda: [self._remove_from_queue(p) for p in removable]
            )
            menu.addAction(act)

        requeueable = [p for p, s in entries if s == VideoStatus.DONE]
        if requeueable:
            label = ("Re-queue (reprocess)" if len(requeueable) == 1
                      else f"Re-queue {len(requeueable)} (reprocess)")
            act = QAction(label, self)
            act.triggered.connect(
                lambda: [self.enqueue(p, force=True) for p in requeueable]
            )
            menu.addAction(act)

        if not menu.isEmpty():
            menu.exec(self._list.viewport().mapToGlobal(pos))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _placeholder_icon() -> QIcon:
    px = QPixmap(THUMB_W, THUMB_H)
    px.fill(QColor("#2a2a2a"))
    return QIcon(px)
