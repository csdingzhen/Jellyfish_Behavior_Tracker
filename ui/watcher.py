"""
ui/watcher.py

Monitors a video folder for new files that finish being written to disk.
Emits file_ready(Path) when a recording is complete.

Completion detection (two-stage):
  1. Size stability — file size must be unchanged across STABLE_CHECKS
     consecutive polls (default 3 × 3 s = 9 s minimum).
  2. Lock probe — on Windows, path.rename(path) raises PermissionError
     while another process holds a write handle.  Both conditions must
     pass before the file is declared ready.

If the folder does not yet exist when watch() is called, polling will
start as soon as the folder appears.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from qtpy.QtCore import QObject, QTimer, Signal

VIDEO_EXTS    = {".mp4", ".avi", ".mov", ".mkv"}
POLL_MS       = 3_000   # poll every 3 s
STABLE_CHECKS = 3       # must see same size 3× → ~9 s before "ready"


class VideoStatus(Enum):
    UNKNOWN    = "unknown"
    RECORDING  = "recording"   # file exists but still growing / locked
    QUEUED     = "queued"      # waiting for GPU to be free
    PROCESSING = "processing"  # pipeline running
    DONE       = "done"        # pipeline completed successfully
    FAILED     = "failed"      # pipeline raised an error
    SKIPPED    = "skipped"     # user manually removed from queue


def _is_locked(path: Path) -> bool:
    """Return True if the file is still held open by another process."""
    try:
        path.rename(path)   # zero-effect rename; PermissionError if locked
        return False
    except OSError:
        return True


class FolderWatcher(QObject):
    """
    Polls a video folder for new files that finish being written.

    Usage
    -----
    watcher = FolderWatcher()
    watcher.file_ready.connect(my_slot)   # slot receives Path
    watcher.watch(Path("/recordings"))
    # …
    watcher.stop()

    Signals
    -------
    file_ready(Path)  — emitted exactly once per file when recording ends
    file_appeared(Path) — emitted when a new file is first seen (still recording)
    """

    file_ready    = Signal(Path)   # recording complete
    file_appeared = Signal(Path)   # first detected (still may be recording)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder:  Path | None              = None
        self._known:   set[Path]                = set()
        self._pending: dict[Path, list[int]]    = {}   # path → recent sizes

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)

    # ── Public API ────────────────────────────────────────────────────────────

    def watch(self, folder: Path) -> None:
        """Start watching *folder*.  Safe to call again to switch folders."""
        self._folder  = folder
        self._pending = {}
        # Seed known set so pre-existing files are never re-queued.
        self._known = (
            {p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS}
            if folder.exists() else set()
        )
        if not self._timer.isActive():
            self._timer.start(POLL_MS)

    def stop(self) -> None:
        self._timer.stop()

    @property
    def is_active(self) -> bool:
        return self._timer.isActive()

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        if self._folder is None:
            return
        if not self._folder.exists():
            return   # wait for folder to appear

        current = {
            p for p in self._folder.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        }

        # Newly appeared files
        for path in current - self._known:
            if path not in self._pending:
                self._pending[path] = []
                self.file_appeared.emit(path)

        # Check each pending file
        for path in list(self._pending):
            if path not in current:
                del self._pending[path]
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue

            sizes = self._pending[path]
            sizes.append(size)

            if len(sizes) < STABLE_CHECKS:
                continue

            recent = sizes[-STABLE_CHECKS:]
            if len(set(recent)) == 1 and recent[0] > 0 and not _is_locked(path):
                del self._pending[path]
                self._known.add(path)
                self.file_ready.emit(path)
