"""
ui/project.py

Project / experiment state — persists to a .cassiopea.json file.

A project bundles:
  - name
  - video folder path
  - calibration JSON path
  - pipeline parameters
  - per-video bell / dye clicks (auto-saved on annotation)

ProjectBar is the thin widget shown above the tab strip.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QPushButton, QLabel, QDialog, QDialogButtonBox,
    QFileDialog, QLineEdit, QComboBox, QMessageBox,
)

_RECENT_FILE  = Path.home() / ".cassiopea_recent.json"
_MAX_RECENT   = 5
CALIB_DIR     = Path(__file__).parent.parent / "calibration"
PROJECT_DIR   = Path(__file__).parent.parent / "project_folder"


def _safe_stem(name: str) -> str:
    """Make a Windows-safe filename stem from a project name."""
    bad = r'\/:*?"<>|'
    return ("".join("_" if c in bad else c for c in name).strip()) or "Untitled"


def _unique_project_path(name: str) -> Path:
    """Return a non-conflicting .cassiopea.json path inside PROJECT_DIR."""
    PROJECT_DIR.mkdir(exist_ok=True)
    stem = _safe_stem(name)
    path = PROJECT_DIR / f"{stem}.cassiopea.json"
    if not path.exists():
        return path
    for i in range(2, 1000):
        path = PROJECT_DIR / f"{stem}_{i}.cassiopea.json"
        if not path.exists():
            return path
    return path   # fallback (will overwrite _999)


# ── Project state dataclass ───────────────────────────────────────────────────

@dataclass
class ProjectState:
    name:         str        = "Untitled"
    video_folder: str        = ""
    calibration:  str        = ""
    parameters:   dict       = field(default_factory=dict)
    videos:       dict       = field(default_factory=dict)

    # Shared annotation — used for all videos unless overridden per-video.
    # Updated automatically after each successful pipeline run (continuity).
    shared_bell_click: list | None = None   # [x, y] in frame-0 pixels
    shared_dye_click:  list | None = None   # [x, y] in frame-0 pixels

    # Not serialised — runtime only
    _path: Path | None = field(default=None, repr=False, compare=False)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        target = path or self._path
        if target is None:
            raise ValueError("No save path set for project.")
        self._path = target
        data = {k: v for k, v in asdict(self).items()
                if not k.startswith("_")}
        target.write_text(json.dumps(data, indent=2))
        _add_recent(target)
        return target

    @classmethod
    def load(cls, path: Path) -> "ProjectState":
        data  = json.loads(path.read_text())
        valid = {k: v for k, v in data.items()
                 if k in cls.__dataclass_fields__}
        obj   = cls(**valid)
        obj._path = path
        _add_recent(path)
        return obj

    # ── Per-video click persistence ───────────────────────────────────────────

    def set_clicks(self, video_path: Path,
                   bell: tuple | None,
                   dye:  tuple | None) -> None:
        entry = self.videos.setdefault(video_path.name, {})
        if bell is not None:
            entry["bell_click"] = list(bell)
        if dye is not None:
            entry["dye_click"] = list(dye)
        if self._path:
            self.save()

    def get_clicks(self, video_path: Path,
                   ) -> tuple[tuple | None, tuple | None]:
        entry = self.videos.get(video_path.name, {})
        bell = tuple(entry["bell_click"]) if "bell_click" in entry else None
        dye  = tuple(entry["dye_click"])  if "dye_click"  in entry else None
        return bell, dye  # type: ignore[return-value]


# ── Recent-projects helpers ───────────────────────────────────────────────────

def _add_recent(path: Path) -> None:
    recent = load_recent()
    s = str(path)
    if s in recent:
        recent.remove(s)
    recent.insert(0, s)
    try:
        _RECENT_FILE.write_text(json.dumps(recent[:_MAX_RECENT]))
    except OSError:
        pass


def extract_continuity_clicks(
    seg_csv: "Path | None",
    track_csv: "Path | None",
) -> tuple[list | None, list | None]:
    """
    Read the last row of seg_csv and track_csv to get updated bell/dye
    positions for the next video.  Returns (bell_click, dye_click) as
    [x, y] lists, or None if the CSV cannot be read.
    """
    bell = None
    dye  = None
    try:
        if seg_csv and seg_csv.exists():
            import csv
            with open(seg_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                bell = [float(last["cx"]), float(last["cy"])]
    except Exception:
        pass
    try:
        if track_csv and track_csv.exists():
            import csv
            with open(track_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                dye = [float(last["x"]), float(last["y"])]
    except Exception:
        pass
    return bell, dye


def load_recent() -> list[str]:
    try:
        return json.loads(_RECENT_FILE.read_text())
    except Exception:
        return []


# ── New-project dialog ────────────────────────────────────────────────────────

class _NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        form   = QFormLayout()

        self._name_edit = QLineEdit("Untitled")
        form.addRow("Project name:", self._name_edit)

        # Video folder row
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select video folder…")
        self._folder_edit.setReadOnly(True)
        folder_btn = QPushButton("Browse…")
        folder_btn.setFixedWidth(72)
        folder_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(folder_btn)
        form.addRow("Video folder:", folder_row)

        # Calibration combo
        self._calib_combo = QComboBox()
        self._populate_calibrations()
        form.addRow("Calibration:", self._calib_combo)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select video folder", str(Path.home()))
        if folder:
            self._folder_edit.setText(folder)

    def _populate_calibrations(self):
        self._calib_combo.clear()
        jsons = sorted(CALIB_DIR.glob("*.json"))
        for jp in jsons:
            self._calib_combo.addItem(jp.stem, userData=str(jp))
        if not jsons:
            self._calib_combo.addItem("(none — run Calibrate first)",
                                      userData="")

    def get_state(self) -> ProjectState | None:
        name   = self._name_edit.text().strip() or "Untitled"
        folder = self._folder_edit.text().strip()
        calib  = self._calib_combo.currentData() or ""
        if not folder:
            QMessageBox.warning(self, "Missing folder",
                                "Select a video folder before continuing.")
            return None
        return ProjectState(name=name, video_folder=folder, calibration=calib)


# ── Project bar ───────────────────────────────────────────────────────────────

class ProjectBar(QWidget):
    """
    Thin header bar shown above the tab strip.
    Emits project_changed(ProjectState) when a project is created or opened.
    """
    project_changed = Signal(object)   # ProjectState

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: ProjectState | None = None
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._new_btn  = QPushButton("New")
        self._open_btn = QPushButton("Open")
        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)

        _bar_btn_style = (
            "QPushButton { padding: 3px 8px; font-size: 11px; min-height: 22px; }"
            "QPushButton:hover { background: #333; }"
            "QPushButton:pressed { background: #1e1e1e; }"
            "QPushButton:disabled { color: #444; }"
        )
        for btn in (self._new_btn, self._open_btn, self._save_btn):
            btn.setFixedWidth(52)
            btn.setStyleSheet(_bar_btn_style)

        self._name_lbl = QLabel("No project")
        self._name_lbl.setStyleSheet("font-weight: bold; color: #aaddff;")

        self._new_btn.clicked.connect(self._on_new)
        self._open_btn.clicked.connect(self._on_open)
        self._save_btn.clicked.connect(self._on_save)

        layout.addWidget(self._new_btn)
        layout.addWidget(self._open_btn)
        layout.addWidget(self._save_btn)
        layout.addWidget(self._name_lbl, stretch=1)

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_new(self):
        dlg = _NewProjectDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        state = dlg.get_state()
        if state is None:
            return
        # Auto-save to project_folder/ — no file dialog needed
        path = _unique_project_path(state.name)
        state.save(path)
        self._apply(state)

    def _on_open(self):
        # Default to project_folder/ so users see their projects immediately
        start = str(PROJECT_DIR) if PROJECT_DIR.exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", start,
            "Cassiopea project (*.cassiopea.json *.json)",
        )
        if not path:
            return
        try:
            self._apply(ProjectState.load(Path(path)))
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))

    def _on_save(self):
        if self._project is None:
            return
        if self._project._path is None:
            # Shouldn't happen with auto-save in _on_new, but handle gracefully
            path = _unique_project_path(self._project.name)
            self._project.save(path)
        else:
            self._project.save()

    def _apply(self, state: ProjectState):
        self._project = state
        self._name_lbl.setText(state.name)
        self._save_btn.setEnabled(True)
        self.project_changed.emit(state)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def project(self) -> ProjectState | None:
        return self._project
