"""
ui/processing.py

Workflow B: video selection → annotation → pipeline run → results display.

Video selection is handled by the VideoSidebarWidget in app.py; this tab
receives the selected video via load_video(path).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QFileDialog,
    QTextEdit,
    QProgressBar,
    QMessageBox,
    QCheckBox,
    QLineEdit,
    QScrollArea,
    QFrame,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from .style import (
    card, step_badge, status_icon, _set_icon_color,
    C_TEXT, C_TEXT_DIM, C_TEXT_MONO,
    C_GREEN, C_RED, C_BLUE, C_ORANGE, C_GRAY, C_BORDER_LO,
    C_CARD_ALT, C_BORDER, _ARROW_SVG,
)

CALIB_DIR      = Path(__file__).parent.parent / "calibration"
_USER_SETTINGS = Path(__file__).parent.parent / "user_settings.json"


def _load_user_settings() -> dict:
    if _USER_SETTINGS.exists():
        try:
            return json.loads(_USER_SETTINGS.read_text())
        except Exception:
            pass
    return {}


def _save_user_settings(data: dict) -> None:
    try:
        _USER_SETTINGS.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _exc_message(exc_info) -> str:
    """Extract a human-readable message from a worker error payload.

    napari's @thread_worker `errored` signal emits the Exception object
    directly (not the (type, value, tb) tuple that sys.exc_info() returns),
    so indexing it with [1] raises TypeError. Be tolerant of both forms.
    """
    if isinstance(exc_info, BaseException):
        return str(exc_info)
    if isinstance(exc_info, tuple) and len(exc_info) >= 2:
        return str(exc_info[1])
    return str(exc_info)


_TASK_NAMES = [
    "SAM2 segmentation",
    "CoTracker tracking",
    "Margin diff (lab frame)",
    "Body-frame rotation",
    "Pulse initiation analysis",
]


class ProcessingTab(QWidget):
    """
    Process tab.

    Steps:
      1. Video loaded externally via load_video(path) from the sidebar
      2. Pick calibration JSON
      3. Click bell (red) and dye (green) on first frame
         → SAM2 mask preview shown immediately after bell click
      4. Parameter panel
      5. Run → progress display
      6. Results loaded automatically on completion

    Signals
    -------
    pipeline_finished(Path, object) — emitted after every manual run started
    from this tab (i.e. via the "Run pipeline" button, not the sidebar
    queue). Second arg is the PipelineResult on success, or None on
    failure/error. app.py listens for this to keep the sidebar's status dot
    in sync and to propagate continuity clicks to the next queued video,
    since a manual run otherwise bypasses both of those.
    """

    pipeline_finished = Signal(Path, object)

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self._video_path:  Path | None  = None
        self._calib_path:  Path | None  = None
        self._bell_click:  tuple | None = None
        self._dye_click:   tuple | None = None
        self._cancel_event:   threading.Event | None = None
        self._worker          = None
        self._run_start_time: float | None = None
        self._project_state   = None   # set via on_project_changed()

        # Re-entrancy guards: assigning layer.data inside an events.data
        # handler re-fires events.data synchronously. In napari 0.7.0 the
        # re-entrant read still sees the pre-assignment point count, so the
        # "keep only the last point" trim below would recurse forever
        # ("maximum recursion depth exceeded"). These flags suppress the
        # handler during our own programmatic data edits.
        self._suppress_bell_event = False
        self._suppress_dye_event  = False

        # SAM2 preview is serialized: only one worker at a time, because SAM2
        # uses Hydra's global singleton and two concurrent previews (e.g. from
        # re-marking the bell) race on it. _preview_pending records that a
        # newer mark arrived while a preview was still running.
        self._preview_worker  = None
        self._preview_pending = False

        # napari layer handles
        self._frame_layer = None
        self._mask_layer  = None
        self._bell_layer  = None
        self._dye_layer   = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Error banner (hidden by default) ─────────────────────────────────
        self._error_banner = self._make_error_banner()
        self._error_banner.setVisible(False)
        layout.addWidget(self._error_banner)

        # ── Video info card ───────────────────────────────────────────────────
        vid_card = card()
        vc_lay = vid_card.layout()
        self._video_name_lbl = QLabel("No video selected")
        self._video_name_lbl.setStyleSheet(
            f"font-weight: bold; color: {C_TEXT}; font-size: 13px;"
        )
        self._video_name_lbl.setWordWrap(True)
        self._video_meta_lbl = QLabel("")
        self._video_meta_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        vc_lay.addWidget(self._video_name_lbl)
        vc_lay.addWidget(self._video_meta_lbl)
        layout.addWidget(vid_card)

        # ── Step 1 – Calibration ──────────────────────────────────────────────
        s1_card = card()
        self._add_step_header(s1_card.layout(), 1, "Calibration")
        calib_row = QHBoxLayout()
        self.calib_combo = QComboBox()
        self.calib_combo.setMinimumWidth(130)
        self.calib_combo.currentIndexChanged.connect(self._on_calib_selected)
        refresh_btn = QPushButton("Refresh")
        calib_row.addWidget(self.calib_combo, stretch=1)
        calib_row.addWidget(refresh_btn)
        refresh_btn.clicked.connect(self._refresh_calibrations)
        s1_card.layout().addLayout(calib_row)
        layout.addWidget(s1_card)
        self._refresh_calibrations()

        # ── Step 2 – Annotation ───────────────────────────────────────────────
        s2_card = card()
        self._add_step_header(s2_card.layout(), 2, "Mark positions on first frame")
        s2_card.layout().setSpacing(8)
        s2_card.layout().addLayout(
            self._annotation_row("Bell", C_RED, self._start_bell_click, "bell")
        )
        s2_card.layout().addLayout(
            self._annotation_row("Dye", C_GREEN, self._start_dye_click, "dye")
        )
        layout.addWidget(s2_card)

        # ── Step 3 – Parameters ───────────────────────────────────────────────
        s3_card = card()
        self._add_step_header(s3_card.layout(), 3, "Configure parameters")
        self._build_param_panel(s3_card.layout())
        layout.addWidget(s3_card)

        # ── Output directory card ─────────────────────────────────────────────
        out_card = card()
        out_card_lay = out_card.layout()
        out_hdr = QLabel("Output directory")
        out_hdr.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {C_TEXT};"
        )
        out_card_lay.addWidget(out_hdr)
        out_row = QHBoxLayout()
        from config import OUTPUTS_DIR
        saved_dir = _load_user_settings().get("output_dir")
        self._output_dir = Path(saved_dir) if saved_dir else OUTPUTS_DIR
        self._out_dir_edit = QLineEdit(str(self._output_dir))
        self._out_dir_edit.setReadOnly(True)
        self._out_dir_edit.setToolTip(
            f"Pipeline outputs (CSVs, masks, plots) are written here.\n"
            f"Default: {OUTPUTS_DIR}"
        )
        out_browse_btn = QPushButton("Browse…")
        out_browse_btn.setMinimumWidth(76)
        out_browse_btn.clicked.connect(self._on_browse_output_dir)
        out_reset_btn = QPushButton("Reset")
        out_reset_btn.setMinimumWidth(60)
        out_reset_btn.setToolTip(f"Reset to default: {OUTPUTS_DIR}")
        out_reset_btn.clicked.connect(self._on_reset_output_dir)
        out_row.addWidget(self._out_dir_edit, stretch=1)
        out_row.addWidget(out_browse_btn)
        out_row.addWidget(out_reset_btn)
        out_card_lay.addLayout(out_row)
        layout.addWidget(out_card)

        # ── Run / Cancel ──────────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run pipeline")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.clicked.connect(self._on_cancel)
        run_row.addWidget(self.run_btn, stretch=1)
        run_row.addWidget(self.cancel_btn)
        layout.addLayout(run_row)

        # ── Recompute row (below Run button) ──────────────────────────────────
        rerun_row = QHBoxLayout()
        rerun_row.setContentsMargins(4, 0, 4, 0)
        rerun_lbl = QLabel("Force recompute:")
        rerun_lbl.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px;"
        )
        self.rerun_sam2_cb = QCheckBox("SAM2")
        self.rerun_sam2_cb.setToolTip(
            "Re-run SAM2 segmentation even if cached outputs exist.\n"
            "Deletes existing mask / contour-radii cache."
        )
        self.rerun_cotrack_cb = QCheckBox("CoTracker")
        self.rerun_cotrack_cb.setToolTip(
            "Re-run CoTracker dye-point tracking even if a cached track CSV exists."
        )
        self.rerun_analysis_cb = QCheckBox("Analysis")
        self.rerun_analysis_cb.setToolTip(
            "Re-run margin diff, body-frame rotation and pulse initiation analysis\n"
            "even if cached analysis outputs exist."
        )
        rerun_row.addWidget(rerun_lbl)
        rerun_row.addWidget(self.rerun_sam2_cb)
        rerun_row.addWidget(self.rerun_cotrack_cb)
        rerun_row.addWidget(self.rerun_analysis_cb)
        rerun_row.addStretch()
        layout.addLayout(rerun_row)

        # ── Progress card ─────────────────────────────────────────────────────
        prog_card = card(padding=10)
        prog_lay = prog_card.layout()

        hdr_row = QHBoxLayout()
        prog_hdr = QLabel("Progress")
        prog_hdr.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {C_TEXT};")
        self._overall_label = QLabel("")
        self._overall_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        self._overall_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr_row.addWidget(prog_hdr)
        hdr_row.addWidget(self._overall_label, stretch=1)
        prog_lay.addLayout(hdr_row)

        self._task_icons:        dict[str, QLabel] = {}
        self._task_status_labels: dict[str, QLabel] = {}
        self._task_start_times:  dict[str, float | None] = {n: None for n in _TASK_NAMES}

        # compat refs used by _reset_progress / _on_cancel_reset legacy paths
        self._progress_bars:   dict[str, QLabel] = {}
        self._progress_labels: dict[str, QLabel] = {}

        for name in _TASK_NAMES:
            row = QHBoxLayout()
            row.setSpacing(8)
            icon = status_icon(C_GRAY)
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f"color: {C_TEXT}; font-size: 12px;")
            stat_lbl = QLabel("Waiting")
            stat_lbl.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 11px; min-width: 90px;"
            )
            stat_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(icon)
            row.addWidget(name_lbl, stretch=1)
            row.addWidget(stat_lbl)
            prog_lay.addLayout(row)
            self._task_icons[name]          = icon
            self._task_status_labels[name]  = stat_lbl
            self._progress_bars[name]       = icon      # compat (not a QProgressBar)
            self._progress_labels[name]     = stat_lbl

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {C_BORDER_LO};")
        prog_lay.addWidget(sep)

        self._timer_label = QLabel("")
        self._timer_label.setAlignment(Qt.AlignCenter)
        self._timer_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 10px;")
        self._timer_label.setVisible(False)
        prog_lay.addWidget(self._timer_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(80)
        self.log_area.setStyleSheet(
            f"font-size: 10px; font-family: monospace; color: {C_TEXT_DIM};"
        )
        self.log_area.setPlaceholderText("Log output…")
        prog_lay.addWidget(self.log_area)

        # invisible overall bar — kept so existing code paths can setValue without error
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 100)
        self.overall_bar.setVisible(False)
        prog_lay.addWidget(self.overall_bar)

        layout.addWidget(prog_card)

        # ── Results card ──────────────────────────────────────────────────────
        res_card = card(padding=10)
        res_lay = res_card.layout()
        res_hdr = QLabel("Results")
        res_hdr.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {C_TEXT};")
        res_lay.addWidget(res_hdr)
        self.result_label = QLabel("No results yet.")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        res_lay.addWidget(self.result_label)
        self.result_plot_label = QLabel()
        self.result_plot_label.setAlignment(Qt.AlignCenter)
        res_lay.addWidget(self.result_plot_label)
        layout.addWidget(res_card)

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self._apply_output_dir(self._output_dir)

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _add_step_header(self, lay: "QVBoxLayout", number: int, title: str) -> None:
        """Prepend a numbered step badge + title row to an existing card layout."""
        hdr = QHBoxLayout()
        hdr.addWidget(step_badge(number))
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-weight: bold; color: {C_TEXT}; font-size: 12px;"
        )
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        lay.addLayout(hdr)

    def _annotation_row(
        self, label: str, color: str, click_fn, attr: str
    ) -> QHBoxLayout:
        """Colored dot · label · coordinate text · Mark button."""
        row = QHBoxLayout()
        row.setSpacing(6)
        row.setContentsMargins(0, 0, 0, 0)

        dot = QLabel("●")
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"color: {color}; font-size: 16px;")

        name_lbl = QLabel(label)
        name_lbl.setFixedWidth(36)
        name_lbl.setStyleSheet(
            f"color: {C_TEXT}; font-weight: bold; font-size: 12px;"
        )

        coord_lbl = QLabel("—")
        coord_lbl.setStyleSheet(
            f"font-family: monospace; color: {C_TEXT_MONO}; font-size: 11px;"
        )

        mark_btn = QPushButton("Mark")
        mark_btn.setFixedWidth(70)
        mark_btn.clicked.connect(click_fn)

        row.addWidget(dot)
        row.addWidget(name_lbl)
        row.addWidget(coord_lbl, stretch=1)
        row.addWidget(mark_btn)

        if attr == "bell":
            self.bell_btn         = mark_btn
            self.bell_coord_label = coord_lbl
        else:
            self.dye_btn         = mark_btn
            self.dye_coord_label = coord_lbl

        return row

    def _make_error_banner(self) -> QFrame:
        f = QFrame()
        f.setStyleSheet("""
            QFrame {
                background: #2a1010;
                border: 1px solid #5a2020;
                border-radius: 6px;
            }
        """)
        lay = QVBoxLayout(f)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)

        top_row = QHBoxLayout()
        icon_lbl = QLabel("⚠")
        icon_lbl.setStyleSheet(f"color: {C_RED}; font-size: 16px;")
        icon_lbl.setFixedWidth(20)
        self._error_title = QLabel("Pipeline error")
        self._error_title.setStyleSheet(
            f"color: {C_RED}; font-weight: bold; font-size: 12px;"
        )
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "background: transparent; border: none; color: #777; font-size: 12px;"
        )
        close_btn.clicked.connect(lambda: f.setVisible(False))
        top_row.addWidget(icon_lbl)
        top_row.addWidget(self._error_title, stretch=1)
        top_row.addWidget(close_btn)
        lay.addLayout(top_row)

        self._error_msg = QLabel("")
        self._error_msg.setStyleSheet(
            "color: #cc8888; font-size: 11px; font-family: monospace;"
        )
        self._error_msg.setWordWrap(True)
        lay.addWidget(self._error_msg)

        btn_row = QHBoxLayout()
        retry_btn = QPushButton("Retry SAM2")
        retry_btn.setObjectName("retryBtn")
        retry_btn.setFixedHeight(26)
        retry_btn.clicked.connect(self._on_retry_sam2)
        viewlog_btn = QPushButton("View log ↓")
        viewlog_btn.setFixedHeight(26)
        viewlog_btn.clicked.connect(
            lambda: self.log_area.setVisible(True)
        )
        btn_row.addWidget(retry_btn)
        btn_row.addWidget(viewlog_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        return f

    def _build_param_panel(self, parent_layout):
        """Simple form built from PipelineParams dataclass fields."""
        from magicgui.widgets import Container, SpinBox, FloatSpinBox
        from .parameters import PipelineParams, Sam2Model

        self._params = PipelineParams()

        def _spin(label, attr, lo, hi, step=1, tooltip=""):
            w = SpinBox(label=label, value=getattr(self._params, attr),
                        min=lo, max=hi, step=step, tooltip=tooltip)
            w.changed.connect(lambda v: setattr(self._params, attr, v))
            return w

        def _fspin(label, attr, lo, hi, step=0.01, tooltip=""):
            w = FloatSpinBox(label=label, value=getattr(self._params, attr),
                             min=lo, max=hi, step=step, tooltip=tooltip)
            w.changed.connect(lambda v: setattr(self._params, attr, v))
            return w

        stride_w = _spin(
            "SAM2 stride", "stride", 1, 16,
            tooltip=(
                "Process every Nth frame with SAM2 for mask propagation.\n"
                "Lower = more accurate but slower. 4 is a good default for 120 fps.\n"
                "Effective temporal resolution = stride / fps seconds."
            ),
        )
        ct_stride_w = _spin(
            "CoTracker stride", "cotracker_stride", 1, 32,
            tooltip=(
                "Temporal stride for CoTracker's sliding-window attention.\n"
                "Larger = faster but may lose the dye point during fast motion.\n"
                "Recommended: 8 for 120 fps recordings."
            ),
        )
        pre_window_w = _spin(
            "Pre-window (frames)", "pre_window", 5, 120,
            tooltip=(
                "Number of frames before each detected pulse peak to include\n"
                "in the initiation analysis window. Should cover at least one\n"
                "full contraction wavefront propagation (~10–30 frames at 120 fps)."
            ),
        )
        inner_frac_w = _fspin(
            "Inner frac", "inner_frac", 0.3, 1.0,
            tooltip=(
                "Inner radius of the annular margin band as a fraction of the\n"
                "bell radius. 0.75 means the band starts 75% of the way out.\n"
                "Keep well inside the true bell margin to avoid petri-dish noise."
            ),
        )
        outer_frac_w = _fspin(
            "Outer frac", "outer_frac", 0.8, 1.5,
            tooltip=(
                "Outer radius of the annular margin band as a fraction of the\n"
                "bell radius. 1.05 slightly overshoots the mask edge to capture\n"
                "the full marginal lappet region."
            ),
        )
        prominence_w = _fspin(
            "Prominence", "prominence", 0.01, 0.5, step=0.01,
            tooltip=(
                "Minimum peak prominence (relative to signal range) for a\n"
                "divergence peak to be counted as a pulse. Raise this to suppress\n"
                "spurious detections; lower it to catch weak pulses."
            ),
        )
        # ── SAM2 model — native QComboBox (magicgui's ComboBox renders incorrectly) ──
        sam2_row = QHBoxLayout()
        sam2_row.setContentsMargins(0, 2, 0, 4)
        sam2_row.setSpacing(8)
        sam2_lbl = QLabel("SAM2 model")
        sam2_lbl.setStyleSheet(f"color: {C_TEXT}; background: transparent; font-size: 12px;")

        self._sam2_model_combo = QComboBox()
        self._sam2_model_combo.setToolTip(
            "SAM2 backbone size. 'tiny' runs fastest and fits in 8 GB VRAM;\n"
            "'large' is more accurate but needs more memory and time.\n"
            "For Cassiopea's high-contrast bell, 'tiny' is usually sufficient."
        )
        for bare, desc in [
            ("tiny",  "tiny — fastest, recommended"),
            ("small", "small — slight accuracy gain"),
            ("base",  "base — use if tiny shows drift"),
            ("large", "large — highest accuracy, slowest"),
        ]:
            self._sam2_model_combo.addItem(desc, userData=bare)
        for i in range(self._sam2_model_combo.count()):
            if self._sam2_model_combo.itemData(i) == self._params.sam2_model.value:
                self._sam2_model_combo.setCurrentIndex(i)
                break
        self._sam2_model_combo.currentIndexChanged.connect(
            lambda _: setattr(
                self._params, "sam2_model",
                Sam2Model(self._sam2_model_combo.currentData())
            )
        )
        sam2_row.addWidget(sam2_lbl)
        sam2_row.addWidget(self._sam2_model_combo, stretch=1)
        parent_layout.addLayout(sam2_row)

        container = Container(widgets=[
            stride_w, ct_stride_w,
            pre_window_w, inner_frac_w, outer_frac_w, prominence_w,
        ])
        container.native.setObjectName("paramContainer")
        container.native.setStyleSheet(f"""
            QWidget {{ background: transparent; }}
            QSpinBox, QDoubleSpinBox, QLineEdit {{
                background: {C_CARD_ALT};
                border: 1px solid {C_BORDER};
                border-radius: 5px;
                color: {C_TEXT};
                padding: 3px 6px;
            }}
            QLabel {{ color: {C_TEXT}; background: transparent; }}
            QPushButton {{
                color: {C_TEXT};
                background: transparent;
                border: none;
                font-weight: 600;
            }}
            QPushButton:disabled {{ color: {C_TEXT_DIM}; }}
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                color: {C_TEXT};
            }}
        """)
        parent_layout.addWidget(container.native)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_video(self, path: Path):
        """Load *path* into the viewer as the current working video."""
        import cv2
        from .thumbnails import read_first_frame
        frame = read_first_frame(path)
        if frame is None:
            QMessageBox.warning(self, "Error", f"Cannot read:\n{path}")
            return

        self._video_path = path
        self._video_name_lbl.setText(path.name)
        self._video_name_lbl.setStyleSheet(
            f"font-weight: bold; color: {C_TEXT}; font-size: 13px;"
        )

        try:
            cap    = cv2.VideoCapture(str(path))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            n_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w_px   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_px   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            dur    = n_fr / fps if fps > 0 else 0.0
            self._video_meta_lbl.setText(
                f"{w_px}×{h_px}  ·  {fps:.0f} fps  ·  {_fmt_time(dur)}  ·  {n_fr} frames"
            )
        except Exception:
            self._video_meta_lbl.setText("")

        self.viewer.layers.clear()
        self._frame_layer = None
        self._mask_layer  = None
        self._bell_layer  = None
        self._dye_layer   = None

        self._frame_layer = self.viewer.add_image(
            frame, name=path.name, rgb=True
        )
        self._bell_click = None
        self._dye_click  = None
        self.bell_coord_label.setText("—")
        self.dye_coord_label.setText("—")

    def on_project_changed(self, state) -> None:
        """Called when the active project changes (from ProjectBar)."""
        self._project_state = state
        if state.calibration:
            calib_path = Path(state.calibration)
            for i in range(self.calib_combo.count()):
                item_path = self.calib_combo.itemData(i)
                if isinstance(item_path, Path) and item_path == calib_path:
                    self.calib_combo.setCurrentIndex(i)
                    break

    # ── Calibration picker ────────────────────────────────────────────────────

    def _refresh_calibrations(self):
        self.calib_combo.clear()
        jsons = sorted(CALIB_DIR.glob("*.json"))
        if not jsons:
            self.calib_combo.addItem("(none found — run Calibrate first)")
            self._calib_path = None
            return
        for jp in jsons:
            self.calib_combo.addItem(jp.stem, userData=jp)
        self._calib_path = jsons[0]

    def _on_calib_selected(self, index):
        path = self.calib_combo.itemData(index)
        self._calib_path = path if isinstance(path, Path) else None

    # ── Bell / Dye annotation ─────────────────────────────────────────────────

    def _start_bell_click(self):
        if self._frame_layer is None:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        if self._bell_layer is None:
            self._bell_layer = self.viewer.add_points(
                data=[], name="Bell click",
                face_color="red", border_color="white",
                symbol="cross", size=20,
            )
            self._bell_layer.events.data.connect(self._on_bell_data)
        # Re-marking means "replace the old point": clear any existing point
        # so exactly one bell marker ever exists. Guarded so this clear does
        # not re-enter _on_bell_data.
        self._suppress_bell_event = True
        self._bell_layer.data = []
        self._suppress_bell_event = False
        self._bell_click = None
        self.bell_coord_label.setText("—")
        self._bell_layer.mode = "add"
        self.viewer.layers.selection.active = self._bell_layer

    def _on_bell_data(self, event=None):
        if self._suppress_bell_event:
            return
        data = self._bell_layer.data
        if len(data) == 0:
            return
        # Keep only the most recent point. Suppress the handler while we do
        # this so the re-fired events.data does not recurse.
        if len(data) > 1:
            self._suppress_bell_event = True
            self._bell_layer.data = data[[-1]]
            self._suppress_bell_event = False
            data = self._bell_layer.data
        row, col = data[-1]
        self._bell_click = (int(col), int(row))
        self.bell_coord_label.setText(
            f"{self._bell_click[0]}, {self._bell_click[1]}"
        )
        self._bell_layer.mode = "pan_zoom"
        self._run_sam2_preview()

    def _start_dye_click(self):
        if self._frame_layer is None:
            QMessageBox.information(self, "No video", "Load a video first.")
            return
        if self._dye_layer is None:
            self._dye_layer = self.viewer.add_points(
                data=[], name="Dye click",
                face_color="#00dc32", border_color="white",
                symbol="disc", size=16,
            )
            self._dye_layer.events.data.connect(self._on_dye_data)
        # Re-marking means "replace the old point": clear any existing point
        # so exactly one dye marker ever exists. Guarded against re-entry.
        self._suppress_dye_event = True
        self._dye_layer.data = []
        self._suppress_dye_event = False
        self._dye_click = None
        self.dye_coord_label.setText("—")
        self._dye_layer.mode = "add"
        self.viewer.layers.selection.active = self._dye_layer

    def _on_dye_data(self, event=None):
        if self._suppress_dye_event:
            return
        data = self._dye_layer.data
        if len(data) == 0:
            return
        if len(data) > 1:
            self._suppress_dye_event = True
            self._dye_layer.data = data[[-1]]
            self._suppress_dye_event = False
            data = self._dye_layer.data
        row, col = data[-1]
        self._dye_click = (int(col), int(row))
        self.dye_coord_label.setText(
            f"{self._dye_click[0]}, {self._dye_click[1]}"
        )
        self._dye_layer.mode = "pan_zoom"

    # ── SAM2 preview ──────────────────────────────────────────────────────────

    def _run_sam2_preview(self):
        if self._bell_click is None or self._video_path is None:
            return

        # Serialize previews. SAM2 builds its model through Hydra's global
        # singleton, and two preview workers running at once race on
        # GlobalHydra.instance().clear() → "GlobalHydra is not initialized".
        # If one is already running (e.g. the user just re-marked the bell),
        # flag a pending re-run instead of starting a second worker; the
        # latest self._bell_click is picked up when the current one finishes.
        if self._preview_worker is not None:
            self._preview_pending = True
            return

        bell = self._bell_click
        video_path = self._video_path
        self._log("SAM2 preview: running segmentation on frame 0…")

        from napari.qt.threading import thread_worker

        @thread_worker
        def _preview_worker():
            import cv2 as cv
            import torch
            from config import SAM2_WEIGHTS, SAM2_CONFIG

            cap = cv.VideoCapture(str(video_path))
            ret, frame_bgr = cap.read()
            cap.release()
            if not ret:
                return None

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from src.tasks import init_sam2_hydra
            # Clear AND re-initialize Hydra. A plain clear() works only on the
            # first preview; on re-mark the cached sam2 import doesn't re-run
            # its one-time Hydra init, so compose() inside build_sam2 raises
            # "GlobalHydra is not initialized". See init_sam2_hydra().
            init_sam2_hydra()
            sam2_model = build_sam2(SAM2_CONFIG, str(SAM2_WEIGHTS))
            with torch.inference_mode():
                img_predictor = SAM2ImagePredictor(sam2_model)
                frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                img_predictor.set_image(frame_rgb)
                masks, scores, _ = img_predictor.predict(
                    point_coords=np.array([[bell[0], bell[1]]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
            from hydra.core.global_hydra import GlobalHydra
            GlobalHydra.instance().clear()
            return masks[0].astype(np.uint8)

        w = _preview_worker()
        self._preview_worker = w

        def _finish_preview():
            self._preview_worker = None
            # If a newer mark arrived mid-run, run it now with the latest click.
            if self._preview_pending:
                self._preview_pending = False
                self._run_sam2_preview()

        def _on_preview_done(mask):
            try:
                if mask is None:
                    self._log("SAM2 preview failed.")
                    return
                if self._mask_layer is not None:
                    try:
                        self.viewer.layers.remove(self._mask_layer)
                    except Exception:
                        pass
                self._mask_layer = self.viewer.add_labels(
                    mask.astype(np.int32), name="Bell mask (preview)",
                    opacity=0.4,
                )
                self._log("SAM2 preview complete.")
            finally:
                _finish_preview()

        def _on_preview_error(exc_info):
            self._log(f"SAM2 preview error: {_exc_message(exc_info)}")
            _finish_preview()

        w.returned.connect(_on_preview_done)
        w.errored.connect(_on_preview_error)
        w.start()

    # ── Output directory ──────────────────────────────────────────────────────

    def _apply_output_dir(self, path: Path) -> None:
        from src.tasks import set_output_root
        from config import OUTPUTS_DIR
        set_output_root(None if path == OUTPUTS_DIR else path)

    def _on_browse_output_dir(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Select output directory", str(self._output_dir)
        )
        if not chosen:
            return
        self._output_dir = Path(chosen)
        self._out_dir_edit.setText(str(self._output_dir))
        self._apply_output_dir(self._output_dir)
        data = _load_user_settings()
        data["output_dir"] = str(self._output_dir)
        _save_user_settings(data)

    def _on_reset_output_dir(self):
        from config import OUTPUTS_DIR
        self._output_dir = OUTPUTS_DIR
        self._out_dir_edit.setText(str(OUTPUTS_DIR))
        self._apply_output_dir(OUTPUTS_DIR)
        data = _load_user_settings()
        data.pop("output_dir", None)
        _save_user_settings(data)

    # ── Run pipeline ──────────────────────────────────────────────────────────

    def _on_run(self):
        if self._video_path is None:
            QMessageBox.warning(self, "No video",
                                "Select a video from the sidebar first.")
            return
        if self._bell_click is None:
            QMessageBox.warning(self, "No bell click",
                                "Click 'Mark' next to Bell and click the bell in the viewer.")
            return
        if self._dye_click is None:
            QMessageBox.warning(self, "No dye click",
                                "Click 'Mark' next to Dye and click the dye mark in the viewer.")
            return
        if self._calib_path is None or not self._calib_path.exists():
            QMessageBox.warning(self, "No calibration",
                                "Select a calibration file, or run Workflow A first.")
            return

        rerun_sam2     = self.rerun_sam2_cb.isChecked()
        rerun_cotrack  = self.rerun_cotrack_cb.isChecked()
        rerun_analysis = self.rerun_analysis_cb.isChecked()
        any_rerun = rerun_sam2 or rerun_cotrack or rerun_analysis

        if any_rerun:
            parts = []
            if rerun_sam2:     parts.append("SAM2")
            if rerun_cotrack:  parts.append("CoTracker")
            if rerun_analysis: parts.append("Analysis")
            reply = QMessageBox.warning(
                self, "Force recompute",
                f"This will DELETE cached outputs for: {', '.join(parts)}\n"
                "and rerun those stages from scratch.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                # uncheck so the retry button can call _on_run cleanly next time
                self.rerun_sam2_cb.setChecked(False)
                return

        self._reset_progress()
        self._run_start_time = time.monotonic()
        self._timer_label.setText("Elapsed  00:00  ·  ETA  calculating…")
        self._timer_label.setVisible(True)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._cancel_event = threading.Event()

        from .workers import run_pipeline_worker
        self._worker = run_pipeline_worker(
            video_path     = self._video_path,
            bell_click     = self._bell_click,
            dye_click      = self._dye_click,
            calib_path     = self._calib_path,
            params         = self._params,
            cancel_event   = self._cancel_event,
            rerun_sam2     = rerun_sam2,
            rerun_cotrack  = rerun_cotrack,
            rerun_analysis = rerun_analysis,
        )
        self._worker.yielded.connect(self._on_progress_event)
        self._worker.returned.connect(self._on_pipeline_done)
        self._worker.errored.connect(self._on_pipeline_error)
        self._worker.start()
        self._log(f"Pipeline started: {self._video_path.name}")

    def _on_cancel(self):
        if self._cancel_event:
            self._cancel_event.set()
        self._log("Cancellation requested…")
        self.cancel_btn.setEnabled(False)

    def _on_retry_sam2(self):
        """Force-rerun SAM2 and restart the pipeline."""
        self._error_banner.setVisible(False)
        self.rerun_sam2_cb.setChecked(True)
        self._on_run()

    # ── Progress updates ──────────────────────────────────────────────────────

    def _reset_progress(self):
        for icon in self._task_icons.values():
            _set_icon_color(icon, C_GRAY)
        for lbl in self._task_status_labels.values():
            lbl.setText("Waiting")
            lbl.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 11px; min-width: 90px;"
            )
        self.overall_bar.setValue(0)
        self._overall_label.setText("")
        self._timer_label.setVisible(False)
        self._run_start_time = None
        self._task_start_times = {n: None for n in _TASK_NAMES}
        self.log_area.clear()
        self.result_label.setText("Running…")
        self.result_plot_label.clear()
        self._error_banner.setVisible(False)

    def _on_progress_event(self, event):
        name = event.task_name
        if name in self._task_icons:
            status = event.status.name.upper()
            icon     = self._task_icons[name]
            stat_lbl = self._task_status_labels[name]

            if status == "RUNNING":
                _set_icon_color(icon, C_BLUE)
                if self._task_start_times[name] is None:
                    self._task_start_times[name] = time.monotonic()
                msg = (event.message or "Running…")[:32]
                stat_lbl.setText(msg)
                stat_lbl.setStyleSheet(
                    f"color: {C_BLUE}; font-size: 11px; min-width: 90px;"
                )
            elif status in ("DONE", "SKIPPED"):
                _set_icon_color(icon, C_GREEN)
                t0 = self._task_start_times.get(name)
                elapsed_s = f" · {_fmt_time(time.monotonic() - t0)}" if t0 else ""
                text = "Skipped" if status == "SKIPPED" else f"Done{elapsed_s}"
                stat_lbl.setText(text)
                stat_lbl.setStyleSheet(
                    f"color: {C_GREEN}; font-size: 11px; min-width: 90px;"
                )
            elif status == "FAILED":
                _set_icon_color(icon, C_RED)
                stat_lbl.setText("Failed")
                stat_lbl.setStyleSheet(
                    f"color: {C_RED}; font-size: 11px; min-width: 90px;"
                )
                self._error_title.setText(f"{name} failed")
                self._error_msg.setText(event.message or "See log for details.")
                self._error_banner.setVisible(True)
            elif status == "CANCELLED":
                _set_icon_color(icon, C_GRAY)
                stat_lbl.setText("Cancelled")
                stat_lbl.setStyleSheet(
                    f"color: {C_TEXT_DIM}; font-size: 11px; min-width: 90px;"
                )
            elif status == "WAITING":
                _set_icon_color(icon, C_GRAY)
                stat_lbl.setText("Waiting")
                stat_lbl.setStyleSheet(
                    f"color: {C_TEXT_DIM}; font-size: 11px; min-width: 90px;"
                )

            if event.message:
                self._log(f"[{name}] {event.message}")

        overall = event.overall_fraction
        self.overall_bar.setValue(int(overall * 100))
        if overall > 0:
            self._overall_label.setText(f"{int(overall * 100)}%")

        if self._run_start_time is not None:
            elapsed = time.monotonic() - self._run_start_time
            if overall > 0.02:
                remaining = elapsed / overall * (1.0 - overall)
                eta_str = f"~{_fmt_time(remaining)}"
            else:
                eta_str = "calculating…"
            self._timer_label.setText(
                f"Elapsed  {_fmt_time(elapsed)}  ·  ETA  {eta_str}"
            )

    def _on_pipeline_done(self, result):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if result.success:
            elapsed = (time.monotonic() - self._run_start_time
                       if self._run_start_time else 0.0)
            self._timer_label.setText(f"Completed in {_fmt_time(elapsed)}")
            self._log("Pipeline completed successfully.")
            self._load_results(result)
            self.pipeline_finished.emit(self._video_path, result)
        else:
            self._on_cancel_reset()
            self._log("Pipeline failed or was cancelled.")
            self.result_label.setText("Pipeline failed — check log.")
            self.pipeline_finished.emit(self._video_path, None)

    def _on_pipeline_error(self, exc_info):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._on_cancel_reset()
        msg = _exc_message(exc_info)
        self._log(f"Pipeline error: {msg}")
        self.result_label.setText(f"Error: {msg}")
        self._error_title.setText("Pipeline error")
        self._error_msg.setText(msg[:200])
        self._error_banner.setVisible(True)
        self.pipeline_finished.emit(self._video_path, None)

    def _on_cancel_reset(self):
        """Reset any still-running tasks to '—'; preserve Done/Skipped tasks."""
        for name, icon in self._task_icons.items():
            lbl = self._task_status_labels[name]
            current = lbl.text()
            # Only reset tasks that weren't successfully finished
            if current not in ("Done", "Skipped") and not current.startswith("Done ·"):
                _set_icon_color(icon, C_GRAY)
                lbl.setText("—")
                lbl.setStyleSheet(
                    f"color: {C_TEXT_DIM}; font-size: 11px; min-width: 90px;"
                )
        self.overall_bar.setValue(0)
        self._overall_label.setText("")
        elapsed = (time.monotonic() - self._run_start_time
                   if self._run_start_time else 0.0)
        self._timer_label.setText(f"Cancelled after {_fmt_time(elapsed)}")
        self._run_start_time = None

    # ── Results display ───────────────────────────────────────────────────────

    def _load_results(self, result):
        lines = ["Results loaded:"]

        if result.track_csv and result.track_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(result.track_csv)
                t_col = "frame_idx" if "frame_idx" in df.columns else df.columns[0]
                track_data = np.column_stack([
                    np.zeros(len(df), dtype=int),
                    df[t_col].values,
                    df["y"].values,
                    df["x"].values,
                ])
                self.viewer.add_tracks(track_data, name="Dye track")
                lines.append(f"  Dye track: {len(df)} frames")
            except Exception as e:
                self._log(f"Track CSV load error: {e}")

        if result.seg_csv and result.seg_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(result.seg_csv)
                pts = np.column_stack([df["cy"].values, df["cx"].values])
                self.viewer.add_points(
                    pts, name="Bell centroid",
                    face_color="yellow", size=6, opacity=0.6,
                )
                lines.append(f"  Bell centroids: {len(df)} frames")
            except Exception as e:
                self._log(f"Seg CSV load error: {e}")

        if result.initiation_csv and result.initiation_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(result.initiation_csv)
                lines.append(f"  Initiation sites: {len(df)} pulses")
                lines.append(df.to_string(index=False, max_rows=8))
            except Exception as e:
                self._log(f"Initiation CSV load error: {e}")

        if result.initiation_plot and result.initiation_plot.exists():
            try:
                pix = QPixmap(str(result.initiation_plot))
                pix = pix.scaledToWidth(400, Qt.SmoothTransformation)
                self.result_plot_label.setPixmap(pix)
            except Exception as e:
                self._log(f"Plot load error: {e}")

        if result.annotated_video and result.annotated_video.exists():
            try:
                import cv2
                frames = []
                cap = cv2.VideoCapture(str(result.annotated_video))
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                cap.release()
                if frames:
                    stack = np.stack(frames)
                    self.viewer.add_image(stack, name="Annotated video", rgb=True)
                    lines.append(f"  Annotated video: {len(frames)} frames")
            except Exception as e:
                self._log(f"Annotated video load error: {e}")

        self.result_label.setText("\n".join(lines))

    # ── Log helper ────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.log_area.append(msg)
