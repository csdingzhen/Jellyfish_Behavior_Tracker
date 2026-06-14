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
    QGroupBox,
    QTextEdit,
    QProgressBar,
    QMessageBox,
    QCheckBox,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap, QImage

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

CALIB_DIR = Path(__file__).parent.parent / "calibration"


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


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
    """

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

        # napari layer handles
        self._frame_layer = None
        self._mask_layer  = None
        self._bell_layer  = None
        self._dye_layer   = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Current video label ───────────────────────────────────────────────
        self._video_label = QLabel("No video selected")
        self._video_label.setStyleSheet("color: #aaa; font-style: italic;")
        self._video_label.setWordWrap(True)
        layout.addWidget(self._video_label)

        # ── Calibration picker ────────────────────────────────────────────────
        calib_box = QGroupBox("Calibration")
        calib_layout = QHBoxLayout(calib_box)
        self.calib_combo = QComboBox()
        self.calib_combo.setMinimumWidth(160)
        self.calib_combo.currentIndexChanged.connect(self._on_calib_selected)
        refresh_calib_btn = QPushButton("Refresh")
        refresh_calib_btn.clicked.connect(self._refresh_calibrations)
        calib_layout.addWidget(QLabel("Calibration:"))
        calib_layout.addWidget(self.calib_combo)
        calib_layout.addWidget(refresh_calib_btn)
        layout.addWidget(calib_box)
        self._refresh_calibrations()

        # ── Bell / Dye annotation ─────────────────────────────────────────────
        ann_box = QGroupBox("Click annotation")
        ann_layout = QVBoxLayout(ann_box)

        bell_row = QHBoxLayout()
        self.bell_btn = QPushButton("Mark bell")
        self.bell_btn.setToolTip("Click on the jellyfish bell in the viewer")
        self.bell_btn.clicked.connect(self._start_bell_click)
        self.bell_coord_label = QLabel("—")
        bell_row.addWidget(self.bell_btn)
        bell_row.addWidget(self.bell_coord_label)
        ann_layout.addLayout(bell_row)

        dye_row = QHBoxLayout()
        self.dye_btn = QPushButton("Mark dye")
        self.dye_btn.setToolTip("Click on the dye mark in the viewer")
        self.dye_btn.clicked.connect(self._start_dye_click)
        self.dye_coord_label = QLabel("—")
        dye_row.addWidget(self.dye_btn)
        dye_row.addWidget(self.dye_coord_label)
        ann_layout.addLayout(dye_row)

        layout.addWidget(ann_box)

        # ── Parameters ────────────────────────────────────────────────────────
        param_box = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_box)
        self._build_param_panel(param_layout)
        layout.addWidget(param_box)

        # ── Per-task recompute + Run ──────────────────────────────────────────
        recompute_box = QGroupBox("Recompute (force re-run)")
        recompute_layout = QHBoxLayout(recompute_box)
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
        recompute_layout.addWidget(self.rerun_sam2_cb)
        recompute_layout.addWidget(self.rerun_cotrack_cb)
        recompute_layout.addWidget(self.rerun_analysis_cb)
        recompute_layout.addStretch()
        layout.addWidget(recompute_box)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run pipeline")
        self.run_btn.setStyleSheet(
            "background: #224488; font-weight: bold; padding: 6px;"
        )
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        layout.addLayout(run_row)

        # ── Progress ──────────────────────────────────────────────────────────
        progress_box = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_box)

        TASK_NAMES = [
            "SAM2 segmentation",
            "CoTracker tracking",
            "Margin diff (lab frame)",
            "Body-frame rotation",
            "Pulse initiation analysis",
        ]
        self._progress_bars:   dict[str, QProgressBar] = {}
        self._progress_labels: dict[str, QLabel]       = {}
        for name in TASK_NAMES:
            row = QHBoxLayout()
            lbl = QLabel(name[:28])
            lbl.setFixedWidth(200)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            status_lbl = QLabel("pending")
            status_lbl.setFixedWidth(80)
            row.addWidget(lbl)
            row.addWidget(bar)
            row.addWidget(status_lbl)
            progress_layout.addLayout(row)
            self._progress_bars[name]   = bar
            self._progress_labels[name] = status_lbl

        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 100)
        self.overall_bar.setValue(0)
        self.overall_bar.setFormat("Overall: %p%")
        progress_layout.addWidget(self.overall_bar)

        self._timer_label = QLabel("")
        self._timer_label.setAlignment(Qt.AlignCenter)
        self._timer_label.setStyleSheet("color: #888; font-size: 10px;")
        self._timer_label.setVisible(False)
        progress_layout.addWidget(self._timer_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        self.log_area.setPlaceholderText("Pipeline log…")
        progress_layout.addWidget(self.log_area)

        layout.addWidget(progress_box)

        # ── Results ───────────────────────────────────────────────────────────
        result_box = QGroupBox("Results")
        result_layout = QVBoxLayout(result_box)
        self.result_label = QLabel("No results yet.")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)

        self.result_plot_label = QLabel()
        self.result_plot_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_plot_label)
        layout.addWidget(result_box)

    def _build_param_panel(self, parent_layout):
        """Simple form built from PipelineParams dataclass fields."""
        from magicgui.widgets import Container, SpinBox, FloatSpinBox, ComboBox
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

        sam2_model_w = ComboBox(
            label="SAM2 model",
            choices=[e.value for e in Sam2Model],
            value=self._params.sam2_model.value,
            tooltip=(
                "SAM2 backbone size. 'tiny' runs fastest and fits in 8 GB VRAM;\n"
                "'large' is more accurate but needs more memory and time.\n"
                "For Cassiopea's high-contrast bell, 'tiny' is usually sufficient."
            ),
        )
        sam2_model_w.changed.connect(
            lambda v: setattr(self._params, "sam2_model", Sam2Model(v))
        )

        container = Container(widgets=[
            stride_w, ct_stride_w, sam2_model_w,
            pre_window_w, inner_frac_w, outer_frac_w, prominence_w,
        ])
        parent_layout.addWidget(container.native)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_video(self, path: Path):
        """Load *path* into the viewer as the current working video."""
        from .thumbnails import read_first_frame
        frame = read_first_frame(path)
        if frame is None:
            QMessageBox.warning(self, "Error", f"Cannot read:\n{path}")
            return

        self._video_path = path
        self._video_label.setText(path.name)
        self._video_label.setStyleSheet("color: #dddddd;")

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
        # Auto-select calibration from the project
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
        self._bell_layer.mode = "add"
        self.viewer.layers.selection.active = self._bell_layer

    def _on_bell_data(self, event):
        data = self._bell_layer.data
        if len(data) == 0:
            return
        row, col = data[-1]
        self._bell_click = (int(col), int(row))
        self.bell_coord_label.setText(
            f"x={self._bell_click[0]}  y={self._bell_click[1]}"
        )
        if len(data) > 1:
            self._bell_layer.data = data[[-1]]
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
        self._dye_layer.mode = "add"
        self.viewer.layers.selection.active = self._dye_layer

    def _on_dye_data(self, event):
        data = self._dye_layer.data
        if len(data) == 0:
            return
        row, col = data[-1]
        self._dye_click = (int(col), int(row))
        self.dye_coord_label.setText(
            f"x={self._dye_click[0]}  y={self._dye_click[1]}"
        )
        if len(data) > 1:
            self._dye_layer.data = data[[-1]]
        self._dye_layer.mode = "pan_zoom"

    # ── SAM2 preview ──────────────────────────────────────────────────────────

    def _run_sam2_preview(self):
        if self._bell_click is None or self._video_path is None:
            return

        bell = self._bell_click
        video_path = self._video_path

        self._log("SAM2 preview: running segmentation on frame 0…")

        from napari.qt.threading import thread_worker

        @thread_worker
        def _preview_worker():
            import torch
            import cv2 as cv
            from config import SAM2_WEIGHTS, SAM2_CONFIG

            cap = cv.VideoCapture(str(video_path))
            ret, frame_bgr = cap.read()
            cap.release()
            if not ret:
                return None

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
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
            return masks[0].astype(np.uint8)

        w = _preview_worker()

        def _on_preview_done(mask):
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

        def _on_preview_error(exc_info):
            self._log(f"SAM2 preview error: {exc_info[1]}")

        w.returned.connect(_on_preview_done)
        w.errored.connect(_on_preview_error)
        w.start()

    # ── Run pipeline ──────────────────────────────────────────────────────────

    def _on_run(self):
        if self._video_path is None:
            QMessageBox.warning(self, "No video", "Select a video from the sidebar first.")
            return
        if self._bell_click is None:
            QMessageBox.warning(self, "No bell click",
                                "Click 'Mark bell' and click the bell in the viewer.")
            return
        if self._dye_click is None:
            QMessageBox.warning(self, "No dye click",
                                "Click 'Mark dye' and click the dye mark in the viewer.")
            return
        if self._calib_path is None or not self._calib_path.exists():
            QMessageBox.warning(self, "No calibration",
                                "Select a calibration file, or run Workflow A first.")
            return

        rerun_sam2    = self.rerun_sam2_cb.isChecked()
        rerun_cotrack = self.rerun_cotrack_cb.isChecked()
        rerun_analysis = self.rerun_analysis_cb.isChecked()
        any_rerun = rerun_sam2 or rerun_cotrack or rerun_analysis

        if any_rerun:
            parts = []
            if rerun_sam2:    parts.append("SAM2")
            if rerun_cotrack: parts.append("CoTracker")
            if rerun_analysis: parts.append("Analysis")
            reply = QMessageBox.warning(
                self, "Force recompute",
                f"This will DELETE cached outputs for: {', '.join(parts)}\n"
                "and rerun those stages from scratch.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
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

    # ── Progress updates ──────────────────────────────────────────────────────

    def _reset_progress(self):
        for bar in self._progress_bars.values():
            bar.setValue(0)
        for lbl in self._progress_labels.values():
            lbl.setText("pending")
        self.overall_bar.setValue(0)
        self._timer_label.setVisible(False)
        self._run_start_time = None
        self.log_area.clear()
        self.result_label.setText("Running…")
        self.result_plot_label.clear()

    def _on_progress_event(self, event):
        name = event.task_name
        if name in self._progress_bars:
            pct = int(event.fraction * 100)
            self._progress_bars[name].setValue(pct)
            self._progress_labels[name].setText(event.status.name.lower())
            if event.message:
                self._log(f"[{name}] {event.message}")
        overall = event.overall_fraction
        self.overall_bar.setValue(int(overall * 100))

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
        else:
            self._on_cancel_reset()
            self._log("Pipeline failed or was cancelled.")
            self.result_label.setText("Pipeline failed — check log.")

    def _on_pipeline_error(self, exc_info):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._on_cancel_reset()
        self._log(f"Pipeline error: {exc_info[1]}")
        self.result_label.setText(f"Error: {exc_info[1]}")

    def _on_cancel_reset(self):
        """Reset all progress bars to zero after a cancel or failure."""
        for bar in self._progress_bars.values():
            bar.setValue(0)
        for lbl in self._progress_labels.values():
            lbl.setText("—")
        self.overall_bar.setValue(0)
        elapsed = (time.monotonic() - self._run_start_time
                   if self._run_start_time else 0.0)
        self._timer_label.setText(f"Cancelled after {_fmt_time(elapsed)}")
        self._run_start_time = None

    # ── Results display ───────────────────────────────────────────────────────

    def _load_results(self, result):
        stem  = self._video_path.stem
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
