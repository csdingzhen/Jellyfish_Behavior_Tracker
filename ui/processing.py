"""
ui/processing.py

Workflow B: video selection → annotation → pipeline run → results display.
"""

from __future__ import annotations

import json
import threading
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
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QTextEdit,
    QProgressBar,
    QSplitter,
    QMessageBox,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
)
from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QPixmap, QImage

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

CALIB_DIR   = Path(__file__).parent.parent / "calibration"
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv"}


class ProcessingTab(QWidget):
    """
    Process tab.

    Steps:
      1. Folder picker → video list with thumbnails
      2. Select video → load first frame into viewer
      3. Pick calibration JSON
      4. Click bell (red) and dye (green) on first frame
         → SAM2 mask preview shown immediately after bell click
      5. Parameter panel
      6. Run → progress display
      7. Results loaded automatically on completion
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self._video_dir:   Path | None  = None
        self._video_path:  Path | None  = None
        self._calib_path:  Path | None  = None
        self._bell_click:  tuple | None = None
        self._dye_click:   tuple | None = None
        self._cancel_event: threading.Event | None = None
        self._worker = None

        # napari layer handles
        self._frame_layer  = None
        self._mask_layer   = None
        self._bell_layer   = None
        self._dye_layer    = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Folder + video list ───────────────────────────────────────────────
        folder_box = QGroupBox("Video folder")
        folder_layout = QVBoxLayout(folder_box)

        folder_row = QHBoxLayout()
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        browse_folder_btn = QPushButton("Browse…")
        browse_folder_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_label)
        folder_row.addWidget(browse_folder_btn)
        folder_layout.addLayout(folder_row)

        self.video_list = QListWidget()
        self.video_list.setIconSize(QSize(120, 90))
        self.video_list.setMaximumHeight(200)
        self.video_list.currentItemChanged.connect(self._on_video_selected)
        folder_layout.addWidget(self.video_list)
        layout.addWidget(folder_box)

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

        # ── Run button + recompute ────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.force_recompute_cb = QCheckBox("Force recompute")
        self.force_recompute_cb.setToolTip(
            "Delete cached outputs and rerun all pipeline stages from scratch."
        )
        self.run_btn = QPushButton("Run pipeline")
        self.run_btn.setStyleSheet("background: #224488; font-weight: bold; padding: 6px;")
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        run_row.addWidget(self.force_recompute_cb)
        run_row.addStretch()
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
        self._progress_bars: dict[str, QProgressBar] = {}
        self._progress_labels: dict[str, QLabel] = {}
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
        from magicgui.widgets import Container, SpinBox, FloatSpinBox, CheckBox, ComboBox
        from .parameters import PipelineParams, Sam2Model

        self._params = PipelineParams()

        def _spin(label, attr, lo, hi, step=1):
            w = SpinBox(label=label, value=getattr(self._params, attr),
                        min=lo, max=hi, step=step)
            w.changed.connect(lambda v: setattr(self._params, attr, v))
            return w

        def _fspin(label, attr, lo, hi, step=0.01):
            w = FloatSpinBox(label=label, value=getattr(self._params, attr),
                             min=lo, max=hi, step=step)
            w.changed.connect(lambda v: setattr(self._params, attr, v))
            return w

        stride_w      = _spin("SAM2 stride",       "stride",           1, 16)
        ct_stride_w   = _spin("CoTracker stride",   "cotracker_stride", 1, 32)
        pre_window_w  = _spin("Pre-window (frames)","pre_window",       5, 120)
        inner_frac_w  = _fspin("Inner frac",        "inner_frac",       0.3, 1.0)
        outer_frac_w  = _fspin("Outer frac",        "outer_frac",       0.8, 1.5)
        prominence_w  = _fspin("Prominence",        "prominence",       0.01, 0.5, 0.01)

        sam2_model_w = ComboBox(
            label="SAM2 model",
            choices=[e.value for e in Sam2Model],
            value=self._params.sam2_model.value,
        )
        sam2_model_w.changed.connect(
            lambda v: setattr(self._params, "sam2_model", Sam2Model(v))
        )

        container = Container(widgets=[
            stride_w, ct_stride_w, sam2_model_w,
            pre_window_w, inner_frac_w, outer_frac_w, prominence_w,
        ])
        parent_layout.addWidget(container.native)

    # ── Folder / video loading ────────────────────────────────────────────────

    def _browse_folder(self):
        from config import VIDEO_DIR
        start = str(self._video_dir or VIDEO_DIR)
        folder = QFileDialog.getExistingDirectory(self, "Select video folder", start)
        if folder:
            self._load_folder(Path(folder))

    def _load_folder(self, folder: Path):
        self._video_dir = folder
        self.folder_label.setText(str(folder))
        self.video_list.clear()

        videos = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        )
        for vp in videos:
            item = QListWidgetItem(vp.name)
            item.setData(Qt.UserRole, vp)
            # Load thumbnail asynchronously (simple: load in place for MVP)
            thumb = self._load_thumb(vp)
            if thumb is not None:
                item.setIcon(self._numpy_to_icon(thumb))
            self.video_list.addItem(item)

    def _load_thumb(self, video_path: Path):
        from .thumbnails import get_thumbnail
        try:
            return get_thumbnail(video_path)
        except Exception:
            return None

    def _numpy_to_icon(self, rgb: np.ndarray):
        from qtpy.QtGui import QImage, QPixmap, QIcon
        h, w, c = rgb.shape
        qi = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        return QIcon(QPixmap.fromImage(qi))

    def _on_video_selected(self, current, previous):
        if current is None:
            return
        vp: Path = current.data(Qt.UserRole)
        self._load_video_first_frame(vp)

    def _load_video_first_frame(self, video_path: Path):
        from .thumbnails import read_first_frame
        frame = read_first_frame(video_path)
        if frame is None:
            QMessageBox.warning(self, "Error", f"Cannot read:\n{video_path}")
            return

        self._video_path = video_path

        # Clear ALL viewer layers — removes the high-res calibration image from
        # the Calibrate tab and any previous video frame.
        self.viewer.layers.clear()
        self._frame_layer = None
        self._mask_layer  = None
        self._bell_layer  = None
        self._dye_layer   = None

        self._frame_layer = self.viewer.add_image(
            frame, name=video_path.name, rgb=True
        )
        self._bell_click = None
        self._dye_click  = None
        self.bell_coord_label.setText("—")
        self.dye_coord_label.setText("—")

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
        self.bell_coord_label.setText(f"x={self._bell_click[0]}  y={self._bell_click[1]}")
        if len(data) > 1:
            self._bell_layer.data = data[[-1]]
        self._bell_layer.mode = "pan_zoom"
        # Kick off SAM2 preview in background
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
        self.dye_coord_label.setText(f"x={self._dye_click[0]}  y={self._dye_click[1]}")
        if len(data) > 1:
            self._dye_layer.data = data[[-1]]
        self._dye_layer.mode = "pan_zoom"

    # ── SAM2 preview ──────────────────────────────────────────────────────────

    def _run_sam2_preview(self):
        """Run SAM2 on frame 0 with the bell click and show a mask overlay."""
        if self._bell_click is None or self._video_path is None:
            return

        bell = self._bell_click
        video_path = self._video_path

        self._log("SAM2 preview: running segmentation on frame 0…")

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
            QMessageBox.warning(self, "No video", "Select a video first.")
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

        force = self.force_recompute_cb.isChecked()
        if force:
            reply = QMessageBox.warning(
                self, "Force recompute",
                "This will DELETE all cached outputs for this video and rerun "
                "every pipeline stage from scratch.\n\nThis can take several minutes.\n\n"
                "Continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return

        self._reset_progress()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._cancel_event = threading.Event()

        from .workers import run_pipeline_worker
        self._worker = run_pipeline_worker(
            video_path         = self._video_path,
            bell_click         = self._bell_click,
            dye_click          = self._dye_click,
            calib_path         = self._calib_path,
            params             = self._params,
            cancel_event       = self._cancel_event,
            delete_old_outputs = force,
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
        self.log_area.clear()
        self.result_label.setText("Running…")
        self.result_plot_label.clear()

    def _on_progress_event(self, event):
        from src.scheduler import TaskStatus
        name = event.task_name
        if name in self._progress_bars:
            pct = int(event.fraction * 100)
            self._progress_bars[name].setValue(pct)
            self._progress_labels[name].setText(event.status.name.lower())
            if event.message:
                self._log(f"[{name}] {event.message}")
        self.overall_bar.setValue(int(event.overall_fraction * 100))

    def _on_pipeline_done(self, result):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if result.success:
            self._log("Pipeline completed successfully.")
            self._load_results(result)
        else:
            self._log("Pipeline failed or was cancelled.")
            self.result_label.setText("Pipeline failed — check log.")

    def _on_pipeline_error(self, exc_info):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._log(f"Pipeline error: {exc_info[1]}")
        self.result_label.setText(f"Error: {exc_info[1]}")

    # ── Results display ───────────────────────────────────────────────────────

    def _load_results(self, result):
        """Load output files into napari layers and the results panel."""
        stem = self._video_path.stem
        lines = ["Results loaded:"]

        # Dye track
        if result.track_csv and result.track_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(result.track_csv)
                # Tracks layer expects (id, t, y, x)
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

        # Bell centroid points
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

        # Initiation table
        if result.initiation_csv and result.initiation_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(result.initiation_csv)
                lines.append(f"  Initiation sites: {len(df)} pulses")
                # Show first few rows in results label
                lines.append(df.to_string(index=False, max_rows=8))
            except Exception as e:
                self._log(f"Initiation CSV load error: {e}")

        # Static plot PNG
        if result.initiation_plot and result.initiation_plot.exists():
            try:
                pix = QPixmap(str(result.initiation_plot))
                pix = pix.scaledToWidth(400, Qt.SmoothTransformation)
                self.result_plot_label.setPixmap(pix)
            except Exception as e:
                self._log(f"Plot load error: {e}")

        # Annotated video — load as napari image stack
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
                    stack = np.stack(frames)   # (T, H, W, 3)
                    self.viewer.add_image(
                        stack, name="Annotated video", rgb=True,
                    )
                    lines.append(f"  Annotated video: {len(frames)} frames")
            except Exception as e:
                self._log(f"Annotated video load error: {e}")

        self.result_label.setText("\n".join(lines))

    # ── Log helper ────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.log_area.append(msg)


# deferred import so napari.qt.threading is only imported after napari is ready
def _thread_worker_import():
    from napari.qt.threading import thread_worker
    return thread_worker


# patch reference used inside _run_sam2_preview
import numpy as np
try:
    from napari.qt.threading import thread_worker
except ImportError:
    thread_worker = None
