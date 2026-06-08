"""
ui/calibration.py

Workflow A: one-time calibration of rhopalia body-frame angles from a
high-resolution still image.

Stage flow
----------
  0  Load image → viewer
  1  Click bell centre (single yellow point)
  2  Click dye mark (single green point)
  3  Click rhopalia sequentially (red points, live angle table)
  4  Save JSON + annotated PNG

Notes
-----
Layer state is polled via QTimer (150 ms) rather than relying on
napari layer.events.data, which is unreliable for interactive additions
in napari 0.7.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration_core import (
    N_RHOPALIA_EXPECTED,
    body_angle,
    build_calibration,
    save_annotated_image,
    write_calibration_json,
)

CALIB_DIR = Path(__file__).parent.parent / "calibration"


class CalibrationTab(QWidget):
    """
    Calibrate tab.

    Stage 0 → pick image → load into viewer
    Stage 1 → click centre (yellow)
    Stage 2 → click dye mark (green)
    Stage 3 → click rhopalia (red, iterative)
    Stage 4 → save
    """

    STAGE_LABELS = [
        "Step 1 of 4: Load a high-resolution image of the jellyfish.",
        f"Step 2 of 4: Click the bell CENTRE in the viewer, then press Next.",
        f"Step 3 of 4: Click the DYE MARK in the viewer (phi = 0°), then press Next.",
        f"Step 4 of 4: Click each RHOPALIUM ({N_RHOPALIA_EXPECTED} expected). "
        "Press 'Remove last' to undo.  Press Save when done.",
    ]

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self._img_path: Path | None       = None
        self._img_data: np.ndarray | None = None
        self._centre:   tuple | None      = None
        self._dye:      tuple | None      = None
        self._rhopalia: list              = []   # list of (x, y) in image coords
        self._stage     = 0
        self._prev_rhop_count = 0

        # napari layer handles
        self._img_layer    = None
        self._centre_layer = None
        self._dye_layer    = None
        self._rhop_layer   = None

        # Poll layer data every 150 ms instead of relying on events.data
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(150)
        self._poll_timer.timeout.connect(self._poll_layers)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Status label
        self.status_label = QLabel(self.STAGE_LABELS[0])
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-weight: bold; color: #aaddff;")
        layout.addWidget(self.status_label)

        # Image picker
        img_box = QGroupBox("Image")
        img_layout = QHBoxLayout(img_box)
        self.img_path_edit = QLineEdit()
        self.img_path_edit.setPlaceholderText("Select a .png / .jpg image…")
        self.img_path_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_image)
        img_layout.addWidget(self.img_path_edit)
        img_layout.addWidget(self.browse_btn)
        layout.addWidget(img_box)

        # Animal name
        name_box = QGroupBox("Animal name")
        name_layout = QHBoxLayout(name_box)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Ethel_Cain")
        name_layout.addWidget(self.name_edit)
        layout.addWidget(name_box)

        # Annotation controls
        ann_box = QGroupBox("Annotation")
        ann_layout = QVBoxLayout(ann_box)

        self.next_btn = QPushButton("Next →")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._on_next)

        self.undo_btn = QPushButton("Remove last rhopalium")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._on_undo)

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("Last click:"))
        self.coord_label = QLabel("—")
        coord_row.addWidget(self.coord_label)
        coord_row.addStretch()

        ann_layout.addWidget(self.next_btn)
        ann_layout.addWidget(self.undo_btn)
        ann_layout.addLayout(coord_row)
        layout.addWidget(ann_box)

        # Live angle table
        table_box = QGroupBox("Rhopalium angles (body frame)")
        table_layout = QVBoxLayout(table_box)
        self.angle_table = QTableWidget(0, 3)
        self.angle_table.setHorizontalHeaderLabels(["#", "phi_body (°)", "px (x,y)"])
        self.angle_table.horizontalHeader().setStretchLastSection(True)
        self.angle_table.setMaximumHeight(220)
        self.angle_table.setEditTriggers(QTableWidget.NoEditTriggers)
        table_layout.addWidget(self.angle_table)
        layout.addWidget(table_box)

        # Save
        self.save_btn = QPushButton("Save calibration")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(
            "background: #226622; font-weight: bold; padding: 6px;"
        )
        self.save_btn.clicked.connect(self._on_save)
        layout.addWidget(self.save_btn)

        layout.addStretch()
        self._update_controls()

    # ── Layer polling (replaces events.data) ──────────────────────────────────

    def _poll_layers(self):
        """
        Called every 150 ms to sync layer data → UI state.
        Replaces events.data which is unreliable for interactive additions
        in napari 0.7.0.
        """
        if self._stage == 1 and self._centre_layer is not None:
            data = self._centre_layer.data
            if len(data) > 0:
                row, col = data[-1]
                pt = (int(col), int(row))
                if pt != self._centre:
                    self._centre = pt
                    self.coord_label.setText(
                        f"Centre  x={pt[0]}  y={pt[1]}"
                    )
                    # Keep only the last point so re-clicks replace it
                    if len(data) > 1:
                        self._centre_layer.data = data[[-1]]
                    self._update_controls()

        elif self._stage == 2 and self._dye_layer is not None:
            data = self._dye_layer.data
            if len(data) > 0:
                row, col = data[-1]
                pt = (int(col), int(row))
                if pt != self._dye:
                    self._dye = pt
                    self.coord_label.setText(
                        f"Dye  x={pt[0]}  y={pt[1]}"
                    )
                    if len(data) > 1:
                        self._dye_layer.data = data[[-1]]
                    self._update_controls()

        elif self._stage == 3 and self._rhop_layer is not None:
            data = self._rhop_layer.data
            n = len(data)
            if n != self._prev_rhop_count:
                self._prev_rhop_count = n
                self._rhopalia = [(int(c), int(r)) for r, c in data]
                if self._rhopalia:
                    last = self._rhopalia[-1]
                    self.coord_label.setText(
                        f"R{n - 1}  x={last[0]}  y={last[1]}"
                    )
                    self._update_text_labels()
                self._refresh_angle_table()
                self._update_controls()

    def _update_text_labels(self):
        n = len(self._rhopalia)
        if n == 0:
            return
        try:
            self._rhop_layer.text = {
                "string": [f"R{i}" for i in range(n)],
                "color": "white",
                "size": 10,
            }
        except Exception:
            pass   # text labels are cosmetic; don't crash if API differs

    # ── UI state management ───────────────────────────────────────────────────

    def _update_controls(self):
        stage = self._stage
        self.status_label.setText(self.STAGE_LABELS[min(stage, 3)])

        # Next button: always enabled once we're in a click stage,
        # validation happens inside _on_next.
        self.next_btn.setEnabled(stage in (1, 2))
        self.next_btn.setVisible(stage in (1, 2))

        self.undo_btn.setEnabled(stage == 3 and len(self._rhopalia) > 0)
        self.undo_btn.setVisible(stage == 3)

        self.save_btn.setEnabled(stage == 3 and len(self._rhopalia) > 0)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select jellyfish image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        self._load_image(Path(path))

    def _load_image(self, path: Path):
        import cv2

        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            QMessageBox.critical(self, "Error", f"Cannot read image:\n{path}")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self._img_path = path
        self._img_data = img_bgr
        self.img_path_edit.setText(str(path))

        # Clear ALL viewer layers — removes any video/processing layers from the
        # Process tab so only the calibration image is shown.
        self.viewer.layers.clear()
        self._img_layer    = None
        self._centre_layer = None
        self._dye_layer    = None
        self._rhop_layer   = None

        self._img_layer = self.viewer.add_image(
            img_rgb, name="Calibration image", rgb=True
        )

        self._centre   = None
        self._dye      = None
        self._rhopalia = []
        self._prev_rhop_count = 0
        self._stage = 1
        self._update_controls()
        self._enter_centre_stage()

    # ── Stage entry ───────────────────────────────────────────────────────────

    def _enter_centre_stage(self):
        self._stage = 1
        self._centre_layer = self.viewer.add_points(
            data=[], name="Centre (C)",
            face_color="yellow", border_color="white",
            symbol="cross", size=18,
        )
        self._centre_layer.mode = "add"
        # Make sure this layer is the active one so clicks register
        self.viewer.layers.selection.active = self._centre_layer
        self._poll_timer.start()
        self._update_controls()

    def _enter_dye_stage(self):
        self._stage = 2
        if self._centre_layer is not None:
            self._centre_layer.mode = "pan_zoom"

        self._dye_layer = self.viewer.add_points(
            data=[], name="Dye mark (D)",
            face_color="#00dc32", border_color="white",
            symbol="disc", size=18,
        )
        self._dye_layer.mode = "add"
        self.viewer.layers.selection.active = self._dye_layer
        self._update_controls()

    def _enter_rhopalia_stage(self):
        self._stage = 3
        self._prev_rhop_count = 0
        if self._dye_layer is not None:
            self._dye_layer.mode = "pan_zoom"

        self._rhop_layer = self.viewer.add_points(
            data=[], name="Rhopalia",
            face_color="#d25000", border_color="white",
            symbol="disc", size=14,
        )
        self._rhop_layer.mode = "add"
        self.viewer.layers.selection.active = self._rhop_layer
        self._update_controls()

    # ── Next / Undo ───────────────────────────────────────────────────────────

    def _on_next(self):
        if self._stage == 1:
            # Read layer state directly in case polling hasn't synced yet
            if self._centre_layer is not None and len(self._centre_layer.data) > 0:
                row, col = self._centre_layer.data[-1]
                self._centre = (int(col), int(row))
            if self._centre is None:
                QMessageBox.information(
                    self, "No point placed",
                    "Click the bell centre in the viewer first.\n\n"
                    "Make sure the 'Centre (C)' layer is selected in the "
                    "layer list (bottom-left of the viewer).",
                )
                return
            self._enter_dye_stage()

        elif self._stage == 2:
            if self._dye_layer is not None and len(self._dye_layer.data) > 0:
                row, col = self._dye_layer.data[-1]
                self._dye = (int(col), int(row))
            if self._dye is None:
                QMessageBox.information(
                    self, "No point placed",
                    "Click the dye mark in the viewer first.\n\n"
                    "Make sure the 'Dye mark (D)' layer is selected in the "
                    "layer list (bottom-left of the viewer).",
                )
                return
            self._enter_rhopalia_stage()

    def _on_undo(self):
        if not self._rhopalia or self._rhop_layer is None:
            return
        data = self._rhop_layer.data
        if len(data) > 0:
            self._rhop_layer.data = data[:-1]
        # Polling will pick up the change on next tick

    # ── Angle table ───────────────────────────────────────────────────────────

    def _refresh_angle_table(self):
        self.angle_table.setRowCount(0)
        if self._centre is None or self._dye is None:
            return
        for i, rp in enumerate(self._rhopalia):
            phi = body_angle(self._centre, self._dye, rp)
            row = self.angle_table.rowCount()
            self.angle_table.insertRow(row)
            self.angle_table.setItem(row, 0, QTableWidgetItem(f"R{i}"))
            self.angle_table.setItem(row, 1, QTableWidgetItem(f"{phi:+.1f}"))
            self.angle_table.setItem(row, 2, QTableWidgetItem(f"{rp[0]}, {rp[1]}"))

    # ── Save ──────────────────────────────────────────────────────────────────

    def _on_save(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required",
                                "Enter an animal name before saving.")
            return

        # Sync final state from layer before saving
        if self._rhop_layer is not None and len(self._rhop_layer.data) > 0:
            self._rhopalia = [
                (int(c), int(r)) for r, c in self._rhop_layer.data
            ]

        if not self._rhopalia:
            QMessageBox.warning(self, "No rhopalia", "Place at least one rhopalium.")
            return

        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        json_out = CALIB_DIR / f"{name}.json"
        png_out  = CALIB_DIR / f"{name}_annotated.png"

        if json_out.exists():
            reply = QMessageBox.question(
                self, "File exists",
                f"{json_out.name} already exists.  Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        calib = build_calibration(self._centre, self._dye, self._rhopalia)
        write_calibration_json(
            calib, json_out,
            img_path=self._img_path,
            img_bgr=self._img_data,
        )
        save_annotated_image(self._img_data, calib, png_out)

        QMessageBox.information(
            self, "Saved",
            f"Calibration saved:\n  {json_out}\n  {png_out}\n\n"
            f"{calib['n_rhopalia']} rhopalia recorded.",
        )
