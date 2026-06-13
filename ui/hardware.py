"""
ui/hardware.py

Collapsible hardware-status panel shown at the top of the right dock.

Displays:
  • GPU name and CUDA status
  • Live VRAM bar (used / total), polled every 2 s
  • GPU-gate slot indicator (how many tasks currently hold the GPU)
  • SAM2 frame batch-size control (tunes peak VRAM per forward pass)
  • Auto-queue toggle (watch folder, process new videos automatically)
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QCheckBox, QSpinBox,
)


class HardwareWidget(QGroupBox):
    """
    Collapsible GPU status + queue-settings panel.

    Signals
    -------
    auto_queue_changed(bool)  — user toggled the auto-queue checkbox
    batch_size_changed(int)   — user changed SAM2 frame batch size
    """

    auto_queue_changed = Signal(bool)
    batch_size_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__("Hardware", parent)
        self.setCheckable(True)
        self.setChecked(False)   # collapsed by default

        from src.resources import HARDWARE, GPU_GATE
        self._hw       = HARDWARE
        self._gpu_gate = GPU_GATE

        self._body = QWidget()
        self._build_ui()
        self.toggled.connect(self._body.setVisible)
        self._body.setVisible(False)

        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._refresh_vram)
        self._vram_timer.start(2000)
        self._refresh_vram()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(self._body)

        layout = QVBoxLayout(self._body)
        layout.setSpacing(4)

        # ── GPU identity ──────────────────────────────────────────────────────
        if self._hw.cuda_available:
            concurrency = self._hw.max_gpu_concurrent
            badge = (
                f"{self._hw.gpu_name}  ·  "
                f"{self._hw.gpu_vram_gb:.1f} GB  ·  "
                f"{concurrency}× GPU concurrency"
            )
            color = "#88ddaa"
            note = (
                "(SAM2 + CoTracker run simultaneously)"
                if concurrency >= 2 else
                "(SAM2 and CoTracker serialise — insufficient VRAM for overlap)"
            )
        else:
            badge = "No CUDA GPU detected — CPU only"
            color = "#dd8844"
            note  = "Expect significantly slower runtimes."

        gpu_lbl = QLabel(badge)
        gpu_lbl.setStyleSheet(f"font-weight: bold; color: {color};")
        layout.addWidget(gpu_lbl)

        note_lbl = QLabel(note)
        note_lbl.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(note_lbl)

        # ── VRAM bar ──────────────────────────────────────────────────────────
        if self._hw.cuda_available:
            vram_row = QHBoxLayout()
            vram_row.addWidget(QLabel("VRAM"))

            self._vram_bar = QProgressBar()
            self._vram_bar.setRange(0, 1000)
            self._vram_bar.setTextVisible(True)
            self._vram_bar.setFixedHeight(14)

            self._vram_detail = QLabel("— / —")
            self._vram_detail.setFixedWidth(120)
            self._vram_detail.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            vram_row.addWidget(self._vram_bar)
            vram_row.addWidget(self._vram_detail)
            layout.addLayout(vram_row)

            self._gate_lbl = QLabel("")
            self._gate_lbl.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(self._gate_lbl)
        else:
            self._vram_bar    = None
            self._vram_detail = None
            self._gate_lbl    = None

        # ── SAM2 batch size ───────────────────────────────────────────────────
        batch_row = QHBoxLayout()
        batch_lbl = QLabel("SAM2 frame batch:")
        batch_lbl.setToolTip(
            "Frames decoded and sent to SAM2 per forward pass.\n"
            "Larger = faster but uses more VRAM. Reduce if you see OOM errors.\n"
            "Recommended: 8 for RTX 4060 (8 GB), 16+ for 20 GB+ GPUs."
        )
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 64)
        default = 8 if self._hw.gpu_vram_gb < 12 else 16
        self._batch_spin.setValue(default)
        self._batch_spin.setFixedWidth(60)
        self._batch_spin.valueChanged.connect(self.batch_size_changed)
        batch_row.addWidget(batch_lbl)
        batch_row.addWidget(self._batch_spin)
        batch_row.addStretch()
        layout.addLayout(batch_row)

        # ── Auto-queue toggle ─────────────────────────────────────────────────
        self._auto_queue_cb = QCheckBox("Auto-queue new recordings")
        self._auto_queue_cb.setToolTip(
            "When the folder watcher detects a newly completed video file,\n"
            "automatically add it to the processing queue. The pipeline starts\n"
            "as soon as the GPU is free — no manual intervention needed."
        )
        self._auto_queue_cb.toggled.connect(self.auto_queue_changed)
        layout.addWidget(self._auto_queue_cb)

    # ── VRAM polling ──────────────────────────────────────────────────────────

    def _refresh_vram(self):
        if not self._hw.cuda_available or self._vram_bar is None:
            return
        try:
            import torch
            used  = torch.cuda.memory_allocated(0) / 1e9
            total = self._hw.gpu_vram_gb
            frac  = used / total if total > 0 else 0.0
            pct   = frac * 100

            self._vram_bar.setValue(int(frac * 1000))
            self._vram_bar.setFormat(f"{pct:.0f}%")

            if pct > 85:
                chunk_color = "#cc4444"
            elif pct > 60:
                chunk_color = "#cc9944"
            else:
                chunk_color = "#44aa66"
            self._vram_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {chunk_color}; }}"
            )
            self._vram_detail.setText(f"{used:.2f} / {total:.1f} GB")

            active = self._gpu_gate.active_count
            limit  = self._hw.max_gpu_concurrent
            self._gate_lbl.setText(f"GPU slot: {active} / {limit} active")
        except Exception:
            pass

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def batch_size(self) -> int:
        return self._batch_spin.value()

    @property
    def auto_queue_enabled(self) -> bool:
        return self._auto_queue_cb.isChecked()
