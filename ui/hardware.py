"""
ui/hardware.py

Hardware-status accordion panel.

A full-width toggle button expands/collapses a body widget showing GPU info,
live VRAM bar, SAM2 batch-size control, and auto-queue toggle.

GPU detection runs in a background QThread so it never blocks the UI thread.
The body shows "Detecting GPU…" until the thread reports back.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QThread, QTimer, Signal, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QCheckBox, QSpinBox, QPushButton, QFrame,
)


class _DetectWorker(QObject):
    """Imports src.resources (triggers torch/CUDA init) off the main thread."""
    finished = Signal(object, object)   # (ResourceInfo, GpuGate)

    def run(self):
        from src.resources import HARDWARE, GPU_GATE
        self.finished.emit(HARDWARE, GPU_GATE)


class HardwareWidget(QWidget):
    """
    Accordion-style hardware status + queue-settings panel.

    Signals
    -------
    auto_queue_changed(bool)   — user toggled auto-queue
    batch_size_changed(int)    — user changed SAM2 frame batch size
    """

    auto_queue_changed = Signal(bool)
    batch_size_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hw       = None
        self._gpu_gate = None

        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._refresh_vram)

        self._build_ui()
        self._start_detection()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Toggle button (full-width header) ─────────────────────────────────
        self._toggle_btn = QPushButton("▶  Hardware")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left; padding: 4px 8px;"
            "  background: #2a2a2a; border: none;"
            "  color: #aaa; font-weight: bold;"
            "}"
            "QPushButton:checked {"
            "  background: #1e2a1e; color: #88ffaa;"
            "}"
            "QPushButton:hover { background: #333; }"
        )
        self._toggle_btn.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle_btn)

        # ── Body (hidden until toggle) ────────────────────────────────────────
        self._body = QFrame()
        self._body.setFrameShape(QFrame.StyledPanel)
        self._body.setVisible(False)
        outer.addWidget(self._body)

        body = QVBoxLayout(self._body)
        body.setContentsMargins(8, 6, 8, 6)
        body.setSpacing(4)

        # GPU name + status
        self._gpu_lbl = QLabel("Detecting GPU…")
        self._gpu_lbl.setStyleSheet("font-weight: bold; color: #888;")
        body.addWidget(self._gpu_lbl)

        self._note_lbl = QLabel("")
        self._note_lbl.setStyleSheet("color: #666; font-size: 10px;")
        body.addWidget(self._note_lbl)

        # VRAM bar
        vram_row = QHBoxLayout()
        vram_row.addWidget(QLabel("VRAM"))
        self._vram_bar = QProgressBar()
        self._vram_bar.setRange(0, 1000)
        self._vram_bar.setTextVisible(True)
        self._vram_bar.setFixedHeight(14)
        self._vram_bar.setEnabled(False)
        self._vram_detail = QLabel("— / —")
        self._vram_detail.setFixedWidth(120)
        self._vram_detail.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vram_row.addWidget(self._vram_bar)
        vram_row.addWidget(self._vram_detail)
        body.addLayout(vram_row)

        self._gate_lbl = QLabel("")
        self._gate_lbl.setStyleSheet("color: #666; font-size: 10px;")
        body.addWidget(self._gate_lbl)

        # SAM2 batch size
        batch_row = QHBoxLayout()
        batch_lbl = QLabel("SAM2 frame batch:")
        batch_lbl.setToolTip(
            "Frames decoded and sent to SAM2 per forward pass.\n"
            "Larger = faster but uses more VRAM. Reduce if OOM errors appear.\n"
            "Recommended: 8 for RTX 4060 (8 GB),  16+ for 20 GB+ GPUs."
        )
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 64)
        self._batch_spin.setValue(8)
        self._batch_spin.setFixedWidth(60)
        self._batch_spin.valueChanged.connect(self.batch_size_changed)
        batch_row.addWidget(batch_lbl)
        batch_row.addWidget(self._batch_spin)
        batch_row.addStretch()
        body.addLayout(batch_row)

        # Auto-queue
        self._auto_queue_cb = QCheckBox("Auto-queue new recordings")
        self._auto_queue_cb.setToolTip(
            "When the folder watcher detects a newly completed video file,\n"
            "automatically add it to the processing queue."
        )
        self._auto_queue_cb.toggled.connect(self.auto_queue_changed)
        body.addWidget(self._auto_queue_cb)

    # ── Background GPU detection ──────────────────────────────────────────────

    def _start_detection(self):
        self._detect_thread = QThread(self)
        self._detect_worker = _DetectWorker()
        self._detect_worker.moveToThread(self._detect_thread)
        self._detect_thread.started.connect(self._detect_worker.run)
        self._detect_worker.finished.connect(self._on_detected)
        self._detect_worker.finished.connect(self._detect_thread.quit)
        self._detect_thread.start()

    def _on_detected(self, hw, gpu_gate):
        self._hw       = hw
        self._gpu_gate = gpu_gate

        if hw.cuda_available:
            badge = (
                f"{hw.gpu_name}  ·  "
                f"{hw.gpu_vram_gb:.1f} GB  ·  "
                f"{hw.max_gpu_concurrent}× GPU concurrency"
            )
            color = "#88ddaa"
            note  = (
                "(SAM2 + CoTracker run simultaneously)"
                if hw.max_gpu_concurrent >= 2 else
                "(SAM2 and CoTracker serialise — insufficient VRAM for overlap)"
            )
            default_batch = 8 if hw.gpu_vram_gb < 12 else 16
            btn_summary = f"{hw.gpu_name}  ·  {hw.gpu_vram_gb:.1f} GB"
        else:
            badge = "No CUDA GPU detected — CPU only"
            color = "#dd8844"
            note  = "Expect significantly slower runtimes."
            default_batch = 4
            btn_summary = "CPU only"

        self._gpu_lbl.setText(badge)
        self._gpu_lbl.setStyleSheet(f"font-weight: bold; color: {color};")
        self._note_lbl.setText(note)
        self._batch_spin.setValue(default_batch)

        arrow = "▼" if self._toggle_btn.isChecked() else "▶"
        self._toggle_btn.setText(f"{arrow}  Hardware")

        if hw.cuda_available:
            self._vram_bar.setEnabled(True)
            self._vram_timer.start(2000)
            self._refresh_vram()

    # ── Toggle body ───────────────────────────────────────────────────────────

    def _on_toggled(self, checked: bool):
        self._body.setVisible(checked)
        text = self._toggle_btn.text()
        # Swap the arrow prefix (first character)
        self._toggle_btn.setText(("▼" if checked else "▶") + text[1:])

    # ── Live VRAM polling ─────────────────────────────────────────────────────

    def _refresh_vram(self):
        if self._hw is None or not self._hw.cuda_available:
            return
        try:
            import torch
            used  = torch.cuda.memory_allocated(0) / 1e9
            total = self._hw.gpu_vram_gb
            # Laptop GPUs can spill into shared system RAM; cap display at dedicated VRAM
            display_used = min(used, total)
            frac  = display_used / total if total > 0 else 0.0
            pct   = frac * 100

            self._vram_bar.setValue(int(frac * 1000))
            self._vram_bar.setFormat(f"{pct:.0f}%")
            chunk = "#cc4444" if pct > 85 else "#cc9944" if pct > 60 else "#44aa66"
            self._vram_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background: {chunk}; }}"
            )
            self._vram_detail.setText(f"{display_used:.2f} / {total:.1f} GB")

            if self._gpu_gate:
                active = self._gpu_gate.active_count
                self._gate_lbl.setText(
                    f"GPU slot: {active} / {self._hw.max_gpu_concurrent} active"
                )
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def batch_size(self) -> int:
        return self._batch_spin.value()

    @property
    def auto_queue_enabled(self) -> bool:
        return self._auto_queue_cb.isChecked()
