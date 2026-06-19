"""
ui/hardware.py

Compact hardware-status bar — always visible, single row.

Shows GPU name, VRAM total, concurrency mode, and live allocated-VRAM.
GPU detection runs off the main thread so it never blocks the UI.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QThread, QTimer, Signal, QObject
from qtpy.QtWidgets import QWidget, QHBoxLayout, QLabel, QCheckBox

from .style import C_BORDER, C_CARD, C_TEXT_DIM, C_TEXT, C_RED, C_ORANGE, C_BLUE


class _DetectWorker(QObject):
    finished = Signal(object, object)   # (ResourceInfo, GpuGate)

    def run(self):
        from src.resources import HARDWARE, GPU_GATE
        self.finished.emit(HARDWARE, GPU_GATE)


class HardwareWidget(QWidget):
    """
    Single-row hardware bar.

    Signals
    -------
    auto_queue_changed(bool)
    batch_size_changed(int)   — kept for API compat
    """

    auto_queue_changed = Signal(bool)
    batch_size_changed = Signal(int)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def auto_queue_enabled(self) -> bool:
        return self._auto_queue_cb.isChecked()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hw         = None
        self._gpu_gate   = None
        self._batch_size = 8

        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._refresh_vram)

        self._build_ui()
        self._start_detection()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setFixedHeight(34)
        self.setStyleSheet(f"""
            HardwareWidget {{
                background: {C_CARD};
                border-bottom: 1px solid {C_BORDER};
            }}
            QCheckBox {{
                color: {C_TEXT_DIM};
                font-size: 10px;
                spacing: 4px;
            }}
            QCheckBox::indicator {{
                width: 13px;
                height: 13px;
                border: 1px solid #555;
                border-radius: 3px;
                background: #1a1a1a;
            }}
            QCheckBox::indicator:checked {{
                background: {C_BLUE};
                border-color: {C_BLUE};
            }}
            QCheckBox::indicator:hover {{
                border-color: #777;
            }}
        """)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 0, 12, 0)
        row.setSpacing(8)

        self._auto_queue_cb = QCheckBox()
        self._auto_queue_cb.setToolTip("Auto-queue new recordings when detected")
        self._auto_queue_cb.toggled.connect(self.auto_queue_changed)
        row.addWidget(self._auto_queue_cb)

        self._info_lbl = QLabel("Detecting GPU…")
        self._info_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        row.addWidget(self._info_lbl, stretch=1)

        self._vram_lbl = QLabel("")
        self._vram_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._vram_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px;")
        self._vram_lbl.setFixedWidth(96)
        row.addWidget(self._vram_lbl)

    # ── GPU detection ─────────────────────────────────────────────────────────

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
            concurrency = ("Serial mode" if hw.max_gpu_concurrent < 2
                           else f"{hw.max_gpu_concurrent}× parallel")
            self._info_lbl.setText(
                f"{hw.gpu_name}  ·  {hw.gpu_vram_gb:.1f} GB  ·  {concurrency}"
            )
            gpu_color = "#76b900" if "nvidia" in hw.gpu_name.lower() else C_TEXT
            self._info_lbl.setStyleSheet(f"color: {gpu_color}; font-size: 11px;")
            self._batch_size = 8 if hw.gpu_vram_gb < 12 else 16
            self._vram_timer.start(2000)
            self._refresh_vram()
        else:
            self._info_lbl.setText("No CUDA GPU — CPU only (slower)")
            self._info_lbl.setStyleSheet(f"color: #e6b800; font-size: 11px;")
            self._batch_size = 4

    # ── Live VRAM polling ─────────────────────────────────────────────────────

    def _refresh_vram(self):
        if self._hw is None or not self._hw.cuda_available:
            return
        try:
            import torch
            used  = torch.cuda.memory_allocated(0) / 1e9
            total = self._hw.gpu_vram_gb
            used  = min(used, total)
            pct   = used / total * 100 if total > 0 else 0
            color = C_RED if pct > 85 else C_ORANGE if pct > 60 else C_TEXT_DIM
            self._vram_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
            self._vram_lbl.setText(f"VRAM {used:.2f} GB")
        except Exception:
            pass
