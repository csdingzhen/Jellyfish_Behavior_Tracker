"""
src/resources.py

Hardware detection and resource gating.

GPU gate
--------
A semaphore that limits how many pipeline stages can hold the GPU
simultaneously.  The limit is set automatically based on detected VRAM:

  >= 20 GB  →  2 concurrent GPU tasks   (RTX 4090 / A100 etc.)
  <  20 GB  →  1 concurrent GPU task    (RTX 4060 laptop etc.)

UI hook
-------
``ResourceInfo`` is a plain dataclass the UI can read at startup to display
hardware information in a settings or status panel.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field


# ── Hardware snapshot ─────────────────────────────────────────────────────────

@dataclass
class ResourceInfo:
    """Snapshot of available hardware — populated once at import time."""
    gpu_name:          str   = "Unknown"
    gpu_vram_gb:       float = 0.0
    cuda_available:    bool  = False
    max_gpu_concurrent: int  = 1
    system_ram_gb:     float = 0.0


def _detect() -> ResourceInfo:
    info = ResourceInfo()
    try:
        import psutil
        info.system_ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.cuda_available  = True
            info.gpu_name        = props.name
            info.gpu_vram_gb     = props.total_memory / 1e9
            info.max_gpu_concurrent = 2 if info.gpu_vram_gb >= 20 else 1
    except Exception:
        pass

    return info


HARDWARE: ResourceInfo = _detect()


# ── GPU gate ──────────────────────────────────────────────────────────────────

class GpuGate:
    """
    Semaphore-based concurrency controller for GPU tasks.

    Usage (within a task function):
        with GPU_GATE:
            ... GPU work ...

    The gate also calls ``torch.cuda.empty_cache()`` on release so VRAM is
    returned to the pool before the next task acquires the lock.

    UI hook: ``GpuGate.active_count`` is safe to read from any thread and
    reflects how many tasks currently hold the GPU.
    """

    def __init__(self, max_concurrent: int):
        self._sem         = threading.Semaphore(max_concurrent)
        self._count_lock  = threading.Lock()
        self._active      = 0

    def acquire(self) -> None:
        self._sem.acquire()
        with self._count_lock:
            self._active += 1

    def release(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        with self._count_lock:
            self._active -= 1
        self._sem.release()

    @contextmanager
    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()

    @property
    def active_count(self) -> int:
        with self._count_lock:
            return self._active


# Singleton — import and use directly
GPU_GATE: GpuGate = GpuGate(HARDWARE.max_gpu_concurrent)
