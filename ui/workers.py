"""
ui/workers.py

Thread worker wrappers for background pipeline execution.
All napari layer updates must happen on the main thread; workers yield
results via napari.qt.threading.thread_worker.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable

from napari.qt.threading import thread_worker

from src.pipeline import PipelineResult, run_pipeline
from src.scheduler import ProgressEvent
from .parameters import PipelineParams


# ── Progress relay via thread-safe queue ──────────────────────────────────────

class ProgressRelay:
    """
    Converts scheduler callbacks (emitted from worker threads) into objects
    that can be safely yielded back to the napari main thread.

    Usage
    -----
    relay = ProgressRelay()
    # pass relay.callback as progress_callback to run_pipeline
    # in the worker, call relay.drain() to yield pending events
    """

    def __init__(self):
        self._q: queue.Queue[ProgressEvent] = queue.Queue()

    def callback(self, event: ProgressEvent) -> None:
        """Called from the scheduler's worker threads."""
        self._q.put(event)

    def drain(self) -> list[ProgressEvent]:
        """Pull all pending events (call from generator worker)."""
        events = []
        try:
            while True:
                events.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return events


# ── Pipeline worker ───────────────────────────────────────────────────────────

@thread_worker
def run_pipeline_worker(
    video_path:   Path,
    bell_click:   tuple[int, int],
    dye_click:    tuple[int, int],
    calib_path:   Path,
    params:       PipelineParams,
    cancel_event: threading.Event,
    delete_old_outputs: bool = False,
):
    """
    Background worker that runs the full pipeline.

    Yields
    ------
    ProgressEvent objects as they arrive from the scheduler, so the UI can
    update its progress bars in real-time.

    Returns
    -------
    PipelineResult on completion (received by the `returned` callback).
    """
    import time

    if delete_old_outputs:
        _delete_cached_outputs(video_path)

    relay = ProgressRelay()

    # resolve SAM2 weights path from model name
    from config import WEIGHTS_DIR
    model_name = params.sam2_model.value
    weights_map = {
        "tiny":      "sam2.1_hiera_tiny.pt",
        "small":     "sam2.1_hiera_small.pt",
        "base_plus": "sam2.1_hiera_base_plus.pt",
        "large":     "sam2.1_hiera_large.pt",
    }
    # image_size override not exposed in UI params; use None (pipeline default)

    result = run_pipeline(
        video_path        = video_path,
        bell_click        = bell_click,
        dye_click         = dye_click,
        calib_path        = calib_path,
        stride            = params.stride,
        cotracker_stride  = params.cotracker_stride,
        pre_window        = params.pre_window,
        inner_frac        = params.inner_frac,
        outer_frac        = params.outer_frac,
        prominence        = params.prominence,
        progress_callback = relay.callback,
        cancel_event      = cancel_event,
    )

    # Drain any remaining events
    for ev in relay.drain():
        yield ev

    return result


def _delete_cached_outputs(video_path: Path) -> None:
    """Remove all cached outputs for this video so the pipeline reruns fully."""
    from src.tasks import run_dir
    from config import OUTPUTS_DIR
    rdir = OUTPUTS_DIR / video_path.stem
    if not rdir.exists():
        return
    stem = video_path.stem
    # Delete the intermediate files the scheduler checks for skipping
    extensions = [
        "_seg.csv", "_contour_radii.npy",
        "_track.csv",
        "_margin_diff.npy",
        "_initiation_b.csv", "_initiation_b_plot.png",
        "_initiation_b_annotated.mp4",
    ]
    for ext in extensions:
        p = rdir / f"{stem}{ext}"
        if p.exists():
            p.unlink()
