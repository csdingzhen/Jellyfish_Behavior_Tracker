"""
src/pipeline.py

Top-level pipeline assembly and execution.

Public API
----------
run_pipeline(...)   — assemble tasks, run the scheduler, return PipelineResult
PipelineResult      — dataclass with paths to all outputs + timing stats

UI integration
--------------
Pass a ``progress_callback`` receiving ``ProgressEvent`` objects.
Pass a ``cancel_event`` threading.Event and set it from a "Cancel" button.
The function is blocking; call it from a background thread (QThread etc.).

CLI integration
---------------
scripts/run_pipeline.py is a thin wrapper that collects click points and
calls run_pipeline().  See that file for a reference implementation.

Parallel execution
------------------
The scheduler automatically runs SAM2, CoTracker, and Margin-diff (Phase 1a)
concurrently wherever the dependency graph and GPU VRAM allow:

  On  < 20 GB VRAM (RTX 4060):
    SAM2 and CoTracker serialise through the GPU gate.
    Phase 1a (CPU) runs as soon as SAM2 produces seg.csv.

  On >= 20 GB VRAM (RTX 4090):
    SAM2 + CoTracker both hold a GPU slot simultaneously.
    Phase 1a starts (CPU) the moment SAM2 finishes.
    Total wall-clock ≈ max(SAM2, CoTracker) rather than SAM2 + CoTracker.

SAM2 vs CoTracker stride mismatch
----------------------------------
SAM2 may run at a finer stride (e.g. 4) than CoTracker (e.g. 8) to improve
centroid/mask accuracy without slowing CoTracker.  The nearest() function in
run_approach_b.py already handles the mismatch: body-frame angle lookup snaps
to the closest available dye frame (max error = half the CoTracker stride =
33 ms at stride 8 / 120 fps).  For Cassiopea which barely rotates, this is
negligible.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .scheduler import ProgressCallback, ProgressEvent, Scheduler, TaskStatus
from .tasks import (
    make_sam2_task,
    make_cotracker_task,
    make_phase1a_task,
    make_phase1b_task,
    make_analysis_task,
    run_dir,
)
from config import OUTPUTS_DIR


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Paths to every output file produced by a successful pipeline run,
    plus timing and status information.

    UI reads this after run_pipeline() returns to populate the Results tab.
    """
    # Core outputs
    seg_csv:        Path | None = None
    contour_npy:    Path | None = None
    track_csv:      Path | None = None
    margin_diff_npy: Path | None = None
    initiation_csv: Path | None = None
    initiation_plot: Path | None = None
    annotated_video: Path | None = None

    # Run metadata
    elapsed_s:      float             = 0.0
    success:        bool              = False
    task_status:    dict[str, str]    = field(default_factory=dict)
    task_elapsed:   dict[str, float]  = field(default_factory=dict)
    errors:         dict[str, str]    = field(default_factory=dict)
    cancelled:      bool              = False

    def summary(self) -> str:
        lines = [
            f"Pipeline {'completed' if self.success else 'failed/cancelled'} "
            f"in {self.elapsed_s:.0f}s",
        ]
        for name, status in self.task_status.items():
            lines.append(f"  {name:35s} {status}")
        if self.errors:
            lines.append("Errors:")
            for name, err in self.errors.items():
                lines.append(f"  {name}: {err.splitlines()[0]}")
        return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(
    video_path:   Path,
    bell_click:   tuple[int, int],
    dye_click:    tuple[int, int],
    calib_path:   Path,
    *,
    stride:           int   = 4,   # SAM2 + Phase 1a/1b stride
    cotracker_stride: int | None = None,   # CoTracker stride (defaults to stride)
    window_size:      int   = 200,
    save_n_masks:     int   = 20,
    inner_frac:       float = 0.75,
    outer_frac:       float = 1.05,
    pre_window:       int   = 30,
    min_distance:     float = 0.42,
    prominence:       float = 0.05,
    delete_frames:    bool  = True,
    # PERF branch options
    sam2_tracks_dye:  bool      = False,   # SAM2 obj_id=2 replaces CoTracker
    image_size:       int | None = None,   # SAM2 internal ViT resolution override
    use_gpu_approach_b: bool    = False,   # GPU grid_sample for margin diff
    progress_callback: ProgressCallback | None = None,
    cancel_event:      threading.Event  | None = None,
) -> PipelineResult:
    """
    Run the full Cassiopea analysis pipeline.

    Parameters
    ----------
    video_path    : path to source video (any OpenCV-readable format)
    bell_click    : (x, y) pixel clicked on the jellyfish bell in frame 0
    dye_click     : (x, y) pixel clicked on the dye mark in frame 0
    calib_path    : path to calibration JSON from calibrate_rhopalia.py
    stride        : frame subsampling (4 = process every 4th frame)
    progress_callback : called with ProgressEvent on every state change
    cancel_event  : set from any thread to abort the run

    Returns
    -------
    PipelineResult with paths to all outputs and run metadata.
    """
    if cancel_event is None:
        cancel_event = threading.Event()
    ct_stride = cotracker_stride if cotracker_stride is not None else stride

    OUTPUTS_DIR.mkdir(exist_ok=True)
    stem = video_path.stem

    # Assemble task graph
    scheduler = Scheduler(
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    scheduler.add(make_sam2_task(
        video_path, bell_click,
        stride=stride, window_size=window_size,
        save_n_masks=save_n_masks, delete_frames=delete_frames,
        dye_click=dye_click if sam2_tracks_dye else None,
        image_size=image_size,
    ))
    if not sam2_tracks_dye:
        # main branch: CoTracker handles dye tracking separately
        scheduler.add(make_cotracker_task(
            video_path, dye_click,
            stride=ct_stride, chunk_size=400,
        ))
    scheduler.add(make_phase1a_task(
        video_path,
        stride=stride, inner_frac=inner_frac, outer_frac=outer_frac,
        use_gpu=use_gpu_approach_b,
    ))
    scheduler.add(make_phase1b_task(video_path, stride=stride,
                                    sam2_tracks_dye=sam2_tracks_dye))
    scheduler.add(make_analysis_task(
        video_path, calib_path,
        stride=stride, pre_window=pre_window,
        min_distance=min_distance, prominence=prominence,
    ))

    t0      = time.time()
    success = scheduler.run()
    elapsed = time.time() - t0

    snap   = scheduler.snapshot()
    rdir   = run_dir(video_path)

    def _maybe(p: Path) -> Path | None:
        return p if p.exists() else None

    return PipelineResult(
        seg_csv         = _maybe(rdir / f"{stem}_seg.csv"),
        contour_npy     = _maybe(rdir / f"{stem}_contour_radii.npy"),
        track_csv       = _maybe(rdir / f"{stem}_track.csv"),
        margin_diff_npy = _maybe(rdir / f"{stem}_margin_diff.npy"),
        initiation_csv  = _maybe(rdir / f"{stem}_initiation_b.csv"),
        initiation_plot = _maybe(rdir / f"{stem}_initiation_b_plot.png"),
        annotated_video = _maybe(rdir / f"{stem}_initiation_b_annotated.mp4"),
        elapsed_s       = elapsed,
        success         = success,
        task_status     = {n: s.name for n, s in snap.status.items()},
        task_elapsed    = snap.elapsed,
        errors          = snap.errors,
        cancelled       = snap.is_cancelled,
    )
