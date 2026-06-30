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

import json
import threading
import time
from datetime import datetime
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
    get_output_root,
)


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


# ── Provenance log ────────────────────────────────────────────────────────────

def _write_run_log(
    video_path: Path,
    calib_path: Path,
    result:     PipelineResult,
    *,
    wall_s:     float,
    config:     dict,
) -> Path:
    """Append a config + timing record to ``<run_dir>/<stem>_run_log.json``.

    Written by both the CLI and the UI (run_pipeline is the single shared
    entry point), so every run — successful or not — leaves a provenance
    trail. Records accumulate as a JSON list across re-runs of the same video.
    """
    from .resources import HARDWARE

    stem     = video_path.stem
    log_path = run_dir(video_path) / f"{stem}_run_log.json"

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video":     video_path.name,
        "calib":     calib_path.name if calib_path else None,
        "config":    config,
        "hardware": {
            "gpu":                HARDWARE.gpu_name,
            "vram_gb":            round(HARDWARE.gpu_vram_gb, 1),
            "max_gpu_concurrent": HARDWARE.max_gpu_concurrent,
        },
        "task_timing": {
            name: {
                "status":    result.task_status.get(name, "-"),
                "elapsed_s": round(result.task_elapsed.get(name, 0.0), 1),
            }
            for name in result.task_status
        },
        "total_wall_s": round(wall_s, 1),
        "success":      result.success,
        "cancelled":    result.cancelled,
        "errors":       {k: v.splitlines()[0] for k, v in result.errors.items()},
    }

    runs = []
    if log_path.exists():
        try:
            runs = json.loads(log_path.read_text())
        except (json.JSONDecodeError, ValueError):
            runs = []
    runs.append(entry)
    log_path.write_text(json.dumps(runs, indent=2))
    return log_path


def _git_short_sha() -> str:
    """Best-effort current commit short SHA for provenance; 'unknown' if unavailable."""
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _write_summary(
    video_path: Path,
    calib_path: Path,
    result:     PipelineResult,
    config:     dict,
) -> Path:
    """Write a per-video ``<stem>_summary.json`` manifest.

    One machine-readable "read me first" file per recording, combining
    recording metadata + params + provenance + the scientific results
    (pulse count, confident fraction, per-rhopalium firing histogram,
    dominant initiator). Downstream analysis and the batch table read this
    rather than re-parsing the initiation CSV each time.
    """
    import csv as _csv

    stem         = video_path.stem
    summary_path = run_dir(video_path) / f"{stem}_summary.json"

    # Recording metadata
    recording = {}
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        recording = {
            "fps":        round(fps, 2),
            "n_frames":   n,
            "duration_s": round(n / fps, 2) if fps > 0 else None,
            "width":      w,
            "height":     h,
        }
    except Exception:
        pass

    # Scientific results — parsed from the initiation CSV. Firing histogram
    # and dominant initiator use CONFIDENT pulses only (the scientifically
    # valid set; see README "Known limitations").
    stats = {
        "n_pulses":            0,
        "n_confident":         0,
        "confident_fraction":  None,
        "dominant_rhopalium":  None,
        "firing_counts":       {},   # rhopalium_id -> count, confident pulses only
    }
    if result.initiation_csv and result.initiation_csv.exists():
        try:
            with open(result.initiation_csv) as f:
                rows = list(_csv.DictReader(f))
            confident = [r for r in rows if r.get("signal_confident") == "1"]
            counts: dict[str, int] = {}
            for r in confident:
                rid = r.get("rhopalium_id")
                if rid not in (None, ""):
                    counts[str(rid)] = counts.get(str(rid), 0) + 1
            dominant = max(counts, key=counts.get) if counts else None
            stats = {
                "n_pulses":           len(rows),
                "n_confident":        len(confident),
                "confident_fraction": round(len(confident) / len(rows), 3) if rows else None,
                "dominant_rhopalium": int(dominant) if dominant is not None else None,
                "firing_counts":      counts,
            }
        except Exception:
            pass

    summary = {
        "video":            video_path.name,
        "stem":             stem,
        "generated":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": _git_short_sha(),
        "success":          result.success,
        "calibration":      calib_path.name if calib_path else None,
        "recording":        recording,
        "config":           config,
        "results":          stats,
        "outputs": {
            "initiation_csv":  result.initiation_csv.name  if result.initiation_csv  else None,
            "initiation_plot": result.initiation_plot.name if result.initiation_plot else None,
            "annotated_video": result.annotated_video.name if result.annotated_video else None,
            "run_log":         f"{stem}_run_log.json",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def _write_videos_table(output_root: Path) -> Path | None:
    """Refresh ``<output_root>/videos.csv`` — one row per processed recording,
    aggregated from each video subfolder's ``<stem>_summary.json``.

    Regenerated from scratch on every call so it always reflects the current
    set of processed videos (and self-heals if a summary changes). One row per
    video keeps it small even for a 24×1-hour batch — we deliberately do NOT
    aggregate every pulse into one giant table. Lives at the project root when
    a project is active (``outputs/<project>/videos.csv``), else under the base.
    """
    import csv as _csv

    if not output_root.exists():
        return None

    rows = []
    for sub in sorted(p for p in output_root.iterdir() if p.is_dir()):
        sjson = sub / f"{sub.name}_summary.json"
        if not sjson.exists():
            continue
        try:
            s = json.loads(sjson.read_text())
        except Exception:
            continue
        rec = s.get("recording", {}) or {}
        res = s.get("results", {}) or {}
        rows.append({
            "stem":               s.get("stem", sub.name),
            "video":              s.get("video", ""),
            "duration_s":         rec.get("duration_s"),
            "fps":                rec.get("fps"),
            "n_pulses":           res.get("n_pulses"),
            "n_confident":        res.get("n_confident"),
            "confident_fraction": res.get("confident_fraction"),
            "dominant_rhopalium": res.get("dominant_rhopalium"),
            "calibration":        s.get("calibration"),
            "generated":          s.get("generated"),
            "success":            s.get("success"),
        })

    if not rows:
        return None

    table_path = output_root / "videos.csv"
    with open(table_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return table_path


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

    get_output_root().mkdir(parents=True, exist_ok=True)
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

    result = PipelineResult(
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

    # Provenance — written for BOTH CLI and UI runs (the UI previously had
    # none). run_log.json accumulates every run; summary.json is the single
    # latest manifest (metadata + params + results). Never let these break
    # the run itself.
    config = {
        "sam2_stride":      stride,
        "cotracker_stride": ct_stride,
        "image_size_px":    image_size,
        "window_size":      window_size,
        "inner_frac":       inner_frac,
        "outer_frac":       outer_frac,
        "pre_window":       pre_window,
        "min_distance_s":   min_distance,
        "prominence":       prominence,
        "save_n_masks":     save_n_masks,
        "gpu_approach_b":   use_gpu_approach_b,
    }
    try:
        _write_run_log(video_path, calib_path, result, wall_s=elapsed, config=config)
    except Exception:
        pass
    try:
        _write_summary(video_path, calib_path, result, config)
    except Exception:
        pass
    try:
        _write_videos_table(get_output_root())
    except Exception:
        pass

    return result
