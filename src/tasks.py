"""
src/tasks.py

Task factory functions — one per pipeline stage.

Each factory returns a ``Task`` object ready to be added to the Scheduler.
All task functions accept ``progress_callback`` and ``cancel_event`` as
keyword arguments (injected by the Scheduler).  The functions are thin
wrappers around the existing scripts so the scripts remain fully usable
standalone from the terminal.

UI notes
--------
- task.name is the display label in the UI sidebar
- task.weight is the relative time cost (used for overall_fraction progress bar)
- task.inputs / task.outputs drive checkpoint detection (skip if up-to-date)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np

# Let scripts be importable from the project root
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.scheduler import Task
from config import OUTPUTS_DIR, SAM2_WEIGHTS, COTRACKER_WEIGHTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stem(video_path: Path) -> str:
    return video_path.stem


def run_dir(video_path: Path, output_root: Path | None = None) -> Path:
    """
    Return the per-recording output directory.
    Default: <OUTPUTS_DIR>/<video_stem>/

    All task outputs live inside this folder so recordings never pollute
    each other's files.  The UI can display one folder per recording.
    """
    root = output_root or OUTPUTS_DIR
    d    = root / video_path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out(video_path: Path, *parts: str, output_root: Path | None = None) -> Path:
    """Build an output path under the per-recording directory."""
    return run_dir(video_path, output_root).joinpath(*[str(p) for p in parts])


# ── Task 1 — SAM2 bell segmentation ──────────────────────────────────────────

def _run_sam2_task(
    video_path:    Path,
    bell_click:    tuple[int, int],
    stride:        int,
    window_size:   int,
    save_n_masks:  int,
    delete_frames: bool = False,
    dye_click:     tuple[int, int] | None = None,    # PERF: track dye inside SAM2
    image_size:    int | None = None,                # PERF: override internal ViT resolution
    progress_callback: Callable | None = None,
    cancel_event:      threading.Event | None = None,
) -> None:
    import torch
    import cv2
    from scripts.run_sam2 import (
        extract_frames, run_sam2, mask_to_stats,
        N_CONTOUR_ANGLES,
    )
    import csv, math

    stem       = _stem(video_path)
    frames_dir = _out(video_path, f"{stem}_frames")
    mask_dir   = _out(video_path, f"{stem}_masks")
    seg_csv    = _out(video_path, f"{stem}_seg.csv")
    cont_npy   = _out(video_path, f"{stem}_contour_radii.npy")

    cap     = cv2.VideoCapture(str(video_path))
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    n_extracted = (total + stride - 1) // stride

    if progress_callback:
        progress_callback(0, n_extracted, "Extracting frames...")

    n_extracted = extract_frames(video_path, frames_dir, stride)

    if cancel_event and cancel_event.is_set():
        return

    # Determine which frames to save as PNG masks
    if save_n_masks > 0:
        idxs         = np.linspace(0, n_extracted - 1, save_n_masks, dtype=int)
        save_indices = set(idxs.tolist())
    else:
        save_indices = set()

    if progress_callback:
        progress_callback(0, n_extracted, "Loading SAM2...")

    from sam2.build_sam import build_sam2_video_predictor
    from config import SAM2_CONFIG
    device    = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    overrides = [f"++model.image_size={image_size}"] if image_size else []
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG, str(SAM2_WEIGHTS), device=device,
        hydra_overrides_extra=overrides if overrides else None,
    )

    seg_stats, contour_arr, dye_track_list = run_sam2(
        predictor, frames_dir, bell_click, mask_dir,
        n_extracted, window_size, save_indices,
        dye_click=dye_click,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    # Write seg CSV
    run_dir(video_path).mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(seg_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["frame_idx", "timestamp_s", "cx", "cy", "radius_px"])
        for i, (cx, cy, r) in enumerate(seg_stats):
            raw = i * stride
            w.writerow([raw, f"{raw / fps_raw:.4f}",
                        f"{cx:.2f}", f"{cy:.2f}", f"{r:.2f}"])

    # Write contour radii
    np.save(str(cont_npy), contour_arr)

    # Write dye track CSV (same format as CoTracker) when dye_click was provided
    if dye_click is not None and dye_track_list:
        track_csv = _out(video_path, f"{_stem(video_path)}_track.csv")
        with open(track_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["frame_idx", "timestamp_s", "x", "y", "visible"])
            for i, (dx, dy) in enumerate(dye_track_list):
                raw = i * stride
                vis = 1 if (dx > 0 or dy > 0) else 0
                w.writerow([raw, f"{raw / fps_raw:.4f}",
                            f"{dx:.2f}", f"{dy:.2f}", vis])

    # Delete extracted JPEG frames if requested (saves 5+ GB for long videos)
    if delete_frames and frames_dir.exists():
        import shutil as _shutil
        if progress_callback:
            progress_callback(n_extracted, n_extracted, "Cleaning up JPEG frames...")
        _shutil.rmtree(frames_dir)

    if progress_callback:
        progress_callback(n_extracted, n_extracted, "SAM2 complete")


def make_sam2_task(
    video_path:   Path,
    bell_click:   tuple[int, int],
    stride:         int   = 4,
    window_size:    int   = 200,
    save_n_masks:   int   = 20,
    delete_frames:  bool  = True,
    dye_click:      tuple[int, int] | None = None,   # PERF: track dye inside SAM2
    image_size:     int | None = None,               # PERF: override ViT resolution
) -> Task:
    stem    = _stem(video_path)
    outputs = [_out(video_path, f"{stem}_seg.csv"),
               _out(video_path, f"{stem}_contour_radii.npy")]
    if dye_click is not None:
        outputs.append(_out(video_path, f"{stem}_track.csv"))
    return Task(
        name        = "SAM2 segmentation",
        fn          = _run_sam2_task,
        deps        = [],
        resource    = "gpu",
        inputs      = [video_path],
        outputs     = outputs,
        weight      = 4.0,
        task_kwargs = dict(
            video_path     = video_path,
            bell_click     = bell_click,
            stride         = stride,
            window_size    = window_size,
            save_n_masks   = save_n_masks,
            delete_frames  = delete_frames,
            dye_click      = dye_click,
            image_size     = image_size,
        ),
    )


# ── Task 2 — CoTracker dye tracking ──────────────────────────────────────────

def _run_cotracker_task(
    video_path:  Path,
    dye_click:   tuple[int, int],
    stride:      int,
    chunk_size:  int,
    progress_callback: Callable | None = None,
    cancel_event:      threading.Event | None = None,
) -> None:
    import cv2, torch
    from scripts.cotracker_test import (
        CoTrackerPredictor, track_chunked, write_csv, render_video,
    )

    stem      = _stem(video_path)
    track_csv = _out(video_path, f"{stem}_track.csv")
    track_vid = _out(video_path, f"{stem}_tracked.mp4")

    cap       = cv2.VideoCapture(str(video_path))
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps       = cap.get(cv2.CAP_PROP_FPS)

    if progress_callback:
        progress_callback(0, total_raw, "Loading CoTracker...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CoTrackerPredictor(checkpoint=str(COTRACKER_WEIGHTS)).to(device).eval()

    tracks, visible = track_chunked(
        model, cap, dye_click, total_raw,
        stride=stride, chunk_size=chunk_size,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    OUTPUTS_DIR.mkdir(exist_ok=True)
    write_csv(track_csv, tracks, visible, fps, stride)

    if not (cancel_event and cancel_event.is_set()):
        if progress_callback:
            progress_callback(total_raw, total_raw, "Rendering tracked video...")
        render_video(cap, tracks, visible, track_vid, fps, stride)

    cap.release()

    if progress_callback:
        progress_callback(total_raw, total_raw, "CoTracker complete")


def make_cotracker_task(
    video_path: Path,
    dye_click:  tuple[int, int],
    stride:     int = 4,
    chunk_size: int = 400,
) -> Task:
    stem = _stem(video_path)
    return Task(
        name        = "CoTracker tracking",
        fn          = _run_cotracker_task,
        deps        = [],               # independent of SAM2
        resource    = "gpu",
        inputs      = [video_path],
        outputs     = [_out(video_path, f"{stem}_track.csv")],
        weight      = 3.0,
        task_kwargs = dict(
            video_path = video_path,
            dye_click  = dye_click,
            stride     = stride,
            chunk_size = chunk_size,
        ),
    )


# ── Task 3 — Approach B Phase 1a (lab-frame margin diff) ─────────────────────

def _run_phase1a_task(
    video_path: Path,
    stride:     int,
    inner_frac: float,
    outer_frac: float,
    use_gpu:    bool = False,    # PERF: use GPU grid_sample instead of CPU warpPolar
    progress_callback: Callable | None = None,
    cancel_event:      threading.Event | None = None,
) -> None:
    from scripts.run_approach_b import load_seg

    stem    = _stem(video_path)
    seg_csv = _out(video_path, f"{stem}_seg.csv")
    lab_npy = _out(video_path, f"{stem}_margin_diff_lab.npy")
    seg = load_seg(seg_csv)

    if use_gpu:
        from scripts.run_approach_b import compute_margin_diff_lab_gpu
        lab = compute_margin_diff_lab_gpu(
            video_path, seg, stride, inner_frac, outer_frac,
            progress_callback=progress_callback, cancel_event=cancel_event,
        )
    else:
        from scripts.run_approach_b import compute_margin_diff_lab
        lab = compute_margin_diff_lab(
            video_path, seg, stride, inner_frac, outer_frac,
            progress_callback=progress_callback, cancel_event=cancel_event,
        )

    run_dir(video_path).mkdir(parents=True, exist_ok=True)
    np.save(str(lab_npy), lab)

    if progress_callback:
        progress_callback(len(lab), len(lab), "Phase 1a complete")


def make_phase1a_task(
    video_path: Path,
    stride:     int   = 4,
    inner_frac: float = 0.75,
    outer_frac: float = 1.05,
    use_gpu:    bool  = False,   # PERF: GPU-accelerated polar transform
) -> Task:
    stem = _stem(video_path)
    seg_csv = _out(video_path, f"{stem}_seg.csv")
    return Task(
        name        = "Margin diff (lab frame)",
        fn          = _run_phase1a_task,
        deps        = ["SAM2 segmentation"],   # needs seg.csv
        resource    = "gpu" if use_gpu else "cpu",
        inputs      = [video_path, seg_csv],
        outputs     = [_out(video_path, f"{stem}_margin_diff_lab.npy")],
        weight      = 1.5,
        task_kwargs = dict(
            video_path = video_path,
            stride     = stride,
            inner_frac = inner_frac,
            outer_frac = outer_frac,
            use_gpu    = use_gpu,
        ),
    )


# ── Task 4 — Phase 1b (body-frame rotation) ───────────────────────────────────

def _run_phase1b_task(
    video_path: Path,
    stride:     int,
    progress_callback: Callable | None = None,
    cancel_event:      threading.Event | None = None,
) -> None:
    from scripts.run_approach_b import (
        load_seg, load_dye, apply_body_frame_rotation,
    )

    stem      = _stem(video_path)
    seg_csv   = _out(video_path, f"{stem}_seg.csv")
    dye_csv   = _out(video_path, f"{stem}_track.csv")
    lab_npy   = _out(video_path, f"{stem}_margin_diff_lab.npy")
    body_npy  = _out(video_path, f"{stem}_margin_diff.npy")

    if progress_callback:
        progress_callback(0, 1, "Applying body-frame rotation...")

    seg      = load_seg(seg_csv)
    dye      = load_dye(dye_csv)
    lab_diff = np.load(str(lab_npy))

    frame_indices = np.arange(len(lab_diff)) * stride
    body_diff     = apply_body_frame_rotation(lab_diff, dye, seg, stride, frame_indices)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    np.save(str(body_npy), body_diff)

    if progress_callback:
        progress_callback(1, 1, "Body-frame rotation complete")


def make_phase1b_task(
    video_path:       Path,
    stride:           int  = 4,
    sam2_tracks_dye:  bool = False,   # PERF: track.csv comes from SAM2, not CoTracker
) -> Task:
    stem = _stem(video_path)
    return Task(
        name        = "Body-frame rotation",
        fn          = _run_phase1b_task,
        # When SAM2 tracks the dye internally, track.csv is produced by SAM2 —
        # depend on SAM2 instead of the (absent) CoTracker task.
        deps        = (["Margin diff (lab frame)", "SAM2 segmentation"]
                       if sam2_tracks_dye else
                       ["Margin diff (lab frame)", "CoTracker tracking"]),
        resource    = "cpu",
        inputs      = [
            _out(video_path, f"{stem}_margin_diff_lab.npy"),
            _out(video_path, f"{stem}_track.csv"),
            _out(video_path, f"{stem}_seg.csv"),
        ],
        outputs     = [_out(video_path, f"{stem}_margin_diff.npy")],
        weight      = 0.1,
        task_kwargs = dict(video_path=video_path, stride=stride),
    )


# ── Task 5 — Approach B analysis (pulse detection + initiation) ───────────────

def _run_analysis_task(
    video_path:   Path,
    calib_path:   Path,
    stride:       int,
    pre_window:   int,
    min_distance: float,
    prominence:   float,
    progress_callback: Callable | None = None,
    cancel_event:      threading.Event | None = None,
) -> None:
    import csv as _csv, json
    from scripts.run_approach_b import (
        load_seg, load_dye, load_calibration,
        detect_pulses, find_initiation_angle, match_rhopalium,
        MAX_ASSIGNMENT_DIST, BASELINE_W, summary_plot,
    )
    from scripts.run_approach_b import render_annotated_video

    stem       = _stem(video_path)
    seg_csv    = _out(video_path, f"{stem}_seg.csv")
    dye_csv    = _out(video_path, f"{stem}_track.csv")
    body_npy   = _out(video_path, f"{stem}_margin_diff.npy")
    init_csv   = _out(video_path, f"{stem}_initiation_b.csv")
    plot_out   = _out(video_path, f"{stem}_initiation_b_plot.png")
    vid_out    = _out(video_path, f"{stem}_initiation_b_annotated.mp4")

    import cv2
    cap     = cv2.VideoCapture(str(video_path))
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps_eff = fps_raw / stride

    seg           = load_seg(seg_csv)
    dye_track     = load_dye(dye_csv)
    calib         = json.loads(calib_path.read_text())
    margin_diff   = np.load(str(body_npy))
    n_frames      = len(margin_diff) + 1
    frame_indices = np.arange(n_frames) * stride   # length n_frames, matching standalone convention

    total_activity = margin_diff.sum(axis=1)
    peaks, props   = detect_pulses(total_activity, fps_eff, min_distance, prominence)

    if progress_callback:
        progress_callback(0, len(peaks), f"Analysing {len(peaks)} pulses...")

    results = []
    for pid, peak_idx in enumerate(peaks):
        if cancel_event and cancel_event.is_set():
            break

        peak_frame = int(frame_indices[min(peak_idx, len(frame_indices) - 1)])
        ts         = peak_frame / fps_raw
        pre_w      = min(pre_window,
                         (peaks[pid] - peaks[pid-1]) // 2 if pid > 0 else pre_window)

        init_angle, sig_val, confident = find_initiation_angle(
            margin_diff, peak_idx, pre_w, BASELINE_W)
        rhop_id, rhop_dist = match_rhopalium(init_angle, calib)
        sig_conf           = confident and rhop_dist <= MAX_ASSIGNMENT_DIST

        results.append({
            "peak_id":          pid,
            "peak_frame":       peak_frame,
            "timestamp_s":      round(ts, 4),
            "activity":         round(float(total_activity[peak_idx]), 2),
            "init_angle_deg":   init_angle,
            "rhopalium_id":     rhop_id,
            "angular_dist_deg": round(rhop_dist, 2),
            "signal_confident": int(sig_conf),
        })

        if progress_callback:
            progress_callback(pid + 1, len(peaks),
                              f"Pulse {pid}: R{rhop_id} dist={rhop_dist:.1f}°")

    # Save CSV
    OUTPUTS_DIR.mkdir(exist_ok=True)
    if results:
        with open(init_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    # Summary plot
    summary_plot(results, calib, total_activity, peaks, frame_indices, fps_raw, plot_out)

    # Annotated video — non-fatal: encoding failure does not fail the task
    if results and not (cancel_event and cancel_event.is_set()):
        if progress_callback:
            progress_callback(len(peaks), len(peaks), "Rendering annotated video...")
        try:
            render_annotated_video(results, calib, video_path, vid_out,
                                   seg, dye_track, fps_raw, stride)
        except Exception as _vid_err:
            print(f"[warn] Annotated video rendering failed (results still saved): {_vid_err}")

    if progress_callback:
        progress_callback(len(peaks), len(peaks), "Analysis complete")


def make_analysis_task(
    video_path:   Path,
    calib_path:   Path,
    stride:       int   = 4,
    pre_window:   int   = 30,
    min_distance: float = 0.42,
    prominence:   float = 0.05,
) -> Task:
    stem = _stem(video_path)
    return Task(
        name        = "Pulse initiation analysis",
        fn          = _run_analysis_task,
        deps        = ["Body-frame rotation"],
        resource    = "cpu",
        inputs      = [
            _out(video_path, f"{stem}_margin_diff.npy"),
            _out(video_path, f"{stem}_seg.csv"),
            _out(video_path, f"{stem}_track.csv"),
            calib_path,
        ],
        outputs     = [
            _out(video_path, f"{stem}_initiation_b.csv"),
            _out(video_path, f"{stem}_initiation_b_plot.png"),
        ],
        weight      = 0.5,
        task_kwargs = dict(
            video_path   = video_path,
            calib_path   = calib_path,
            stride       = stride,
            pre_window   = pre_window,
            min_distance = min_distance,
            prominence   = prominence,
        ),
    )
