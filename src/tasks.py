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


# ── Output directory (runtime override) ──────────────────────────────────────
#
# Two layers compose the output root:
#   base           — the user-chosen output directory (set_output_root), default
#                    OUTPUTS_DIR from config.py.
#   project subdir — the active project's name (set_project_subdir). When set,
#                    all outputs live under <base>/<project>/ so a project's
#                    recordings are grouped together and project-level summaries
#                    (videos.csv) sit alongside the per-video folders.
#
# With no project active the subdir is None and outputs go straight under base
# (the historical layout), so non-project / ad-hoc runs are unaffected.

_output_root_override: Path | None = None
_project_subdir:       str  | None = None


def _safe_name(name: str) -> str:
    """Sanitize a project name into a filesystem-safe folder name."""
    bad = r'\/:*?"<>|'
    return ("".join("_" if c in bad else c for c in name).strip()) or "Untitled"


def set_output_root(path: Path | None) -> None:
    """Override the base output directory at runtime (called by the UI)."""
    global _output_root_override
    _output_root_override = path


def set_project_subdir(name: str | None) -> None:
    """Set the active project name as an output subfolder (called by the UI).

    Pass None (or empty) to clear it — outputs then go straight under the base.
    """
    global _project_subdir
    _project_subdir = _safe_name(name) if name else None


def get_output_base() -> Path:
    """The base output directory, ignoring any project subfolder."""
    return _output_root_override if _output_root_override is not None else OUTPUTS_DIR


def get_output_root() -> Path:
    """The effective output root: ``<base>/<project>`` if a project is active,
    else just ``<base>``."""
    base = get_output_base()
    return base / _project_subdir if _project_subdir else base


# ── SAM2 / Hydra initialization ───────────────────────────────────────────────

def init_sam2_hydra() -> None:
    """Make SAM2's Hydra-based builders safe to call repeatedly in one process.

    ``build_sam2`` / ``build_sam2_video_predictor`` call ``hydra.compose()``,
    which asserts Hydra is initialized with SAM2's config module. That init runs
    exactly once, in ``sam2/__init__.py`` at first import — but the import is
    cached, so it does NOT re-run on later builds. A plain
    ``GlobalHydra.instance().clear()`` (the previous approach) therefore left
    Hydra *uninitialized* on the 2nd+ build in a process (e.g. re-marking the
    bell, or running the pipeline after a SAM2 preview), raising
    "GlobalHydra is not initialized".

    Clearing and then explicitly re-initializing guarantees a valid, fresh
    Hydra state before every build, regardless of import caching or whatever
    prior Hydra state exists.
    """
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_module
    GlobalHydra.instance().clear()
    initialize_config_module("sam2", version_base="1.2")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stem(video_path: Path) -> str:
    return video_path.stem


def run_dir(video_path: Path, output_root: Path | None = None) -> Path:
    """
    Return the per-recording output directory.
    Default: <OUTPUTS_DIR>/<video_stem>/  (overrideable via set_output_root)

    All task outputs live inside this folder so recordings never pollute
    each other's files.  The UI can display one folder per recording.
    """
    root = output_root or get_output_root()
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
    delete_frames: bool = False,     # kept for API compat; streaming never creates frames_dir
    dye_click:     tuple[int, int] | None = None,    # PERF: track dye inside SAM2
    image_size:    int | None = None,                # PERF: override internal ViT resolution
    sentinel_path: Path | None = None,               # written only on successful completion
    progress_callback: Callable | None = None,
    cancel_event:      threading.Event | None = None,
) -> None:
    import torch
    import cv2
    from scripts.run_sam2 import (
        run_sam2_streaming, mask_to_stats,
        N_CONTOUR_ANGLES,
    )
    stem     = _stem(video_path)
    mask_dir = _out(video_path, f"{stem}_masks")
    seg_csv  = _out(video_path, f"{stem}_seg.csv")
    cont_npy = _out(video_path, f"{stem}_contour_radii.npy")

    cap     = cv2.VideoCapture(str(video_path))
    fps_raw = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    n_strided = (total + stride - 1) // stride

    # Determine which frames to save as PNG masks
    if save_n_masks > 0:
        idxs         = np.linspace(0, n_strided - 1, save_n_masks, dtype=int)
        save_indices = set(idxs.tolist())
    else:
        save_indices = set()

    if progress_callback:
        progress_callback(0, n_strided, "Loading SAM2...")

    from sam2.build_sam import build_sam2_video_predictor
    from config import SAM2_CONFIG
    # Initialize Hydra for SAM2's compose()-based builder. Must clear AND
    # re-initialize (not just clear) so this works on the 2nd+ build in a
    # process — e.g. after a SAM2 preview, or another queued video — since
    # sam2's one-time import init does not re-run. See init_sam2_hydra().
    init_sam2_hydra()

    device    = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    overrides = [f"++model.image_size={image_size}"] if image_size else []
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG, str(SAM2_WEIGHTS), device=device,
        hydra_overrides_extra=overrides if overrides else None,
    )

    seg_stats, contour_arr, dye_track_list = run_sam2_streaming(
        predictor, video_path, stride, bell_click, mask_dir,
        n_strided, window_size, save_indices,
        dye_click=dye_click,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    # Do not write partial outputs if the run was cancelled
    if cancel_event and cancel_event.is_set():
        return

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

    # Sentinel written last — scheduler skips this task only when it exists
    if sentinel_path is not None:
        sentinel_path.touch()

    if progress_callback:
        progress_callback(n_strided, n_strided, "SAM2 complete")


def _migrate_sentinel(sentinel: Path, data_outputs: list[Path], video_path: Path) -> None:
    """Write sentinel for a pre-existing complete run that predates the sentinel system."""
    if sentinel.exists():
        return
    try:
        video_mtime = video_path.stat().st_mtime
        if all(p.exists() and p.stat().st_mtime >= video_mtime for p in data_outputs):
            sentinel.touch()
    except OSError:
        pass


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
    stem     = _stem(video_path)
    sentinel = _out(video_path, f"{stem}_sam2.complete")
    data_out = [_out(video_path, f"{stem}_seg.csv"),
                _out(video_path, f"{stem}_contour_radii.npy")]
    if dye_click is not None:
        data_out.append(_out(video_path, f"{stem}_track.csv"))
    _migrate_sentinel(sentinel, data_out, video_path)
    outputs  = data_out + [sentinel]
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
            sentinel_path  = sentinel,
        ),
    )


# ── Task 2 — CoTracker dye tracking ──────────────────────────────────────────

def _run_cotracker_task(
    video_path:  Path,
    dye_click:   tuple[int, int],
    stride:      int,
    chunk_size:  int,
    sentinel_path: Path | None = None,               # written only on successful completion
    render_annotated_s: float = 0.0,   # 0 = skip render; >0 = render first N seconds
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

    # Do not write partial outputs if the run was cancelled
    if cancel_event and cancel_event.is_set():
        cap.release()
        return

    get_output_root().mkdir(parents=True, exist_ok=True)
    write_csv(track_csv, tracks, visible, fps, stride)

    # Sentinel written immediately after CSV — before optional render
    if sentinel_path is not None:
        sentinel_path.touch()

    if render_annotated_s > 0:
        if progress_callback:
            progress_callback(total_raw, total_raw, "Rendering tracked video...")
        max_strided = int(render_annotated_s * fps / stride)
        render_video(cap, tracks[:max_strided], visible[:max_strided],
                     track_vid, fps, stride)

    cap.release()

    if progress_callback:
        progress_callback(total_raw, total_raw, "CoTracker complete")


def make_cotracker_task(
    video_path: Path,
    dye_click:  tuple[int, int],
    stride:     int   = 4,
    chunk_size: int   = 400,
    render_annotated_s: float = 0.0,   # 0 = skip render; >0 = first N seconds only
) -> Task:
    stem     = _stem(video_path)
    sentinel = _out(video_path, f"{stem}_cotrack.complete")
    track_csv = _out(video_path, f"{stem}_track.csv")
    _migrate_sentinel(sentinel, [track_csv], video_path)
    return Task(
        name        = "CoTracker tracking",
        fn          = _run_cotracker_task,
        deps        = [],               # independent of SAM2
        resource    = "gpu",
        inputs      = [video_path],
        outputs     = [track_csv, sentinel],
        weight      = 3.0,
        task_kwargs = dict(
            video_path          = video_path,
            dye_click           = dye_click,
            stride              = stride,
            chunk_size          = chunk_size,
            sentinel_path       = sentinel,
            render_annotated_s  = render_annotated_s,
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

    get_output_root().mkdir(parents=True, exist_ok=True)
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
    max_annotated_s: float = 60.0,   # 0 = skip annotated video; >0 = first N seconds
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
    get_output_root().mkdir(parents=True, exist_ok=True)
    if results:
        with open(init_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    # Summary plot
    summary_plot(results, calib, total_activity, peaks, frame_indices, fps_raw, plot_out)

    # Annotated video — non-fatal: encoding failure does not fail the task
    if results and max_annotated_s > 0 and not (cancel_event and cancel_event.is_set()):
        if progress_callback:
            progress_callback(len(peaks), len(peaks), "Rendering annotated video...")
        max_raw_frames = int(max_annotated_s * fps_raw)
        try:
            render_annotated_video(results, calib, video_path, vid_out,
                                   seg, dye_track, fps_raw, stride,
                                   max_raw_frames=max_raw_frames)
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
    max_annotated_s: float = 60.0,   # 0 = skip annotated video; >0 = first N seconds
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
            video_path       = video_path,
            calib_path       = calib_path,
            stride           = stride,
            pre_window       = pre_window,
            min_distance     = min_distance,
            prominence       = prominence,
            max_annotated_s  = max_annotated_s,
        ),
    )
