# Codebase Map

Quick reference for which file to read or edit for each concern.

---

## Entry points

| Task | Entry point |
| --- | --- |
| Launch GUI | `scripts/run_ui.py` → `ui/app.py: main()` |
| Full pipeline (CLI) | `scripts/run_pipeline.py` |
| SAM2 only | `scripts/run_sam2.py` |
| CoTracker only | `scripts/cotracker_test.py` |
| Approach B only | `scripts/run_approach_b.py` |
| Calibration (CLI) | `scripts/calibrate_rhopalia.py` |
| Validation video | `scripts/validate_tracking.py` |

---

## src/ — core library

### `src/pipeline.py`

`run_pipeline(video_path, bell_click, dye_click, calib_path, *, stride, cotracker_stride, ...)` — assembles all tasks into a DAG and executes via the scheduler. Returns a `PipelineResult` object. Both the CLI and the UI call this function.

### `src/scheduler.py`

DAG task runner. Key classes:

- `Task(name, fn, deps, gpu_required)` — a node in the DAG.
- `Scheduler` — runs tasks whose dependencies are complete; respects GPU gate; calls `progress_callback(ProgressEvent)` on each update.
- `ProgressEvent(task_name, status, fraction, message)` — progress update object forwarded to UI.
- `TaskStatus` enum: `PENDING | RUNNING | DONE | FAILED | SKIPPED`.

To add a new stage: create a `Task` with the right `deps`, add it to the list in `pipeline.py`.

### `src/tasks.py`

One factory function per pipeline stage. Each function returns a `Task` object with the stage's logic as a closure. Stages:

- `task_sam2(video_path, bell_click, stride, ...)` — runs SAM2 segmentation.
- `task_cotracker(video_path, dye_click, stride, ...)` — runs CoTracker tracking.
- `task_margin_diff(video_path, seg_csv, contour_npy, ...)` — computes lab-frame margin diff.
- `task_bodyframe(margin_diff_lab, track_csv, ...)` — rotates to body frame.
- `task_approach_b(margin_diff_npy, calib_path, ...)` — pulse detection + initiation assignment + rendering.

To change algorithm parameters: edit the task factory and update the `PipelineParams` dataclass in `ui/parameters.py` and the argparse defaults in `scripts/run_pipeline.py`.

### `src/calibration_core.py`

Shared calibration math used by both the CLI (`scripts/calibrate_rhopalia.py`) and the UI (`ui/calibration.py`). Key exports:

- `phi_deg(centre, point)` — angle in degrees from centre to point (lab frame, 0° = right).
- `body_angle(centre, dye, rhopalium)` — body-frame angle (0° = toward dye). Range (−180°, +180°].
- `build_calibration(centre, dye, rhopalia)` — returns calibration dict.
- `write_calibration_json(calib, path, ...)` — writes JSON + optional annotated image.
- `save_annotated_image(img_bgr, calib, path)` — draws calibration overlay on image.
- `N_RHOPALIA_EXPECTED = 16`.

### `src/resources.py`

GPU detection and VRAM estimation. Key exports:

- `HARDWARE` — named tuple with `gpu_name`, `vram_gb`, `has_gpu`.
- `GpuGate` — asyncio semaphore that limits how many GPU tasks run simultaneously.

### `src/scheduler.py` — `ProgressEvent`

```python
@dataclass
class ProgressEvent:
    task_name: str
    status: TaskStatus
    fraction: float        # 0.0–1.0 within this task
    overall_fraction: float
    message: str = ""
```

---

## scripts/ — CLI entry points

Each script is self-contained: it parses arguments, collects any required user input (click windows), and calls the relevant `src/` function. Scripts are thin wrappers — the logic lives in `src/`.

### `scripts/run_pipeline.py`

Full pipeline. Key sections:

- Argument parsing (lines ~270–295): all pipeline parameters with defaults.
- Click collection (lines ~320–360): opens OpenCV windows for bell and dye clicks.
- Progress printing (lines ~60–130): ANSI terminal progress table.
- Calls `src/pipeline.py: run_pipeline()`.

### `scripts/run_approach_b.py`

Approach B standalone. Also contains:

- `_probe_nvenc()` — functional test (encodes a 16×16 frame) to detect working NVENC.
- `_FFmpegWriter` — streaming video writer that falls back to CPU encoding if NVENC fails.
- `render_annotated_video()` — overlays centroid, dye axis, and initiation markers on each frame.

---

## ui/ — napari graphical interface

### `ui/app.py`

Creates the napari viewer, hides the "layer controls" and "layer list" dock panels (to reduce Photoshop-like complexity), and mounts the `CassiopeaWidget` on the right panel.

### `ui/widget.py`

`CassiopeaWidget(QWidget)` — two-tab container. Lazily imports `CalibrationTab` and `ProcessingTab` to avoid import-time circular dependencies.

### `ui/calibration.py`

`CalibrationTab(QWidget)` — four-stage annotation workflow:

1. Load image → viewer.
2. Click bell centre (yellow point).
3. Click dye mark (green point).
4. Click each rhopalium (red points, live angle table).
5. Save JSON + annotated PNG.

**Important napari 0.7.0 quirk:** `layer.events.data` does not fire when points are added interactively. Layer state is polled via `QTimer` at 150 ms instead.

### `ui/processing.py`

`ProcessingTab(QWidget)` — video processing workflow:

1. Browse video folder → thumbnail list.
2. Select video → loads first frame.
3. Select calibration JSON.
4. Mark bell click (SAM2 preview on click).
5. Mark dye click.
6. Set parameters.
7. Run pipeline (background thread via `workers.py`).
8. Load results into viewer layers + sidebar table.

### `ui/workers.py`

`run_pipeline_worker` — a `@thread_worker` (napari superqt) that calls `src/pipeline.py: run_pipeline()` in a background thread and yields `ProgressEvent` objects to the main thread. `ProgressRelay` bridges the scheduler's callback interface to the generator-based worker interface.

### `ui/parameters.py`

`PipelineParams` dataclass — canonical parameter set for the UI. Mirrors the argparse defaults in `scripts/run_pipeline.py`.

### `ui/thumbnails.py`

`get_thumbnail(video_path)` — returns a 120×90 RGB numpy array, cached in `<video_dir>/.thumbnails/<stem>.png`. `read_first_frame(video_path)` — returns the full-resolution first frame.

---

## calibration/ — persistent annotation files

`<animal>.json` and `<animal>_annotated.png` for each calibrated animal. These are **tracked in git** so calibration is not lost when outputs are cleaned.

JSON format:

```json
{
  "n_rhopalia": 16,
  "centre_px": [320, 256],
  "dye_px": [380, 200],
  "rhopalia": [
    {"id": 0, "angle_deg": -45.2, "px": [410, 180]},
    ...
  ]
}
```

---

## config.py — path configuration

Edit `VIDEO_DIR` to point at your recordings. All other paths are derived from the project root. `FPS = 120` should match your actual recording frame rate.

---

## Key data formats

| File | Shape / format | Notes |
| --- | --- | --- |
| `_seg.csv` | `frame, cx, cy, radius` | Bell centroid and radius per frame |
| `_track.csv` | `frame_idx, x, y, visibility` | Dye position at tracked frames |
| `_contour_radii.npy` | `(n_frames, 360)` float32 | Bell boundary radius per degree |
| `_margin_diff_lab.npy` | `(360, n_frames)` float32 | Lab-frame margin activity |
| `_margin_diff.npy` | `(360, n_frames)` float32 | Body-frame margin activity (cached) |
| `_initiation_b.csv` | one row per pulse | See README for column descriptions |
