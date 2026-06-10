# Pipeline Architecture

The pipeline is a five-stage DAG. Stages 1 and 2 can run in parallel; stages 3–5 are sequential but all depend on both earlier stages.

```text
Video
  │
  ├─── Stage 1: SAM2 bell segmentation ─────────────────────────────┐
  │        Per-frame binary mask + centroid + radius                 │
  │                                                                  ▼
  ├─── Stage 2: CoTracker dye tracking ─────────────────────────► Stage 3: Margin diff (lab frame)
  │        Per-frame dye (x, y) position                            │
  │                                                                  ▼
  └─── [One-time] Calibration ─────────────────────────────────► Stage 4: Body-frame rotation
           Body-frame angles for all 16 rhopalia                    │
                                                                     ▼
                                                                  Stage 5: Pulse initiation
                                                                     │
                                                                  initiation_b.csv
                                                                  initiation_b_plot.png
                                                                  annotated video
```

---

## Stage 1 — Bell segmentation (SAM2)

**Input:** video, one click on the bell body in frame 0

**Model:** SAM2 video predictor (`sam2.1_hiera_tiny.pt` by default; see [04_performance.md](04_performance.md) for model comparison)

**What it does:**
- User clicks the bell centre in frame 0; this is the prompt for SAM2.
- SAM2 propagates the mask through the entire video using its memory bank, handling deformation and partial occlusion.
- Processes in windows of ~200 frames (memory-bounded); windows overlap to avoid boundary artifacts.
- After all frames are processed, JPEG frames are deleted to recover disk space.

**Outputs:**
- `<stem>_seg.csv`: per-frame `(cx, cy, radius)` — bell centroid and approximate radius derived from mask area.
- `<stem>_contour_radii.npy`: shape `(n_frames, 360)` — bell boundary radius at each 1° angle (polar representation). Used by Approach B for the margin strip.

**Recommended stride:** 1 (every frame). Stride=4 saves time but introduces linear interpolation of contour radii between sampled frames, which can smear the initiation signal.

**Key files:**
- `scripts/run_sam2.py` — standalone CLI
- `src/tasks.py: task_sam2()` — task wrapper for the parallel scheduler
- `config.py: SAM2_WEIGHTS, SAM2_CONFIG` — model selection

---

## Stage 2 — Dye mark tracking (CoTracker)

**Input:** video, one click on the dye mark in frame 0

**Model:** CoTracker3 offline (`scaled_offline.pth`), a transformer-based video point tracker from Meta AI.

**What it does:**
- User clicks the dye mark in frame 0.
- CoTracker processes the video in overlapping chunks, using temporal context from many frames to maintain track even when the dye is faint.
- Runs in bfloat16 with `torch.compile` for ~2× speedup.

**Outputs:**
- `<stem>_track.csv`: per-frame `(frame_idx, x, y, visibility)` — dye position at every tracked frame.

**Recommended stride:** 4 (every 4th frame = 30 fps effective). The dye moves slowly relative to the frame rate; stride=4 is sufficient for accurate angle estimation without the large memory cost of stride=1.

**Key files:**
- `scripts/cotracker_test.py` — standalone CLI
- `src/tasks.py: task_cotracker()` — task wrapper

**License note:** CoTracker uses CC-BY-NC 4.0. For commercial use, replace with TAPIR (Apache 2.0) — API-compatible.

---

## Calibration (one-time per animal)

**Input:** high-resolution still photo with the dye mark visible

**What it does:**
- User clicks: (1) bell centre, (2) dye mark (defines 0°), (3) each rhopalium in any order.
- Computes body-frame angle for each rhopalium: `phi_body = atan2(dy, dx) - atan2(dye_dy, dye_dx)`, wrapped to (−180°, +180°].
- The dye mark defines 0°; angles increase counter-clockwise when viewed from above.

**Outputs:**
- `calibration/<name>.json`: `{n_rhopalia, centre_px, dye_px, rhopalia: [{id, angle_deg, px}]}`
- `calibration/<name>_annotated.png`: labelled verification image.

**16 rhopalia** are expected for Cassiopea, but the count is not enforced. Clicking fewer is allowed (the assignment still works; angular resolution is reduced).

**Key files:**
- `scripts/calibrate_rhopalia.py` — standalone CLI (OpenCV window, click to annotate)
- `src/calibration_core.py` — shared math (`body_angle`, `build_calibration`, `write_calibration_json`)
- `ui/calibration.py` — napari UI tab that calls the same core functions

---

## Stage 3 — Margin difference (lab frame)

**Input:** video + `_seg.csv` + `_contour_radii.npy`

**What it does:**
- For each consecutive frame pair, applies `cv2.warpPolar` centred on the bell centroid.
- Extracts an annular strip from the polar image (radii = `inner_frac` to `outer_frac` × bell radius).
- Computes `margin_diff[θ, t] = mean(|strip(t+1) − strip(t)|)` across the strip radial axis.
- Result is a 2D array of shape `(360, n_frames)` in **lab frame** (angle 0 = fixed direction in image).

**Outputs:**
- `<stem>_margin_diff_lab.npy` — deleted after Stage 4 to save disk space.

**Key parameters:**
- `inner_frac` (default 0.75): inner radius as fraction of bell radius. Controls how much of the bell interior is excluded.
- `outer_frac` (default 1.05): outer radius. Values > 1.0 include tissue just outside the bell boundary, which can show the expansion wave before the bell itself moves.

---

## Stage 4 — Body-frame rotation

**Input:** `_margin_diff_lab.npy` + `_track.csv` (dye positions)

**What it does:**
- For each frame, computes `phi_dye(t)` = angle from bell centroid to dye position (in lab frame).
- Rotates the lab-frame margin diff by `−phi_dye(t)` so that angle 0 always points toward the dye mark.
- The result is a body-frame signal where the same body-frame angle maps to the same physical location on the bell in every frame, regardless of animal rotation.

**Outputs:**
- `<stem>_margin_diff.npy` — shape `(360, n_frames)`, cached. All downstream analysis uses this.

---

## Stage 5 — Pulse initiation analysis

**Input:** `_margin_diff.npy` + calibration JSON

**What it does:**

**5a. Pulse detection:**
- Sums `_margin_diff` across all angles → 1D total activity signal.
- Detects peaks with `scipy.signal.find_peaks` using `prominence` threshold.

**5b. Initiation site per pulse:**
- For each peak, scans backward through the last `pre_window` frames.
- Finds the earliest frame where any single-angle activity exceeds 25% of the peak-frame maximum.
- The angle of this first elevated signal is the initiation site.

**5c. Rhopalium assignment:**
- Converts initiation angle from body-frame degrees to rhopalium ID using the calibration file.
- Assigns the nearest rhopalium; stores angular distance.
- `signal_confident = 1` if angular distance < 11.25° (half of 360°/16 rhopalium spacing).

**Outputs:**
- `<stem>_initiation_b.csv`
- `<stem>_initiation_b_plot.png`
- `<stem>_initiation_b_annotated.mp4`
- `<stem>_spacetime_pulse_b/pulse_NNN.png` — per-pulse angle × time heatmap

**Key files:**
- `scripts/run_approach_b.py` — standalone CLI
- `src/tasks.py: task_approach_b()` — task wrapper

---

## Parallel execution

`src/scheduler.py` runs stages 1, 2, and 3 (Phase 1a of Approach B) concurrently, gated by a `GpuGate` semaphore to prevent VRAM overflow. On an 8 GB GPU: SAM2 and CoTracker each use ~3 GB; Approach B Phase 1 is CPU-bound, so all three can overlap.

`src/pipeline.py: run_pipeline()` is the high-level entry point used by both the CLI (`scripts/run_pipeline.py`) and the UI (`ui/workers.py`).
