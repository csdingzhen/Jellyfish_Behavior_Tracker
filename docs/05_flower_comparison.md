# FLOWER.py — Comparison with the Main Pipeline

FLOWER.py is an independent script written by the lab mentor for detecting pulse initiation sites. It was developed in parallel with this pipeline and reaches similar conclusions via a different technical route. This document summarises how it works, how it compares, and how the two could be combined.

---

## What FLOWER.py does

FLOWER is a two-stage script. It does **not** use SAM2, CoTracker, or any deep learning models — it is a pure computer-vision approach based on frame differencing and optical flow.

### Stage 1 — Pulse anchor detection

**Goal:** find the frame index of each contraction peak.

**Method:**
1. Read video at a downsampled rate (effective ~30 fps via `DETECT_FPS` parameter).
2. For each frame pair, compute a frame-difference image (gaussian blur → morphological opening → threshold at a percentile → count changed-pixel area).
3. The "contraction score" = negative of this changed-pixel area (so contractions appear as peaks).
4. Apply a rolling min-max normalisation over a sliding window.
5. Use `scipy.signal.find_peaks` with `ANCHOR_NORM_THR ≈ 0.55` to detect normalised peaks.

**Outputs:** `stage1_diff_trace.csv`, `pulse_anchors.csv` (one row per pulse, with frame index and timestamp).

### Stage 2 — Initiation site detection (per pulse)

**Goal:** find the spatial origin of the contraction wave within a window before each pulse peak.

**Method:**
1. For each anchor, extract a window of frames starting ~0.35 s before the peak and ending ~0.10 s before the peak.
2. Build a "DIFF-SUM ROI": accumulate all pixels that change frequently across the window → morphological cleanup → keep largest connected component. This is a cheap, data-driven surrogate for the SAM2 bell mask.
3. Compute Farneback dense optical flow between frames spaced `lag_flow` apart within the ROI.
4. For each pixel, compute the "inward flow" = dot product of the flow vector with the unit vector pointing **toward the DIFF-SUM centroid** (not toward the geometric bell centre). Positive inward flow means tissue is moving toward the centroid.
5. Initiation detection: find the first frame(s) where ≥ `EARLY_N_PIXELS` pixels have inward flow ≥ `ABS_INWARD_THR` (1.0 px/frame).
6. Use connected components around the strongest pixel to find a precise `(init_x, init_y)` location.

**Outputs:** `pulse_inits_ABS_THR.csv` with `init_x, init_y, peak_x, peak_y` in full-frame pixel coordinates.

---

## Similarities to the main pipeline

| Aspect | FLOWER.py | Main pipeline |
| --- | --- | --- |
| Pulse timing | Frame-difference area, `find_peaks` | Polar margin diff, `find_peaks` |
| Initiation detection | Inward optical flow before peak | Earliest elevated angle in margin diff |
| Pre-peak search window | ~0.25 s before peak | `pre_window` frames before peak |
| ROI for analysis | DIFF-SUM hotspot mask | SAM2 bell mask + polar strip |
| Contraction signal | Frame difference | Frame difference on polar strip |

Both approaches detect the pulse timing using a frame-difference-based activity signal, scan backward before the peak to find initiation, and use some form of optical motion signal. The scientific question and the algorithmic structure are the same.

---

## Key differences

### 1. No body-frame reference

FLOWER outputs `(init_x, init_y)` in raw pixel coordinates. Without a body-frame angle, you cannot:
- Assign a rhopalium ID to the initiation site.
- Compare initiation locations across pulses where the animal has rotated.
- Aggregate statistics across recordings.

The main pipeline's dye-anchored body frame solves this problem. FLOWER is a "pixel-coordinate" approach; the main pipeline is an "angle-in-body-frame" approach.

### 2. No bell segmentation

FLOWER uses the DIFF-SUM ROI as a cheap surrogate for the bell mask. This is clever but fragile: if chamber walls or tentacles produce large frame differences, they will dominate the ROI and produce spurious initiation detections. SAM2's semantic mask is more reliable in recordings with high tentacle activity or chamber reflections.

### 3. Inward flow toward ROI centroid vs polar strip margin diff

FLOWER measures radial inward flow toward the diff-sum centroid. The main pipeline measures frame difference on the polar margin strip. These capture slightly different signals:

- FLOWER's inward flow is better at detecting **which direction** tissue is moving.
- The main pipeline's margin diff is better at detecting **when** the margin first deforms (because the outer fringe beyond the bell edge shows the expansion wave before the bell contracts inward).

### 4. No deep learning dependency

FLOWER has no model weights to download and no GPU requirement. It runs on any machine with OpenCV and scipy. This is an important practical advantage for shared machines or environments where PyTorch installation is difficult.

### 5. Batch/parallel processing

FLOWER is written to process whole directories with `multiprocessing`. The main pipeline processes one video at a time (though the UI makes batch processing easy via the folder browser).

---

## Combination strategies

### Option 1: Use FLOWER's Stage 1 to replace Approach B Phase 1a (pulse timing only)

FLOWER's frame-difference anchor detection is simpler and faster than the polar margin diff approach for the sole purpose of finding pulse peak frames. If pulse timing accuracy is equivalent, FLOWER Stage 1 could replace the full margin diff computation for a ~10× speedup on pulse detection.

**Integration point:** `src/tasks.py: task_approach_b()`, Phase 1a.

### Option 2: Assign rhopalium IDs to FLOWER's detected sites

FLOWER gives `(init_x, init_y)` in pixel coordinates. Given the main pipeline's per-frame outputs (`_seg.csv` for centroid, `_track.csv` for dye position), you can convert any pixel coordinate to a body-frame angle:

```python
phi_dye = phi_deg(centroid, dye_position)
phi_init = phi_deg(centroid, (init_x, init_y))
body_frame_angle = (phi_init - phi_dye + 180) % 360 - 180
```

Then look up the nearest rhopalium in the calibration JSON. This gives FLOWER's detections the same rhopalium-ID assignment as the main pipeline, enabling cross-validation.

### Option 3: Cross-validation

Run both FLOWER and the main pipeline on the same recordings. Compare:
- Detected pulse times (should agree closely).
- Initiation angles (should agree within ~20° if both are working correctly).

Disagreement flags either a bug or a recording quality issue (e.g., poor dye contrast → main pipeline loses the body-frame reference; heavy tentacle motion → FLOWER's DIFF-SUM ROI is wrong).

### Option 4: Hybrid pipeline for speed

Use FLOWER's Stage 1 as a fast pre-filter: detect all pulse anchor frames cheaply. Then run the expensive SAM2 + CoTracker only on windows around confirmed pulse events (±2 s). For a 10-minute recording with 20 pulses, this could reduce processing time by ~80%.

### Option 5: FLOWER's DIFF-SUM ROI as automatic SAM2 seed

The DIFF-SUM ROI identifies the most active region of the frame — which is where the bell is. This ROI could be used to automatically compute a bell-centre click for SAM2, removing the need for manual annotation in the processing workflow.

---

## Summary assessment

FLOWER is accurate and elegant for its purpose (initiation pixel coordinates), but it cannot answer the scientific question in body-frame terms without additional information from the main pipeline. The main pipeline provides body-frame assignment but requires SAM2 and CoTracker infrastructure.

The two approaches are complementary. The most valuable near-term integration is **Option 2** (assigning rhopalium IDs to FLOWER's outputs) — it requires no changes to either script, just a post-processing step that uses both output files.
