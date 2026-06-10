# Cassiopea Pipeline — UI Guide

A napari-based interface for analysing *Cassiopea* jellyfish behaviour from top-view video recordings.  
Launch with: `venv\Scripts\python scripts\run_ui.py`

---

## Overview

The UI has two tabs:

| Tab | Purpose |
|-----|---------|
| **Calibrate** | One-time annotation of rhopalia positions from a still photo |
| **Process** | Run the full analysis pipeline on a video recording |

Work through Calibrate first for any new animal, then use Process for each video.

---

## Workflow A — Calibrate

Use a high-resolution still photo of the jellyfish taken with the dye mark clearly visible.  
You only need to do this **once per animal**. The result is a `calibration/<name>.json` file that can be reused for all videos of that animal.

### Steps

**1. Load image**  
Click **Browse…** and select your `.jpg` / `.png` photo.  
The image appears in the viewer.

**2. Enter animal name**  
Type an identifier in the **Animal name** field (e.g. `Ethel_Cain`).  
The calibration file will be saved as `calibration/<name>.json`.  
You will be warned if a file with that name already exists.

**3. Click the bell centre**  
The instruction banner reads *Step 2 of 4 — Click the bell CENTRE*.  
Single-click the centre of mass of the jellyfish bell in the viewer.  
The **Last click** coordinate updates within 0.2 s.  
Press **Next →** to confirm and advance.

**4. Click the dye mark**  
The banner reads *Step 3 of 4 — Click the DYE MARK*.  
Single-click the dye spot on the bell surface.  
This point defines the **phi = 0°** reference direction (body-frame origin).  
Press **Next →** to confirm and advance.

**5. Click each rhopalium**  
The banner reads *Step 4 of 4 — Click each RHOPALIUM*.  
Click each rhopalium in any order around the bell margin.  
After each click the **Rhopalium angles** table updates with the body-frame angle (degrees from the dye direction) and pixel coordinates.  
Use **Remove last rhopalium** (or just re-click) to correct mistakes.  
Cassiopea has 16 rhopalia, but the count is not enforced.

**6. Save**  
Press **Save calibration**.  
Two files are written to `calibration/`:
- `<name>.json` — body-frame angles, used by the pipeline
- `<name>_annotated.png` — annotated verification diagram

> **Tip**: zoom into the viewer with the scroll wheel before clicking to place points accurately. Pan by holding Space and dragging, or by middle-click drag.

---

## Workflow B — Process

### Steps

**1. Select a video folder**  
Click **Browse…** next to *Video folder*.  
All `.mp4 / .avi / .mov` files in that folder are listed with thumbnail previews.

**2. Select a video**  
Click any video in the list.  
Its first frame loads into the viewer. Any previously loaded calibration image is automatically closed.

**3. Select calibration**  
Choose the animal's calibration from the **Calibration** dropdown.  
Click **Refresh** if you just created a new one in the Calibrate tab.  
If no calibrations exist, a warning directs you to Workflow A.

**4. Mark the bell**  
Click **Mark bell**, then click the jellyfish bell in the viewer.  
A SAM2 segmentation preview (semi-transparent mask) appears within a few seconds to confirm the bell was detected correctly.  
If the mask looks wrong, click **Mark bell** again and click a better position.

**5. Mark the dye**  
Click **Mark dye**, then click the dye mark in the viewer.  
The green point and its coordinates appear in the sidebar.

**6. Set parameters** *(optional — defaults usually work)*

| Parameter | Default | Meaning |
|-----------|---------|---------|
| SAM2 stride | 4 | Process every Nth frame for segmentation (4 = 30 fps effective at 120 fps) |
| CoTracker stride | 8 | Dye tracking frame interval |
| SAM2 model | tiny | SAM2 model size (tiny is fastest; larger gives better masks on difficult footage) |
| Pre-window (frames) | 30 | Frames before a pulse peak to search for initiation |
| Inner frac | 0.75 | Inner radius of the polar ring (fraction of bell radius) |
| Outer frac | 1.05 | Outer radius of the polar ring |
| Prominence | 0.08 | Minimum peak prominence for pulse detection |

**7. Run pipeline**  
Click **Run pipeline**.  
Five progress bars show each stage:  
`SAM2 segmentation → CoTracker tracking → Margin diff → Body-frame rotation → Pulse initiation analysis`  
Stages whose outputs are already cached show *skipped* and complete instantly.

> **Force recompute**: tick this checkbox to delete all cached outputs and re-run every stage from scratch. A confirmation dialog appears before any files are deleted.

**8. View results**  
When the pipeline finishes:
- The viewer loads the bell centroids (yellow dots) and dye trajectory as layers
- The sidebar shows the **initiation table** with one row per detected pulse
- A static summary plot appears below the table

### Output files

All outputs are written to `outputs/<video_stem>/`:

| File | Contents |
|------|---------|
| `<stem>_seg.csv` | Per-frame bell centroid (cx, cy) and radius |
| `<stem>_track.csv` | Per-frame dye mark position (x, y) |
| `<stem>_initiation_b.csv` | Per-pulse: peak frame, timestamp, activity, init angle, rhopalium ID, angular distance |
| `<stem>_initiation_b_plot.png` | Summary figure (signal trace + polar initiation map) |
| `<stem>_initiation_b_annotated.mp4` | Annotated video with centroid, dye axis, and initiation overlay |
| `<stem>_run_log.json` | Config and timing for every run (appends, never overwrites) |

---

## Tips and troubleshooting

**The bell click doesn't place a point**  
Make sure the correct layer is selected in napari's layer list. After clicking **Mark bell** or **Mark dye**, the appropriate layer is activated automatically — just click in the canvas.

**Pipeline stage shows "failed" in red**  
Check the log text area for the error message. Common causes:
- Video file cannot be read (codec not supported by OpenCV)
- Calibration JSON not found at the selected path
- GPU out of memory — reduce SAM2 stride or switch to a smaller model

**SAM2 mask preview looks wrong (wrong object segmented)**  
Re-click **Mark bell** and place the click closer to the centre of the bell disc, away from tentacles and chamber walls.

**Fewer pulses detected than expected**  
Lower the **Prominence** parameter (try 0.03–0.05). If the signal looks noisy, increase **Outer frac** slightly to capture more of the bell margin.

**Calibration file not appearing in the dropdown**  
Click **Refresh** — the dropdown re-scans the `calibration/` folder each time.

**Results from a previous run still appear**  
Tick **Force recompute** before clicking Run pipeline to delete cached outputs and re-analyse from scratch.

---

## Keyboard shortcuts (napari viewer)

| Key / action | Effect |
|--------------|--------|
| Scroll wheel | Zoom in / out |
| Space + drag | Pan |
| `[` / `]` | Decrease / increase point size |
| Ctrl+Z | Undo last action in active layer |
| F11 | Toggle full-screen |
