# Cassiopea Behavior Analysis Pipeline

Automated video analysis pipeline for *Cassiopea* (upside-down jellyfish) recordings. Given a top-view video of the animal with a small dye mark applied to its bell, this pipeline extracts:

- **Bell orientation over time** — which way the animal is facing at each moment
- **Contraction timing** — when the bell pulses and how strongly
- **Pulse initiation site** — which rhopalium (marginal neurocluster) triggered each contraction

---

## Background

*Cassiopea* has 16 rhopalia distributed around its bell margin. These act as independent pacemakers — any one of them can trigger a contraction wave that spreads across the whole bell. A key scientific question is: which rhopalium fires most often, and do they follow a pattern?

Because the animal has no obvious anatomical landmark for orientation (unlike *Aurelia*, which has a visible gonad cross), we apply a small **dye mark** to the bell surface. This mark serves as a reference point from which all angular measurements are made.

---

## How it works

```text
Your video
    │
    ├─► Track the dye mark through every frame            (CoTracker)
    │       → bell orientation at each moment
    │
    ├─► Outline the jellyfish bell in every frame         (SAM2)
    │       → bell centre, size, and boundary
    │       → contour radii r(θ, t): bell-edge position per angle
    │
    ├─► [One-time] From a hi-res photo, click each rhopalium
    │       → fixed body-frame angles saved to calibration file
    │
    └─► For each frame pair, measure optical change at the bell margin
            → decomposed by angle (polar unwrap)
            → peaks in total margin activity = contraction events
            → first angle to show elevated activity = initiation site
            → match to nearest rhopalium from calibration
```

---

## How the initiation detection works (Approach B)

This is the novel analysis step and deserves a detailed explanation.

### Polar unwrapping

`cv2.warpPolar` remaps the circular bell image into a rectangle where rows = angle
(0–360°) and columns = radius. This "unrolls" the bell ring so that the margin
becomes a flat horizontal band:

```text
Original frame           After polar unwrap
                         0°  ──────────────────── 360°
   ┌───────────┐         │  inner  [margin] outer  │
   │     ○     │  ──►    │  ░░░░░░ █████████ ░░░░  │
   └───────────┘         └─────────────────────────┘
                                   ▲
                             strip we extract
```

We keep only the outer annular band (default: 75–105% of bell radius).
Setting the outer edge beyond 100% captures tissue **outside** the bell boundary —
this is where an expansion wave appears first when a rhopalium fires.

### Frame-to-frame difference per angle

For each consecutive frame pair, we compute the absolute pixel intensity change
in the margin strip, averaged radially:

```text
margin_diff[θ, t] = mean( |strip(t+1, θ) − strip(t, θ)| )
```

This 2D signal (angle × time) measures optical activity at each margin position
at each moment. It detects tissue deformation, texture change, and boundary
movement simultaneously — sensitive to events that occur before the macroscopic
contraction is visible.

### Body-frame rotation

The signal is rotated by −φ\_dye(t) so that angle 0 always points toward the
dye mark. This makes measurements from different frames directly comparable
regardless of slow animal rotation.

### Pulse detection

Summing across all angles gives total margin activity. Peaks = contraction events.

### Initiation detection

For each pulse peak t\*:

1. Compute per-angle resting baseline from the frames just before the pre-window
2. Subtract baseline → excess activity above noise floor
3. Smooth angularly (Gaussian, circular) → suppress pixel-level noise
4. Scan pre-window frames from earliest: the first frame where any angle exceeds
   25% of the peak activity level = initiation frame
5. argmax of that frame → initiation angle in body frame
6. Match to the nearest rhopalium from the calibration file

The result is a **space-time plot** (angle × time heatmap) that directly shows
the contraction wave spreading outward from the initiation rhopalium:

```text
angle
360°│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    │░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░  ← wave spreading
    │░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓░░░
    │░░░░░░░░░░░░▓▓░░░░░░░░░░░░░░░░  ← initiation point
    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  0°└─────────────────────────────► time → peak
```

---

## Requirements

### Hardware

- NVIDIA GPU with ≥ 6 GB VRAM (tested on RTX 4060 8 GB)
- ≥ 16 GB system RAM (32 GB recommended)
- CUDA 12.x driver installed

### Video format

- Top-view recording, camera fixed relative to the chamber
- Recommended: ≥ 30 fps (120 fps tested and preferred)
- Grayscale or colour; resolution ≥ 640 × 512

### Dye mark

- A small visible mark applied to the bell surface
- Fluorescent dye under UV illumination gives best contrast, but standard dye works
- Place the mark off-centre so it can define an orientation angle

---

## Installation

All commands are run in **PowerShell** from the project folder.

### 1. Clone or download this repository

```powershell
cd C:\Projects
git clone <repo-url> Jellyfish
cd Jellyfish
```

### 2. Run the setup script

```powershell
.\setup.ps1
```

Creates a virtual environment, installs all packages, downloads model weights (~400 MB),
and runs a GPU smoke test. After it finishes you should see:

```text
All checks passed. Environment is ready.
```

### 3. Set your video folder

Open [config.py](config.py) and change `VIDEO_DIR`:

```python
VIDEO_DIR = Path(r"C:\Users\YourName\Videos\Cassiopea")
```

---

## Step-by-step guide

### Prepare a test clip

```powershell
& "C:\path\to\ffmpeg.exe" -i "C:\path\to\recording.mp4" `
    -ss 00:00:00 -t 60 -c copy data\test_clip.mp4
```

---

### Step 1 — Segment the bell (SAM2)

Outlines the jellyfish bell in every frame. Also computes the bell contour
profile r(θ, t) on the fly, which is needed for Approach A.

```powershell
.\venv\Scripts\python.exe scripts\run_sam2.py --video data\test_clip.mp4
```

A window opens. **Click anywhere on the bell body**, then press **ENTER**.

**Outputs:**

- `outputs\test_clip_seg.csv` — bell centre (cx, cy) and radius per frame
- `outputs\test_clip_contour_radii.npy` — bell boundary profile r(θ, t)
- `outputs\test_clip_sam2_validation.png` — spot-check image

---

### Step 2 — Track the dye mark (CoTracker)

```powershell
.\venv\Scripts\python.exe scripts\cotracker_test.py --video data\test_clip.mp4
```

A window opens. **Click on the dye mark**, then press **ENTER**.

For a faster first run, use every 4th frame (30 fps effective):

```powershell
.\venv\Scripts\python.exe scripts\cotracker_test.py --stride 4
```

**Outputs:**

- `outputs\test_clip_track.csv` — dye mark position (x, y) and confidence per frame
- `outputs\test_clip_tracked.mp4` — video with dye mark overlaid

---

### Step 3 — Validate tracking

```powershell
.\venv\Scripts\python.exe scripts\validate_tracking.py
```

Generates `outputs\test_clip_annotated.mp4`. Check that the red dot (bell centre)
stays on the animal and the green dot (dye mark) follows the mark throughout.

---

### Step 4 — Calibrate rhopalium positions (one-time per animal)

Take a **high-resolution still photo** of the jellyfish with the dye mark visible.

```powershell
.\venv\Scripts\python.exe scripts\calibrate_rhopalia.py "C:\path\to\photo.jpg"
```

**Three-step process in the window:**

| Step | Action | Key |
| --- | --- | --- |
| 1 | Click the bell centre of mass | ENTER |
| 2 | Click the dye mark (0° reference) | ENTER |
| 3 | Click each rhopalium one by one | BACKSPACE to undo, ENTER when done |

**Outputs:**

- `calibration\photo_name.json` — rhopalium body-frame angles (permanent record)
- `calibration\photo_name_annotated.png` — labelled diagram

---

### Step 5 — Pulse initiation analysis (Approach B)

Detects contraction events and identifies which rhopalium initiated each pulse
using polar margin intensity differences.

```powershell
.\venv\Scripts\python.exe scripts\run_approach_b.py
```

Phase 1 computes and caches `_margin_diff.npy` (~2–3 min, done once).
All subsequent runs with different parameters reuse the cache and finish in seconds.

**Tunable parameters:**

| Parameter | Default | Effect |
| --- | --- | --- |
| `--inner-frac` | 0.75 | Inner edge of margin ring (fraction of bell radius). Decrease to include more of the bell interior. |
| `--outer-frac` | 1.05 | Outer edge. Values > 1.0 include tissue outside the bell where the expansion wave first appears. |
| `--pre-window` | 30 | Frames before peak to scan for initiation. Increase for slow pulses; decrease if pulses are rapid. |
| `--min-distance` | 0.42 s | Minimum time between detected pulses. Increase to suppress double-detections. |
| `--prominence` | 0.05 | Peak detection threshold (fraction of signal range). Increase to filter weak events. |
| `--recompute` | off | Force recomputation of margin_diff.npy. Only needed when changing inner-frac or outer-frac. |

**Outputs:**

- `outputs\test_clip_margin_diff.npy` — cached per-angle margin activity (N_frames × 360)
- `outputs\test_clip_initiation_b.csv` — per-pulse rhopalium assignments
- `outputs\test_clip_initiation_b_plot.png` — activity signal + firing histogram
- `outputs\test_clip_spacetime_pulse_b\pulse_000.png` — angle × time heatmap per pulse
- `outputs\test_clip_initiation_b_annotated.mp4` — full video with pulse labels

---

### Reading the initiation CSV

| Column | Meaning |
| --- | --- |
| `peak_frame` | Video frame at peak contraction |
| `timestamp_s` | Time in seconds |
| `init_angle_deg` | Body-frame angle of initiation site (0° = dye mark) |
| `rhopalium_id` | Matched rhopalium (0–15) |
| `angular_dist_deg` | Arc distance to matched rhopalium — values below 11.25° (half of 360°/16) indicate confident assignment |
| `signal_confident` | 1 = confident (strong signal AND angular_dist ≤ 11.25°), 0 = uncertain |

---

## Interpreting the space-time plot

Each `spacetime_pulse_b/pulse_NNN.png` shows a heatmap of margin activity
(angle on Y axis, time on X axis). Warm colours indicate high pixel change rate.

**What to look for:**

- A localised bright region appearing early in the pre-window at the initiation
  angle, then spreading to adjacent angles = clean initiation detection
- The cyan horizontal line marks the detected initiation angle; dashed lines
  mark rhopalium positions
- If the bright region is diffuse from the start, the initiation happened before
  the pre-window — try increasing `--pre-window`

---

## Known limitations

### Initiation detection accuracy

The polar margin approach works best with:

- Frame rate ≥ 60 fps (120 fps preferred)
- Visible texture at the bell margin (zooxanthellae pattern helps)
- Consistent camera position relative to the animal

Some pulses will still show `signal_confident = 0`, meaning the initiation
could not be reliably attributed to a single rhopalium. The `angular_dist_deg`
column quantifies the confidence; use only confident assignments for statistics.

### Long recordings

For recordings longer than a few minutes, process in temporal segments:

- SAM2 batches internally (200-frame windows, memory-bounded)
- `margin_diff.npy` is computed frame-by-frame (memory is not a constraint)
- Concatenate per-segment CSVs afterward for a full-recording analysis

---

## Output file reference

```text
outputs\
  <clip>_seg.csv                    bell centre + radius per frame  (SAM2)
  <clip>_contour_radii.npy          bell boundary r(θ,t)  (SAM2)
  <clip>_track.csv                  dye mark position per frame  (CoTracker)
  <clip>_margin_diff.npy            per-angle margin activity  (Approach B cache)
  <clip>_initiation_b.csv           per-pulse rhopalium assignments
  <clip>_annotated.mp4              validation video  (dye + centroid)
  <clip>_sam2_validation.png        SAM2 segmentation spot-check
  <clip>_initiation_b_plot.png      activity signal + firing histogram
  <clip>_initiation_b_annotated.mp4 full video with pulse labels
  <clip>_spacetime_pulse_b\         per-pulse angle × time heatmaps
    pulse_000.png
    pulse_001.png
    ...

calibration\
  <animal>.json                     rhopalium body-frame angles  (permanent)
  <animal>_annotated.png            labelled calibration diagram
```

---

## License notes

- **SAM2** (bell segmentation): Apache 2.0
- **CoTracker** (dye tracking): CC-BY-NC 4.0 — research use only. For commercial use, replace with TAPIR (Apache 2.0).
- **RAFT** (optical flow, via torchvision): BSD — included in the environment but not used in the primary pipeline

---

## Citation

Models used:

- SAM2: Ravi et al., Meta AI (2024)
- CoTracker: Karaev et al., Meta AI + Oxford VGG (2023)
